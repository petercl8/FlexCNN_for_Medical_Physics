import numpy as np
import torch
from skimage.transform import radon, resize
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from functions.helper.display_images import show_multiple_matched_tensors
from classes.dataset_resizing import (
    bilinear_resize_sino,
    crop_pad_sino,
)


def project_attenuation(atten_image, target_height, circle=False, theta=None):
    '''
    Project attenuation image to create attenuation sinogram on-the-fly.
    
    atten_image:    attenuation image as numpy array (H, W)
    target_height:  target vertical dimension for output sinogram (horizontal preserved)
    circle:         circle mode for radon transform
    theta:          projection angles (must match activity sinogram angular sampling)
    
    Returns: attenuation sinogram as numpy array (target_height, width_from_theta)
    '''
    # Ensure image is positive and numpy
    atten_image = np.clip(atten_image, 0, None)
    
    # Perform forward Radon transform (creates sinogram from image)
    # Output shape is (num_detector_elements, num_angles) where num_angles = len(theta)
    atten_sino = radon(
        atten_image,
        circle=circle,
        preserve_range=True,
        theta=theta
    )
    
    # Resize vertically only to target_height, preserve horizontal dimension
    # This ensures the attenuation sinogram has the same angular sampling as the activity sinogram
    current_height, current_width = atten_sino.shape
    if current_height != target_height:
        atten_sino = resize(
            atten_sino,
            output_shape=(target_height, current_width),
            order=1,  # linear interpolation (fast, sufficient quality)
            mode='edge',
            preserve_range=True,
            anti_aliasing=True
        )
    
    return atten_sino


def visualize_sinogram_alignment(
    paths,
    settings,
    num_examples=5,
    scale_num_examples=None,
    start_index=0,
    fig_size=3,
    cmap='inferno',
    circle=False,
    # Resizing / alignment options
    use_crop_pad=False,
    crop_vert_size=None,
    crop_target_width=None,
    crop_pool_size=2,
    crop_pad_type='sinogram',
    use_bilinear_resize=False,
    sino_size=None
):
    '''
    Visualize alignment between projected attenuation sinograms and activity sinograms.
    
    This function:
    1. Loads attenuation images and activity sinograms
    2. Projects attenuation images to sinograms using the Radon transform
    3. Optionally applies crop/pad or bilinear resizing (from dataset_resizing) to both sinogram types
    4. Scales both sinogram types so they have common counts/totals
    5. Displays activity and attenuation sinograms side-by-side (matched scales)
    6. Displays a normalized overlay for visual alignment inspection
    7. Prints atten_sino_scale factor for use in dataloaders/training (no return)
    
    paths:         paths dictionary containing atten_image_path and sino_path
    settings:      settings dictionary containing atten_image_scale and sino_scale
    num_examples:        number of examples to visualize
    scale_num_examples:  number of examples to use when estimating atten_sino_scale (defaults to num_examples)
    start_index:   index to start from in dataset
    fig_size:      figure size for display
    cmap:          colormap for display
    circle:        circle mode for radon transform
    use_crop_pad:  apply crop/pad/pool from dataset_resizing.crop_pad_sino
    crop_vert_size: vertical size to crop/pad to (required if use_crop_pad)
    crop_target_width: horizontal width target (required if use_crop_pad)
    crop_pool_size: pooling width for crop_pad_sino
    crop_pad_type: padding type for crop_pad_sino ('sinogram' or 'zeros')
    use_bilinear_resize: apply bilinear resize from dataset_resizing.bilinear_resize_sino
    sino_size:      target sinogram size if bilinear resize is used (square)
    '''
    # Get file paths from paths dictionary
    atten_image_path = paths['atten_image_path']
    activity_sino_path = paths['sino_path']
    
    # Check if files exist
    if not os.path.exists(atten_image_path):
        raise FileNotFoundError(f"Attenuation image file not found: {atten_image_path}")
    if not os.path.exists(activity_sino_path):
        raise FileNotFoundError(f"Activity sinogram file not found: {activity_sino_path}")
    
    # Load data (memory-mapped for efficiency)
    print(f"Loading attenuation images from: {atten_image_path}")
    atten_images = np.load(atten_image_path, mmap_mode='r')
    
    print(f"Loading activity sinograms from: {activity_sino_path}")
    activity_sinos = np.load(activity_sino_path, mmap_mode='r')
    
    # Get scale factors from settings
    atten_image_scale = settings['atten_image_scale']
    sino_scale = settings['sino_scale']
    
    # Determine how many examples to display vs. to use for scale estimation
    if scale_num_examples is None:
        scale_num_examples = num_examples

    end_index_view = min(start_index + num_examples, len(atten_images), len(activity_sinos))
    end_index_scale = min(start_index + scale_num_examples, len(atten_images), len(activity_sinos))
    actual_num_view = end_index_view - start_index
    actual_num_scale = end_index_scale - start_index
    
    print(f"\nProcessing {actual_num_scale} examples for scale estimation (indices {start_index} to {end_index_scale-1})")
    print(f"Displaying first {actual_num_view} examples for visualization")
    
    # Initialize lists for collecting tensors and scale factors
    activity_sino_scaled_list = []
    projected_atten_sino_list = []
    overlay_list = []
    atten_sino_scale_factors = []
    
    # ===== MAIN LOOP: Process each example =====
    for idx in range(start_index, end_index_scale):
        # --- Extract activity sinogram ---
        # Handle different array shapes (4D with channels, 3D without, etc.)
        if len(activity_sinos.shape) == 4:  # (N, C, H, W)
            activity_sino = activity_sinos[idx, 0, :, :].squeeze()
        elif len(activity_sinos.shape) == 3:  # (N, H, W)
            activity_sino = activity_sinos[idx, :, :].squeeze()
        else:
            activity_sino = activity_sinos[idx].squeeze()
        
        # --- Calculate projection angles from activity sinogram dimensions ---
        # The horizontal dimension of a sinogram equals the number of projection angles
        # This ensures attenuation sinogram has same angular sampling as activity sinogram
        num_angles = activity_sino.shape[1]
        theta = np.linspace(0, 180, num_angles, endpoint=False)
        
        # Target height for attenuation sinogram (must match activity sinogram)
        target_height = activity_sino.shape[0]
        
        # --- Project attenuation image to sinogram ---
        atten_img = atten_images[idx].squeeze() * atten_image_scale
        atten_sino = project_attenuation(atten_img, target_height, circle=circle, theta=theta)

        # --- Optional geometric alignment using dataset_resizing helpers ---
        # Convert to torch tensors with channel dimension for helper functions
        act_torch = torch.from_numpy(activity_sino).unsqueeze(0).float()
        atten_torch = torch.from_numpy(atten_sino).unsqueeze(0).float()

        # Apply crop/pad/pool if requested
        if use_crop_pad:
            if crop_vert_size is None or crop_target_width is None:
                raise ValueError("crop_vert_size and crop_target_width must be provided when use_crop_pad=True")
            act_torch, atten_torch = crop_pad_sino(
                act_torch,
                atten_torch,
                vert_size=crop_vert_size,
                target_width=crop_target_width,
                pool_size=crop_pool_size,
                pad_type=crop_pad_type,
            )

        # Apply bilinear resize if requested
        if use_bilinear_resize:
            if sino_size is None:
                raise ValueError("sino_size must be provided when use_bilinear_resize=True")
            act_torch, atten_torch = bilinear_resize_sino(
                act_torch,
                atten_torch,
                sino_size,
            )

        # Back to numpy for scaling/overlay
        activity_sino = act_torch.squeeze().cpu().numpy()
        atten_sino = atten_torch.squeeze().cpu().numpy()
        
        # --- Scale sinograms for matched visualization ---
        # Activity sinogram: scale by sino_scale factor from settings
        activity_sino_scaled = activity_sino * sino_scale
        
        # Calculate atten_sino_scale: ratio of total activity to total attenuation
        # This makes the scaled attenuation sinogram have same total counts as activity sinogram
        atten_sino_scale = activity_sino_scaled.sum() / (atten_sino.sum() + 1e-8)
        atten_sino_scale_factors.append(atten_sino_scale)
        
        # Attenuation sinogram: scale by atten_image_scale (dataset) and atten_sino_scale (computed)
        atten_sino_scaled = atten_sino * atten_image_scale * atten_sino_scale
        
        # --- Create normalized overlay for visual alignment inspection ---
        # Normalize each sinogram by its own total (unscaled) to make them comparable
        activity_sino_norm = activity_sino / (activity_sino.sum() + 1e-8)
        atten_sino_norm = atten_sino / (atten_sino.sum() + 1e-8)
        
        # Overlay: normalized activity + normalized attenuation
        # If sinograms are well-aligned, features should overlap clearly
        overlay = activity_sino_norm + atten_sino_norm
        
        # --- Convert to PyTorch tensors with batch and channel dimensions ---
        activity_sino_tensor = torch.from_numpy(activity_sino_scaled).unsqueeze(0).unsqueeze(0).float()
        atten_sino_tensor = torch.from_numpy(atten_sino_scaled).unsqueeze(0).unsqueeze(0).float()
        overlay_tensor = torch.from_numpy(overlay).unsqueeze(0).unsqueeze(0).float()
        
        # Append to lists only for the examples we intend to visualize
        if idx < end_index_view:
            activity_sino_scaled_list.append(activity_sino_tensor)
            projected_atten_sino_list.append(atten_sino_tensor)
            overlay_list.append(overlay_tensor)
    
    # ===== END MAIN LOOP =====
    
    # Concatenate all examples into batch tensors
    activity_sino_batch = torch.cat(activity_sino_scaled_list, dim=0)
    atten_sino_batch = torch.cat(projected_atten_sino_list, dim=0)
    overlay_batch = torch.cat(overlay_list, dim=0)
    
    # === DISPLAY 1: Activity vs Attenuation (scaled to match) ===
    print(f"\nDisplaying scaled alignment visualization:")
    print(f"Row 1: Activity sinograms (scaled by sino_scale)")
    print(f"Row 2: Projected attenuation sinograms (scaled by atten_image_scale * atten_sino_scale)")
    
    show_multiple_matched_tensors(
        activity_sino_batch,
        atten_sino_batch,
        cmap=cmap,
        fig_size=fig_size
    )
    
    # === DISPLAY 2: Normalized overlay (for alignment inspection) ===
    print(f"\nDisplaying normalized overlay visualization:")
    print(f"Overlay (normalized sum of activity and attenuation)")
    
    show_multiple_matched_tensors(
        overlay_batch,
        cmap=cmap,
        fig_size=fig_size
    )
    
    # === Print statistics for verification ===
    print(f"\nStatistics:")
    print(f"Activity sinogram (scaled) - min: {activity_sino_batch.min():.4f}, max: {activity_sino_batch.max():.4f}, mean: {activity_sino_batch.mean():.4f}")
    print(f"Attenuation sinogram (scaled) - min: {atten_sino_batch.min():.4f}, max: {atten_sino_batch.max():.4f}, mean: {atten_sino_batch.mean():.4f}")
    print(f"Overlay (normalized) - min: {overlay_batch.min():.4f}, max: {overlay_batch.max():.4f}, mean: {overlay_batch.mean():.4f}")
    
    # === Compute and print attenuation sinogram scale factor ===
    # This value can be used in dataloaders/training to scale attenuation sinograms correctly
    avg_atten_sino_scale = np.mean(atten_sino_scale_factors)
    print(f"\nAttenuation sinogram scale factor:")
    print(f"atten_sino_scale = {avg_atten_sino_scale:.6f}")
    print(f"(Use this value in settings['atten_sino_scale'] for training)")