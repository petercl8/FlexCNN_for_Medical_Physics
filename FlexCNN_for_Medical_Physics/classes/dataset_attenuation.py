import numpy as np
import torch
from skimage.transform import radon, resize
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from functions.helper.display_images import show_multiple_matched_tensors


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
    start_index=0,
    fig_size=3,
    cmap='inferno',
    circle=False
):
    '''
    Visualize alignment between projected attenuation sinograms and activity sinograms.
    
    This function:
    1. Loads attenuation images and activity sinograms
    2. Projects attenuation images to sinograms using the Radon transform
    3. Scales both sinogram types so they have common counts/totals
    4. Displays activity and attenuation sinograms side-by-side (matched scales)
    5. Displays a normalized overlay for visual alignment inspection
    6. Returns atten_sino_scale factor for use in dataloaders/training
    
    paths:         paths dictionary containing atten_image_path and sino_path
    settings:      settings dictionary containing atten_image_scale and sino_scale
    num_examples:  number of examples to visualize
    start_index:   index to start from in dataset
    fig_size:      figure size for display
    cmap:          colormap for display
    circle:        circle mode for radon transform
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
    
    # Select examples
    end_index = min(start_index + num_examples, len(atten_images), len(activity_sinos))
    actual_num = end_index - start_index
    
    print(f"\nProcessing {actual_num} examples (indices {start_index} to {end_index-1})")
    
    # Initialize lists for collecting tensors and scale factors
    activity_sino_scaled_list = []
    projected_atten_sino_list = []
    overlay_list = []
    atten_sino_scale_factors = []
    
    # ===== MAIN LOOP: Process each example =====
    for idx in range(start_index, end_index):
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
        
        # Append to lists
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
    
    return activity_sino_batch, atten_sino_batch, overlay_batch, avg_atten_sino_scale