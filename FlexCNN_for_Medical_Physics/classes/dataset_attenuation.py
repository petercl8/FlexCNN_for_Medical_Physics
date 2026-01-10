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
    randomize=False,
    random_seed=None,
    fig_size=3,
    cmap='inferno',
    circle=False,
    theta_type='symmetrical', # Set to 'speed' to match activity sinogram angular sampling after pooling
                        # Set to 'symmetrical' to match sampling before pooling.
    # Activity resize/pad options
    act_resize_type='crop_pad',   # 'crop_pad', 'bilinear', or None
    act_pad_type='zeros', # 'sinoram' or 'zeros'
    act_vert_size=288,  # Target vertical size. Also used for horizontal size if bilinear resizing is selected.
    act_target_width=288,
    act_pool_size=2,
    # Attenuation resize/pad options
    atten_resize_type='crop_pad', # 'crop_pad', 'bilinear', or None
    atten_pad_type='zeros',
    atten_vert_size=288, # Target vertical size. Also used for horizontal size if bilinear resizing is selected.
    atten_target_width=288,
    atten_pool_size=2,
):
    '''
    Quick visual alignment helper: load sinograms, optionally resize/pad both activity and attenuation,
    scale to common totals, show matched pairs and overlay, print atten_sino_scale_factor.
    '''
    atten_images = np.load(paths['train_atten_image_path'], mmap_mode='r')
    activity_sinos = np.load(paths['train_sino_path'], mmap_mode='r')
    atten_image_scale = settings['atten_image_scale']
    sino_scale = settings['sino_scale']
    
    # Determine how many examples to display vs. use for scale estimation
    if scale_num_examples is None:
        scale_num_examples = num_examples

    total_examples = activity_sinos.shape[0]
    if randomize:
        rng = np.random.default_rng(random_seed)
        total_needed = max(scale_num_examples, num_examples)
        chosen = rng.choice(total_examples, size=total_needed, replace=False)
        scale_indices = chosen[:scale_num_examples]
        view_indices = chosen[:num_examples]
    else:
        end_index_view = min(start_index + num_examples, total_examples)
        end_index_scale = min(start_index + scale_num_examples, total_examples)
        scale_indices = np.arange(start_index, end_index_scale)
        view_indices = scale_indices[: end_index_view - start_index]
    view_indices_set = set(view_indices.tolist())
    
    # Initialize lists for collecting tensors and scale factors
    activity_sino_scaled_list = []
    projected_atten_sino_list = []
    overlay_list = []
    atten_sino_scale_factors = []
    
    # ===== MAIN LOOP: Process each example =====
    for idx in scale_indices:
        # Extract activity sinogram (N, C, H, W)
        activity_sino = activity_sinos[idx, 0, :, :].squeeze()
        
        # Calculate theta from activity sinogram width (and pool size) if needed
        if theta_type == 'speed':
            num_angles = int(activity_sino.shape[1]/act_pool_size)
            theta = np.linspace(0, 180, num_angles, endpoint=False)
        elif theta_type == 'symmetrical':
            num_angles = activity_sino.shape[1]
            theta = np.linspace(0, 180, num_angles, endpoint=False)

        # Target height = sino height, to get us in the right ballpark
        target_height = activity_sino.shape[0]
        
        # Project attenuation
        atten_img = atten_images[idx].squeeze() * atten_image_scale
        atten_sino = project_attenuation(atten_img, target_height, circle=circle, theta=theta)

        # Print shapes for debugging
        print(f"Example {idx} (before resize): activity_sino shape: {activity_sino.shape}, atten_sino shape: {atten_sino.shape}")

        # Resize/pad activity
        if act_resize_type == 'crop_pad':
            act_torch, _ = crop_pad_sino(
                torch.from_numpy(activity_sino).unsqueeze(0).float(), None,
                vert_size=act_vert_size,
                target_width=act_target_width,
                pool_size=act_pool_size,
                pad_type=act_pad_type,
            )
            activity_sino = act_torch.squeeze().cpu().numpy()
        elif act_resize_type == 'bilinear':
            act_torch, _ = bilinear_resize_sino(
                torch.from_numpy(activity_sino).unsqueeze(0).float(), None, act_vert_size
            )
            activity_sino = act_torch.squeeze().cpu().numpy()

        # Resize/pad attenuation
        if atten_resize_type == 'crop_pad':
            _, atten_torch = crop_pad_sino(
                None, torch.from_numpy(atten_sino).unsqueeze(0).float(),
                vert_size=atten_vert_size,
                target_width=atten_target_width,
                pool_size=atten_pool_size,
                pad_type=atten_pad_type,
            )
            atten_sino = atten_torch.squeeze().cpu().numpy()
        elif atten_resize_type == 'bilinear':
            _, atten_torch = bilinear_resize_sino(
                None, torch.from_numpy(atten_sino).unsqueeze(0).float(), atten_vert_size
            )
            atten_sino = atten_torch.squeeze().cpu().numpy()

        print(f"Example {idx} (after resize): activity_sino shape: {activity_sino.shape}, atten_sino shape: {atten_sino.shape}")

        # Scale sinograms and create overlay
        activity_sino_scaled = activity_sino * sino_scale
        atten_sino_scale_factor = activity_sino_scaled.sum() / (atten_sino.sum() + 1e-8)
        atten_sino_scale_factors.append(atten_sino_scale_factor)
        atten_sino_rescaled = atten_sino * atten_image_scale * atten_sino_scale_factor
        
        activity_sino_norm = activity_sino / (activity_sino.sum() + 1e-8)
        atten_sino_norm = atten_sino / (atten_sino.sum() + 1e-8)
        overlay = activity_sino_norm + atten_sino_norm
        
        # Store tensors for visualization examples only
        if idx in view_indices_set:
            activity_sino_scaled_list.append(torch.from_numpy(activity_sino_scaled).unsqueeze(0).unsqueeze(0).float())
            projected_atten_sino_list.append(torch.from_numpy(atten_sino_rescaled).unsqueeze(0).unsqueeze(0).float())
            overlay_list.append(torch.from_numpy(overlay).unsqueeze(0).unsqueeze(0).float())
    
    # Concatenate and display
    activity_sino_batch = torch.cat(activity_sino_scaled_list, dim=0)
    atten_sino_batch = torch.cat(projected_atten_sino_list, dim=0)
    overlay_batch = torch.cat(overlay_list, dim=0)
    
    # Print scale factor
    avg_atten_sino_scale_factor = np.mean(atten_sino_scale_factors)
    print(f"atten_sino_scale_factor = {avg_atten_sino_scale_factor:.6f}")


    show_multiple_matched_tensors(activity_sino_batch, atten_sino_batch, cmap=cmap, fig_size=fig_size)
    show_multiple_matched_tensors(overlay_batch, cmap=cmap, fig_size=fig_size)