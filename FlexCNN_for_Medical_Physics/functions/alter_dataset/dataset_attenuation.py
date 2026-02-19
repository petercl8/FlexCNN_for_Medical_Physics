import numpy as np
from numpy.lib.format import open_memmap
import torch
from skimage.transform import radon, resize
import os
import sys

from ..helper.display_images import show_multiple_matched_tensors, show_multiple_unmatched_tensors
from ...classes.dataset_resizing import resize_sino_data


def project_attenuation(atten_image, sino_height, circle=False, theta=None):
    '''
    Project attenuation image to create attenuation sinogram on-the-fly.
    
    atten_image:    attenuation image as numpy array (H, W)
    sino_height:  target vertical dimension for output sinogram (horizontal preserved)
    circle:         circle mode for radon transform
    theta:          projection angles (must match activity sinogram angular sampling)
    
    Returns: attenuation sinogram as numpy array (sino_height, width_from_theta)
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
    
    # Resize vertically only to sino_height, preserve horizontal dimension
    # This ensures the attenuation sinogram has the same angular sampling as the activity sinogram
    current_height, current_width = atten_sino.shape
    if current_height != sino_height:
        atten_sino = resize(
            atten_sino,
            output_shape=(sino_height, current_width),
            order=1,  # linear interpolation (fast, sufficient quality)
            mode='edge',
            preserve_range=True,
            anti_aliasing=True
        )
    
    return atten_sino

def generate_attenuation_sinogram(
    atten_img,
    sino_height=382,
    sino_width=513,
    circle=False,
    theta_type='symmetrical', # Set to 'speed' to match activity sinogram angular sampling after pooling
                        # Set to 'symmetrical' to match sampling before pooling.
    atten_creation_pool_size=2, # Only used if theta_type is 'speed'
):
    '''
    Generate attenuation sinogram for a given index by projecting the corresponding
    attenuation image to match the activity sinogram dimensions and angular sampling.   
    '''   
    # Ensure image is positive and numpy
    atten_image = np.clip(atten_img, 0, None)

    # Calculate theta from activity sinogram width (and pool size) if needed
    if theta_type == 'speed':
        num_angles = int(sino_width/atten_creation_pool_size)
        theta = np.linspace(0, 180, num_angles, endpoint=False)
    else:
        num_angles = sino_width
        theta = np.linspace(0, 180, num_angles, endpoint=False)

    # Project attenuation
    atten_sino = project_attenuation(
        atten_img, 
        sino_height, 
        circle=circle, 
        theta=theta, 
        )

    return atten_sino

def precompute_atten_sinos(
    dataset_path,
    atten_image_fileName,
    atten_sino_fileName,
    sino_height=382,
    sino_width=513,
    theta_type='symmetrical',
    atten_creation_pool_size=2,
    circle=False,
):
    """
    Precompute attenuation sinograms to disk using memmap-based writes.

    Args:
        dataset_path (str): Absolute path to dataset directory.
        atten_image_fileName (str): File name of attenuation images (.npy). Shape expected (N, H, W) or (N, 1, H, W).
        atten_sino_fileName (str): Output file name for attenuation sinograms (.npy). Saved under data_dirName.
        sino_height (int): Target sinogram height (detector elements).
        sino_width (int): Target sinogram width (number of angles).
        theta_type (str): 'symmetrical' or 'speed' (passed to generate_attenuation_sinogram).
        atten_creation_pool_size (int): Pool size used when theta_type == 'speed'.
        circle (bool): Radon circle mode.
    """

    from tqdm import trange

    atten_image_path = os.path.join(dataset_path, atten_image_fileName)
    atten_sino_path = os.path.join(dataset_path, atten_sino_fileName)

    atten_images = np.load(atten_image_path, mmap_mode='r')
    if atten_images.ndim == 4:
        atten_images = atten_images[:, 0]  # drop channel if present

    num_samples = atten_images.shape[0]

    # Enforce .npy output for compatibility with np.load(..., mmap_mode='r')
    if not atten_sino_path.endswith('.npy'):
        raise ValueError(f"atten_sino_fileName must end with .npy, got: {atten_sino_path}")

    # Create .npy memmap with channel dimension (N, 1, H, W)
    out = open_memmap(atten_sino_path, mode='w+', dtype=np.float32, shape=(num_samples, 1, sino_height, sino_width))

    for i in trange(num_samples, desc='Precomputing atten sinos'):
        atten_sino = generate_attenuation_sinogram(
            atten_images[i],
            sino_height=sino_height,
            sino_width=sino_width,
            circle=circle,
            theta_type=theta_type,
            atten_creation_pool_size=atten_creation_pool_size,
        )
        out[i, 0] = atten_sino

    # Ensure data is written
    out.flush()

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
    theta_type='symmetrical',  # 'speed' matches pooled angular sampling; 'symmetrical' matches pre-pooling sampling
    # Sinogram resize/pad options (applied to both activity and attenuation)
    sino_resize_type='bilinear',  # 'pool', 'bilinear', or None
    sino_pad_type='sinogram',  # 'sinogram' or 'zeros'
    sino_init_vert_cut=None,
    vert_pool_size=1,
    horiz_pool_size=1,
    bilinear_intermediate_size=180,
    # Target size (square)
    sino_size=288,
    # Attenuation generation options
    atten_creation_pool_size=2,  # Only used if theta_type is 'speed'
):
    """
    Visual alignment helper for activity and attenuation sinograms.

    Workflow:
    1. Load activity sinograms and attenuation images
    2. Generate attenuation sinograms on-the-fly
    3. Scale both to common magnitude
    4. Visualize scaled sinograms (before resize)
    5. Optionally resize/pad to target dimensions
    6. Visualize again (after resize)
    7. Print the attenuation sinogram scale factor

    Args:
        paths (dict): Must include 'train_act_sino_path' and 'train_atten_image_path'.
        settings (dict): Must include 'act_sino_scale' and 'atten_image_scale'.
        num_examples (int): Number of examples to display.
        scale_num_examples (int or None): Number of examples to estimate scale factor.
        start_index (int): Start index for sequential sampling when randomize is False.
        randomize (bool): Whether to sample indices randomly.
        random_seed (int or None): Seed for reproducible random sampling.
        fig_size (int or tuple): Figure size passed to display helpers.
        cmap (str): Colormap for display.
        circle (bool): Radon circle mode.
        theta_type (str): 'speed' or 'symmetrical' sampling for attenuation sinogram generation.
        sino_resize_type (str or None): 'pool', 'bilinear', or None. Applied to both sinograms.
        sino_pad_type (str): 'sinogram' or 'zeros'. Applied to both sinograms.
        sino_init_vert_cut (int or None): Pre-resize vertical crop height.
        vert_pool_size (int): Vertical pooling factor.
        horiz_pool_size (int): Horizontal pooling factor.
        bilinear_intermediate_size (int or None): Intermediate size for bilinear resize.
        sino_size (int): Target square sinogram size for resize_sino_data.
        atten_creation_pool_size (int): Pool size used when theta_type is 'speed'.

    Returns:
        None. Displays figures and prints the attenuation sinogram scale factor.
    """

    # ===== PHASE 1: Load Data and Determine Indices =====
    activity_sinos = np.load(paths['train_act_sino_path'], mmap_mode='r')
    atten_images = np.load(paths['train_atten_image_path'], mmap_mode='r')

    act_sino_scale = settings['act_sino_scale']

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

    # ===== PHASE 2: Generate and Scale Sinograms (Before Resize) =====
    activity_sino_scaled_list = []
    atten_sino_list = []

    for idx in scale_indices:
        # Extract raw sinograms
        activity_sino = activity_sinos[idx, 0, :, :].squeeze()
        atten_img = atten_images[idx].squeeze()
        
        # Generate attenuation sinogram to match activity dimensions
        sino_height_raw, sino_width_raw = activity_sino.shape
        atten_sino = generate_attenuation_sinogram(
            atten_img,
            sino_height_raw,
            sino_width_raw,
            circle=circle,
            theta_type=theta_type,
            atten_creation_pool_size=atten_creation_pool_size,
        )

        print(f"Example {idx}: Raw sinogram shape: {activity_sino.shape}")

        # Convert to torch [1, C, H, W] - need two unsqueezes to go from [H,W] to [1,1,H,W]
        activity_sino_torch = torch.from_numpy(activity_sino).unsqueeze(0).unsqueeze(0).float()
        atten_sino_torch = torch.from_numpy(atten_sino).unsqueeze(0).unsqueeze(0).float()

        # Scale activity sinogram
        activity_sino_scaled = activity_sino_torch * act_sino_scale

        # Store scaled sinograms before resizing
        activity_sino_scaled_list.append(activity_sino_scaled)
        atten_sino_list.append(atten_sino_torch)

    # Build batches from pre-resized sinograms
    activity_sino_batch_scaled_preResize = torch.cat(activity_sino_scaled_list, dim=0)
    atten_sino_batch_preResize = torch.cat(atten_sino_list, dim=0)

    # Calculate scale factor from raw (pre-resized) sinograms
    atten_sino_scale_factor = activity_sino_batch_scaled_preResize.mean().item() / atten_sino_batch_preResize.mean().item()
    print(f"atten_sino_scale_factor = {atten_sino_scale_factor}")

    # Scale attenuation sinograms
    atten_sino_batch_scaled_preResize = atten_sino_batch_preResize * atten_sino_scale_factor

    # Create overlay from scaled batches
    overlay_batch_preResize = activity_sino_batch_scaled_preResize + atten_sino_batch_scaled_preResize

    # ===== VISUALIZATION 1: Before Resize =====
    print("\n" + "="*60)
    print("VISUALIZATION 1: SCALED SINOGRAMS (BEFORE RESIZE)")
    print("="*60)
    
    view_indices_batch = np.array([np.where(scale_indices == idx)[0][0] for idx in view_indices])
    
    show_multiple_matched_tensors(
        activity_sino_batch_scaled_preResize[view_indices_batch], 
        atten_sino_batch_scaled_preResize[view_indices_batch], 
        cmap=cmap, fig_size=fig_size
    )
    show_multiple_matched_tensors(
        overlay_batch_preResize[view_indices_batch], 
        cmap=cmap, fig_size=fig_size
    )

    # ===== PHASE 3: Resize Sinograms =====
    print("\n" + "="*60)
    print("RESIZING SINOGRAMS TO TARGET DIMENSIONS")
    print("="*60)
    
    if sino_resize_type is not None:
        # Resize each example individually (resize_sino_data expects [C, H, W] not batches)
        activity_resized_list = []
        atten_resized_list = []
        
        for i in range(activity_sino_batch_scaled_preResize.shape[0]):
            act_resized, atten_resized = resize_sino_data(
                activity_sino_batch_scaled_preResize[i],
                atten_sino_batch_scaled_preResize[i],
                sino_size=sino_size,
                resize_sino=True,
                sino_resize_type=sino_resize_type,
                sino_pad_type=sino_pad_type,
                sino_init_vert_cut=sino_init_vert_cut,
                vert_pool_size=vert_pool_size,
                horiz_pool_size=horiz_pool_size,
                bilinear_intermediate_size=bilinear_intermediate_size,
            )
            activity_resized_list.append(act_resized)
            atten_resized_list.append(atten_resized)
        
        # Rebuild batches
        activity_sino_batch_scaled_postResize = torch.stack(activity_resized_list, dim=0)
        atten_sino_batch_scaled_postResize = torch.stack(atten_resized_list, dim=0)

        print(f"Resized to target dimensions: ({sino_size}, {sino_size})")

        # Create overlay from resized scaled batches
        overlay_batch_postResize = activity_sino_batch_scaled_postResize + atten_sino_batch_scaled_postResize

        # ===== VISUALIZATION 2: After Resize =====
        print("\n" + "="*60)
        print("VISUALIZATION 2: SCALED SINOGRAMS (AFTER RESIZE)")
        print("="*60)
        
        show_multiple_matched_tensors(
            activity_sino_batch_scaled_postResize[view_indices_batch], 
            atten_sino_batch_scaled_postResize[view_indices_batch], 
            cmap=cmap, fig_size=fig_size
        )
        show_multiple_matched_tensors(
            overlay_batch_postResize[view_indices_batch], 
            cmap=cmap, fig_size=fig_size
        )
    else:
        print("No resizing requested (all resize_type parameters are None)")