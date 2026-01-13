import numpy as np
from numpy.lib.format import open_memmap
import torch
from skimage.transform import radon, resize
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from functions.helper.display_images import show_multiple_matched_tensors, show_multiple_unmatched_tensors
from classes.dataset_resizing import (
    bilinear_resize_sino,
    crop_pad_sino,
)


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
    atten_creation_pool_size=2, # Only used if theta_type is 'speed'
    atten_pool_size=2,
):
    '''
    Quick visual alignment helper: load sinograms, optionally resize/pad both activity and attenuation,
    scale to common totals, show matched pairs and overlay, print atten_sino_scale_factor.
    '''

    # Load Raw Data
    activity_sinos = np.load(paths['train_sino_path'], mmap_mode='r')
    atten_images = np.load(paths['train_atten_image_path'], mmap_mode='r')

    # Load Scales
    act_sino_scale = settings['act_sino_scale']
    atten_image_scale = settings['atten_image_scale']

    
    # Determine which examples to show
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
    
    # Initialize lists for collecting tensors and scale factors
    activity_sino_list = []
    atten_sino_list = []
    overlay_list = []
    
    # ===== MAIN LOOP: Process each example =====
    for idx in scale_indices:
        # Extract activity sinogram and attenuation image
        activity_sino = activity_sinos[idx, 0, :, :].squeeze()
        atten_img = atten_images[idx].squeeze()
        
        # Generate attenuation sinogram
        sino_width = activity_sino.shape[1]
        sino_height = activity_sino.shape[0]

        atten_sino = generate_attenuation_sinogram(
            atten_img,
            sino_height,
            sino_width,
            circle=circle,
            theta_type=theta_type,
            atten_creation_pool_size=atten_creation_pool_size,
        )

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

        # Scale activity sinogram
        activity_sino = activity_sino * act_sino_scale

        # Store Sinograms
        activity_sino_list.append(torch.from_numpy(activity_sino).unsqueeze(0).unsqueeze(0).float())
        atten_sino_list.append(torch.from_numpy(atten_sino).unsqueeze(0).unsqueeze(0).float())

        # Create Overlays
        activity_sino_norm = activity_sino / (activity_sino.sum() + 1e-8)
        atten_sino_norm = atten_sino / (atten_sino.sum() + 1e-8)
        overlay_list.append(torch.from_numpy(activity_sino_norm + atten_sino_norm).unsqueeze(0).unsqueeze(0).float())
        
    # Concatenate Lists into Batches
    activity_sino_batch = torch.cat(activity_sino_list, dim=0)
    atten_sino_batch = torch.cat(atten_sino_list, dim=0)
    overlay_batch = torch.cat(overlay_list, dim=0)

    # Calculate and store scale factors
    atten_sino_scale_factor = activity_sino_batch.mean().item() / atten_sino_batch.mean().item()
    print(f"atten_sino_scale_factor = {atten_sino_scale_factor}")

    # Scale attenuation sinograms for display
    atten_sino_batch = atten_sino_batch * atten_sino_scale_factor
    
    # Map view_indices (original dataset indices) to batch positions (0, 1, 2, ...)
    view_indices_batch = np.array([np.where(scale_indices == idx)[0][0] for idx in view_indices])

    show_multiple_matched_tensors(activity_sino_batch[view_indices_batch], atten_sino_batch[view_indices_batch], cmap=cmap, fig_size=fig_size)
    show_multiple_matched_tensors(overlay_batch[view_indices_batch], cmap=cmap, fig_size=fig_size)



def precompute_atten_sinos(
    project_dirPath,
    data_dirName,
    atten_image_fileName,
    atten_sino_fileName,
    sino_height=382,
    sino_width=512,
    theta_type='symmetrical',
    atten_creation_pool_size=2,
    circle=False,
):
    """
    Precompute attenuation sinograms to disk using memmap-based writes.

    Args:
        project_dirPath (str): Absolute path to project root.
        data_dirName (str): Dataset directory relative to project root.
        atten_image_fileName (str): File name of attenuation images (.npy). Shape expected (N, H, W) or (N, 1, H, W).
        atten_sino_fileName (str): Output file name for attenuation sinograms (.npy). Saved under data_dirName.
        sino_height (int): Target sinogram height (detector elements).
        sino_width (int): Target sinogram width (number of angles).
        theta_type (str): 'symmetrical' or 'speed' (passed to generate_attenuation_sinogram).
        atten_creation_pool_size (int): Pool size used when theta_type == 'speed'.
        circle (bool): Radon circle mode.
    """

    from tqdm import trange

    dataset_path = os.path.join(project_dirPath, data_dirName)
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