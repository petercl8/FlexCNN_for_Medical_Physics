import numpy as np

def compute_reconstruction_scales(paths, sample_mode='full', sample_size=1000):
    """
    Compute scaling factors on the training set to make reconstructions quantitatively match ground truth.

    This should be run once per dataset; the training set is largest and avoids using ground-truth test data.

    Args:
        paths: paths dictionary from setup_paths() containing data file paths
        sample_mode: 'full' (use entire dataset) or 'even' (evenly spaced samples)
        sample_size: if sample_mode='even', number of samples to use

    Returns:
        dict with keys 'recon1_scale' and 'recon2_scale'

    Example:
        # After running setup_paths() in notebook:
        scales = compute_reconstruction_scales(paths, sample_mode='full')
        recon1_scale = scales['recon1_scale']
        recon2_scale = scales['recon2_scale']
    """
    image_path = paths['train_image_path']
    recon1_path = paths['train_recon1_path']
    recon2_path = paths['train_recon2_path']
    
    # Load arrays with memory mapping
    image_array = np.load(image_path, mmap_mode='r')
    
    # Compute image sum based on sampling mode
    if sample_mode == 'full':
        print(f"Computing scales from full dataset ({len(image_array)} images)...")
        image_sum = image_array.sum()
        sample_description = f"{len(image_array)} images (full dataset)"
    elif sample_mode == 'even':
        array_len = len(image_array)
        step = max(1, array_len // sample_size)
        indices = np.arange(0, array_len, step)
        print(f"Computing scales from {len(indices)} evenly sampled images (every {step}th image)...")
        image_sum = image_array[indices].sum()
        sample_description = f"{len(indices)} evenly sampled images"
    else:
        raise ValueError(f"sample_mode must be 'full' or 'even', got '{sample_mode}'")
    
    scales = {}
    
    # Compute recon1 scale
    if recon1_path is not None:
        recon1_array = np.load(recon1_path, mmap_mode='r')
        if sample_mode == 'full':
            recon1_sum = recon1_array.sum()
        else:
            recon1_sum = recon1_array[indices].sum()
        scales['recon1_scale'] = float(image_sum / recon1_sum)
    else:
        scales['recon1_scale'] = 1.0
    
    # Compute recon2 scale
    if recon2_path is not None:
        recon2_array = np.load(recon2_path, mmap_mode='r')
        if sample_mode == 'full':
            recon2_sum = recon2_array.sum()
        else:
            recon2_sum = recon2_array[indices].sum()
        scales['recon2_scale'] = float(image_sum / recon2_sum)
    else:
        scales['recon2_scale'] = 1.0
    
    print(f"\nScaling factors computed from {sample_description}:")
    if scales['recon1_scale'] != 1.0:
        print(f"  recon1_scale: {scales['recon1_scale']:.6f}")
    if scales['recon2_scale'] != 1.0:
        print(f"  recon2_scale: {scales['recon2_scale']:.6f}")
    
    return scales
