import numpy as np

def compute_reconstruction_scales(paths, dataset='train', sample_mode='full', sample_size=1000):
    """
    Compute scaling factors to make reconstructions quantitatively match ground truth.

    Use the training set by default (largest, avoids contaminating test ground truth). Optionally compute from
    the test set to compare scale drift.

    Args:
        paths: paths dictionary from setup_paths() containing data file paths
        dataset: 'train' (default) or 'test' – which set to sample
        sample_mode: 'full' (use entire dataset) or 'even' (evenly spaced samples)
        sample_size: if sample_mode='even', number of samples to use

    Returns:
        dict with keys 'recon1_scale' and 'recon2_scale'

    Example:
        # After running setup_paths() in notebook:
        scales = compute_reconstruction_scales(paths, dataset='train', sample_mode='full')
        recon1_scale = scales['recon1_scale']
        recon2_scale = scales['recon2_scale']
    """
    if dataset == 'train':
        image_path = paths['train_image_path']
        recon1_path = paths['train_recon1_path']
        recon2_path = paths['train_recon2_path']
    elif dataset == 'test':
        image_path = paths['test_image_path']
        recon1_path = paths['test_recon1_path']
        recon2_path = paths['test_recon2_path']
    else:
        raise ValueError("dataset must be 'train' or 'test'")
    
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
