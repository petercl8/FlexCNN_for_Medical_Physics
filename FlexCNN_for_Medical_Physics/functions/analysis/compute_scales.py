import numpy as np

def compute_quantitative_reconstruction_scale(paths, dataset='train'):
    """
    Compute the global scaling factor to match reconstructions to ground truth.
    
    Computes the ratio of total ground truth activity to total reconstruction activity
    across the entire dataset. This single global scale is used for quantitative matching.
    
    Args:
        paths: paths dictionary from setup_paths() containing data file paths
        dataset: 'train' (default) or 'test'
    
    Returns:
        dict with keys 'recon1_scale' and 'recon2_scale'
    
    Example:
        scales = compute_reconstruction_scales_simple(paths, dataset='train')
        recon1_scale = scales['recon1_scale']
        recon2_scale = scales['recon2_scale']
    """
    # Select paths based on dataset
    if dataset == 'train':
        image_path = paths['train_image_path']
        recon1_path = paths['train_recon1_path']
        recon2_path = paths['train_recon2_path']
        sinogram_path = paths.get('train_sino_path', None)
    elif dataset == 'test':
        image_path = paths['test_image_path']
        recon1_path = paths['test_recon1_path']
        recon2_path = paths['test_recon2_path']
        sinogram_path = paths.get('test_sino_path', None)
    else:
        raise ValueError("dataset must be 'train' or 'test'")

    # Load ground truth images and compute total activity
    image_array = np.load(image_path, mmap_mode='r')
    total_image_activity = image_array.sum()

    scales = {}

    # Compute recon1 scale: ratio of ground truth to reconstruction activity
    if recon1_path is not None:
        recon1_array = np.load(recon1_path, mmap_mode='r')
        total_recon1_activity = recon1_array.sum()
        scales['recon1_scale'] = float(total_image_activity / total_recon1_activity)
    else:
        scales['recon1_scale'] = 1.0

    # Compute recon2 scale: ratio of ground truth to reconstruction activity
    if recon2_path is not None:
        recon2_array = np.load(recon2_path, mmap_mode='r')
        total_recon2_activity = recon2_array.sum()
        scales['recon2_scale'] = float(total_image_activity / total_recon2_activity)
    else:
        scales['recon2_scale'] = 1.0

    # Compute sinogram scale: match average of nonzero sinogram pixels to average of nonzero image pixels
    if sinogram_path is not None:
        sinogram_array = np.load(sinogram_path, mmap_mode='r')
        image_nonzero = image_array[image_array > 0]
        sinogram_nonzero = sinogram_array[sinogram_array > 0]
        avg_nonzero_image = float(image_nonzero.mean()) if image_nonzero.size > 0 else 0.0
        avg_nonzero_sinogram = float(sinogram_nonzero.mean()) if sinogram_nonzero.size > 0 else 1.0
        scales['sinogram_scale'] = avg_nonzero_image / avg_nonzero_sinogram
    else:
        scales['sinogram_scale'] = 1.0

    # Print results
    print(f"\nSimple scaling factors for {dataset} dataset:")
    if scales['recon1_scale'] != 1.0:
        print(f"  recon1_scale: {scales['recon1_scale']:.6f}")
    if scales['recon2_scale'] != 1.0:
        print(f"  recon2_scale: {scales['recon2_scale']:.6f}")
    if 'sinogram_scale' in scales and scales['sinogram_scale'] != 1.0:
        print(f"  sinogram_scale: {scales['sinogram_scale']:.6f}")

    return scales


def analyze_reconstruction_scale_distribution(paths, dataset='train', sample_mode='full', sample_size=1000, ratio_cap_multiple=None):
    """
    Analyze the distribution of per-example reconstruction scaling ratios.

    Uses per-example sums to compute individual scaling ratios, revealing spread and outliers.
    Optionally caps ratios at (ratio_cap_multiple * uncapped median ratio) to blunt near-zero
    denominators. Reports both uncapped and capped statistics to assess scaling consistency.

    Args:
        paths: paths dictionary from setup_paths() containing data file paths
        dataset: 'train' (default) or 'test'
        sample_mode: 'full' (use entire dataset) or 'even' (evenly spaced samples)
        sample_size: if sample_mode='even', number of samples to use
        ratio_cap_multiple: float or None. If set, ratios are capped at
            ratio_cap_multiple * (uncapped median ratio) before computing capped mean/std.

    Returns:
        dict with keys recon1_capped_mean, recon2_capped_mean, recon1_capped_std, recon2_capped_std,
        recon1_scale_uncapped, recon2_scale_uncapped, recon1_scale_uncapped_std,
        recon2_scale_uncapped_std, recon1_max_ratio, recon2_max_ratio,
        recon1_max_capped_ratio, recon2_max_capped_ratio

    Example:
        scales = analyze_reconstruction_scale_distribution(paths, dataset='train', sample_mode='full', ratio_cap_multiple=5)
        recon1_capped_mean = scales['recon1_capped_mean']
        recon2_capped_mean = scales['recon2_capped_mean']
    """

    def _per_example_sums(array, indices):
        sampled = array if indices is None else array[indices]
        flat = sampled.reshape(len(sampled), -1)
        return flat.sum(axis=1)

    def _compute_scale(image_sums, recon_sums):
        valid_mask = recon_sums > 0
        if not valid_mask.any():
            return 1.0, 0.0, 1.0, 0.0, 0, 0, None, None, None, None

        ratios = image_sums[valid_mask] / recon_sums[valid_mask]
        uncapped_mean = float(ratios.mean())
        uncapped_std = float(ratios.std())
        uncapped_median = float(np.median(ratios))
        uncapped_max = float(ratios.max())
        cap_value = None if ratio_cap_multiple is None else uncapped_median * ratio_cap_multiple

        if cap_value is None:
            capped_ratios = ratios
            capped_count = 0
        else:
            capped_ratios = np.minimum(ratios, cap_value)
            capped_count = int((ratios > cap_value).sum())

        capped_mean = float(capped_ratios.mean())
        capped_std = float(capped_ratios.std())
        capped_max = float(capped_ratios.max())
        valid_count = int(valid_mask.sum())
        return capped_mean, capped_std, uncapped_mean, uncapped_std, valid_count, capped_count, cap_value, uncapped_median, uncapped_max, capped_max

    if ratio_cap_multiple is not None and ratio_cap_multiple <= 0:
        raise ValueError("ratio_cap_multiple must be positive")

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

    image_array = np.load(image_path, mmap_mode='r')

    if sample_mode == 'full':
        indices = None
        sample_description = f"{len(image_array)} images (full dataset)"
        print(f"Computing scales from full dataset ({len(image_array)} images)...")
    elif sample_mode == 'even':
        array_len = len(image_array)
        step = max(1, array_len // sample_size)
        indices = np.arange(0, array_len, step)
        sample_description = f"{len(indices)} evenly sampled images"
        print(f"Computing scales from {len(indices)} evenly sampled images (every {step}th image)...")
    else:
        raise ValueError(f"sample_mode must be 'full' or 'even', got '{sample_mode}'")

    image_sums = _per_example_sums(image_array, indices)
    scales = {}

    if recon1_path is not None:
        recon1_array = np.load(recon1_path, mmap_mode='r')
        recon1_sums = _per_example_sums(recon1_array, indices)
        recon1_capped_mean, recon1_capped_std, recon1_uncapped, recon1_uncapped_std, recon1_valid, recon1_num_capped, recon1_calculated_cap_value, recon1_median, recon1_max_ratio, recon1_max_capped_ratio = _compute_scale(image_sums, recon1_sums)
        scales['recon1_capped_mean'] = recon1_capped_mean
        scales['recon1_capped_std'] = recon1_capped_std
        scales['recon1_scale_uncapped'] = recon1_uncapped
        scales['recon1_scale_uncapped_std'] = recon1_uncapped_std
        scales['recon1_max_ratio'] = recon1_max_ratio
        scales['recon1_max_capped_ratio'] = recon1_max_capped_ratio
    else:
        recon1_valid = 0
        recon1_num_capped = 0
        recon1_calculated_cap_value = None
        recon1_median = None
        recon1_max_ratio = None
        recon1_max_capped_ratio = None
        scales['recon1_capped_mean'] = 1.0
        scales['recon1_capped_std'] = 0.0
        scales['recon1_scale_uncapped'] = 1.0
        scales['recon1_scale_uncapped_std'] = 0.0
        scales['recon1_max_ratio'] = 1.0
        scales['recon1_max_capped_ratio'] = 1.0

    if recon2_path is not None:
        recon2_array = np.load(recon2_path, mmap_mode='r')
        recon2_sums = _per_example_sums(recon2_array, indices)
        recon2_capped_mean, recon2_capped_std, recon2_uncapped, recon2_uncapped_std, recon2_valid, recon2_num_capped, recon2_calculated_cap_value, recon2_median, recon2_max_ratio, recon2_max_capped_ratio = _compute_scale(image_sums, recon2_sums)
        scales['recon2_capped_mean'] = recon2_capped_mean
        scales['recon2_capped_std'] = recon2_capped_std
        scales['recon2_scale_uncapped'] = recon2_uncapped
        scales['recon2_scale_uncapped_std'] = recon2_uncapped_std
        scales['recon2_max_ratio'] = recon2_max_ratio
        scales['recon2_max_capped_ratio'] = recon2_max_capped_ratio
    else:
        recon2_valid = 0
        recon2_num_capped = 0
        recon2_calculated_cap_value = None
        recon2_median = None
        recon2_max_ratio = None
        recon2_max_capped_ratio = None
        scales['recon2_capped_mean'] = 1.0
        scales['recon2_capped_std'] = 0.0
        scales['recon2_scale_uncapped'] = 1.0
        scales['recon2_scale_uncapped_std'] = 0.0
        scales['recon2_max_ratio'] = 1.0
        scales['recon2_max_capped_ratio'] = 1.0

    print(f"\nScaling factors computed from {sample_description}:")
    if scales['recon1_capped_mean'] != 1.0:
        cap_msg = f", calculated_cap_value={recon1_calculated_cap_value:.6f} (median-based)" if recon1_calculated_cap_value is not None else ""
        print(
            f"  recon1_scale: capped_mean={scales['recon1_capped_mean']:.6f}, capped_std={scales['recon1_capped_std']:.6f}, "
            f"uncapped_mean={scales['recon1_scale_uncapped']:.6f}, uncapped_std={scales['recon1_scale_uncapped_std']:.6f}, "
            f"max_ratio={scales['recon1_max_ratio']:.6f}, max_capped_ratio={scales['recon1_max_capped_ratio']:.6f}, "
            f"valid={recon1_valid}, num_capped={recon1_num_capped}{cap_msg}"
        )
    if scales['recon2_capped_mean'] != 1.0:
        cap_msg = f", calculated_cap_value={recon2_calculated_cap_value:.6f} (median-based)" if recon2_calculated_cap_value is not None else ""
        print(
            f"  recon2_scale: capped_mean={scales['recon2_capped_mean']:.6f}, capped_std={scales['recon2_capped_std']:.6f}, "
            f"uncapped_mean={scales['recon2_scale_uncapped']:.6f}, uncapped_std={scales['recon2_scale_uncapped_std']:.6f}, "
            f"max_ratio={scales['recon2_max_ratio']:.6f}, max_capped_ratio={scales['recon2_max_capped_ratio']:.6f}, "
            f"valid={recon2_valid}, num_capped={recon2_num_capped}{cap_msg}"
        )

    return scales


def compute_average_activity_per_image(paths, dataset='train'):
    """
    Compute the average total activity per image across the dataset.
    
    Calculates the mean of per-image sums (total activity per image) to provide
    a typical scale for the data. Useful for setting initial learned scale ranges.
    
    Args:
        paths: paths dictionary from setup_paths() containing data file paths
        dataset: 'train' (default) or 'test'
    
    Returns:
        float: mean activity per image
    
    Example:
        avg_activity = compute_average_activity_per_image(paths, dataset='train')
        print(f"Average activity per image: {avg_activity:.6f}")
    """
    # Select paths based on dataset
    if dataset == 'train':
        image_path = paths['train_image_path']
    elif dataset == 'test':
        image_path = paths['test_image_path']
    else:
        raise ValueError("dataset must be 'train' or 'test'")
    
    # Load ground truth images
    image_array = np.load(image_path, mmap_mode='r')
    
    # Compute per-image activity sums
    flat = image_array.reshape(len(image_array), -1)
    per_image_sums = flat.sum(axis=1)
    
    # Compute average
    avg_activity = float(per_image_sums.mean())
    std_activity = float(per_image_sums.std())
    min_activity = float(per_image_sums.min())
    max_activity = float(per_image_sums.max())
    
    # Print results
    print(f"\nActivity statistics for {dataset} dataset ({len(image_array)} images):")
    print(f"  Average activity per image: {avg_activity:.6f}")
    print(f"  Std dev activity per image: {std_activity:.6f}")
    print(f"  Min activity per image: {min_activity:.6f}")
    print(f"  Max activity per image: {max_activity:.6f}")
    
    return avg_activity