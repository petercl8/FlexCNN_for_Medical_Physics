import numpy as np


def compute_average_activity_per_image(paths, dataset='train'):
    """
    Compute the average total activity per image across the dataset.
    
    Calculates the mean of per-image sums (total activity per image) to provide
    a typical scale for the data. Also computes global scaling factors to match
    reconstructions and attenuation images to ground truth.
    
    Args:
        paths: paths dictionary from setup_paths() containing data file paths
        dataset: 'train' (default) or 'test'
    
    Returns:
        dict with keys 'image', 'recon1', 'recon2', 'atten_image', 'scales', where:
            - 'image', 'recon1', 'recon2', 'atten_image' each contain:
                - 'avg': mean activity per image
                - 'std': standard deviation of activity per image
                - 'min': minimum activity per image
                - 'max': maximum activity per image
            - 'scales' contains:
                - 'recon1_scale': ratio of image to recon1 average activity
                - 'recon2_scale': ratio of image to recon2 average activity
                - 'atten_image_scale': ratio of image to atten_image average activity
    
    Example:
        stats = compute_average_activity_per_image(paths, dataset='train')
        avg_activity = stats['image']['avg']
        recon1_scale = stats['scales']['recon1_scale']
    """
    # Select paths based on dataset
    if dataset == 'train':
        image_path = paths['train_image_path']
        recon1_path = paths['train_recon1_path']
        recon2_path = paths['train_recon2_path']
        atten_image_path = paths['train_atten_image_path']
    elif dataset == 'test':
        image_path = paths['test_image_path']
        recon1_path = paths['test_recon1_path']
        recon2_path = paths['test_recon2_path']
        atten_image_path = paths['test_atten_image_path']
    else:
        raise ValueError("dataset must be 'train' or 'test'")
    
    def _compute_stats(array):
        """Helper function to compute statistics for an array."""
        flat = array.reshape(len(array), -1)
        per_image_sums = flat.sum(axis=1)
        return {
            'avg': float(per_image_sums.mean()),
            'std': float(per_image_sums.std()),
            'min': float(per_image_sums.min()),
            'max': float(per_image_sums.max())
        }
    
    # Load ground truth images and compute statistics
    image_array = np.load(image_path, mmap_mode='r')
    stats = {}
    stats['image'] = _compute_stats(image_array)
    
    # Compute recon1 statistics
    if recon1_path is not None:
        recon1_array = np.load(recon1_path, mmap_mode='r')
        stats['recon1'] = _compute_stats(recon1_array)
    else:
        stats['recon1'] = None
    
    # Compute recon2 statistics
    if recon2_path is not None:
        recon2_array = np.load(recon2_path, mmap_mode='r')
        stats['recon2'] = _compute_stats(recon2_array)
    else:
        stats['recon2'] = None
    
    # Compute atten_image statistics
    if atten_image_path is not None:
        atten_image_array = np.load(atten_image_path, mmap_mode='r')
        stats['atten_image'] = _compute_stats(atten_image_array)
    else:
        stats['atten_image'] = None
    
    # Compute scales (ratios of average activities)
    stats['scales'] = {}
    if stats['recon1'] is not None:
        stats['scales']['recon1_scale'] = float(stats['image']['avg'] / stats['recon1']['avg'])
    else:
        stats['scales']['recon1_scale'] = 1.0
    
    if stats['recon2'] is not None:
        stats['scales']['recon2_scale'] = float(stats['image']['avg'] / stats['recon2']['avg'])
    else:
        stats['scales']['recon2_scale'] = 1.0
    
    if stats['atten_image'] is not None:
        stats['scales']['atten_image_scale'] = float(stats['image']['avg'] / stats['atten_image']['avg'])
    else:
        stats['scales']['atten_image_scale'] = 1.0
    
    # Print results
    print(f"\nActivity statistics for {dataset} dataset ({len(image_array)} images):")
    print(f"  Image (ground truth):")
    print(f"    Average activity per image: {stats['image']['avg']:.6f}")
    print(f"    Std dev activity per image: {stats['image']['std']:.6f}")
    print(f"    Min activity per image: {stats['image']['min']:.6f}")
    print(f"    Max activity per image: {stats['image']['max']:.6f}")
    
    if stats['recon1'] is not None:
        print(f"  Recon1:")
        print(f"    Average activity per image: {stats['recon1']['avg']:.6f}")
        print(f"    Std dev activity per image: {stats['recon1']['std']:.6f}")
        print(f"    Min activity per image: {stats['recon1']['min']:.6f}")
        print(f"    Max activity per image: {stats['recon1']['max']:.6f}")
    
    if stats['recon2'] is not None:
        print(f"  Recon2:")
        print(f"    Average activity per image: {stats['recon2']['avg']:.6f}")
        print(f"    Std dev activity per image: {stats['recon2']['std']:.6f}")
        print(f"    Min activity per image: {stats['recon2']['min']:.6f}")
        print(f"    Max activity per image: {stats['recon2']['max']:.6f}")
    
    if stats['atten_image'] is not None:
        print(f"  Attenuation image:")
        print(f"    Average activity per image: {stats['atten_image']['avg']:.6f}")
        print(f"    Std dev activity per image: {stats['atten_image']['std']:.6f}")
        print(f"    Min activity per image: {stats['atten_image']['min']:.6f}")
        print(f"    Max activity per image: {stats['atten_image']['max']:.6f}")
    
    print(f"\n  Scaling factors (ratio of image to reconstruction/attenuation average activity):")
    if stats['scales']['recon1_scale'] != 1.0:
        print(f"    recon1_scale: {stats['scales']['recon1_scale']:.6f}")
    if stats['scales']['recon2_scale'] != 1.0:
        print(f"    recon2_scale: {stats['scales']['recon2_scale']:.6f}")
    if stats['scales']['atten_image_scale'] != 1.0:
        print(f"    atten_image_scale: {stats['scales']['atten_image_scale']:.6f}")
    
    return stats


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


def compute_sinogram_to_image_scale(paths, dataset='train', sample_number=None, clip_percentile_low=1, clip_percentile_high=99, scale_stat='median'):
    """
    Compute scaling factor to make sinograms comparable in magnitude to images.
    
    Computes the ratio of typical image pixel values to typical sinogram pixel values,
    making sinogram inputs roughly match image target scale for training.
    
    Args:
        paths: paths dictionary from setup_paths() containing data file paths
        dataset: 'train' (default) or 'test'
        sample_number: number of examples to sample for calculation (None = use all)
        clip_percentile_low: lower percentile for clipping outliers (default 1)
        clip_percentile_high: upper percentile for clipping outliers (default 99)
        scale_stat: 'mean' or 'median' for computing typical values (default 'median')
    
    Returns:
        float: sinogram_scale factor
    
    Example:
        sinogram_scale = compute_sinogram_to_image_scale(paths, dataset='train', sample_number=1000)
    """
    # Select paths based on dataset
    if dataset == 'train':
        image_path = paths['train_image_path']
        sinogram_path = paths.get('train_sino_path', None)
    elif dataset == 'test':
        image_path = paths['test_image_path']
        sinogram_path = paths.get('test_sino_path', None)
    else:
        raise ValueError("dataset must be 'train' or 'test'")

    if sinogram_path is None:
        print(f"\nNo sinogram path found for {dataset} dataset, returning scale=1.0")
        return 1.0

    # Load arrays
    image_array = np.load(image_path, mmap_mode='r')
    sinogram_array = np.load(sinogram_path, mmap_mode='r')

    # Optionally sample for calculation
    if sample_number is not None:
        total_examples = len(sinogram_array)
        rng = np.random.default_rng()
        indices = rng.choice(total_examples, size=min(sample_number, total_examples), replace=False)
        sampled_sinograms = np.stack([sinogram_array[i] for i in indices], axis=0)
        sampled_images = np.stack([image_array[i] for i in indices], axis=0)
        sinogram_nonzero = sampled_sinograms[sampled_sinograms > 0]
        image_nonzero = sampled_images[sampled_images > 0]
    else:
        sinogram_nonzero = sinogram_array[sinogram_array > 0]
        image_nonzero = image_array[image_array > 0]

    # Clip nonzero values before computing stat
    if image_nonzero.size > 0:
        image_low = np.percentile(image_nonzero, clip_percentile_low)
        image_high = np.percentile(image_nonzero, clip_percentile_high)
        image_clipped = np.clip(image_nonzero, image_low, image_high)
        if scale_stat == 'mean':
            stat_nonzero_image = float(np.mean(image_clipped))
        else:
            stat_nonzero_image = float(np.median(image_clipped))
    else:
        stat_nonzero_image = 0.0

    if sinogram_nonzero.size > 0:
        sino_low = np.percentile(sinogram_nonzero, clip_percentile_low)
        sino_high = np.percentile(sinogram_nonzero, clip_percentile_high)
        sinogram_clipped = np.clip(sinogram_nonzero, sino_low, sino_high)
        if scale_stat == 'mean':
            stat_nonzero_sinogram = float(np.mean(sinogram_clipped))
        else:
            stat_nonzero_sinogram = float(np.median(sinogram_clipped))
    else:
        stat_nonzero_sinogram = 1.0

    sinogram_scale = stat_nonzero_image / stat_nonzero_sinogram if stat_nonzero_sinogram != 0 else 1.0

    # Print results
    print(f"\nSinogram-to-image scaling factor for {dataset} dataset:")
    print(f"  sinogram_scale: {sinogram_scale:.6f}")
    print(f"  (stat={scale_stat}, clip={clip_percentile_low}-{clip_percentile_high}%, samples={'all' if sample_number is None else sample_number})")

    return sinogram_scale