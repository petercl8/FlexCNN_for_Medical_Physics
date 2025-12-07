import numpy as np


def compute_reconstruction_scales(paths, dataset='train', sample_mode='full', sample_size=1000, ratio_cap_multiple=None):
    """
    Compute scaling factors (mean and std) to match reconstructions to ground truth.

    Uses per-example sums so ratios and spread are visible. Optionally caps ratios at
    (ratio_cap_multiple * uncapped median ratio) to blunt near-zero denominators.
    Reports both uncapped and capped mean/std so the impact of capping is visible.

    Args:
        paths: paths dictionary from setup_paths() containing data file paths
        dataset: 'train' (default) or 'test'
        sample_mode: 'full' (use entire dataset) or 'even' (evenly spaced samples)
        sample_size: if sample_mode='even', number of samples to use
        ratio_cap_multiple: float or None. If set, ratios are capped at
            ratio_cap_multiple * (uncapped median ratio) before computing capped mean/std.

    Returns:
        dict with keys recon1_scale, recon2_scale, recon1_scale_std, recon2_scale_std,
        recon1_scale_uncapped, recon2_scale_uncapped, recon1_scale_uncapped_std,
        recon2_scale_uncapped_std

    Example:
        scales = compute_reconstruction_scales(paths, dataset='train', sample_mode='full', ratio_cap_multiple=5)
        recon1_scale = scales['recon1_scale']
        recon2_scale = scales['recon2_scale']
    """

    def _per_example_sums(array, indices):
        sampled = array if indices is None else array[indices]
        flat = sampled.reshape(len(sampled), -1)
        return flat.sum(axis=1)

    def _compute_scale(image_sums, recon_sums):
        valid_mask = recon_sums > 0
        if not valid_mask.any():
            return 1.0, 0.0, 1.0, 0.0, 0, 0, None, None

        ratios = image_sums[valid_mask] / recon_sums[valid_mask]
        uncapped_mean = float(ratios.mean())
        uncapped_std = float(ratios.std())
        uncapped_median = float(np.median(ratios))
        cap_value = None if ratio_cap_multiple is None else uncapped_median * ratio_cap_multiple

        if cap_value is None:
            capped_ratios = ratios
            capped_count = 0
        else:
            capped_ratios = np.minimum(ratios, cap_value)
            capped_count = int((ratios > cap_value).sum())

        capped_mean = float(capped_ratios.mean())
        capped_std = float(capped_ratios.std())
        valid_count = int(valid_mask.sum())
        return capped_mean, capped_std, uncapped_mean, uncapped_std, valid_count, capped_count, cap_value, uncapped_median

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
        recon1_scale, recon1_std, recon1_uncapped, recon1_uncapped_std, recon1_valid, recon1_capped, recon1_cap, recon1_median = _compute_scale(image_sums, recon1_sums)
        scales['recon1_scale'] = recon1_scale
        scales['recon1_scale_std'] = recon1_std
        scales['recon1_scale_uncapped'] = recon1_uncapped
        scales['recon1_scale_uncapped_std'] = recon1_uncapped_std
    else:
        recon1_valid = 0
        recon1_capped = 0
        recon1_cap = None
        recon1_median = None
        scales['recon1_scale'] = 1.0
        scales['recon1_scale_std'] = 0.0
        scales['recon1_scale_uncapped'] = 1.0
        scales['recon1_scale_uncapped_std'] = 0.0

    if recon2_path is not None:
        recon2_array = np.load(recon2_path, mmap_mode='r')
        recon2_sums = _per_example_sums(recon2_array, indices)
        recon2_scale, recon2_std, recon2_uncapped, recon2_uncapped_std, recon2_valid, recon2_capped, recon2_cap, recon2_median = _compute_scale(image_sums, recon2_sums)
        scales['recon2_scale'] = recon2_scale
        scales['recon2_scale_std'] = recon2_std
        scales['recon2_scale_uncapped'] = recon2_uncapped
        scales['recon2_scale_uncapped_std'] = recon2_uncapped_std
    else:
        recon2_valid = 0
        recon2_capped = 0
        recon2_cap = None
        recon2_median = None
        scales['recon2_scale'] = 1.0
        scales['recon2_scale_std'] = 0.0
        scales['recon2_scale_uncapped'] = 1.0
        scales['recon2_scale_uncapped_std'] = 0.0

    print(f"\nScaling factors computed from {sample_description}:")
    if scales['recon1_scale'] != 1.0:
        cap_msg = f", cap={recon1_cap:.6f} (median-based)" if recon1_cap is not None else ""
        print(
            f"  recon1_scale: mean={scales['recon1_scale']:.6f}, std={scales['recon1_scale_std']:.6f}, "
            f"uncapped_mean={scales['recon1_scale_uncapped']:.6f}, uncapped_std={scales['recon1_scale_uncapped_std']:.6f}, "
            f"valid={recon1_valid}, capped={recon1_capped}{cap_msg}"
        )
    if scales['recon2_scale'] != 1.0:
        cap_msg = f", cap={recon2_cap:.6f} (median-based)" if recon2_cap is not None else ""
        print(
            f"  recon2_scale: mean={scales['recon2_scale']:.6f}, std={scales['recon2_scale_std']:.6f}, "
            f"uncapped_mean={scales['recon2_scale_uncapped']:.6f}, uncapped_std={scales['recon2_scale_uncapped_std']:.6f}, "
            f"valid={recon2_valid}, capped={recon2_capped}{cap_msg}"
        )

    return scales
