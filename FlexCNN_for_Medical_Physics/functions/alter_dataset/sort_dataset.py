import os
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader

from FlexCNN_for_Medical_Physics.classes.dataset.dataset_classes import NpArrayDataSet
from FlexCNN_for_Medical_Physics.functions.helper.image_processing.reconstruction_projection import reconstruct
from FlexCNN_for_Medical_Physics.functions.helper.image_processing.display_images import show_multiple_matched_tensors


def sort_DataSet(
    config,
    load_dir,
    load_act_image_file,
    load_act_sino_file,
    load_act_recon1_file,
    load_act_recon2_file,
    load_atten_image_file,
    load_atten_sino_file,
    save_dir,
    save_act_image_file,
    save_act_sino_file,
    save_act_recon1_file,
    save_act_recon2_file,
    save_atten_image_file,
    save_atten_sino_file,
    max_save_size,
    metric_function,
    threshold,
    threshold_min_max,
    num_examples=-1,
    visualize=False,
    settings=None,
    data_opts=None,
    recon_variant=None,
):
    """
    Filter and save subsets of image/sinogram pairs based on a reconstruction metric.

    Parameters
    ----------
    config : dict
        Must contain: 'train_SI', 'SI_fixedScale', 'SI_fixedScale', 'SI_normalize', 'IS_normalize',
        'gen_image_size', 'gen_sino_size', 'gen_image_channels', 'gen_sino_channels_SI'/'gen_sino_channels_IS'.
    load_image_path : str
        Path to source image .npy file.
    load_sino_path : str
        Path to source sinogram .npy file.
    save_image_path : str
        Path to output memmap file for filtered images (written float32).
    save_sino_path : str
        Path to output memmap file for filtered sinograms (written float32).
    max_save_size : int
        Capacity (maximum number of examples to save).
    metric_function : callable
        Function taking (image_ground_scaled, FBP_output) and returning scalar metric.
    threshold : float
        Threshold value used for filtering.
    threshold_min_max : {'min','max'}
        If 'min': keep items where metric > threshold.
        If 'max': keep items where metric < threshold.
    num_examples : int
        Limit number of examples to iterate from dataset (-1 means all).
    visualize : bool
        If True, prints and displays intermediate tensors.

    Returns
    -------
    total_matches : int
        Number of examples that matched the threshold.
    stored_count : int
        Number of examples actually written to the output files.
    overflow : bool
        True if more examples matched than `max_save_size`.

    Notes
    -----
    - Uses batch_size=1 for simplicity.
    - Reconstruction uses FBP via `reconstruct()` to compute metric baseline.
    - All saved arrays are float32, channel-first.
    - No early stop; if more items pass than capacity, excess are ignored once full.
    - Visualization invokes matplotlib; can slow execution.
    """
    def build_path(directory, file_name):
        if file_name is None:
            return None
        return os.path.join(directory, file_name)

    load_act_image_path = build_path(load_dir, load_act_image_file)
    load_act_sino_path = build_path(load_dir, load_act_sino_file)
    load_act_recon1_path = build_path(load_dir, load_act_recon1_file)
    load_act_recon2_path = build_path(load_dir, load_act_recon2_file)
    load_atten_image_path = build_path(load_dir, load_atten_image_file)
    load_atten_sino_path = build_path(load_dir, load_atten_sino_file)

    save_act_image_path = build_path(save_dir, save_act_image_file)
    save_act_sino_path = build_path(save_dir, save_act_sino_file)
    save_act_recon1_path = build_path(save_dir, save_act_recon1_file)
    save_act_recon2_path = build_path(save_dir, save_act_recon2_file)
    save_atten_image_path = build_path(save_dir, save_atten_image_file)
    save_atten_sino_path = build_path(save_dir, save_atten_sino_file)

    # Conservative defaults for partitioning (do not inherit project-wide aggressive pooling)
    default_data_opts = {
        'sino_resize_type': 'pool',
        'sino_pad_type': 'zeros',
        'image_pad_type': 'zeros',
        'sino_init_vert_cut': None,
        'vert_pool_size': 1,
        'horiz_pool_size': 1,
        'bilinear_intermediate_size': None,
    }

    effective_data_opts = dict(default_data_opts)
    if data_opts:
        effective_data_opts.update(data_opts)
    # Merge into a local settings copy so other callers are unaffected
    settings_local = dict(settings) if settings is not None else {}
    settings_local['data_opts'] = effective_data_opts

    dataset = NpArrayDataSet(
        act_sino_path=load_act_sino_path,
        act_image_path=load_act_image_path,
        atten_image_path=load_atten_image_path,
        atten_sino_path=load_atten_sino_path,
        act_recon1_path=load_act_recon1_path,
        act_recon2_path=load_act_recon2_path,
        config=config,
        settings=settings_local,
        augment=(None, False),
        num_examples=num_examples,
    )

    # Use a simple collate_fn that returns the single-sample element directly.
    # This avoids the default collate attempting to stack/convert None entries.
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=False,
        collate_fn=lambda batch: batch[0],
    )

    first = True
    saved_idx = 0
    overflow = False
    save_act_image_array = None
    save_act_sino_array = None
    save_act_recon1_array = None
    save_act_recon2_array = None
    save_atten_image_array = None
    save_atten_sino_array = None

    for batch in iter(dataloader):
        # New dataset API returns nested tuples: (act_data, atten_data, recon_data)
        act_data, atten_data, recon_data = batch
        # act_data is (sino_scaled, image_scaled) with a batch dim
        sino_ground_scaled = act_data[0].squeeze(0)
        image_ground_scaled = act_data[1].squeeze(0)
        atten_sino_scaled = atten_data[0].squeeze(0) if atten_data[0] is not None else None
        atten_image_scaled = atten_data[1].squeeze(0) if atten_data[1] is not None else None
        recon1_scaled = recon_data[0].squeeze(0) if recon_data[0] is not None else None
        recon2_scaled = recon_data[1].squeeze(0) if recon_data[1] is not None else None

        if first:
            save_act_image_array_shape = (max_save_size, *tuple(image_ground_scaled.shape))
            save_act_sino_array_shape = (max_save_size, *tuple(sino_ground_scaled.shape))

            save_act_recon1_array_shape = (max_save_size, *tuple(recon1_scaled.shape)) if recon1_scaled is not None else None
            save_act_recon2_array_shape = (max_save_size, *tuple(recon2_scaled.shape)) if recon2_scaled is not None else None
            save_atten_image_array_shape = (max_save_size, *tuple(atten_image_scaled.shape)) if atten_image_scaled is not None else None
            save_atten_sino_array_shape = (max_save_size, *tuple(atten_sino_scaled.shape)) if atten_sino_scaled is not None else None

            print('save_act_image_array_shape: ', save_act_image_array_shape)
            print('save_act_sino_array_shape: ', save_act_sino_array_shape)
            if save_act_recon1_array_shape is not None:
                print('save_act_recon1_array_shape: ', save_act_recon1_array_shape)
            if save_act_recon2_array_shape is not None:
                print('save_act_recon2_array_shape: ', save_act_recon2_array_shape)
            if save_atten_image_array_shape is not None:
                print('save_atten_image_array_shape: ', save_atten_image_array_shape)
            if save_atten_sino_array_shape is not None:
                print('save_atten_sino_array_shape: ', save_atten_sino_array_shape)

            def _ensure_dir_and_shape(path, shape):
                if path is None:
                    return False
                d = os.path.dirname(path)
                if d and not os.path.exists(d):
                    os.makedirs(d, exist_ok=True)
                # validate shape
                if not isinstance(shape, tuple) or not all(isinstance(x, int) for x in shape):
                    raise RuntimeError(f"Invalid memmap shape {shape} for path {path}")
                return True

            def _open_memmap_overwrite(path, shape):
                if not _ensure_dir_and_shape(path, shape):
                    return None
                if os.path.exists(path) and os.path.isdir(path):
                    raise RuntimeError(f"Cannot create memmap: target path exists and is a directory: {path}")
                try:
                    print(f"Creating memmap at {path} with shape={shape}")
                    return np.lib.format.open_memmap(path, mode='w+', shape=shape, dtype=np.float32)
                except OSError as e:
                    info = {
                        'path': path,
                        'exists': os.path.exists(path),
                        'isfile': os.path.isfile(path),
                        'isdir': os.path.isdir(path),
                        'shape': shape,
                    }
                    raise RuntimeError(f"open_memmap failed for {path}: {e}; diagnostics: {info}") from e

            save_act_image_array = _open_memmap_overwrite(save_act_image_path, save_act_image_array_shape)
            save_act_sino_array = _open_memmap_overwrite(save_act_sino_path, save_act_sino_array_shape)
            if save_act_recon1_array_shape is not None and save_act_recon1_path is not None:
                save_act_recon1_array = _open_memmap_overwrite(save_act_recon1_path, save_act_recon1_array_shape)
            if save_act_recon2_array_shape is not None and save_act_recon2_path is not None:
                save_act_recon2_array = _open_memmap_overwrite(save_act_recon2_path, save_act_recon2_array_shape)
            if save_atten_image_array_shape is not None and save_atten_image_path is not None:
                save_atten_image_array = _open_memmap_overwrite(save_atten_image_path, save_atten_image_array_shape)
            if save_atten_sino_array_shape is not None and save_atten_sino_path is not None:
                save_atten_sino_array = _open_memmap_overwrite(save_atten_sino_path, save_atten_sino_array_shape)
            first = False

        # Reconstruction selection for IQA metric.
        # If user requested a specific recon_variant, attempt to load it and error on failure.
        if recon_variant is not None:
            try:
                recon_candidate = recon_data[recon_variant - 1] if isinstance(recon_data, (list, tuple)) else recon_data
                recon_used = recon_candidate.squeeze(0)
            except Exception as e:
                raise RuntimeError(f"sort_DataSet: requested recon_variant={recon_variant} but failed to load reconstruction: {e}")
        else:
            recon_used = reconstruct(
                sino_ground_scaled,
                config['gen_image_size'],
                config['SI_normalize'],
                config['SI_fixedScale'],
                recon_type='FBP',
            )

        # Ensure tensors have a batch dimension and channel dimension expected by metric functions
        def _ensure_min3d(tensor):
            if tensor is None:
                return None
            if not isinstance(tensor, torch.Tensor):
                try:
                    tensor = torch.from_numpy(tensor)
                except Exception:
                    raise RuntimeError('sort_DataSet: metric inputs must be torch tensors or numpy arrays convertible to tensors')
            # Expect images only: convert (H,W) -> (1,H,W); keep (C,H,W) as-is
            if tensor.dim() == 2:
                return tensor.unsqueeze(0)
            return tensor

        img_for_metric = _ensure_min3d(image_ground_scaled)
        recon_for_metric = _ensure_min3d(recon_used)

        image_metric = metric_function(img_for_metric, recon_for_metric)

        if threshold_min_max == 'min':
            keep = image_metric > threshold
        else:
            keep = image_metric < threshold

        if keep:
            if saved_idx >= max_save_size:
                overflow = True
            if overflow==False:
                save_act_sino_array[saved_idx] = sino_ground_scaled.cpu().numpy()
                save_act_image_array[saved_idx] = image_ground_scaled.cpu().numpy()
                if save_act_recon1_array is not None and recon1_scaled is not None:
                    save_act_recon1_array[saved_idx] = recon1_scaled.cpu().numpy()
                if save_act_recon2_array is not None and recon2_scaled is not None:
                    save_act_recon2_array[saved_idx] = recon2_scaled.cpu().numpy()
                if save_atten_image_array is not None and atten_image_scaled is not None:
                    save_atten_image_array[saved_idx] = atten_image_scaled.cpu().numpy()
                if save_atten_sino_array is not None and atten_sino_scaled is not None:
                    save_atten_sino_array[saved_idx] = atten_sino_scaled.cpu().numpy()
            saved_idx += 1
            print('Current index (for next image): ', saved_idx)
            print('Stored count: ', min(saved_idx, max_save_size))
            print('Overflow: ', overflow)


        if visualize:
            current_stored_count = min(saved_idx, max_save_size)
            print('==================================')
            print('Image Metric: ', image_metric)
            print('Threshold: ', threshold)
            print('Keep?: ', keep)
            print('Current index (for next image): ', saved_idx)
            print('Saved Arrays:')
            print('image_ground_scaled / recon_used / sino_ground_scaled')
            show_multiple_matched_tensors(image_ground_scaled, recon_used)
            show_multiple_matched_tensors(sino_ground_scaled)
            show_multiple_matched_tensors(torch.from_numpy(save_act_sino_array[0: min(current_stored_count, 9)]))
            show_multiple_matched_tensors(torch.from_numpy(save_act_image_array[0: min(current_stored_count, 9)]))
        # end loop

    # Simple reporting and cleanup: distinguish total matches from stored rows
    total_matches = saved_idx
    stored_count = min(total_matches, max_save_size)

    arrays = [
        ('save_act_image_array', save_act_image_array),
        ('save_act_sino_array', save_act_sino_array),
        ('save_act_recon1_array', save_act_recon1_array),
        ('save_act_recon2_array', save_act_recon2_array),
        ('save_atten_image_array', save_atten_image_array),
        ('save_atten_sino_array', save_atten_sino_array),
    ]

    def _finalize_arrays(arrays, path_map=None, remove_files=False):
        for name, arr in arrays:
            path = path_map.get(name) if path_map is not None else None
            try:
                if arr is not None and hasattr(arr, 'flush'):
                    try:
                        arr.flush()
                    except Exception:
                        pass
                if arr is not None and hasattr(arr, '_mmap') and getattr(arr, '_mmap') is not None:
                    try:
                        arr._mmap.close()
                    except Exception:
                        pass
            except Exception:
                pass

            if remove_files:
                try:
                    gc.collect()
                except Exception:
                    pass
                try:
                    if path is not None and os.path.exists(path):
                        os.remove(path)
                        print(f"Removed file: {path}")
                except Exception:
                    print(f"Failed to remove file: {path}")

    print(f"sort_DataSet: total_matches={total_matches}, stored_count={stored_count}, max_save_size={max_save_size}")

    def _cleanup_and_raise(message):
        _finalize_arrays(arrays, path_map=path_map, remove_files=True)
        raise RuntimeError(message)

    path_map = {
        'save_act_image_array': save_act_image_path,
        'save_act_sino_array': save_act_sino_path,
        'save_act_recon1_array': save_act_recon1_path,
        'save_act_recon2_array': save_act_recon2_path,
        'save_atten_image_array': save_atten_image_path,
        'save_atten_sino_array': save_atten_sino_path,
    }

    if overflow:
        _cleanup_and_raise(
            f"sort_DataSet: overflow detected; more than max_save_size={max_save_size} examples matched"
        )

    if stored_count != max_save_size:
        _cleanup_and_raise(
            f"sort_DataSet: stored_count={stored_count} does not match max_save_size={max_save_size}"
        )

    _finalize_arrays(arrays, path_map=path_map, remove_files=False)
    print("sort_DataSet: all created files flushed and closed")

    # Final shape print (no duplicate flush)
    for name, arr in arrays:
        try:
            if arr is None:
                print(f"{name}: None")
            else:
                print(f"{name}: shape={getattr(arr, 'shape', None)}")
        except Exception as e:
            print(f"{name}: unable to read shape: {e}")

    return total_matches, stored_count, overflow