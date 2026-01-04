import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from FlexCNN_for_Medical_Physics.classes.dataset_classes import NpArrayDataSet
from FlexCNN_for_Medical_Physics.functions.helper.reconstruction_projection import reconstruct
from FlexCNN_for_Medical_Physics.functions.helper.display_images import show_multiple_matched_tensors


def sort_DataSet(
    config,
    load_image_path,
    load_sino_path,
    save_image_path,
    save_sino_path,
    max_save_index,
    metric_function,
    threshold,
    threshold_min_max,
    num_examples=-1,
    visualize=False,
    sino_scale=1
):
    """
    Filter and save subsets of image/sinogram pairs based on a reconstruction metric.

    Parameters
    ----------
    config : dict
        Must contain: 'train_SI', 'SI_fixedScale', 'SI_fixedScale', 'SI_normalize', 'IS_normalize',
        'image_size', 'sino_size', 'image_channels', 'sino_channels'.
    load_image_path : str
        Path to source image .npy file.
    load_sino_path : str
        Path to source sinogram .npy file.
    save_image_path : str
        Path to output memmap file for filtered images (written float32).
    save_sino_path : str
        Path to output memmap file for filtered sinograms (written float32).
    max_save_index : int
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
    save_sino_array : np.memmap
        Memmapped sinogram array (capacity max_save_index).
    save_image_array : np.memmap
        Memmapped image array (capacity max_save_index).

    Notes
    -----
    - Uses batch_size=1 for simplicity.
    - Reconstruction uses FBP via `reconstruct()` to compute metric baseline.
    - All saved arrays are float32, channel-first.
    - No early stop; if more items pass than capacity, excess are ignored once full.
    - Visualization invokes matplotlib; can slow execution.
    """
    train_SI = config['train_SI']
    scale = config['SI_fixedScale'] if train_SI else config['SI_fixedScale']

    dataloader = DataLoader(
        NpArrayDataSet(image_path=load_image_path, sino_path=load_sino_path, config=config, settings=settings, num_examples=num_examples),
        batch_size=1,
        shuffle=True,
        pin_memory=True,
    )

    first = True
    saved_idx = 0

    for sino_ground, sino_ground_scaled, image_ground, image_ground_scaled in iter(dataloader):
        if first:
            save_image_array_shape = (
                max_save_index,
                image_ground_scaled.shape[1],
                image_ground_scaled.shape[2],
                image_ground_scaled.shape[3]
            )
            save_sino_array_shape = (
                max_save_index,
                sino_ground_scaled.shape[1],
                sino_ground_scaled.shape[2],
                sino_ground_scaled.shape[3]
            )
            print('save_image_array_shape: ', save_image_array_shape)
            print('save_sino_array_shape: ', save_sino_array_shape)

            save_image_array = np.lib.format.open_memmap(
                save_image_path, mode='w+', shape=save_image_array_shape, dtype=np.float32
            )
            save_sino_array = np.lib.format.open_memmap(
                save_sino_path, mode='w+', shape=save_sino_array_shape, dtype=np.float32
            )
            first = False

        # FBP reconstruction for metric comparison
        FBP_output = reconstruct(
            sino_ground_scaled,
            config['image_size'],
            config['SI_normalize'],
            config['SI_fixedScale'],
            recon_type='FBP'
        )

        image_metric = metric_function(image_ground_scaled, FBP_output)

        if threshold_min_max == 'min':
            keep = image_metric > threshold
        else:
            keep = image_metric < threshold

        if keep and saved_idx < max_save_index:
            save_sino_array[saved_idx] = sino_ground_scaled.cpu().numpy()
            save_image_array[saved_idx] = image_ground_scaled.cpu().numpy()
            saved_idx += 1
            print('Current index (for next image): ', saved_idx)

        if visualize:
            print('==================================')
            print('Image Metric: ', image_metric)
            print('Threshold: ', threshold)
            print('Keep?: ', keep)
            print('Current index (for next image): ', saved_idx)
            print('Saved Arrays:')
            print('image_ground_scaled / FBP_output / sino_ground_scaled')
            show_multiple_matched_tensors(image_ground_scaled, FBP_output)
            show_multiple_matched_tensors(sino_ground_scaled)
            show_multiple_matched_tensors(torch.from_numpy(save_sino_array[0: min(saved_idx, 9)]))
            show_multiple_matched_tensors(torch.from_numpy(save_image_array[0: min(saved_idx, 9)]))

    return save_sino_array, save_image_array