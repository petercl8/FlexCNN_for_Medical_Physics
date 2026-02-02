"""
Helpers for evaluation data loading during tuning with fresh random batches.

Supports two evaluation modes:
- 'val': report on fresh validation batches (dataset cached, no augmentation)
- 'qa': report on fresh QA phantom batches (dataset cached, with augmentation for variety)

Each evaluation call samples new random samples from cached datasets, providing natural
regularization against overfitting to a fixed batch. Datasets are cached per (paths, config)
to avoid expensive re-instantiation on every evaluation. QA batches are augmented to
exploit limited phantom data; validation batches use the large validation set as-is.
"""

import numpy as np
import torch
from FlexCNN_for_Medical_Physics.classes.dataset_classes import NpArrayDataSet
from FlexCNN_for_Medical_Physics.functions.helper.metrics import SSIM, MSE, custom_metric
from FlexCNN_for_Medical_Physics.functions.helper.metrics_wrappers import calculate_metric
from FlexCNN_for_Medical_Physics.functions.helper.roi import ROI_simple_phantom


# Module-level caches: persistent for the lifetime of the Ray worker process
val_dataset = None
qa_dataset = None


def load_validation_batch(paths, config, settings):
    """
    Load a fresh random validation batch without augmentation.
    
    Each call loads a new random sample from the validation set. No deterministic
    seeding, so batches vary across evaluations. No augmentation since the validation
    set is large and diverse.
    
    Args:
        paths: dict with 'tune_val_act_sino_path', 'tune_val_act_image_path'
        config: dict with 'gen_image_size', 'gen_sino_size', 'gen_image_channels', 'gen_sino_channels'
        settings: dict with 'tune_eval_batch_size' (e.g., 32)
    
    Returns:
        dict with keys 'sino' and 'image' (tensors on CPU)
    
    Raises:
        ValueError: if validation paths are not set
    """
    # Check paths
    if paths.get('tune_val_act_sino_path') is None or paths.get('tune_val_act_image_path') is None:
        raise ValueError(
            "tune_report_for='val' requires tune_val_act_sino_path and tune_val_act_image_path to be set."
        )
    
    tune_eval_batch_size = settings.get('tune_eval_batch_size', 32)
    
    global val_dataset
    
    # Load validation dataset on first call; reuse on subsequent calls
    if val_dataset is None:
        val_dataset = NpArrayDataSet(
            act_image_path=paths['tune_val_act_image_path'],
            act_sino_path=paths['tune_val_act_sino_path'],
            config=config,
            settings=settings,
            augment=(None, False),  # No augmentation for validation
            offset=0,
            num_examples=-1,  # Load entire validation set
            sample_division=1,
            device='cpu',  # Load to CPU; caller handles device placement
            act_recon1_path=None,
            act_recon2_path=None,
            atten_image_path=None,
            atten_sino_path=None
        )
    
    # Sample a fresh random batch (no fixed seed; differs each call)
    # Allow replacement when requested batch size exceeds dataset size
    indices = np.random.choice(
        len(val_dataset),
        size=tune_eval_batch_size,
        replace=(tune_eval_batch_size > len(val_dataset))
    )
    
    # Extract the batch
    sino_batch = []
    image_batch = []
    for idx in indices:
        act_data, atten_data, recon_data = val_dataset[idx]
        sino, image = act_data
        sino_batch.append(sino)
        image_batch.append(image)
    
    return {
        'sino': torch.stack(sino_batch),
        'image': torch.stack(image_batch),
    }


def load_qa_batch(paths, config, settings, augment=('SI', True)):
    """
    Load a fresh random QA phantom batch with augmentation and masks.
    
    Each call loads a new random sample from the QA set. Augmentations are applied
    to exploit limited phantom data. Masks are loaded via recon1_path to ensure
    augmentations are consistent between masks and images.
    
    Args:
         paths: dict with 'tune_qa_act_sino_path', 'tune_qa_act_image_path',
               'tune_qa_hotMask_path', 'tune_qa_hotBackgroundMask_path'
               (hotMask passed as recon1_path; hotBackgroundMask as recon2_path
               to ensure augmentations align across all tensors)
         config: dict with 'gen_image_size', 'gen_sino_size', 'gen_image_channels', 'gen_sino_channels'
        settings: dict with 'tune_eval_batch_size' (e.g., 32) and 'augment' matching training
        augment: type of augmentation to apply to images, sinograms & masks
    
    Returns:
        dict with keys 'sino', 'image', 'hotMask', 'hotBackgroundMask' (all tensors on CPU)
    
    Raises:
        ValueError: if QA paths are not set
    """
    # Check paths
    required_qa_paths = [
        'tune_qa_act_sino_path', 'tune_qa_act_image_path', 'tune_qa_hotMask_path',
        'tune_qa_hotBackgroundMask_path'
    ]
    if not all(paths.get(p) is not None for p in required_qa_paths):
        raise ValueError(
            f"tune_report_for='qa' requires all of {required_qa_paths} to be set."
        )
    
    tune_eval_batch_size = settings.get('tune_eval_batch_size', 32)
    
    global qa_dataset
    
    # Load QA dataset on first call; reuse on subsequent calls
    if qa_dataset is None:
        qa_dataset = NpArrayDataSet(
            act_image_path=paths['tune_qa_act_image_path'],
            act_sino_path=paths['tune_qa_act_sino_path'],
            config=config,
            settings=settings,
            augment=augment,  # Enable augmentation for QA using training pipeline setting
            offset=0,
            num_examples=-1,  # Load entire QA set
            sample_division=1,
            device='cpu',  # Load to CPU; caller handles device placement
            act_recon1_path=paths['tune_qa_hotMask_path'],      # Hot mask via recon1_path
            act_recon2_path=paths['tune_qa_hotBackgroundMask_path'],  # Hot background via recon2_path
            atten_image_path=None, 
            atten_sino_path=None,
        )
    
    # Sample a fresh random batch (no fixed seed; differs each call)
    indices = np.random.choice(len(qa_dataset), size=tune_eval_batch_size, replace=True) # Allow replacement due to small QA set
    
    # Extract the batch
    sino_batch = []
    image_batch = []
    hotMask_batch = []
    hotBackgroundMask_batch = []
    for idx in indices:
        # Unpack nested structure: act_data, atten_data, recon_data
        act_data, atten_data, recon_data = qa_dataset[idx]
        sino, image = act_data
        hotMask, hotBackgroundMask = recon_data
        sino_batch.append(sino)
        image_batch.append(image)
        hotMask_batch.append(hotMask)
        hotBackgroundMask_batch.append(hotBackgroundMask)
    
    return {
        'sino': torch.stack(sino_batch),
        'image': torch.stack(image_batch),
        'hotMask': torch.stack(hotMask_batch),
        'hotBackgroundMask': torch.stack(hotBackgroundMask_batch),
    }


def evaluate_val(gen, batch, device, train_SI):
    """
    Evaluate network on a validation batch and compute metrics.
    
    Batch is moved to device, evaluated, and metrics are computed.
    Batch remains on CPU until evaluation to minimize GPU memory pressure.
    
    Args:
        gen: Generator network (assumed in eval mode)
        batch: dict with 'sino' and 'image' tensors (on CPU)
        device: torch device to move tensors to
        train_SI: bool, True for sino→image, False for image→sino
    
    Returns:
        dict with metric results: {'MSE': float, 'SSIM': float, 'CUSTOM': float}
    """
    eval_sino = batch['sino'].to(device)
    eval_image = batch['image'].to(device)
    
    # Assign inputs/targets based on train_SI
    if train_SI:
        eval_input = eval_sino
        eval_target = eval_image
    else:
        eval_input = eval_image
        eval_target = eval_sino
    
    # Generate network output
    with torch.no_grad():
        eval_output = gen(eval_input)
    
    # Calculate metrics
    mse_val = calculate_metric(eval_target, eval_output, MSE)
    ssim_val = calculate_metric(eval_target, eval_output, SSIM)
    custom_val = custom_metric(eval_target, eval_output)
    
    # Explicit cleanup to prevent memory accumulation during tuning
    del eval_sino, eval_image, eval_input, eval_target, eval_output
    
    # Return metrics
    return {
        'MSE': mse_val,
        'SSIM': ssim_val,
        'CUSTOM': custom_val
    }


def evaluate_qa(gen, batch, device, use_ground_truth_rois=False):
    """
    Evaluate network on a QA phantom batch and compute contrast metrics.
    
    Batch is moved to device, evaluated, and ROI metrics are computed.
    Batch remains on CPU until evaluation to minimize GPU memory pressure.
    
    Args:
        gen: Generator network (assumed in eval mode)
        batch: dict with 'sino', 'image', 'hotMask', 'hotBackgroundMask' tensors (on CPU)
        device: torch device to move tensors to
        use_ground_truth_rois: bool, if True use ground truth for ROI checks (for validation)

    Returns:
        dict with metric results:
            - 'CR_symmetric': float
            - 'hot_underestimation': float
            - 'cold_overestimation': float
    """
    eval_sino = batch['sino'].to(device)
    eval_image = batch['image'].to(device)
    hotMask = batch['hotMask'].to(device)
    
    # Generate network output (sino → reconstructed image)
    with torch.no_grad():
        network_output = gen(eval_sino)
    
    # Optionally substitute ground truth for ROI checks to validate mask correctness
    eval_output = eval_image if use_ground_truth_rois else network_output
    
    # Symmetric phantom metrics (returns dict)
    cr_metrics = ROI_simple_phantom(eval_image, eval_output, hotMask)
    
    # Explicit cleanup to prevent memory accumulation during tuning
    del eval_sino, eval_image, hotMask, network_output, eval_output
    
    # Return metrics
    return cr_metrics
