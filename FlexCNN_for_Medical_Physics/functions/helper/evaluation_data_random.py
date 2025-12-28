"""
Helpers for evaluation data loading during tuning with fresh random batches.

Supports two evaluation modes:
- 'val': report on fresh validation batches (no caching, no augmentation)
- 'qa': report on fresh QA phantom batches (no caching, with augmentation for variety)

Each evaluation call loads new random samples, providing natural regularization
against overfitting to a fixed batch. QA batches are augmented to exploit limited
phantom data; validation batches use the large validation set as-is.
"""

import numpy as np
import torch
from FlexCNN_for_Medical_Physics.classes.dataset_classes import NpArrayDataSet
from FlexCNN_for_Medical_Physics.functions.helper.metrics import SSIM, MSE, custom_metric
from FlexCNN_for_Medical_Physics.functions.helper.metrics_wrappers import calculate_metric
from FlexCNN_for_Medical_Physics.functions.helper.roi import ROI_simple_phantom

# Number of batches to sample and average per evaluation
NUM_EVAL_BATCHES = 3


def load_validation_batches(paths, config, settings):
    """
    Load multiple fresh random validation batches without augmentation.
    
    Each call loads a new random sample from the validation set. No deterministic
    seeding, so batches vary across evaluations. No augmentation since the validation
    set is large and diverse.
    
    Args:
        paths: dict with 'tune_val_sino_path', 'tune_val_image_path'
        config: dict with 'image_size', 'sino_size', 'image_channels', 'sino_channels'
        settings: dict with 'tune_eval_batch_size' (e.g., 32)
    
    Returns:
        list of dicts, each with keys 'sino' and 'image' (tensors on CPU)
    
    Raises:
        ValueError: if validation paths are not set
    """
    # Check paths
    if paths.get('tune_val_sino_path') is None or paths.get('tune_val_image_path') is None:
        raise ValueError(
            "tune_report_for='val' requires tune_val_sino_path and tune_val_image_path to be set."
        )
    
    tune_eval_batch_size = settings.get('tune_eval_batch_size', 32)
    
    # Load full validation set (memory-mapped for efficiency)
    val_dataset = NpArrayDataSet(
        image_path=paths['tune_val_image_path'],
        sino_path=paths['tune_val_sino_path'],
        config=config,
        augment=(None, False),  # No augmentation for validation
        offset=0,
        num_examples=-1,  # Load entire validation set
        sample_division=1,
        device='cpu',  # Load to CPU; caller handles device placement
        recon1_path=None,
        recon2_path=None,
    )
    
    # Load NUM_EVAL_BATCHES fresh random batches
    batches = [] # a list of dicts with 'sino' and 'image' tensors
    for _ in range(NUM_EVAL_BATCHES):
        # Sample a fresh random batch (no fixed seed; differs each call)
        indices = np.random.choice(len(val_dataset), size=tune_eval_batch_size, replace=False)
        
        # Extract the batch
        sino_batch = []
        image_batch = []
        for idx in indices:
            sino, image = val_dataset[idx][:2]  # Unpack sino, image (ignore recon1/recon2)
            sino_batch.append(sino)
            image_batch.append(image)
        
        batches.append({
            'sino': torch.stack(sino_batch),
            'image': torch.stack(image_batch),
        })
    
    return batches


def load_qa_batches(paths, config, settings, augment=('SI', True)):
    """
    Load multiple fresh random QA phantom batches with augmentation and masks.
    
    Each call loads a new random sample from the QA set. Augmentations are applied
    to exploit limited phantom data. Masks are loaded via recon1_path to ensure
    augmentations are consistent between masks and images.
    
    Args:
        paths: dict with 'tune_qa_sino_path', 'tune_qa_image_path',
               'tune_qa_hotMask_path', 'tune_qa_hotBackgroundMask_path'
               (hotMask passed as recon1_path; hotBackgroundMask as recon2_path
               to ensure augmentations align across all tensors)
        config: dict with 'image_size', 'sino_size', 'image_channels', 'sino_channels'
        settings: dict with 'tune_eval_batch_size' (e.g., 32) and 'augment' matching training
        augment: type of augmentation to apply to images, sinograms & masks
    
    Returns:
        list of dicts, each with keys 'sino', 'image', 'hotMask', 'hotBackgroundMask'
        (all tensors on CPU)
    
    Raises:
        ValueError: if QA paths are not set
    """
    # Check paths
    required_qa_paths = [
        'tune_qa_sino_path', 'tune_qa_image_path', 'tune_qa_hotMask_path',
        'tune_qa_hotBackgroundMask_path'
    ]
    if not all(paths.get(p) is not None for p in required_qa_paths):
        raise ValueError(
            f"tune_report_for='qa' requires all of {required_qa_paths} to be set."
        )
    
    tune_eval_batch_size = settings.get('tune_eval_batch_size', 32)
    
    # Load full QA set
    # Use training-time augmentations for QA (same pipeline)
    # Pass hotMask via recon1_path and hotBackgroundMask via recon2_path so augmentations apply consistently
    qa_dataset = NpArrayDataSet(
        image_path=paths['tune_qa_image_path'],
        sino_path=paths['tune_qa_sino_path'],
        config=config,
        augment=augment,  # Enable augmentation for QA using training pipeline setting
        offset=0,
        num_examples=-1,  # Load entire QA set
        sample_division=1,
        device='cpu',  # Load to CPU; caller handles device placement
        recon1_path=paths['tune_qa_hotMask_path'],      # Hot mask via recon1_path
        recon2_path=paths['tune_qa_hotBackgroundMask_path'],  # Hot background via recon2_path
    )
    
    # Load NUM_EVAL_BATCHES fresh random batches with augmentation
    batches = []
    for _ in range(NUM_EVAL_BATCHES):
        # Sample a fresh random batch (no fixed seed; differs each call)
        indices = np.random.choice(len(qa_dataset), size=tune_eval_batch_size, replace=True) # Allow replacement due to small QA set

        # THRE HERE

        # Extract the batch
        sino_batch = []
        image_batch = []
        hotMask_batch = []
        hotBackgroundMask_batch = []
        for idx in indices:
            # Unpack sino, image, hotMask (recon1), hotBackgroundMask (recon2)
            sino, image, hotMask, hotBackgroundMask = qa_dataset[idx][:4]
            sino_batch.append(sino)
            image_batch.append(image)
            hotMask_batch.append(hotMask)
            hotBackgroundMask_batch.append(hotBackgroundMask)
        
        batches.append({
            'sino': torch.stack(sino_batch),
            'image': torch.stack(image_batch),
            'hotMask': torch.stack(hotMask_batch),
            'hotBackgroundMask': torch.stack(hotBackgroundMask_batch),
        })
    
    return batches


def evaluate_val(gen, batches, device, train_SI):
    """
    Evaluate network on multiple validation batches and average metrics.
    
    Each batch is moved to device, evaluated, and metrics are averaged across all batches.
    
    Args:
        gen: Generator network (assumed in eval mode)
        batches: list of dicts, each with 'sino' and 'image' tensors (on CPU)
        device: torch device to move tensors to
        train_SI: bool, True for sino→image, False for image→sino
    
    Returns:
        dict with averaged metric results: {'MSE': float, 'SSIM': float, 'CUSTOM': float}
    """
    mse_sum = 0.0
    ssim_sum = 0.0
    custom_sum = 0.0
    num_batches = len(batches)
    
    for batch in batches:
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
        
        # Accumulate metrics
        mse_sum += calculate_metric(eval_target, eval_output, MSE)
        ssim_sum += calculate_metric(eval_target, eval_output, SSIM)
        custom_sum += custom_metric(eval_target, eval_output)
    
    # Return averaged metrics
    return {
        'MSE': mse_sum / num_batches,
        'SSIM': ssim_sum / num_batches,
        'CUSTOM': custom_sum / num_batches
    }


def evaluate_qa(gen, batches, device, settings, use_ground_truth_rois=False):
    """
    Evaluate network on multiple QA phantom batches and average contrast metrics.
    
    Each batch is moved to device, evaluated, and ROI metrics are averaged across all batches.
    
    Args:
        gen: Generator network (assumed in eval mode)
        batches: list of dicts, each with 'sino', 'image', 'hotMask', 'hotBackgroundMask' tensors (on CPU)
        device: torch device to move tensors to
        settings: dict with optional 'tune_qa_hot_weight' (kept for compatibility)
        use_ground_truth_rois: bool, if True use ground truth for ROI checks (for validation)

    Returns:
        dict with averaged metric results:
            - 'CR_symmetric': float
            - 'hot_underestimation': float
            - 'cold_overestimation': float
    """
    num_batches = len(batches)
    CR_symmetric_sum = 0.0
    hot_under_sum = 0.0
    cold_over_sum = 0.0

    for batch in batches:
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
        CR_symmetric_sum += cr_metrics['CR_symmetric']
        hot_under_sum += cr_metrics['hot_underestimation']
        cold_over_sum += cr_metrics['cold_overestimation']
    
    # Return averaged metrics
    return {
        'CR_symmetric': CR_symmetric_sum / num_batches,
        'hot_underestimation': hot_under_sum / num_batches,
        'cold_overestimation': cold_over_sum / num_batches,
    }
