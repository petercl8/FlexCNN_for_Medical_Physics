"""
Runtime evaluation for training and tuning modes.

Provides batch loading and evaluation functions for validation and QA phantom
evaluation used during both Ray Tune hyperparameter optimization and training
with learning curve logging.

Supports:
- Simple networks (ACT, ATTEN, CONCAT)
- Frozen flow networks (FROZEN_COFLOW, FROZEN_COUNTERFLOW)
- Both validation and QA phantom modes (auto-detected by batch content)
- Evaluation on training, holdout, and QA splits

Datasets are cached per worker to avoid expensive re-instantiation on every
evaluation. Validation uses the full unaugmented dataset; QA uses augmented
phantoms to exploit limited data.
"""

import time
import numpy as np
import torch
from ray.tune import report

from FlexCNN_for_Medical_Physics.classes.dataset.dataset_classes import NpArrayDataSet
from FlexCNN_for_Medical_Physics.functions.helper.metrics.metrics import SSIM, MSE
from FlexCNN_for_Medical_Physics.custom_criteria import custom_metric
from FlexCNN_for_Medical_Physics.functions.helper.metrics.metrics_wrappers import calculate_metric, update_tune_dataframe
from FlexCNN_for_Medical_Physics.functions.helper.metrics.roi import ROI_simple_phantom, ROI_NEMA_hot, ROI_NEMA_cold
from FlexCNN_for_Medical_Physics.functions.helper.utilities.timing import display_times
from FlexCNN_for_Medical_Physics.functions.helper.image_processing.display_images import show_single_unmatched_tensor

# Module-level caches: persistent for the lifetime of the Ray worker process

_train_dataset = None
_holdout_dataset = None
_qa_dataset = None

# Timing control for report_cross_validation_metrics
PRINT_REPORT_TIMING = False  # Set to False to suppress timing output during evaluations

# Control whether to use ground truth for ROI checks (QA mode only)
use_ground_truth_rois = False


def _sample_eval_indices(dataset, eval_batch_size):
    """Sample random indices for evaluation batch."""
    return np.random.choice(
        len(dataset),
        size=eval_batch_size,
        replace=(eval_batch_size > len(dataset))
    )

def _extract_standard_eval_batch(dataset, indices):
    """Extract act/atten tensors from sampled dataset indices (no QA masks)."""
    act_sino_batch = []
    act_image_batch = []
    atten_sino_batch = []
    atten_image_batch = []

    for idx in indices:
        act_data, atten_data, recon_data = dataset[idx]
        if act_data is not None:
            act_sino, act_image = act_data
            act_sino_batch.append(act_sino)
            act_image_batch.append(act_image)
        if atten_data is not None:
            atten_sino, atten_image = atten_data
            if atten_sino is not None:
                atten_sino_batch.append(atten_sino)
            if atten_image is not None:
                atten_image_batch.append(atten_image)

    result = {}
    if act_sino_batch:
        result['act_sino_scaled'] = torch.stack(act_sino_batch)
    if act_image_batch:
        result['act_image_scaled'] = torch.stack(act_image_batch)
    if atten_sino_batch:
        result['atten_sino_scaled'] = torch.stack(atten_sino_batch)
    if atten_image_batch:
        result['atten_image_scaled'] = torch.stack(atten_image_batch)

    return result

def _get_or_create_eval_dataset(split, paths, config, settings):
    """Get cached train/holdout evaluation dataset or create it on first call."""
    global _train_dataset, _holdout_dataset

    if split == 'train':
        if _train_dataset is None:
            _train_dataset = NpArrayDataSet(
                act_image_path=paths['act_image_path'],
                act_sino_path=paths['act_sino_path'],
                config=config,
                settings=settings,
                augment=(None, False),
                offset=0,
                num_examples=-1,
                sample_division=1,
                device='cpu',
                act_recon1_path=None,
                act_recon2_path=None,
                atten_image_path=paths.get('atten_image_path'),
                atten_sino_path=paths.get('atten_sino_path')
            )
        return _train_dataset

    if split == 'holdout':
        if _holdout_dataset is None:
            _holdout_dataset = NpArrayDataSet(
                act_image_path=paths['eval_holdout_act_image_path'],
                act_sino_path=paths['eval_holdout_act_sino_path'],
                config=config,
                settings=settings,
                augment=(None, False),
                offset=0,
                num_examples=-1,
                sample_division=1,
                device='cpu',
                act_recon1_path=None,
                act_recon2_path=None,
                atten_image_path=paths.get('eval_holdout_atten_image_path'),
                atten_sino_path=paths.get('eval_holdout_atten_sino_path')
            )
        return _holdout_dataset

    raise ValueError(f"Unsupported split for cached eval dataset: '{split}'")

def load_eval_batch(split, paths, config, settings, augment=None):
    """
    Load evaluation batch for train/holdout/QA splits with unified interface.
    
    Routes to appropriate dataset based on split value:
    - split='train': Random sample from training set (no augmentation)
    - split='holdout': Random sample from holdout/validation set (no augmentation)
    - split='qa': QA phantom batch with optional augmentation and masks
    
    Each call loads a fresh random sample. Datasets are cached per worker to avoid
    re-instantiation. QA mode supports multiple load modes (random, sequential, slice_range).
    
    Args:
        split: str, 'train', 'holdout', or 'qa'
        paths: dict with appropriate paths based on split:
               - 'train': needs 'act_sino_path', 'act_image_path', 'atten_image_path', 'atten_sino_path' (from training set)
               - 'holdout': needs 'eval_holdout_act_sino_path', 'eval_holdout_act_image_path', etc.
               - 'qa': needs 'eval_qa_*' paths and mask paths
        config: dict with 'gen_image_size', 'gen_sino_size', 'gen_image_channels', 'gen_sino_channels'
        settings: dict with 'eval_batch_size', and for QA: 'qa_load_mode', optional 'qa_slice_range'
        augment: tuple (type, flip_bool) or None. Only used for QA split in random mode; holdout/train always no augment
    
    Returns:
        dict with keys:
        - 'act_sino_scaled', 'act_image_scaled' (always present, tensors on CPU)
        - 'atten_sino_scaled', 'atten_image_scaled' (if available)
        - 'hotMask', 'hotBackgroundMask' (for QA split only)
    
    Raises:
        ValueError: if split is invalid, required paths missing, or eval_batch_size invalid
        KeyError: if required settings keys missing
    """
    if split not in ['train', 'holdout', 'qa']:
        raise ValueError(f"split must be 'train', 'holdout', or 'qa', got '{split}'")
    
    if 'eval_batch_size' not in settings:
        raise KeyError("Missing required settings['eval_batch_size'] for evaluation batch loading")
    eval_batch_size = settings['eval_batch_size']

    if eval_batch_size is None or eval_batch_size <= 0:
        raise ValueError(f"settings['eval_batch_size'] must be > 0, got {eval_batch_size}")
    
    # ===== TRAIN/HOLDOUT splits: Shared non-QA path =====
    if split in ('train', 'holdout'):
        dataset = _get_or_create_eval_dataset(split, paths, config, settings)
        indices = _sample_eval_indices(dataset, eval_batch_size)
        return _extract_standard_eval_batch(dataset, indices)
    
    # ===== QA split: Load phantom with masks =====
    else:  # split == 'qa'
        # Check required paths
        required_qa_paths = [
            'eval_qa_act_sino_path', 'eval_qa_act_image_path', 'eval_qa_hotMask_path',
            'eval_qa_hotBackgroundMask_path'
        ]
        if not all(paths.get(p) is not None for p in required_qa_paths):
            raise ValueError(
                f"split='qa' requires all of {required_qa_paths} to be set."
            )
        
        qa_load_mode = settings.get('qa_load_mode')
        qa_slice_range = settings.get('qa_slice_range')
        
        # Default augmentation for QA batch
        if augment is None:
            augment = ('SI', True)
        
        # Determine augmentation based on load mode
        qa_augment = augment if qa_load_mode == 'random' else (None, False)
        
        global _qa_dataset

        # Load QA dataset on first call; reuse on subsequent calls
        if _qa_dataset is None:
            _qa_dataset = NpArrayDataSet(
                act_image_path=paths['eval_qa_act_image_path'],
                act_sino_path=paths['eval_qa_act_sino_path'],
                config=config,
                settings=settings,
                augment=qa_augment,
                offset=0,
                num_examples=-1,
                sample_division=1,
                device='cpu',
                act_recon1_path=paths['eval_qa_hotMask_path'],
                act_recon2_path=paths['eval_qa_hotBackgroundMask_path'],
                atten_image_path=paths.get('eval_qa_atten_image_path'),
                atten_sino_path=paths.get('eval_qa_atten_sino_path')
            )
        
        # Determine indices based on slice range and load mode
        if qa_slice_range is not None:
            slice_start, slice_end = qa_slice_range
            indices = np.arange(slice_start, slice_end)
        elif qa_load_mode == 'sequential':
            indices = np.arange(len(_qa_dataset))
        else:
            indices = np.random.choice(len(_qa_dataset), size=eval_batch_size, replace=True)
        
        # Extract batch
        act_sino_batch = []
        act_image_batch = []
        hotMask_batch = []
        hotBackgroundMask_batch = []
        atten_sino_batch = []
        atten_image_batch = []
        for idx in indices:
            act_data, atten_data, recon_data = _qa_dataset[idx]
            act_sino, act_image = act_data
            hotMask, hotBackgroundMask = recon_data
            act_sino_batch.append(act_sino)
            act_image_batch.append(act_image)
            hotMask_batch.append(hotMask)
            hotBackgroundMask_batch.append(hotBackgroundMask)
            
            if atten_data is not None:
                atten_sino, atten_image = atten_data
                if atten_sino is not None:
                    atten_sino_batch.append(atten_sino)
                if atten_image is not None:
                    atten_image_batch.append(atten_image)
        
        result = {
            'act_sino_scaled': torch.stack(act_sino_batch),
            'act_image_scaled': torch.stack(act_image_batch),
            'hotMask': torch.stack(hotMask_batch),
            'hotBackgroundMask': torch.stack(hotBackgroundMask_batch),
        }
        
        if atten_sino_batch:
            result['atten_sino_scaled'] = torch.stack(atten_sino_batch)
        if atten_image_batch:
            result['atten_image_scaled'] = torch.stack(atten_image_batch)
        
        return result

def evaluate_metrics(generators, batch, device, train_SI, network_type, tune_metric='SSIM', evaluate_on='val', run_mode='tune', compute_standard_metrics=False):
    """
    Evaluate network(s) on holdout or QA batch with unified interface.
    
    Routes between simple and frozen flow networks based on network_type parameter.
    Supports mixed metric outputs: standard metrics (MSE/SSIM/CUSTOM) or split-specific
    metrics (ROI for QA) depending on compute_standard_metrics flag.
    
    Evaluation mode:
    - If evaluate_on='val': Always returns validation metrics (MSE/SSIM/CUSTOM as configured)
    - If evaluate_on='qa' and compute_standard_metrics=False: Returns ROI metrics
    - If evaluate_on='qa' and compute_standard_metrics=True: Returns MSE/SSIM/CUSTOM instead
    
    Args:
        generators: Tuple containing:
                   - (gen,) for simple networks (ACT, ATTEN, CONCAT)
                   - (gen_act, gen_atten) for frozen flow networks (FROZEN_COFLOW, FROZEN_COUNTERFLOW)
        batch: dict with tensors (on CPU). Expected keys:
               - 'act_sino_scaled', 'act_image_scaled' (always present)
               - 'atten_sino_scaled' or 'atten_image_scaled' (for attenuation networks)
               - 'hotMask', 'hotBackgroundMask' (for QA modes)
        device: torch device to move tensors to
        train_SI: bool, True for sino→image, False for image→sino
        network_type: str, 'ACT', 'ATTEN', 'CONCAT', 'FROZEN_COFLOW', or 'FROZEN_COUNTERFLOW'
        tune_metric: str, which metric to optimize ('MSE', 'SSIM', 'CUSTOM', 'qa-simple', 'qa-nema')
        evaluate_on: str, 'val' or 'qa'
        run_mode: str, 'tune' or 'train' (affects CUSTOM metric computation)
        compute_standard_metrics: bool, if True force MSE/SSIM/CUSTOM instead of QA-specific metrics
    
    Returns:
        dict with metrics:
        - Validation: {MSE: float, SSIM: float, ...}
        - QA (no compute_standard_metrics): {CR_...: float, ...}
        - QA (compute_standard_metrics=True): {MSE: float, SSIM: float, CUSTOM: float}
    
    Raises:
        ValueError: if network_type is invalid or required data missing
    """
    # Route to appropriate evaluator based on network type
    is_frozen_flow = network_type in ('FROZEN_COFLOW', 'FROZEN_COUNTERFLOW')
    
    if is_frozen_flow:
        # Frozen flow network path
        flow_mode = 'coflow' if network_type == 'FROZEN_COFLOW' else 'counterflow'
        return _evaluate_metrics_frozen(generators, batch, device, flow_mode, 
                                        tune_metric=tune_metric, evaluate_on=evaluate_on, 
                                        run_mode=run_mode, compute_standard_metrics=compute_standard_metrics)
    else:
        # Simple network path  
        return _evaluate_metrics_simple(generators, batch, device, train_SI, network_type,
                                         tune_metric=tune_metric, evaluate_on=evaluate_on,
                                         run_mode=run_mode, compute_standard_metrics=compute_standard_metrics)

def _evaluate_metrics_simple(generators, batch, device, train_SI, network_type, tune_metric='SSIM', evaluate_on='val', run_mode='tune', compute_standard_metrics=False):
    """
    Evaluate simple network (ACT, ATTEN, or CONCAT) on batch.
    
    Internal implementation for simple networks; called by evaluate_metrics().
    """
    from FlexCNN_for_Medical_Physics.functions.main_run_functions.train_utils import route_batch_inputs
    
    # Unpack generator
    gen = generators[0]
    
    # Route inputs and targets
    eval_input, eval_target = route_batch_inputs(train_SI, batch, network_type)
    
    # Move to device
    eval_input = eval_input.to(device)
    eval_target = eval_target.to(device)
    
    # Generate output
    with torch.no_grad():
        eval_output = gen(eval_input)
    
    # Compute metrics based on mode
    if evaluate_on == 'val' or compute_standard_metrics:
        # Validation mode or forced standard metrics: always compute MSE/SSIM
        mse_val = calculate_metric(eval_target, eval_output, MSE)
        ssim_val = calculate_metric(eval_target, eval_output, SSIM)
        
        # Compute CUSTOM when required
        compute_custom = (tune_metric == 'CUSTOM') or (run_mode == 'train') or compute_standard_metrics
        if compute_custom:
            custom_val = custom_metric(eval_output, eval_target)
            metrics = {
                'MSE': mse_val,
                'SSIM': ssim_val,
                'CUSTOM': custom_val
            }
        else:
            metrics = {
                'MSE': mse_val,
                'SSIM': ssim_val
            }
        
        del eval_input, eval_target, eval_output
        return metrics
    
    elif evaluate_on == 'qa':
        # QA mode: Compute ROI metrics based on tune_metric type
        hotMask = batch['hotMask'].to(device)
        eval_qa_output = eval_target if use_ground_truth_rois else eval_output
        
        if tune_metric == 'qa-simple':
            # QA Simple mode: compute simple ROI metrics
            cr_metrics = ROI_simple_phantom(eval_target, eval_qa_output, hotMask)
            del eval_input, eval_target, eval_output, hotMask, hotBackgroundMask, eval_qa_output
            return cr_metrics
        elif tune_metric == 'qa-nema':
            # QA NEMA mode: compute NEMA ROI metrics

            hotBackgroundMask = batch['hotBackgroundMask'].to(device)

            nema_hot_metric = ROI_NEMA_hot(eval_target, eval_qa_output, hotBackgroundMask, hotMask)
            del eval_input, eval_target, eval_output, hotMask, hotBackgroundMask, eval_qa_output
            return {'NEMA_hot': nema_hot_metric}
        else:
            raise ValueError(f"Invalid tune_metric='{tune_metric}' for QA mode (expected 'qa-simple' or 'qa-nema')")
    
    else:
        raise ValueError(f"Unknown evaluate_on='{evaluate_on}' (expected 'val' or 'qa')")

def _evaluate_metrics_frozen(generators, batch, device, flow_mode, tune_metric='SSIM', evaluate_on='val', run_mode='tune', compute_standard_metrics=False):
    """
    Evaluate frozen flow network on batch.
    
    Internal implementation for frozen flow networks; called by evaluate_metrics().
    """
    # Unpack generators
    gen_act, gen_atten = generators
    
    # Validate required data
    if 'act_sino_scaled' not in batch or batch['act_sino_scaled'] is None:
        raise ValueError("Frozen flow evaluation requires 'act_sino_scaled' data")
    if 'act_image_scaled' not in batch or batch['act_image_scaled'] is None:
        raise ValueError("Frozen flow evaluation requires 'act_image_scaled' data")
    
    # Load activity data
    eval_act_sino = batch['act_sino_scaled'].to(device)
    eval_act_image = batch['act_image_scaled'].to(device)
    
    # Determine attenuation input based on flow mode
    if flow_mode == 'coflow':
        if 'atten_sino_scaled' not in batch or batch['atten_sino_scaled'] is None:
            raise ValueError("COFLOW mode requires 'atten_sino_scaled' data")
        atten_input = batch['atten_sino_scaled'].to(device)
    elif flow_mode == 'counterflow':
        if 'atten_image_scaled' not in batch or batch['atten_image_scaled'] is None:
            raise ValueError("COUNTERFLOW mode requires 'atten_image_scaled' data")
        atten_input = batch['atten_image_scaled'].to(device)
    else:
        raise ValueError(f"Invalid flow_mode='{flow_mode}' (expected 'coflow' or 'counterflow')")
    
    # Run frozen attenuation network to get features
    with torch.no_grad():
        result = gen_atten(atten_input, return_features=True)
        frozen_enc_feats = result['encoder']
        frozen_dec_feats = result['decoder']
        
        # Run activity network with frozen features
        eval_output = gen_act(
            eval_act_sino,
            frozen_encoder_features=frozen_enc_feats,
            frozen_decoder_features=frozen_dec_feats
        )
    
    eval_target = eval_act_image
    
    # Compute metrics based on mode
    if evaluate_on == 'val' or compute_standard_metrics:
        # Validation mode or forced standard metrics
        mse_val = calculate_metric(eval_target, eval_output, MSE)
        ssim_val = calculate_metric(eval_target, eval_output, SSIM)
        
        # Compute CUSTOM when required
        compute_custom = (tune_metric == 'CUSTOM') or (run_mode == 'train') or compute_standard_metrics
        if compute_custom:
            custom_val = custom_metric(eval_output, eval_target)
            metrics = {
                'MSE': mse_val,
                'SSIM': ssim_val,
                'CUSTOM': custom_val
            }
        else:
            # Tune mode with non-CUSTOM metric: skip expensive CUSTOM computation
            metrics = {
                'MSE': mse_val,
                'SSIM': ssim_val
            }
        
        # Explicit cleanup
        del eval_act_sino, eval_act_image, atten_input, frozen_enc_feats, frozen_dec_feats, eval_target, eval_output
        
        return metrics
    
    elif evaluate_on == 'qa':
        # QA mode: Compute ROI metrics based on tune_metric type
        hotMask = batch['hotMask'].to(device)
        eval_qa_output = eval_act_image if use_ground_truth_rois else eval_output
        
        if tune_metric == 'qa-simple':
            # QA Simple mode: compute simple ROI metrics
            cr_metrics = ROI_simple_phantom(eval_act_image, eval_qa_output, hotMask)
            del eval_act_sino, eval_act_image, atten_input, frozen_enc_feats, frozen_dec_feats, eval_target, eval_output, hotMask, hotBackgroundMask, eval_qa_output
            return cr_metrics
        elif tune_metric == 'qa-nema':
            # QA NEMA mode: compute NEMA ROI metrics
            hotBackgroundMask = batch['hotBackgroundMask'].to(device)
            nema_hot_metric = ROI_NEMA_hot(eval_act_image, eval_qa_output, hotBackgroundMask, hotMask)
            del eval_act_sino, eval_act_image, atten_input, frozen_enc_feats, frozen_dec_feats, eval_target, eval_output, hotMask, hotBackgroundMask, eval_qa_output
            return {'NEMA_hot': nema_hot_metric}
        else:
            raise ValueError(f"Invalid tune_metric='{tune_metric}' for QA mode (expected 'qa-simple' or 'qa-nema')")
    
    else:
        raise ValueError(f"Unknown evaluate_on='{evaluate_on}' (expected 'val' or 'qa')")

def report_cross_validation_metrics(generators, paths, config, settings, tune_dataframe, tune_dataframe_path, 
                        train_SI, tune_dataframe_fraction, tune_max_t, report_num, 
                        example_num, batch_step, epoch, device, report_to_ray=True, 
                        return_metrics=False):
    """
    Load evaluation batch, compute metrics, update dataframe, and report to Ray Tune.
    
    Args:
        generators: Tuple of generator model(s):
                    - (gen,) for simple networks (ACT, ATTEN, CONCAT)
                    - (gen_act, gen_atten) for frozen flow networks
                    All generators will be set to eval mode and restored to train mode
        paths: Path dictionary with validation/QA data paths
        config: Configuration dictionary
        settings: Settings dictionary with evaluate_on, tune_metric, etc.
        tune_dataframe: Current tune dataframe (optional)
        tune_dataframe_path: Path to save dataframe (optional)
        train_SI: Training direction (True for sino→image)
        tune_dataframe_fraction: Fraction of tune_max_t at which to save dataframe
        tune_max_t: Maximum number of reports
        report_num: Current report number
        example_num: Current example number
        batch_step: Current batch step
        epoch: Current epoch
        device: Device string
        report_to_ray: If True, report metrics to Ray Tune
        return_metrics: If True, return metrics dict
    
    Returns:
        Metrics dict if return_metrics=True, else None
    
    Raises:
        ValueError: if NaN detected in optimization metric or invalid evaluate_on value
    """
    # ===== SETUP: Generator mode & configuration =====
    for gen in generators:
        gen.eval()
    
    t_total = t_start = time.time()
    evaluate_on = settings.get('evaluate_on', 'val')
    network_type = config.get('network_type', 'ACT')
    tune_metric = settings.get('tune_metric', 'SSIM')
    run_mode = settings.get('run_mode', 'tune')
    
    # ===== LOAD: Batch data based on split type =====
    if evaluate_on == 'val':
        batch = load_eval_batch('holdout', paths, config, settings)
    elif evaluate_on == 'qa':
        batch = load_eval_batch('qa', paths, config, settings, augment=('SI', True))
    else:
        raise ValueError(f"Invalid evaluate_on='{evaluate_on}'. Expected 'val' or 'qa'")
    
    t_start = display_times(f"[TUNE_REPORT #{report_num}] Batch load time", t_start, PRINT_REPORT_TIMING)
    
    # ===== EVALUATE: Compute metrics on batch =====
    metrics = evaluate_metrics(generators, batch, device, train_SI, network_type, 
                               tune_metric=tune_metric, evaluate_on=evaluate_on, 
                               run_mode=run_mode, compute_standard_metrics=False)
    t_start = display_times(f"[TUNE_REPORT #{report_num}] Evaluation time", t_start, PRINT_REPORT_TIMING)
    
    # ===== UPDATE: Save dataframe checkpoint if reached fraction =====
    if tune_dataframe is not None and tune_dataframe_fraction is not None and tune_max_t is not None:
        if int(tune_dataframe_fraction * tune_max_t) == report_num:
            tune_dataframe = update_tune_dataframe(
                tune_dataframe, tune_dataframe_path, generators[0], config, metrics
            )
    
    # ===== CLEANUP: Memory management =====
    del batch
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    
    # Restore trainable generator to train mode (frozen generator stays eval if present)
    generators[0].train()
    
    # ===== VALIDATE: Check for NaN in metric =====
    if tune_metric in metrics and (
        metrics[tune_metric] is None or 
        torch.isnan(torch.tensor(metrics[tune_metric])).item()
    ):
        raise ValueError(f"NaN detected in {tune_metric}, terminating trial")
    
    # ===== REPORT: Add metadata and send to Ray Tune =====
    metrics.update({
        'example_num': example_num,
        'batch_step': batch_step,
        'epoch': epoch
    })
    
    t_start = display_times(f"[TUNE_REPORT #{report_num}] Report time", t_start, PRINT_REPORT_TIMING)
    if report_to_ray:
        report(metrics)

    t_start = display_times(f"[TUNE_REPORT #{report_num}] TOTAL Eval Time", t_total, PRINT_REPORT_TIMING)

    if return_metrics:
        return metrics