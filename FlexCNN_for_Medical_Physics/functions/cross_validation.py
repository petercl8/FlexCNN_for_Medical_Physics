"""
Cross-validation evaluation during Ray Tune training.

Provides batch loading and evaluation functions for validation and QA phantom
reporting during Ray Tune hyperparameter optimization. 

Supports:
- Simple networks (ACT, ATTEN, CONCAT)
- Frozen flow networks (FROZEN_COFLOW, FROZEN_COUNTERFLOW)
- Both validation and QA phantom modes (auto-detected by batch content)

Datasets are cached per worker to avoid expensive re-instantiation on every
evaluation. Validation uses the full unaugmented dataset; QA uses augmented
phantoms to exploit limited data.
"""

import time
import numpy as np
import torch
from ray.tune import report

from FlexCNN_for_Medical_Physics.classes.dataset_classes import NpArrayDataSet
from FlexCNN_for_Medical_Physics.functions.helper.metrics import SSIM, MSE, custom_metric
from FlexCNN_for_Medical_Physics.functions.helper.metrics_wrappers import calculate_metric, update_tune_dataframe
from FlexCNN_for_Medical_Physics.functions.helper.roi import ROI_simple_phantom
from FlexCNN_for_Medical_Physics.functions.helper.timing import display_times


# Module-level caches: persistent for the lifetime of the Ray worker process
_val_dataset = None
_qa_dataset = None

# Timing control for report_tune_metrics
PRINT_REPORT_TIMING = False  # Set to False to suppress timing output during tuning reports


def load_validation_batch(paths, config, settings):
    """
    Load a fresh random validation batch without augmentation.
    
    Each call loads a new random sample from the validation set. No deterministic
    seeding, so batches vary across evaluations. No augmentation since the validation
    set is large and diverse. May include attenuation data if available (for frozen flow networks).
    
    Args:
        paths: dict with 'tune_val_act_sino_path', 'tune_val_act_image_path', and optionally
               'tune_val_atten_sino_path', 'tune_val_atten_image_path' for frozen flow networks
        config: dict with 'gen_image_size', 'gen_sino_size', 'gen_image_channels', 'gen_sino_channels'
        settings: dict with 'tune_eval_batch_size' (e.g., 32)
    
    Returns:
        dict with keys 'act_sino_scaled' and 'act_image_scaled' (always present, tensors on CPU),
        and 'atten_sino_scaled' and 'atten_image_scaled' if available
    
    Raises:
        ValueError: if validation paths are not set
    """    
    tune_eval_batch_size = settings['tune_eval_batch_size']
    
    global _val_dataset
    
    # Load validation dataset on first call; reuse on subsequent calls
    if _val_dataset is None:
        _val_dataset = NpArrayDataSet(
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
            atten_image_path=paths.get('tune_val_atten_image_path'),  # Load if available
            atten_sino_path=paths.get('tune_val_atten_sino_path')  # Load if available
        )
    
    # Sample a fresh random batch (no fixed seed; differs each call)
    # Allow replacement when requested batch size exceeds dataset size
    indices = np.random.choice(
        len(_val_dataset),
        size=tune_eval_batch_size,
        replace=(tune_eval_batch_size > len(_val_dataset))
    )
    
    # Extract the batch
    act_sino_batch = []
    act_image_batch = []
    atten_sino_batch = []
    atten_image_batch = []
    for idx in indices:
        act_data, atten_data, recon_data = _val_dataset[idx]

        if act_data is not None:
            act_sino, act_image = act_data
            act_sino_batch.append(act_sino)
            act_image_batch.append(act_image)
        
        # Extract attenuation data if available (frozen flow networks)
        if atten_data is not None:
            atten_sino, atten_image = atten_data
            # Only append if tensors are actually valid (not None)
            if atten_sino is not None:
                atten_sino_batch.append(atten_sino)
            if atten_image is not None:
                atten_image_batch.append(atten_image)
    
    result = {}
 
    # Add activity data if loaded (with _scaled suffix to match training convention)
    if act_sino_batch:
        result['act_sino_scaled'] = torch.stack(act_sino_batch)
    if act_image_batch:
        result['act_image_scaled'] = torch.stack(act_image_batch)

    # Add attenuation data if loaded (with _scaled suffix to match training convention)
    if atten_sino_batch:
        result['atten_sino_scaled'] = torch.stack(atten_sino_batch)
    if atten_image_batch:
        result['atten_image_scaled'] = torch.stack(atten_image_batch)
    
    return result


def load_qa_batch(paths, config, settings, augment=('SI', True)):
    """
    Load a fresh random QA phantom batch with augmentation and masks.
    
    Each call loads a new random sample from the QA set. Augmentations are applied
    to exploit limited phantom data. Masks are loaded via recon1_path to ensure
    augmentations are consistent between masks and images. May include attenuation data
    if available (for frozen flow networks).
    
    Args:
         paths: dict with 'tune_qa_act_sino_path', 'tune_qa_act_image_path',
               'tune_qa_hotMask_path', 'tune_qa_hotBackgroundMask_path',
               and optionally 'tune_qa_atten_sino_path', 'tune_qa_atten_image_path'
               (hotMask passed as recon1_path; hotBackgroundMask as recon2_path
               to ensure augmentations align across all tensors)
         config: dict with 'gen_image_size', 'gen_sino_size', 'gen_image_channels', 'gen_sino_channels'
        settings: dict with 'tune_eval_batch_size' (e.g., 32) and 'augment' matching training
        augment: type of augmentation to apply to images, sinograms & masks
    
    Returns:
        dict with keys 'act_sino_scaled', 'act_image_scaled', 'hotMask', 'hotBackgroundMask' (all tensors on CPU),
        and 'atten_sino_scaled' and 'atten_image_scaled' if available
    
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
    
    global _qa_dataset
    
    # Load QA dataset on first call; reuse on subsequent calls
    if _qa_dataset is None:
        _qa_dataset = NpArrayDataSet(
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
            atten_image_path=paths.get('tune_qa_atten_image_path'),  # Load if available
            atten_sino_path=paths.get('tune_qa_atten_sino_path'),  # Load if available
        )
    
    # Sample a fresh random batch (no fixed seed; differs each call)
    indices = np.random.choice(len(_qa_dataset), size=tune_eval_batch_size, replace=True) # Allow replacement due to small QA set
    
    # Extract the batch
    act_sino_batch = []
    act_image_batch = []
    hotMask_batch = []
    hotBackgroundMask_batch = []
    atten_sino_batch = []
    atten_image_batch = []
    for idx in indices:
        # Unpack nested structure: act_data, atten_data, recon_data
        act_data, atten_data, recon_data = _qa_dataset[idx]
        act_sino, act_image = act_data
        hotMask, hotBackgroundMask = recon_data
        act_sino_batch.append(act_sino)
        act_image_batch.append(act_image)
        hotMask_batch.append(hotMask)
        hotBackgroundMask_batch.append(hotBackgroundMask)
        
        # Extract attenuation data if available (frozen flow networks)
        if atten_data is not None:
            atten_sino, atten_image = atten_data
            # Only append if tensors are actually valid (not None)
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
    
    # Add attenuation data if loaded (with _scaled suffix to match training convention)
    if atten_sino_batch:
        result['atten_sino_scaled'] = torch.stack(atten_sino_batch)
    if atten_image_batch:
        result['atten_image_scaled'] = torch.stack(atten_image_batch)
    
    return result


def evaluate_val(generators, batch, device, train_SI, network_type, tune_metric='SSIM', use_ground_truth_rois=False):
    """
    Evaluate simple network (ACT, ATTEN, or CONCAT) on validation or QA batch.
    
    For frozen flow networks, use evaluate_val_frozen() instead.
    
    Evaluation mode is automatically detected:
    - If batch contains 'hotMask': QA mode (returns CR metrics via ROI_simple_phantom)
    - Otherwise: Validation mode (returns specified tune_metric)
    
    Args:
        generators: Tuple containing single generator (gen,) for simple networks
        batch: dict with tensors (on CPU). Expected keys:
               - 'act_sino_scaled', 'act_image_scaled' for activity/CONCAT networks
               - 'atten_sino_scaled', 'atten_image_scaled' for attenuation networks
               - 'hotMask', 'hotBackgroundMask' for QA mode (optional)
        device: torch device to move tensors to
        train_SI: bool, True for sino→image, False for image→sino
        network_type: str, 'ACT', 'ATTEN', or 'CONCAT'
        tune_metric: str, which metric to compute for validation ('MSE', 'SSIM', or 'CUSTOM')
        use_ground_truth_rois: bool, if True use ground truth for ROI checks (QA mode only)
    
    Returns:
        For validation: dict with single metric: {tune_metric: float}
        For QA: dict with CR metrics (CR_symmetric, hot_underestimation, cold_overestimation)
    """
    # Import here to avoid circular dependency
    from FlexCNN_for_Medical_Physics.functions.main_run_functions.train_utils import route_batch_inputs
    
    # Unpack generator (simple networks use single generator)
    gen = generators[0]
    
    # Detect QA mode by presence of hotMask
    is_qa_mode = 'hotMask' in batch and batch['hotMask'] is not None
    
    # Route inputs and targets using shared logic (batch already uses _scaled naming)
    eval_input, eval_target = route_batch_inputs(train_SI, batch, network_type)
    
    # Move to device
    eval_input = eval_input.to(device)
    eval_target = eval_target.to(device)
    
    # Generate output (simple single-input forward pass)
    with torch.no_grad():
        eval_output = gen(eval_input)
    
    # Compute metrics based on eval mode
    if is_qa_mode:
        # QA mode: use ROI metrics with hotMask
        hotMask = batch['hotMask'].to(device)
        eval_qa_output = eval_target if use_ground_truth_rois else eval_output
        cr_metrics = ROI_simple_phantom(eval_target, eval_qa_output, hotMask)
        
        # Explicit cleanup
        del eval_input, eval_target, eval_output, hotMask, eval_qa_output
        
        return cr_metrics
    
    else:
        # Validation mode: use specified metric
        if tune_metric == 'MSE':
            metric_val = calculate_metric(eval_target, eval_output, MSE)
        elif tune_metric == 'SSIM':
            metric_val = calculate_metric(eval_target, eval_output, SSIM)
        elif tune_metric == 'CUSTOM':
            metric_val = custom_metric(eval_target, eval_output)
        else:
            raise ValueError(f"Unknown tune_metric='{tune_metric}'")
        
        # Explicit cleanup
        del eval_input, eval_target, eval_output
        
        return {tune_metric: metric_val}


def evaluate_val_frozen(generators, batch, device, flow_mode, tune_metric='SSIM', use_ground_truth_rois=False):
    """
    Evaluate frozen flow network on validation or QA batch.
    
    Runs frozen attenuation network to extract features, then passes them to
    trainable activity network.
    
    Evaluation mode is automatically detected:
    - If batch contains 'hotMask': QA mode (returns CR metrics via ROI_simple_phantom)
    - Otherwise: Validation mode (returns specified tune_metric)
    
    Args:
        generators: Tuple containing (gen_act, gen_atten) for frozen flow networks
        batch: dict with tensors (on CPU). Must contain:
               - 'act_sino_scaled', 'act_image_scaled' for activity target
               - Either 'atten_sino_scaled' (coflow) or 'atten_image_scaled' (counterflow)
               - 'hotMask', 'hotBackgroundMask' for QA mode (optional)
        device: torch device to move tensors to
        flow_mode: str, 'coflow' or 'counterflow'
        tune_metric: str, which metric to compute for validation ('MSE', 'SSIM', or 'CUSTOM')
        use_ground_truth_rois: bool, if True use ground truth for ROI checks (QA mode only)
    
    Returns:
        For validation: dict with single metric: {tune_metric: float}
        For QA: dict with CR metrics (CR_symmetric, hot_underestimation, cold_overestimation)
    """
    # Unpack generators (frozen flow uses two generators)
    gen_act, gen_atten = generators
    
    # Detect QA mode by presence of hotMask
    is_qa_mode = 'hotMask' in batch and batch['hotMask'] is not None
    
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
    
    # Compute metrics based on eval mode
    if is_qa_mode:
        # QA mode: use ROI metrics with hotMask
        hotMask = batch['hotMask'].to(device)
        eval_qa_output = eval_act_image if use_ground_truth_rois else eval_output
        cr_metrics = ROI_simple_phantom(eval_act_image, eval_qa_output, hotMask)
        
        # Explicit cleanup
        del eval_act_sino, eval_act_image, atten_input, frozen_enc_feats, frozen_dec_feats, eval_target, eval_output, hotMask, eval_qa_output

        return cr_metrics
    else:
        # Validation mode: use specified metric
        if tune_metric == 'MSE':
            metric_val = calculate_metric(eval_target, eval_output, MSE)
        elif tune_metric == 'SSIM':
            metric_val = calculate_metric(eval_target, eval_output, SSIM)
        elif tune_metric == 'CUSTOM':
            metric_val = custom_metric(eval_target, eval_output)
        else:
            raise ValueError(f"Unknown tune_metric='{tune_metric}'")
        
        # Explicit cleanup
        del eval_act_sino, eval_act_image, atten_input, frozen_enc_feats, frozen_dec_feats, eval_target, eval_output
        
        return {tune_metric: metric_val}


def evaluate_qa(gen, batch, device, tune_metric='SSIM', use_ground_truth_rois=False):
    """
    Deprecated: Use evaluate_val() instead.
    
    This function is kept for backward compatibility but evaluate_val() now handles both
    validation and QA modes automatically by detecting the presence of 'hotMask' in batch.
    """
    return evaluate_val((gen,), batch, device, train_SI=True, network_type='ACT', 
                       tune_metric=tune_metric, use_ground_truth_rois=use_ground_truth_rois)


def evaluate_qa_frozen(gen_act, gen_atten, batch, device, flow_mode, tune_metric='SSIM', use_ground_truth_rois=False):
    """
    Deprecated: Use evaluate_val_frozen() instead.
    
    This function is kept for backward compatibility but evaluate_val_frozen() now handles both
    validation and QA modes automatically by detecting the presence of 'hotMask' in batch.
    """
    return evaluate_val_frozen((gen_act, gen_atten), batch, device, flow_mode, 
                              tune_metric=tune_metric, use_ground_truth_rois=use_ground_truth_rois)


def _evaluate_batch(generators, batch, device, train_SI, network_type, tune_metric, report_num):
    """
    Evaluate batch on generator(s) and return metrics.
    
    Internal helper for report_tune_metrics. Routes between simple and frozen flow evaluators
    based on network_type. QA mode is auto-detected by presence of hotMask in batch.
    """
    is_frozen_flow = network_type in ('FROZEN_COFLOW', 'FROZEN_COUNTERFLOW')
    
    if is_frozen_flow:
        flow_mode = 'coflow' if network_type == 'FROZEN_COFLOW' else 'counterflow'
        metrics = evaluate_val_frozen(generators, batch, device, flow_mode, 
                                     tune_metric=tune_metric, use_ground_truth_rois=False)
    else:
        metrics = evaluate_val(generators, batch, device, train_SI, network_type, 
                              tune_metric=tune_metric, use_ground_truth_rois=False)
    
    return metrics


def report_tune_metrics(generators, paths, config, settings, tune_dataframe, tune_dataframe_path, 
                        train_SI, tune_dataframe_fraction, tune_max_t, report_num, 
                        example_num, batch_step, epoch, device):
    """
    Load evaluation batch, compute metrics, update dataframe, and report to Ray Tune.
    
    Args:
        generators: Tuple of generator model(s):
                    - (gen,) for simple networks (ACT, ATTEN, CONCAT)
                    - (gen_act, gen_atten) for frozen flow networks
                    All generators will be set to eval mode and restored to train mode
        paths: Path dictionary with validation/QA data paths
        config: Configuration dictionary
        settings: Settings dictionary with tune_report_for, tune_metric, etc.
        tune_dataframe: Current tune dataframe
        tune_dataframe_path: Path to save dataframe
        train_SI: Training direction (True for sino→image)
        tune_dataframe_fraction: Fraction of tune_max_t at which to save dataframe
        tune_max_t: Maximum number of reports
        report_num: Current report number
        example_num: Current example number
        batch_step: Current batch step
        epoch: Current epoch
        device: Device string
    
    Returns:
        Updated tune_dataframe
    
    Raises:
        ValueError: if NaN detected in optimization metric or invalid tune_report_for
    """
    # ===== SETUP: Generator mode & configuration =====
    for gen in generators:
        gen.eval()
    
    t_total = t_start = time.time()
    tune_report_for = settings.get('tune_report_for', 'val')
    network_type = config.get('network_type', 'ACT')
    tune_metric = settings.get('tune_metric', 'SSIM')
    
    # ===== LOAD: Batch data based on report type =====
    if tune_report_for == 'val':
        batch = load_validation_batch(paths, config, settings)
    elif tune_report_for == 'qa':
        batch = load_qa_batch(paths, config, settings, augment=('SI', True))
    else:
        raise ValueError(f"Invalid tune_report_for='{tune_report_for}'")
    
    t_start = display_times(f"[TUNE_REPORT #{report_num}] Batch load time", t_start, PRINT_REPORT_TIMING)
    
    # ===== EVALUATE: Compute metrics on batch =====
    metrics = _evaluate_batch(generators, batch, device, train_SI, network_type, tune_metric, report_num)
    t_start = display_times(f"[TUNE_REPORT #{report_num}] Evaluation time", t_start, PRINT_REPORT_TIMING)
    
    # ===== UPDATE: Save dataframe checkpoint if reached fraction =====
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
    report(metrics)
    

    t_start = display_times(f"[TUNE_REPORT #{report_num}] TOTAL Eval Time", t_total, PRINT_REPORT_TIMING)

    return tune_dataframe
