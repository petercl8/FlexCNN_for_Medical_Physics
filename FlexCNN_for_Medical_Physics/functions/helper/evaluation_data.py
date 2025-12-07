"""
Helpers for evaluation data loading during tuning.

Supports three evaluation modes:
- 'same': report on current training batch (no separate loader)
- 'val': report on fixed validation batch (deterministic, cached)
- 'qa': report on QA phantom data with masks (stub for future CRC metrics)
"""

import numpy as np
import torch
from FlexCNN_for_Medical_Physics.classes.dataset import NpArrayDataSet

def load_validation_batch(paths, config, settings):
    """
    Load a fixed validation batch with deterministic sampling.
    
    Uses a fixed random seed (42) to ensure deterministic sampling across trials.
    Validation data is loaded to CPU; the caller is responsible for device placement.
    
    Args:
        paths: dict with 'tune_val_sino_path', 'tune_val_image_path'
        config: dict with 'image_size', 'sino_size', 'image_channels', 'sino_channels'
        settings: dict with 'eval_batch_size' (e.g., 32)
    
    Returns:
        dict with keys 'sino' and 'image' (both tensors on CPU)
    
    Raises:
        ValueError: if validation paths are not set
    """
    # Check paths
    if paths.get('tune_val_sino_path') is None or paths.get('tune_val_image_path') is None:
        raise ValueError(
            "tune_report_for='val' requires tune_val_sino_file and tune_val_image_file to be set in the notebook."
        )
    
    eval_batch_size = settings.get('eval_batch_size', 32)
    
    # Load full validation set
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
    
    # Deterministically sample a fixed batch
    np.random.seed(42)  # Fixed seed ensures same indices across all trials
    indices = np.random.choice(len(val_dataset), size=eval_batch_size, replace=False)
    
    # Extract the batch
    sino_batch = []
    image_batch = []
    for idx in indices:
        sino, image = val_dataset[idx][:2]  # Unpack sino, image (ignore recon1/recon2)
        sino_batch.append(sino)
        image_batch.append(image)
    
    return {
        'sino': torch.stack(sino_batch),
        'image': torch.stack(image_batch),
    }

def load_qa_phantom_data(paths, config):
    """
    Load QA phantom sinogram and masks.
    
    QA phantoms are used to compute contrast recovery coefficients (CRC) and other
    QA-specific metrics. Unlike validation, we don't need ground truth images since
    CRC is computed directly from the reconstructed images and the masks.
    
    Loads to CPU; caller handles device placement.
    
    Args:
        paths: dict with 'tune_qa_sino_path', 'tune_qa_backMask_path', 'tune_qa_hotMask_path'
        config: dict with 'image_size', 'sino_size', 'image_channels', 'sino_channels'
    
    Returns:
        dict with keys 'sino', 'backMask', 'hotMask' (all tensors on CPU)
    
    Raises:
        ValueError: if QA paths are not set
    """
    # Check paths
    if (paths.get('tune_qa_sino_path') is None or 
        paths.get('tune_qa_backMask_path') is None or 
        paths.get('tune_qa_hotMask_path') is None):
        raise ValueError(
            "tune_report_for='qa' requires tune_qa_sino_file, tune_qa_backMask_file, and tune_qa_hotMask_file to be set in the notebook."
        )
    
    # Load QA sinogram and masks
    sino_qa = np.load(paths['tune_qa_sino_path'], mmap_mode='r')
    backMask = np.load(paths['tune_qa_backMask_path'], mmap_mode='r')
    hotMask = np.load(paths['tune_qa_hotMask_path'], mmap_mode='r')
    
    # Convert to tensors on CPU
    return {
        'sino': torch.from_numpy(sino_qa).float(),
        'backMask': torch.from_numpy(backMask).float(),
        'hotMask': torch.from_numpy(hotMask).float(),
    }

def eval_data(paths, config, settings, current_batch=None):
    """
    Load evaluation data based on tune_report_for setting.
    
    Returns a dict with 'sino' key (and either 'image' for val/same, or masks for qa).
    Note: Does NOT cache—caching is handled by get_eval_batch().
    
    Args:
        paths: dict with all data paths
        config: dict with network config
        settings: dict with 'tune_report_for' and 'eval_batch_size'
        current_batch: tuple (sino_scaled, act_map_scaled) for 'same' mode; required if tune_report_for='same'
    
    Returns:
        dict with 'sino' and either 'image' (for val/same) or 'backMask'/'hotMask' (for qa)
    
    Raises:
        ValueError: if tune_report_for is invalid or current_batch is None when needed
    """
    tune_report_for = settings.get('tune_report_for', 'same')
    
    if tune_report_for == 'same':
        if current_batch is None:
            raise ValueError("tune_report_for='same' requires current_batch to be provided.")
        sino_scaled, act_map_scaled = current_batch
        return {'sino': sino_scaled, 'image': act_map_scaled}
    elif tune_report_for == 'val':
        return load_validation_batch(paths, config, settings)
    elif tune_report_for == 'qa':
        return load_qa_phantom_data(paths, config)
    else:
        raise ValueError(
            f"Invalid tune_report_for='{tune_report_for}'. "
            "Must be 'same', 'val', or 'qa'."
        )


def get_eval_batch(gen, paths, config, settings, cache, current_batch=None, device='cpu', train_SI=True):
    """
    Complete evaluation pipeline: load data, assign inputs/targets, move to device, generate network output.
    
    Handles all caching: 
    - First call: loads data, moves to device, assigns inputs/targets, caches them, generates output
    - Subsequent calls: reuses cached eval_input/target, only regenerates eval_output
    
    Args:
        gen: Generator network (in eval mode)
        paths: dict with all data paths
        config: dict with network config
        settings: dict with 'tune_report_for' and 'eval_batch_size'
        cache: dict to store/retrieve cached data (populated with 'eval_input', 'eval_target', and mode-specific keys)
        current_batch: tuple (sino_scaled, act_map_scaled) for 'same' mode
        device: torch device to move tensors to
        train_SI: bool, True for sino→image, False for image→sino
    
    Returns:
        (eval_input, eval_target, eval_output, cache): 
            eval_input/target/output are tensors on device
            cache is populated with cached data
    """
    tune_report_for = settings.get('tune_report_for', 'same')
    
    # Load evaluation data on first call; reuse loaded data on subsequent calls
    if 'eval_data_dict' not in cache:
        cache['eval_data_dict'] = eval_data(paths, config, settings, current_batch=current_batch)
    eval_data_dict = cache['eval_data_dict']
    
    # Initialize eval_input and eval_target on first call; reuse on subsequent calls
    if 'eval_input' not in cache:
        # Move sinogram to device
        eval_sino = eval_data_dict['sino'].to(device)
        
        # Handle different data types
        if 'image' in eval_data_dict:
            # val or same mode: has ground truth image
            eval_image = eval_data_dict['image'].to(device)
            if train_SI:
                cache['eval_target'] = eval_image
                cache['eval_input'] = eval_sino
            else:
                cache['eval_target'] = eval_sino
                cache['eval_input'] = eval_image
        else:
            # qa mode: no ground truth image, only sino and masks
            cache['eval_target'] = None
            cache['eval_input'] = eval_sino
    
    eval_input = cache['eval_input']
    eval_target = cache['eval_target']
    
    # Generate network output (only part that changes on each report)
    with torch.no_grad():
        eval_output = gen(eval_input)
    
    return eval_input, eval_target, eval_output, cache
