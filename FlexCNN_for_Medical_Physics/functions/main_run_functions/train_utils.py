"""
Shared training utilities for supervisory trainables.

Provides infrastructure functions (generator/optimizer creation, checkpointing),
reporting functions (tuning metrics, visualization), and batch collation for
reuse across trainable_supervisory.py, trainable_frozen_backbone.py, and trainable.py.
"""

import os
import torch
import logging

from FlexCNN_for_Medical_Physics.classes.generators import Generator_180, Generator_288, Generator_320
from FlexCNN_for_Medical_Physics.functions.helper.metrics_wrappers import calculate_metric, update_tune_dataframe
from FlexCNN_for_Medical_Physics.functions.helper.metrics import SSIM, MSE, patchwise_moment_metric
from FlexCNN_for_Medical_Physics.functions.helper.display_images import show_single_unmatched_tensor, show_multiple_matched_tensors
from FlexCNN_for_Medical_Physics.functions.helper.evaluation_data_random import load_validation_batch, load_qa_batch, evaluate_val, evaluate_qa

# Module logger for optional Tune debug output
logger = logging.getLogger(__name__)


def collate_nested(batch):
    """
    Custom collate function for nested tuple batches with optional None fields.
    Expects each sample to be: (act_data, atten_data, recon_data) where:
      act_data = (sino, image)
      atten_data = (atten_sino, atten_image)
      recon_data = (recon1, recon2)
    
    Stacks tensors when all samples provide them; keeps None when all samples have None.
    Raises ValueError if a field has mixed None/tensor across the batch (fail-fast).
    """
    def _stack_or_none(items):
        """Stack if all tensors, None if all None, else raise."""
        non_none = [item for item in items if item is not None]
        if len(non_none) == 0:
            return None
        elif len(non_none) == len(items):
            return torch.stack(non_none, dim=0)
        else:
            raise ValueError(f"Mixed None/tensor in batch field: {len(non_none)}/{len(items)} non-None")
    
    # Unpack nested structure from each sample
    act_data_list = [sample[0] for sample in batch]
    atten_data_list = [sample[1] for sample in batch]
    recon_data_list = [sample[2] for sample in batch]
    
    # Stack each field independently
    sino_batch = _stack_or_none([act[0] for act in act_data_list])
    image_batch = _stack_or_none([act[1] for act in act_data_list])
    
    atten_sino_batch = _stack_or_none([atten[0] for atten in atten_data_list])
    atten_image_batch = _stack_or_none([atten[1] for atten in atten_data_list])
    
    recon1_batch = _stack_or_none([recon[0] for recon in recon_data_list])
    recon2_batch = _stack_or_none([recon[1] for recon in recon_data_list])
    
    # Return nested structure
    act_data = (sino_batch, image_batch)
    atten_data = (atten_sino_batch, atten_image_batch)
    recon_data = (recon1_batch, recon2_batch)
    
    return act_data, atten_data, recon_data


def create_generator(config: dict, device: str):
    """
    Instantiate the appropriate Generator class based on config input size.
    
    Args:
        config: Configuration dictionary containing 'image_size', 'sino_size', 'train_SI'.
        device: Device to place generator on (e.g., 'cuda:0' or 'cpu').
    
    Returns:
        Generator instance (Generator_180, Generator_288, or Generator_320) on specified device.
    
    Raises:
        ValueError if input_size is not 180, 288, or 320.
    """
    train_SI = config['train_SI']
    image_size = config['image_size']
    sino_size = config['sino_size']
    
    # Determine input size based on train_SI direction
    input_size = sino_size if train_SI else image_size
    
    # Select appropriate Generator class
    if input_size == 180:
        GeneratorClass = Generator_180
    elif input_size == 288:
        GeneratorClass = Generator_288
    elif input_size == 320:
        GeneratorClass = Generator_320
    else:
        raise ValueError(f"No Generator class available for input_size={input_size}. Supported sizes: 180, 288, 320")
    
    # Instantiate and move to device
    gen = GeneratorClass(config=config, gen_SI=train_SI).to(device)
    return gen


def create_optimizer(model, config: dict) -> torch.optim.Adam:
    """
    Create optimizer for generator with optional separate learning rate group for scale parameters.
    
    Args:
        model: Generator model instance.
        config: Configuration dictionary containing 'gen_b1', 'gen_b2', 'gen_lr', 
                'SI_output_scale_lr_mult' or 'IS_output_scale_lr_mult', and 'train_SI'.
    
    Returns:
        torch.optim.Adam optimizer, potentially with parameter groups for learnable scale.
    """
    train_SI = config['train_SI']
    betas = (config['gen_b1'], config['gen_b2'])
    base_lr = config['gen_lr']
    
    # Determine scale LR multiplier based on direction
    if train_SI:
        scale_lr_mult = config.get('SI_output_scale_lr_mult', 1.0)
    else:
        scale_lr_mult = config.get('IS_output_scale_lr_mult', 1.0)
    
    # If model has learnable output scale, create separate param group for it
    if getattr(model, 'output_scale_learnable', False):
        scale_param = [model.log_output_scale]
        main_params = [p for n, p in model.named_parameters() if n != 'log_output_scale']
        optimizer = torch.optim.Adam(
            [
                {'params': main_params, 'lr': base_lr, 'betas': betas},
                {'params': scale_param, 'lr': scale_lr_mult * base_lr, 'betas': betas, 'weight_decay': 0.0},
            ]
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=betas)
    
    return optimizer


def build_checkpoint_dict(model, optimizer, config: dict, epoch: int, batch_step: int) -> dict:
    """
    Assemble checkpoint dictionary with model state, optimizer state, and metadata.
    
    Args:
        model: Generator model instance.
        optimizer: Optimizer instance.
        config: Configuration dictionary (not saved; provided fresh at load time).
        epoch: Current epoch number.
        batch_step: Current batch step number.
    
    Returns:
        Dictionary with keys: 'epoch', 'batch_step', 'gen_state_dict', 'gen_opt_state_dict'.
    """
    checkpoint = {
        'epoch': epoch,
        'batch_step': batch_step,
        'gen_state_dict': model.state_dict(),
        'gen_opt_state_dict': optimizer.state_dict(),
    }
    return checkpoint


def save_checkpoint(checkpoint_dict: dict, checkpoint_path: str) -> None:
    """
    Save checkpoint dictionary to file.
    
    Args:
        checkpoint_dict: Checkpoint dictionary to save.
        checkpoint_path: Path to save checkpoint to.
    """
    torch.save(checkpoint_dict, checkpoint_path)


def load_checkpoint(checkpoint_path: str) -> tuple:
    """
    Load checkpoint from file.
    
    Args:
        checkpoint_path: Path to checkpoint file.
    
    Returns:
        Tuple of (epoch, batch_step, gen_state_dict, gen_opt_state_dict).
    
    Raises:
        FileNotFoundError if checkpoint file does not exist.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']
    batch_step = checkpoint['batch_step']
    gen_state_dict = checkpoint['gen_state_dict']
    gen_opt_state_dict = checkpoint['gen_opt_state_dict']
    
    return epoch, batch_step, gen_state_dict, gen_opt_state_dict


def log_tune_debug(gen, epoch: int, batch_step: int, gen_loss, device: str) -> None:
    """
    Log per-step debug information for Tune runs.
    
    Args:
        gen: Generator model instance.
        epoch: Current epoch number.
        batch_step: Current batch step number.
        gen_loss: Loss tensor for current batch.
        device: Device string for context.
    """
    try:
        param_norm = sum(p.detach().norm().item() for p in gen.parameters())
    except Exception:
        param_norm = float('nan')
    
    try:
        grad_norm = sum(p.grad.detach().norm().item() for p in gen.parameters() if p.grad is not None)
    except Exception:
        grad_norm = float('nan')
    
    try:
        output_scale = torch.exp(gen.log_output_scale).item() if getattr(gen, 'output_scale_learnable', False) else gen.fixed_output_scale.item()
    except Exception:
        output_scale = float('nan')
    
    try:
        output_scale_grad = gen.log_output_scale.grad.detach().item() if getattr(gen, 'output_scale_learnable', False) and gen.log_output_scale.grad is not None else 0.0
    except Exception:
        output_scale_grad = float('nan')
    
    logger.debug(
        f"[TUNE_DEBUG] epoch={epoch} batch_step={batch_step} loss={gen_loss.item():.6f} "
        f"param_norm={param_norm:.6f} grad_norm={grad_norm:.6f} "
        f"output_scale={output_scale:.6f} output_scale_grad={output_scale_grad:.6f} device={device}"
    )


def report_tune_metrics(gen, paths, config, settings, tune_dataframe, tune_dataframe_path, 
                        train_SI, tune_dataframe_fraction, tune_max_t, report_num, 
                        example_num, batch_step, epoch, session, device):
    """
    Load evaluation batch, compute metrics, update dataframe, and report to Ray Tune.
    
    Args:
        gen: Generator model (will be set to eval mode and restored to train mode)
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
        session: Ray Tune session object
        device: Device string
    
    Returns:
        Updated tune_dataframe
    
    Raises:
        ValueError: if NaN detected in optimization metric or invalid tune_report_for
    """
    # Set generator to eval mode for validation
    gen.eval()
    
    # Load fresh random batch and evaluate
    tune_report_for = settings.get('tune_report_for', 'val')
    if tune_report_for == 'val':
        batch = load_validation_batch(paths, config, settings)
        metrics = evaluate_val(gen, batch, device, train_SI)

        # Update dataframe at specified fraction
        if int(tune_dataframe_fraction * tune_max_t) == report_num:
            tune_dataframe = update_tune_dataframe(
                tune_dataframe, tune_dataframe_path, gen, config, metrics
            )

    elif tune_report_for == 'qa':
        batch = load_qa_batch(paths, config, settings, augment=('SI', True))
        metrics = evaluate_qa(gen, batch, device, use_ground_truth_rois=False)

        # Update dataframe at specified fraction
        if int(tune_dataframe_fraction * tune_max_t) == report_num:
            tune_dataframe = update_tune_dataframe(
                tune_dataframe, tune_dataframe_path, gen, config, metrics
            )

    else:
        raise ValueError(f"Invalid tune_report_for='{tune_report_for}'")
    
    # Explicit memory cleanup to prevent accumulation across trials
    del batch
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    
    # Restore generator to train mode
    gen.train()
    
    # NaN check
    metric_to_check = settings.get('tune_metric', 'SSIM')
    if metric_to_check in metrics and (
        metrics[metric_to_check] is None or 
        torch.isnan(torch.tensor(metrics[metric_to_check])).item()
    ):
        raise ValueError(f"NaN detected in {metric_to_check}, terminating trial")

    # Add metadata to metrics
    metrics.update({
        'example_num': example_num,
        'batch_step': batch_step,
        'epoch': epoch
    })
    
    # Report to Ray Tune
    session.report(metrics)
    
    return tune_dataframe


def visualize_train(batch_data, target, CNN_output, mean_gen_loss, mean_CNN_MSE, mean_CNN_SSIM, 
                    epoch, batch_step, example_num):
    """
    Display training progress: metrics, input/target/output visualizations.
    
    Args:
        batch_data: Dict with 'input', 'target', 'atten_image', 'atten_sino' (some may be None)
        target: Target tensor for current batch
        CNN_output: Network output for current batch
        mean_gen_loss: Mean generator loss over display_step
        mean_CNN_MSE: Mean MSE over display_step
        mean_CNN_SSIM: Mean SSIM over display_step
        epoch: Current epoch
        batch_step: Current batch step
        example_num: Current example number
    """
    input_ = batch_data['input']
    atten_image_scaled = batch_data.get('atten_image')
    atten_sino_scaled = batch_data.get('atten_sino')
    
    print('================Training===================')
    print(f'CURRENT PROGRESS: epoch: {epoch} / batch_step: {batch_step} / image #: {example_num}')
    print(f'mean_gen_loss: {mean_gen_loss}')
    print(f'mean_CNN_MSE : {mean_CNN_MSE}')
    print(f'mean_CNN_SSIM: {mean_CNN_SSIM}')
    print('===========================================')
    print('Last Batch MSE: ', calculate_metric(target, CNN_output, MSE))
    print('Last Batch SSIM: ', calculate_metric(target, CNN_output, SSIM))
    print('Last Batch LDM: ', patchwise_moment_metric(target, CNN_output, return_per_moment=True))
    print('Input Sinogram:')
    if input_.shape[1] == 1:
        show_single_unmatched_tensor(input_[0:2], fig_size=5)
    else:
        show_single_unmatched_tensor(input_[0:2], fig_size=15)
    print(input_.shape)
    print('Target/Output:')
    show_multiple_matched_tensors(target[0:8], CNN_output[0:8])

    if atten_image_scaled is not None and atten_sino_scaled is not None:
        show_single_unmatched_tensor(atten_image_scaled[0:2], fig_size=2)
        show_single_unmatched_tensor(atten_sino_scaled[0:2], fig_size=2)


def visualize_test(batch_data, target, CNN_output, mean_CNN_MSE, mean_CNN_SSIM, 
                   mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM):
    """
    Display test results: metrics and comparisons with reconstructions.
    
    Args:
        batch_data: Dict with 'input', 'recon1_output', 'recon2_output'
        target: Target tensor for current batch
        CNN_output: Network output for current batch
        mean_CNN_MSE: Mean network MSE
        mean_CNN_SSIM: Mean network SSIM
        mean_recon1_MSE: Mean recon1 MSE
        mean_recon1_SSIM: Mean recon1 SSIM
        mean_recon2_MSE: Mean recon2 MSE
        mean_recon2_SSIM: Mean recon2 SSIM
    """
    input_ = batch_data['input']
    recon1_output = batch_data['recon1_output']
    recon2_output = batch_data['recon2_output']
    
    print('==================Testing==================')
    print(f'mean_CNN_MSE/mean_recon2_MSE/mean_recon1_MSE : {mean_CNN_MSE}/{mean_recon2_MSE}/{mean_recon1_MSE}')
    print(f'mean_CNN_SSIM/mean_recon2_SSIM/mean_recon1_SSIM: {mean_CNN_SSIM}/{mean_recon2_SSIM}/{mean_recon1_SSIM}')
    print('===========================================')
    print('Input Sinogram:')
    show_single_unmatched_tensor(input_[0:2], fig_size=25)
    print('Target/Output/Recon2/Recon1:')
    show_multiple_matched_tensors(target[0:9], CNN_output[0:9], recon2_output[0:9], recon1_output[0:9])


def visualize_mode(batch_data, target, CNN_output, visualize_batch_size, visualize_offset):
    """
    Display visualization mode outputs: input, target, reconstructions, network output.
    
    Args:
        batch_data: Dict with 'input', 'recon1_output', 'recon2_output' (some may be None)
        target: Target tensor for current batch
        CNN_output: Network output for current batch
        visualize_batch_size: Number of images to display
        visualize_offset: Offset for display numbering
    """
    input_ = batch_data['input']
    recon1_output = batch_data.get('recon1_output')
    recon2_output = batch_data.get('recon2_output')
    
    if visualize_batch_size == 120:
        batch_step = batch_data.get('batch_step', 0)
        print(f'visualize_offset: {visualize_offset}, Image Number (batch_step*120): {batch_step*120}')
        show_single_unmatched_tensor(target, grid=True, cmap='inferno', fig_size=1)
    else:
        print('Input:')
        show_single_unmatched_tensor(input_[0:visualize_batch_size])
        
        # Handle None reconstructions gracefully
        if recon1_output is not None and recon2_output is not None:
            print('Target/Recon2/Recon1/Output:')
            show_multiple_matched_tensors(
                target[0:visualize_batch_size], 
                recon1_output[0:visualize_batch_size], 
                recon2_output[0:visualize_batch_size], 
                CNN_output[0:visualize_batch_size]
            )
        else:
            print('Target/Output:')
            show_multiple_matched_tensors(target[0:visualize_batch_size], CNN_output[0:visualize_batch_size])
