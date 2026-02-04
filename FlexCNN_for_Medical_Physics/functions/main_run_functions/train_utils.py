"""
Shared training utilities for supervisory trainables.

Provides infrastructure functions (generator/optimizer creation, checkpointing),
visualization functions, batch collation, and shared utilities for training pipelines.

Cross-validation (tuning metrics reporting) has been moved to cross_validation module.
"""
import os
import time
import torch
import logging

from FlexCNN_for_Medical_Physics.classes.generators import Generator_180, Generator_288, Generator_320
from FlexCNN_for_Medical_Physics.functions.helper.timing import display_times
from FlexCNN_for_Medical_Physics.functions.helper.metrics_wrappers import calculate_metric, reconstruct_images_and_update_test_dataframe
from FlexCNN_for_Medical_Physics.functions.helper.metrics import SSIM, MSE, patchwise_moment_metric
from FlexCNN_for_Medical_Physics.functions.helper.display_images import show_single_unmatched_tensor, show_multiple_matched_tensors
from FlexCNN_for_Medical_Physics.functions.helper.reconstruction_projection import reconstruct


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


def init_checkpoint_state(load_state, run_mode, checkpoint_path, num_epochs, device):
    """
    Load checkpoint if needed and initialize epoch/batch state based on run_mode.
    
    Args:
        load_state: Boolean indicating whether to load from checkpoint
        run_mode: String indicating run mode ('train', 'tune', 'test', 'visualize')
        checkpoint_path: Path to checkpoint file
        num_epochs: Number of epochs to run
        device: Device string for torch.load map_location
    
    Returns:
        Tuple of (start_epoch, end_epoch, batch_step, gen_state_dict, gen_opt_state_dict)
        For test/visualize, gen_state_dict and gen_opt_state_dict are returned for proper loading.
    
    Raises:
        FileNotFoundError if checkpoint file does not exist and load_state is True
    """
    if load_state:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        epoch_loaded = checkpoint['epoch']
        batch_step_loaded = checkpoint['batch_step']
        gen_state_dict = checkpoint['gen_state_dict']
        gen_opt_state_dict = checkpoint['gen_opt_state_dict']
        if run_mode in ('test', 'visualize'):
            start_epoch = 0
            end_epoch = 1
            batch_step = 0
        else:  # train (for tune mode, load_state = False)
            start_epoch = epoch_loaded
            end_epoch = num_epochs
            batch_step = batch_step_loaded
    else:
        start_epoch = 0
        batch_step = 0
        end_epoch = num_epochs
        gen_state_dict = None
        gen_opt_state_dict = None
    return start_epoch, end_epoch, batch_step, gen_state_dict, gen_opt_state_dict


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


def visualize_train(batch_data, mean_gen_loss, current_CNN_MSE, current_CNN_SSIM, 
                    epoch, batch_step, example_num):
    """
    Display training progress: loss and current batch metrics, input/target/output visualizations.
    
    Args:
        batch_data: Dict with 'input', 'target', 'CNN_output', 'recon1_output', 'recon2_output'
        mean_gen_loss: Mean generator loss over display_step (accumulated from training batches)
        current_CNN_MSE: MSE on current batch only
        current_CNN_SSIM: SSIM on current batch only
        epoch: Current epoch
        batch_step: Current batch step
        example_num: Current example number
    """
    input_ = batch_data['input']
    target = batch_data['target']
    CNN_output = batch_data['CNN_output']
    recon1_output = batch_data['recon1_output']
    recon2_output = batch_data['recon2_output']
    
    print('================Training===================')
    print(f'CURRENT PROGRESS: epoch: {epoch} / batch_step: {batch_step} / image #: {example_num}')
    print(f'mean_gen_loss: {mean_gen_loss}')
    print(f'current_CNN_MSE : {current_CNN_MSE}')
    print(f'current_CNN_SSIM: {current_CNN_SSIM}')
    print('===========================================')
    print('Current Batch LDM: ', patchwise_moment_metric(target, CNN_output, return_per_moment=True))
    print('Input Sinogram:')
    if input_.shape[1] == 1:
        show_single_unmatched_tensor(input_[0:2], fig_size=5)
    else:
        show_single_unmatched_tensor(input_[0:2], fig_size=20)
    print(input_.shape)
    print('Target/Output:')
    show_multiple_matched_tensors(target[0:8], CNN_output[0:8])
    
    if recon1_output is not None and recon2_output is not None:
        print('Recon1/Recon2:')
        show_multiple_matched_tensors(recon1_output[0:8], recon2_output[0:8])


def visualize_test(batch_data, mean_CNN_MSE, mean_CNN_SSIM, 
                   mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM):
    """
    Display test results: metrics and comparisons with reconstructions.
    
    Args:
        batch_data: Dict with 'input', 'target', 'CNN_output', 'recon1_output', 'recon2_output'
        mean_CNN_MSE: Mean network MSE
        mean_CNN_SSIM: Mean network SSIM
        mean_recon1_MSE: Mean recon1 MSE
        mean_recon1_SSIM: Mean recon1 SSIM
        mean_recon2_MSE: Mean recon2 MSE
        mean_recon2_SSIM: Mean recon2 SSIM
    """
    input_ = batch_data['input']
    target = batch_data['target']
    CNN_output = batch_data['CNN_output']
    recon1_output = batch_data['recon1_output']
    recon2_output = batch_data['recon2_output']
    
    print('==================Testing==================')
    if recon1_output is not None and recon2_output is not None:
        print(f'mean_CNN_MSE/mean_recon2_MSE/mean_recon1_MSE : {mean_CNN_MSE}/{mean_recon2_MSE}/{mean_recon1_MSE}')
        print(f'mean_CNN_SSIM/mean_recon2_SSIM/mean_recon1_SSIM: {mean_CNN_SSIM}/{mean_recon2_SSIM}/{mean_recon1_SSIM}')
    else:
        print(f'mean_CNN_MSE : {mean_CNN_MSE}')
        print(f'mean_CNN_SSIM: {mean_CNN_SSIM}')
    print('===========================================')
    print('Input Sinogram:')
    show_single_unmatched_tensor(input_[0:2], fig_size=25)
    
    if recon1_output is not None and recon2_output is not None:
        print('Target/Output/Recon2/Recon1:')
        show_multiple_matched_tensors(target[0:9], CNN_output[0:9], recon2_output[0:9], recon1_output[0:9])
    else:
        print('Target/Output:')
        show_multiple_matched_tensors(target[0:9], CNN_output[0:9])


def visualize_visualize(batch_data, visualize_batch_size, visualize_offset):
    """
    Display visualization mode outputs: input, target, reconstructions, network output.
    
    Args:
        batch_data: Dict with 'input', 'target', 'CNN_output', 'recon1_output', 'recon2_output'
        visualize_batch_size: Number of images to display
        visualize_offset: Offset for display numbering
    """
    input_ = batch_data['input']
    target = batch_data['target']
    CNN_output = batch_data['CNN_output']
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


def route_batch_inputs(train_SI, batch_tensors, network_type=None):
    """
    Route batch inputs to correct input/target based on domain and direction.
    For CONCAT networks, concatenates activity and attenuation sinograms.
    
    Args:
        train_SI: Boolean indicating sinogram->image direction (only checked if network_type is not ATTEN)
        batch_tensors: Dict with 'sino_scaled', 'act_map_scaled', 'atten_sino_scaled', 'atten_image_scaled'
        network_type: String indicating network type (enables CONCAT concatenation)
    
    Returns:
        Tuple of (input_, target)
    """
    if network_type == 'CONCAT':
        # CONCAT mode: concatenate activity sinogram (3ch) + attenuation sinogram (1ch) -> 4ch
        input_ = torch.cat([batch_tensors['act_sino_scaled'], 
                   batch_tensors['atten_sino_scaled']], dim=1)
        target = batch_tensors['act_image_scaled']
    elif network_type == 'ATTEN':
        # Attenuation domain: honor train_SI to support both directions
        if train_SI:
            input_ = batch_tensors['atten_sino_scaled']
            target = batch_tensors['atten_image_scaled']
        else:
            input_ = batch_tensors['atten_image_scaled']
            target = batch_tensors['atten_sino_scaled']
    elif network_type == 'ACT':
        # Activity domain: use configured train_SI direction
        if train_SI:
            input_ = batch_tensors['act_sino_scaled']
            target = batch_tensors['act_image_scaled']
        else:
            input_ = batch_tensors['act_image_scaled']
            target = batch_tensors['act_sino_scaled']
    else:
        raise ValueError(f"Invalid network_type='{network_type}' for routing batch inputs")
    
    return input_, target


def generate_reconstructions_for_visualization(recon1, recon2, input_, config):
    """
    Generate or pass-through reconstructions for visualization and test mode.
    
    Args:
        recon1: Precomputed recon1 or None
        recon2: Precomputed recon2 or None
        input_: Input tensor (sinogram) for FBP/MLEM reconstruction
        config: Configuration dict with 'gen_image_size', 'SI_normalize', 'SI_fixedScale'
    
    Returns:
        Tuple of (recon1_output, recon2_output)
    """
    if recon1 is not None:
        recon1_output = recon1
    else:
        recon1_output = reconstruct(input_, config['gen_image_size'], config['SI_normalize'], config['SI_fixedScale'], recon_type='FBP')
    
    if recon2 is not None:
        recon2_output = recon2
    else:
        recon2_output = reconstruct(input_, config['gen_image_size'], config['SI_normalize'], config['SI_fixedScale'], recon_type='MLEM')
    
    return recon1_output, recon2_output


def compute_test_metrics(network_type, input_, CNN_output, target, act_image_scaled, test_dataframe, config, recon1, recon2):
    """
    Compute test metrics and update dataframe based on domain.
    
    Args:
        network_type: String indicating network type (determines attenuation vs activity behavior)
        input_: Input tensor
        CNN_output: Network output tensor
        target: Target tensor
        act_map_scaled: Activity map (used for activity domain only)
        test_dataframe: Test results dataframe
        config: Configuration dict
        recon1: Precomputed recon1 or None
        recon2: Precomputed recon2 or None
    
    Returns:
        Tuple of (test_dataframe, mean_CNN_MSE, mean_CNN_SSIM, mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM, recon1_output, recon2_output)
    """

    if network_type == 'ATTEN':
        # Attenuation: no recon comparisons; compute network metrics only
        mean_CNN_MSE = calculate_metric(target, CNN_output, MSE)
        mean_CNN_SSIM = calculate_metric(target, CNN_output, SSIM)
        recon1_output = None
        recon2_output = None
        mean_recon1_MSE = None
        mean_recon1_SSIM = None
        mean_recon2_MSE = None
        mean_recon2_SSIM = None
    else:
        # Activity: full recon comparisons
        test_dataframe, mean_CNN_MSE, mean_CNN_SSIM, mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM, recon1_output, recon2_output = \
            reconstruct_images_and_update_test_dataframe(input_, CNN_output, act_image_scaled, test_dataframe, config, compute_MLEM=False, recon1=recon1, recon2=recon2)
    
    return test_dataframe, mean_CNN_MSE, mean_CNN_SSIM, mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM, recon1_output, recon2_output
