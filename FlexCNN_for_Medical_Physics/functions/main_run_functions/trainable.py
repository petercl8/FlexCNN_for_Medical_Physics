
import os
import time
import torch
import pandas as pd
import logging
from torch.utils.data import DataLoader

from FlexCNN_for_Medical_Physics.classes.generators import Generator_180, Generator_288, Generator_320
from FlexCNN_for_Medical_Physics.classes.dataset_classes import NpArrayDataSet
from FlexCNN_for_Medical_Physics.classes.losses import HybridLoss
from FlexCNN_for_Medical_Physics.functions.helper.timing import display_times

from FlexCNN_for_Medical_Physics.functions.helper.metrics_wrappers import (
    calculate_metric,
    reconstruct_images_and_update_test_dataframe,
    update_tune_dataframe
)

from FlexCNN_for_Medical_Physics.functions.helper.metrics_wrappers import (
    reconstruct_images_and_update_test_dataframe,
    update_tune_dataframe
)

from FlexCNN_for_Medical_Physics.functions.helper.metrics import (
    SSIM,
    MSE,
    patchwise_moment_metric
)
from FlexCNN_for_Medical_Physics.functions.helper.reconstruction_projection import reconstruct
from FlexCNN_for_Medical_Physics.functions.helper.display_images import (
    show_single_unmatched_tensor,
    show_multiple_matched_tensors
)
from FlexCNN_for_Medical_Physics.functions.helper.weights_init import weights_init, weights_init_he
from FlexCNN_for_Medical_Physics.functions.helper.displays_and_reports import (
    compute_display_step,
    get_tune_session
)
from FlexCNN_for_Medical_Physics.functions.helper.evaluation_data_random import load_validation_batches, load_qa_batches, evaluate_val, evaluate_qa

# Module logger for optional Tune debug output
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


def _create_generator(config: dict, device: str):
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


def _create_optimizer(model, config: dict) -> torch.optim.Adam:
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


def _build_checkpoint_dict(model, optimizer, config: dict, epoch: int, batch_step: int) -> dict:
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


def _save_checkpoint(checkpoint_dict: dict, checkpoint_path: str) -> None:
    """
    Save checkpoint dictionary to file.
    
    Args:
        checkpoint_dict: Checkpoint dictionary to save.
        checkpoint_path: Path to save checkpoint to.
    """
    torch.save(checkpoint_dict, checkpoint_path)


def _load_checkpoint(checkpoint_path: str) -> tuple:
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


def _log_tune_debug(gen, epoch: int, batch_step: int, gen_loss, device: str) -> None:
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


def run_trainable(config, paths, settings):
    """
    Train, test, or visualize a network using the unified trainable pipeline.
    """

    # ========================================================================================
    # SECTION 1: EXTRACT CONFIGURATION VARIABLES
    # ========================================================================================
    run_mode = settings['run_mode']
    device = settings['device']
    tune_debug = settings.get('tune_debug', False)
    if tune_debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"[TUNE_DEBUG] Entering run_trainable; initial device request: {device}")
    
    # Network configuration
    train_SI = config['train_SI']
    image_size = config['image_size']
    sino_size = config['sino_size']
    image_channels = config['image_channels']
    sino_channels = config['sino_channels']
    
    # Data loading configuration
    augment = settings.get('augment', False)
    shuffle = settings.get('shuffle', True)
    offset = settings.get('offset', 0)
    num_examples = settings.get('num_examples', -1)
    sample_division = settings.get('sample_division', 1)
    
    # Training/Tuning configuration
    num_epochs = settings.get('num_epochs', 1)
    load_state = settings.get('load_state', False)
    save_state = settings.get('save_state', False)
    show_times = settings.get('show_times', False)
    
    # Display/Reporting configuration
    visualize_batch_size = settings.get('visualize_batch_size', 9)
    tune_dataframe_fraction = settings.get('tune_dataframe_fraction', 1.0)
    tune_max_t = settings.get('tune_max_t', 0)
    tune_restore = settings.get('tune_restore', False)
    
    # File paths
    checkpoint_path = paths.get('checkpoint_path', None)
    tune_dataframe_path = paths.get('tune_dataframe_path', None)

    # ========================================================================================
    # SECTION 2: COMPUTE BATCH SIZE AND RUNTIME PARAMETERS
    # ========================================================================================
    # Convert batch_base2_exponent to batch_size for tune/train modes
    if 'batch_base2_exponent' in config and settings['run_mode'] in ('tune', 'train'):
        config['batch_size'] = 2 ** config['batch_base2_exponent']

    batch_size = config['batch_size']
    display_step = compute_display_step(config, settings)
    session = get_tune_session()

    # ========================================================================================
    # SECTION 3: INITIALIZE DATAFRAMES (for tune/test modes)
    # ========================================================================================
    if run_mode == 'tune':
        if tune_restore == False:
            tune_dataframe = pd.DataFrame({
                'SI_dropout': [], 'SI_exp_kernel': [], 'SI_gen_fill': [], 'SI_gen_hidden_dim': [],
                'SI_gen_neck': [], 'SI_layer_norm': [], 'SI_normalize': [], 'SI_pad_mode': [],
                'batch_size': [], 'gen_lr': [], 'num_params': [], 'mean_CNN_MSE': [],
                'mean_CNN_SSIM': [], 'mean_CNN_CUSTOM': []
            })
            tune_dataframe.to_csv(tune_dataframe_path, index=False)
        else:
            tune_dataframe = pd.read_csv(tune_dataframe_path)

    if run_mode == 'test':
        test_dataframe = pd.DataFrame({
            'MSE (Network)': [], 'MSE (Recon1)': [], 'MSE (Recon2)': [],
            'SSIM (Network)': [], 'SSIM (Recon1)': [], 'SSIM (Recon2)': []
        })

    # ========================================================================================
    # SECTION 4: INSTANTIATE MODEL AND LOSS FUNCTION
    # ========================================================================================
    gen = _create_generator(config, device)

    base_criterion = config['sup_base_criterion']
    stats_criterion = config.get('sup_stats_criterion', None)
    alpha_min = config.get('sup_alpha_min', 0.2)
    half_life_examples = config.get('sup_half_life_examples', 2000)
    hybrid_loss = HybridLoss(
        base_loss=base_criterion,
        stats_loss=stats_criterion,
        alpha_min=alpha_min,
        half_life_examples=half_life_examples
    )

    # ========================================================================================
    # SECTION 5: CREATE OPTIMIZER
    # ========================================================================================
    gen_opt = _create_optimizer(gen, config)

    # ========================================================================================
    # SECTION 6: BUILD DATA LOADER
    # ========================================================================================
    dataloader = DataLoader(
        NpArrayDataSet(
            image_path=paths['image_path'],
            sino_path=paths['sino_path'],
            config=config,
            settings=settings,
            augment=augment,
            offset=offset,
            num_examples=num_examples,
            sample_division=sample_division,
            device=device,
            recon1_path=paths.get('recon1_path', None),
            recon2_path=paths.get('recon2_path', None),
            atten_image_path=paths.get('atten_image_path', None),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        collate_fn=collate_nested,
    )

    # ========================================================================================
    # SECTION 7: LOAD OR INITIALIZE CHECKPOINT AND WEIGHTS
    # ========================================================================================
    if load_state:
        epoch_loaded, batch_step_loaded, gen_state_dict, gen_opt_state_dict = _load_checkpoint(checkpoint_path)
        gen.load_state_dict(gen_state_dict)
        gen_opt.load_state_dict(gen_opt_state_dict)
        if run_mode in ('test', 'visualize'):
            gen.eval()
            start_epoch = 0
            end_epoch = 1
            batch_step = 0
        else:  # train/tune mode
            start_epoch = epoch_loaded
            end_epoch = num_epochs
            batch_step = batch_step_loaded
    else:
        gen = gen.apply(weights_init_he)
        start_epoch = 0
        batch_step = 0
        end_epoch = num_epochs

    # ========================================================================================
    # SECTION 8: INITIALIZE RUNNING METRICS AND TIMERS
    # ========================================================================================
    mean_gen_loss = 0
    mean_CNN_SSIM = 0
    mean_CNN_MSE = 0
    report_num = 1  # First report to RayTune is report_num = 1
    eval_cache = {}  # Evaluation cache for 'val' and 'qa' modes

    # Timing trackers
    time_init_full = time.time()    # Full step time (reset at display)
    time_init_loader = time.time()  # Data loading time (reset at display and batch end)


    # ========================================================================================
    # SECTION 9: EPOCH LOOP
    # ========================================================================================

    for epoch in range(start_epoch, end_epoch):

        # ========================================================================================
        # SECTION 10: BATCH LOOP - FORWARD/BACKWARD PASS
        # ========================================================================================
        
        for act_data, atten_data, recon_data in iter(dataloader):
            # _____ SUBSECTION 10A: UNPACK BATCH DATA _____
            sino_scaled, act_map_scaled = act_data
            atten_sino, atten_image = atten_data
            recon1, recon2 = recon_data
            
            # _____ SUBSECTION 10B: TIMING AND INPUT ROUTING _____
            _ = display_times('loader time', time_init_loader, show_times)
            time_init_full = display_times('FULL STEP TIME', time_init_full, show_times)
            
            if train_SI:
                target = act_map_scaled
                input_ = sino_scaled
            else:
                target = sino_scaled
                input_ = act_map_scaled

            # _____ SUBSECTION 10C: FORWARD PASS & TRAINING STEP (tune/train only) _____
            if run_mode in ('tune', 'train'):
                time_init_train = time.time()

                gen_opt.zero_grad()
                CNN_output = gen(input_)

                if run_mode == 'train' and torch.sum(CNN_output[1, 0, :]) < 0:
                    print('PIXEL VALUES SUM TO A NEGATIVE NUMBER. IF THIS CONTINUES FOR AWHILE, YOU MAY NEED TO RESTART')

                gen_loss = hybrid_loss(CNN_output, target)
                gen_loss.backward()
                gen_opt.step()

                if tune_debug:
                    _log_tune_debug(gen, epoch, batch_step, gen_loss, device)

                # Keep track of the average generator loss
                mean_gen_loss += gen_loss.item() / display_step
                _ = display_times('training time', time_init_train, show_times)

            # _____ SUBSECTION 10D: FORWARD PASS ONLY (test/visualize) _____
            else:
                CNN_output = gen(input_).detach()

            # _____ SUBSECTION 10E: INCREMENT BATCH COUNTER _____
            batch_step += 1

            # ========================================================================================
            # SECTION 11: RUN-TYPE SPECIFIC OPERATIONS
            # ========================================================================================

            time_init_metrics = time.time()

            # Tuning or Training: we only calculate the mean value of the metrics, but not dataframes or reconstructions. Mean values are used to calculate the optimization metrics #

            if run_mode in ('tune', 'train'):
                mean_CNN_SSIM += calculate_metric(target, CNN_output, SSIM) / display_step # The SSIM function can only take single images as inputs, not batches, so we use a wrapper function and pass batches to it.
                mean_CNN_MSE += calculate_metric(target, CNN_output, MSE) / display_step # The MSE function can take either single images or batches. We use the wrapper for consistency.

            # Test: Calculate individual image metrics and store in dataframe
            if run_mode == 'test':
                test_dataframe, mean_CNN_MSE, mean_CNN_SSIM, mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM, recon1_output, recon2_output = \
                    reconstruct_images_and_update_test_dataframe(input_, CNN_output, act_map_scaled, test_dataframe, config, compute_MLEM=False, recon1=recon1, recon2=recon2)

            # Visualize: Generate reconstructions for display if necessary
            if run_mode == 'visualize':
                if recon1 is not None:
                    recon1_output = recon1
                else:
                    recon1_output = reconstruct(input_, config['image_size'], config['SI_normalize'], config['SI_fixedScale'], recon_type='FBP')
                
                if recon2 is not None:
                    recon2_output = recon2
                else:
                    recon2_output = reconstruct(input_, config['image_size'], config['SI_normalize'], config['SI_fixedScale'], recon_type='MLEM')

            _ = display_times('metrics time', time_init_metrics, show_times)

            # ========================================================================================
            # SECTION 12: REPORTING AND VISUALIZATION (runs at display_step intervals)
            # ========================================================================================

            if batch_step % display_step == 0:
                time_init_visualization = time.time()
                example_num = batch_step * batch_size

                # _____ REPORTING: Ray Tune Validation (tune mode) _____
                if run_mode == 'tune' and session is not None:
                    # Set generator to eval mode for validation
                    gen.eval()
                    
                    # Load fresh random batches and evaluate
                    tune_report_for = settings.get('tune_report_for', 'val')
                    if tune_report_for == 'val':
                        batches = load_validation_batches(paths, config, settings)
                        metrics = evaluate_val(gen, batches, device, train_SI)

                        # Update dataframe at specified fraction
                        if int(tune_dataframe_fraction * tune_max_t) == report_num:
                            tune_dataframe = update_tune_dataframe(
                                tune_dataframe, tune_dataframe_path, gen, config,
                                metrics.get('MSE'), metrics.get('SSIM'), metrics.get('CUSTOM')
                            )

                    elif tune_report_for == 'qa':
                        batches = load_qa_batches(paths, config, settings, augment=('SI', True))
                        metrics = evaluate_qa(gen, batches, device, use_ground_truth_rois=False)
                    else:
                        raise ValueError(f"Invalid tune_report_for='{tune_report_for}'")
                    
                    # Explicit memory cleanup to prevent accumulation across trials
                    del batches
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

                    report_num += 1

                # _____ VISUALIZATION: Training Mode _____
                if run_mode == 'train':
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

                # _____ VISUALIZATION: Test Mode _____
                if run_mode == 'test':
                    print('==================Testing==================')
                    print(f'mean_CNN_MSE/mean_recon2_MSE/mean_recon1_MSE : {mean_CNN_MSE}/{mean_recon2_MSE}/{mean_recon1_MSE}')
                    print(f'mean_CNN_SSIM/mean_recon2_SSIM/mean_recon1_SSIM: {mean_CNN_SSIM}/{mean_recon2_SSIM}/{mean_recon1_SSIM}')
                    print('===========================================')
                    print('Input Sinogram:')
                    show_single_unmatched_tensor(input_[0:2], fig_size=25)
                    print('Target/Output/Recon2/Recon1:')
                    show_multiple_matched_tensors(target[0:9], CNN_output[0:9], recon2_output[0:9], recon1_output[0:9])

                # _____ VISUALIZATION: Visualization Mode _____
                if run_mode == 'visualize':
                    visualize_offset = settings.get('visualize_offset', 0)
                    if visualize_batch_size == 120:
                        print(f'visualize_offset: {visualize_offset}, Image Number (batch_step*120): {batch_step*120}')
                        show_single_unmatched_tensor(target, grid=True, cmap='inferno', fig_size=1)
                    else:
                        print('Input:')
                        show_single_unmatched_tensor(input_[0:visualize_batch_size])
                        print('Target/Recon2/Recon1/Output:')
                        show_multiple_matched_tensors(target[0:visualize_batch_size], recon1_output[0:visualize_batch_size], recon2_output[0:visualize_batch_size], CNN_output[0:visualize_batch_size])

                # _____ STATE SAVING _____
                if save_state:
                    print('Saving model!')
                    checkpoint_dict = _build_checkpoint_dict(gen, gen_opt, config, epoch, batch_step)
                    _save_checkpoint(checkpoint_dict, checkpoint_path)

                # _____ RESET RUNNING METRICS _____
                mean_gen_loss = 0
                mean_CNN_SSIM = 0
                mean_CNN_MSE = 0

                _ = display_times('visualization time', time_init_visualization, show_times)

            # _____ END OF BATCH: RESET LOADER TIMER _____
            time_init_loader = time.time()

    # ========================================================================================
    # SECTION 13: FINAL STATE SAVING (after all epochs complete)
    # ========================================================================================
    if save_state:
        print('Saving model!')
        checkpoint_dict = _build_checkpoint_dict(gen, gen_opt, config, epoch + 1, batch_step)
        _save_checkpoint(checkpoint_dict, checkpoint_path)

    # ========================================================================================
    # SECTION 14: RETURN TEST DATAFRAME (if applicable)
    # ========================================================================================
    if run_mode == 'test':
        return test_dataframe# Verification edit: Copilot modified file on 2025-11-23
