
import os
import time
import torch
import pandas as pd
import logging
from torch.utils.data import DataLoader

from FlexCNN_for_Medical_Physics.classes.generators import Generator
from FlexCNN_for_Medical_Physics.classes.dataset_classes import NpArrayDataSet
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


def run_SUP(config, paths, settings):
    """
    Train, test, or visualize a supervisory-loss network using explicit dicts.
    """
    # Core settings
    run_mode = settings['run_mode']
    device = settings['device']
    tune_debug = settings.get('tune_debug', False)
    if tune_debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"[TUNE_DEBUG] Entering run_SUP; initial device request: {device}")
    train_SI = config['train_SI']
    image_size = config['image_size']
    sino_size = config['sino_size']
    image_channels = config['image_channels']
    sino_channels = config['sino_channels']
    augment = settings.get('augment', False)
    shuffle = settings.get('shuffle', True)
    show_times = settings.get('show_times', False)
    visualize_batch_size = settings.get('visualize_batch_size', 9)
    num_epochs = settings.get('num_epochs', 1)
    load_state = settings.get('load_state', False)
    save_state = settings.get('save_state', False)
    checkpoint_path = paths.get('checkpoint_path', None)
    tune_dataframe_path = paths.get('tune_dataframe_path', None)
    tune_dataframe_fraction = settings.get('tune_dataframe_fraction', 1.0)
    tune_max_t = settings.get('tune_max_t', 0)
    tune_restore= settings.get('tune_restore', False)
    sup_criterion = config['sup_criterion']
    # Dataset slicing controls
    offset = settings.get('offset', 0)
    num_examples = settings.get('num_examples', -1)
    sample_division = settings.get('sample_division', 1)
    
    # Convert batch_base2_exponent to batch_size for tune/train modes
    if 'batch_base2_exponent' in config and settings['run_mode'] in ('tune', 'train'):
        config['batch_size'] = 2 ** config['batch_base2_exponent']

    batch_size = config['batch_size']

    # Compute batch_size and display_step using helper (with integer rounding)
    display_step = compute_display_step(config, settings)

    # Initialize Ray Tune session if available
    session = get_tune_session()


    # Tuning/Test specific initializations
    if run_mode == 'tune':
        if tune_restore==False:
            tune_dataframe = pd.DataFrame({'SI_dropout': [], 'SI_exp_kernel': [], 'SI_gen_fill': [], 'SI_gen_hidden_dim': [], 'SI_gen_neck': [], 'SI_layer_norm': [], 'SI_normalize': [],'SI_pad_mode': [], 'batch_size': [], 'gen_lr': [], 'num_params': [], 'mean_CNN_MSE': [], 'mean_CNN_SSIM': [], 'mean_CNN_CUSTOM': []})
            tune_dataframe.to_csv(tune_dataframe_path, index=False)
        else:
            tune_dataframe = pd.read_csv(tune_dataframe_path)

    if run_mode == 'test':
        test_dataframe = pd.DataFrame({'MSE (Network)' : [],  'MSE (Recon1)': [],  'MSE (Recon2)': [], 'SSIM (Network)' : [], 'SSIM (Recon1)': [], 'SSIM (Recon2)': []})


    # Model and optimizer
    gen = Generator(config=config, gen_SI=train_SI).to(device)

    # Optimizer with separate group for learnable output scale
    betas = (config['gen_b1'], config['gen_b2'])
    base_lr = config['gen_lr']
    if train_SI:
        scale_lr_mult = config.get('SI_output_scale_lr_mult', 1.0)
    else:
        scale_lr_mult = config.get('IS_output_scale_lr_mult', 1.0)
    if getattr(gen, 'output_scale_learnable', False):
        scale_param = [gen.log_output_scale]
        main_params = [p for n, p in gen.named_parameters() if n != 'log_output_scale']
        gen_opt = torch.optim.Adam(
            [
                {'params': main_params, 'lr': base_lr, 'betas': betas},
                {'params': scale_param, 'lr': scale_lr_mult * base_lr, 'betas': betas, 'weight_decay': 0.0},
            ]
        )
    else:
        gen_opt = torch.optim.Adam(gen.parameters(), lr=base_lr, betas=betas)

    # Data loader
    dataloader = DataLoader(
        NpArrayDataSet(
            image_path=paths['image_path'],
            sino_path=paths['sino_path'],
            config=config,
            augment=augment,
            offset=offset,
            num_examples=num_examples,
            sample_division=sample_division,
            device=device,
            recon1_path=paths.get('recon1_path', None),
            recon2_path=paths.get('recon2_path', None),
            recon1_scale=settings.get('recon1_scale', 1.0),
            recon2_scale=settings.get('recon2_scale', 1.0),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )

    # Checkpoint handling
    if load_state:
        checkpoint = torch.load(checkpoint_path)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        gen_opt.load_state_dict(checkpoint['gen_opt_state_dict'])
        if run_mode in ('test', 'visualize'):
            gen.eval()
            start_epoch = 0
            end_epoch = 1
            batch_step = 0
        else:  # train
            start_epoch = checkpoint['epoch']
            end_epoch = num_epochs
            batch_step = checkpoint['batch_step']
    else:
        gen = gen.apply(weights_init_he)
        start_epoch = 0
        batch_step = 0
        end_epoch = num_epochs

    # Running metrics
    mean_gen_loss = 0
    mean_CNN_SSIM = 0
    mean_CNN_MSE = 0
    report_num = 1 # First report to RayTune is report_num = 1
    
    # Evaluation cache (for 'val' and 'qa' modes, or 'same' mode caching)
    eval_cache = {}

    # Timing
    time_init_full = time.time() # This is reset at the display time so that the full step time is displayed (see below).
    time_init_loader = time.time() # This is reset at the display time, but also reset at the end of the inner "for loop", so that only displays the data loading time.

    ########################
    ### Loop over Epochs ###
    ########################

    for epoch in range(start_epoch, end_epoch):

        #########################
        ### Loop Over Batches ###
        #########################
        
        for sino_scaled, act_map_scaled, *recon_data in iter(dataloader):
            # Unpack optional reconstructions if provided
            recon1 = recon_data[0] if len(recon_data) > 0 else None
            recon2 = recon_data[1] if len(recon_data) > 1 else None
            
            # Times
            _ = display_times('loader time', time_init_loader, show_times)             # _ is a dummy variable that isn't used in this loop
            time_init_full = display_times('FULL STEP TIME', time_init_full, 
            show_times) # This step resets time_init_full after displaying the time so this displays the full time to run the loop over a batch.
            
            # Inputs/targets
            if train_SI:
                target = act_map_scaled
                input_ = sino_scaled
            else:
                target = sino_scaled
                input_ = act_map_scaled

            # Forward/optimize
            if run_mode in ('tune', 'train'):
                time_init_train = time.time()

                gen_opt.zero_grad()
                CNN_output = gen(input_)

                if run_mode == 'train' and torch.sum(CNN_output[1, 0, :]) < 0:
                    print('PIXEL VALUES SUM TO A NEGATIVE NUMBER. IF THIS CONTINUES FOR AWHILE, YOU MAY NEED TO RESTART')

                # Update gradients                    
                gen_loss = sup_criterion(CNN_output, target)
                gen_loss.backward()
                gen_opt.step()

                # Optional per-step debug logging for Tune runs
                if tune_debug:
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

                # Keep track of the average generator loss
                mean_gen_loss += gen_loss.item() / display_step
                _ = display_times('training time', time_init_train, show_times)

            ## If Testing or Vizualizing, calculate output only ## 
            else:
                CNN_output = gen(input_).detach()

            # Increment batch_step
            batch_step += 1

            ####################################
            ### Run-Type Specific Operations ###
            ####################################

            time_init_metrics = time.time()

            # If Tuning or Training, we only calculate the mean value of the metrics, but not dataframes or reconstructions. Mean values are used to calculate the optimization metrics #

            if run_mode in ('tune', 'train'):
                mean_CNN_SSIM += calculate_metric(target, CNN_output, SSIM) / display_step # The SSIM function can only take single images as inputs, not batches, so we use a wrapper function and pass batches to it.
                mean_CNN_MSE += calculate_metric(target, CNN_output, MSE) / display_step # The MSE function can take either single images or batches. We use the wrapper for consistency.

            # If Testing, we calculate and store reconstructions and metrics in a dataframe #
            if run_mode == 'test':
                test_dataframe, mean_CNN_MSE, mean_CNN_SSIM, mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM, recon1_output, recon2_output = \
                    reconstruct_images_and_update_test_dataframe(input_, CNN_output, act_map_scaled, test_dataframe, config, compute_MLEM=False, recon1=recon1, recon2=recon2)

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

            # Reporting / visualization
            if batch_step % display_step == 0:
                time_init_visualization = time.time()
                example_num = batch_step * batch_size

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
                        batches = load_qa_batches(paths, config, settings)
                        metrics = evaluate_qa(gen, batches, device, settings)
                    else:
                        raise ValueError(f"Invalid tune_report_for='{tune_report_for}'")
                    
                    # Restore generator to train mode
                    gen.train()
                    
                    # Check for NaN and terminate if found
                    metric_to_check = settings.get('tune_metric', 'SSIM')  # or 'ROI_NEMA_weighted', etc.
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
                    show_single_unmatched_tensor(input_[0:3], fig_size=15)
                    print(input_.shape)
                    print('Target/Output:')
                    show_multiple_matched_tensors(target[0:8], CNN_output[0:8])
                    #show_multiple_matched_tensors(target[0:8], CNN_output[0:8], recon1[0:8], recon2[0:8])

                if run_mode == 'test':
                    print('==================Testing==================')
                    print(f'mean_CNN_MSE/mean_recon2_MSE/mean_recon1_MSE : {mean_CNN_MSE}/{mean_recon2_MSE}/{mean_recon1_MSE}')
                    print(f'mean_CNN_SSIM/mean_recon2_SSIM/mean_recon1_SSIM: {mean_CNN_SSIM}/{mean_recon2_SSIM}/{mean_recon1_SSIM}')
                    print('===========================================')
                    print('Input Sinogram:')
                    show_single_unmatched_tensor(input_[0:9], fig_size=15)
                    print('Target/Output/Recon2/Recon1:')
                    show_multiple_matched_tensors(target[0:9], CNN_output[0:9], recon2_output[0:9], recon1_output[0:9])

                if run_mode == 'visualize':
                    visualize_offset = settings.get('visualize_offset', 0)
                    if visualize_batch_size == 120:
                        print(f'visualize_offset: {visualize_offset}, Image Number (batch_step*120): {batch_step*120}')
                        show_single_unmatched_tensor(target, grid=True, cmap='inferno', fig_size=1)
                    else:
                        print('Input:')
                        show_single_unmatched_tensor(input_[0:visualize_batch_size])
                        print('Target/Recon2/Recon1/Output:')
                        show_multiple_matched_tensors(target[0:visualize_batch_size], recon2_output[0:visualize_batch_size], recon1_output[0:visualize_batch_size], CNN_output[0:visualize_batch_size])

                if save_state:
                    print('Saving model!')
                    torch.save({'epoch': epoch, 'batch_step': batch_step, 'gen_state_dict': gen.state_dict(), 'gen_opt_state_dict': gen_opt.state_dict()}, checkpoint_path)

                # Reset running stats
                mean_gen_loss = 0
                mean_CNN_SSIM = 0
                mean_CNN_MSE = 0

                _ = display_times('visualization time', time_init_visualization, show_times)

            # Reset loader timer
            time_init_loader = time.time()

    # Save final state (training)
    if save_state:
        print('Saving model!')
        torch.save({'epoch': epoch + 1, 'batch_step': batch_step, 'gen_state_dict': gen.state_dict(), 'gen_opt_state_dict': gen_opt.state_dict()}, checkpoint_path)

    if run_mode == 'test':
        return test_dataframe# Verification edit: Copilot modified file on 2025-11-23
