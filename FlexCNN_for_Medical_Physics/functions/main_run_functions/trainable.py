
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
from FlexCNN_for_Medical_Physics.functions.main_run_functions.train_utils import (
    collate_nested,
    create_generator,
    create_optimizer,
    build_checkpoint_dict,
    save_checkpoint,
    load_checkpoint,
    log_tune_debug,
    report_tune_metrics,
    visualize_train,
    visualize_test,
    visualize_mode
)

# Module logger for optional Tune debug output
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    network_type = config['network_type']
    is_atten = (network_type == 'SUP_ATTEN')
    
    # Force train_SI=True for attenuation domain
    if is_atten:
        config['train_SI'] = True
    
    train_SI = config['train_SI']
    
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
                'batch_size': [], 'gen_lr': [], 'num_params': []
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
    gen = create_generator(config, device)

    base_criterion = config['sup_base_criterion']
    # Use SI or IS prefixed stats params based on training direction
    prefix = 'SI' if train_SI else 'IS'
    stats_criterion = config.get(f'{prefix}_stats_criterion', None)
    alpha_min = config.get(f'{prefix}_alpha_min', -1)
    half_life_examples = config.get(f'{prefix}_half_life_examples', 2000)
    hybrid_loss = HybridLoss(
        base_loss=base_criterion,
        stats_loss=stats_criterion,
        alpha_min=alpha_min,
        half_life_examples=half_life_examples
    )

    # ========================================================================================
    # SECTION 5: CREATE OPTIMIZER
    # ========================================================================================
    gen_opt = create_optimizer(gen, config)

    # ========================================================================================
    # SECTION 5A: FILTER PATHS BY NETWORK TYPE (avoid loading unnecessary data)
    # ========================================================================================
    if network_type == 'SUP_ACT':
        paths['atten_image_path'] = None
        paths['atten_sino_path'] = None
    elif network_type == 'SUP_ATTEN':
        paths['image_path'] = None
        paths['sino_path'] = None
        paths['recon1_path'] = None
        paths['recon2_path'] = None

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
            recon1_path=paths['recon1_path'],
            recon2_path=paths['recon2_path'],
            atten_image_path=paths['atten_image_path'],
            atten_sino_path=paths['atten_sino_path'],
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
        epoch_loaded, batch_step_loaded, gen_state_dict, gen_opt_state_dict = load_checkpoint(checkpoint_path)
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
            atten_sino_scaled, atten_image_scaled = atten_data
            recon1, recon2 = recon_data
            recon1_output = recon1
            recon2_output = recon2
            
            # _____ SUBSECTION 10B: TIMING AND INPUT ROUTING _____
            _ = display_times('loader time', time_init_loader, show_times)
            time_init_full = display_times('FULL STEP TIME', time_init_full, show_times)
            
            if is_atten:
                # Attenuation domain: always sino->image (train_SI forced to True)
                target = atten_image_scaled
                input_ = atten_sino_scaled
            else:
                # Activity domain: use configured train_SI direction
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
                    log_tune_debug(gen, epoch, batch_step, gen_loss, device)

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
                if is_atten:
                    # Attenuation: no recon comparisons; compute network metrics only
                    mean_CNN_MSE += calculate_metric(target, CNN_output, MSE) / display_step
                    mean_CNN_SSIM += calculate_metric(target, CNN_output, SSIM) / display_step
                else:
                    # Activity: full recon comparisons
                    test_dataframe, mean_CNN_MSE, mean_CNN_SSIM, mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM, recon1_output, recon2_output = \
                        reconstruct_images_and_update_test_dataframe(input_, CNN_output, act_map_scaled, test_dataframe, config, compute_MLEM=False, recon1=recon1, recon2=recon2)

            # Visualize: Generate reconstructions for display if necessary
            if run_mode == 'visualize':
                if is_atten:
                    # Attenuation: no recon generation; visualize_mode will handle missing recon gracefully
                    pass
                else:
                    # Activity: generate recon comparisons
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

                # Construct unified batch_data for all visualization calls
                batch_data = {
                    'input': input_,
                    'target': target,
                    'CNN_output': CNN_output,
                    'recon1_output': recon1_output,
                    'recon2_output': recon2_output
                }

                # _____ REPORTING: Ray Tune Validation (tune mode) _____
                if run_mode == 'tune' and session is not None:
                    tune_dataframe = report_tune_metrics(
                        gen, paths, config, settings, tune_dataframe, tune_dataframe_path,
                        train_SI, tune_dataframe_fraction, tune_max_t, report_num,
                        example_num, batch_step, epoch, session, device
                    )
                    report_num += 1

                # _____ VISUALIZATION: Training Mode _____
                if run_mode == 'train':
                    visualize_train(batch_data, mean_gen_loss, mean_CNN_MSE, 
                                  mean_CNN_SSIM, epoch, batch_step, example_num)

                # _____ VISUALIZATION: Test Mode _____
                if run_mode == 'test':
                    visualize_test(batch_data, mean_CNN_MSE, mean_CNN_SSIM,
                                 mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM)

                # _____ VISUALIZATION: Visualization Mode _____
                if run_mode == 'visualize':
                    visualize_offset = settings.get('visualize_offset', 0)
                    batch_data['batch_step'] = batch_step
                    visualize_mode(batch_data, visualize_batch_size, visualize_offset)

                # _____ STATE SAVING _____
                if save_state:
                    print('Saving model!')
                    checkpoint_dict = build_checkpoint_dict(gen, gen_opt, config, epoch, batch_step)
                    save_checkpoint(checkpoint_dict, checkpoint_path)

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
        checkpoint_dict = build_checkpoint_dict(gen, gen_opt, config, epoch + 1, batch_step)
        save_checkpoint(checkpoint_dict, checkpoint_path)

    # ========================================================================================
    # SECTION 14: RETURN TEST DATAFRAME (if applicable)
    # ========================================================================================
    if run_mode == 'test':
        return test_dataframe# Verification edit: Copilot modified file on 2025-11-23
