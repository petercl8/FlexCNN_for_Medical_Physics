import time
import torch
import pandas as pd
import logging
from torch.utils.data import DataLoader

from FlexCNN_for_Medical_Physics.classes.dataset_classes import NpArrayDataSet
from FlexCNN_for_Medical_Physics.classes.losses import HybridLoss
from FlexCNN_for_Medical_Physics.functions.helper.timing import display_times

from FlexCNN_for_Medical_Physics.functions.helper.metrics_wrappers import (
    calculate_metric,
)

from FlexCNN_for_Medical_Physics.functions.helper.metrics import (
    SSIM,
    MSE,
)
from FlexCNN_for_Medical_Physics.functions.helper.weights_init import weights_init_he
from FlexCNN_for_Medical_Physics.functions.helper.displays_and_reports import (
    compute_display_step,
    get_tune_session
)
from FlexCNN_for_Medical_Physics.functions.main_run_functions.train_utils import (
    collate_nested,
    build_checkpoint_dict,
    save_checkpoint,
    log_tune_debug,
    report_tune_metrics,
    visualize_train,
    visualize_test,
    visualize_visualize,
    visualize_visualize,
    route_batch_inputs,
    generate_reconstructions_for_visualization,
    compute_test_metrics,
    init_checkpoint_state,
)

from FlexCNN_for_Medical_Physics.functions.helper.setup_generators_optimizer import (
    create_generator,
    create_optimizer,
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
    train_SI = config['train_SI']
    
    # Data loading configuration
    augment = settings['augment']
    shuffle = settings['shuffle']
    offset = settings['offset']
    num_examples = settings['num_examples']
    sample_division = settings['sample_division']
    
    # Training/Tuning configuration
    num_epochs = settings['num_epochs']
    load_state = settings['load_state']
    save_state = settings['save_state']
    show_times = settings['show_times']
    
    # Display/Reporting configuration (mode-specific)
    if run_mode == 'tune':
        tune_dataframe_fraction = settings['tune_dataframe_fraction']
        tune_max_t = settings['tune_max_t']
        tune_restore = settings['tune_restore']
        tune_dataframe_path = paths['tune_dataframe_path']
    checkpoint_path = paths['checkpoint_path']


    # ========================================================================================
    # SECTION 2: COMPUTE BATCH SIZE AND RUNTIME PARAMETERS
    # ========================================================================================
    # Convert batch_base2_exponent to batch_size for tune/train modes
    if 'batch_base2_exponent' in config and run_mode in ('tune', 'train'):
        config['batch_size'] = 2 ** config['batch_base2_exponent']

    batch_size = config['batch_size']
    display_step = compute_display_step(config, settings)
    session = get_tune_session()

    # ========================================================================================
    # SECTION 3: INITIALIZE DATAFRAMES (for tune/test modes)
    # ========================================================================================
    if run_mode == 'tune':
        if tune_restore == False:
            # Create new tuning dataframe
            tune_dataframe = pd.DataFrame({
                'SI_dropout': [], 'SI_exp_kernel': [], 'SI_gen_fill': [], 'SI_gen_hidden_dim': [],
                'SI_gen_neck': [], 'SI_layer_norm': [], 'SI_normalize': [], 'SI_pad_mode': [],
                'batch_size': [], 'gen_lr': [], 'num_params': []
            })
            tune_dataframe.to_csv(tune_dataframe_path, index=False)
        else:
            # Load existing tuning dataframe
            tune_dataframe = pd.read_csv(tune_dataframe_path)

    if run_mode == 'test':
        # Create test results dataframe
        test_dataframe = pd.DataFrame({
            'MSE (Network)': [], 'MSE (Recon1)': [], 'MSE (Recon2)': [],
            'SSIM (Network)': [], 'SSIM (Recon1)': [], 'SSIM (Recon2)': []
        })

    # ========================================================================================
    # SECTION 4: INSTANTIATE MODEL AND OPTIMIZER
    # ========================================================================================
    gen = create_generator(config, device)
    gen_opt = create_optimizer(gen, config)

    # ========================================================================================
    # SECTION 5: LOAD OR INITIALIZE CHECKPOINT AND WEIGHTS
    # ========================================================================================
    start_epoch, end_epoch, batch_step, gen_state_dict, gen_opt_state_dict = init_checkpoint_state(
        load_state, run_mode, checkpoint_path, num_epochs, device
    )
    if load_state:
        gen.load_state_dict(gen_state_dict)
        gen_opt.load_state_dict(gen_opt_state_dict)
    else:
        gen = gen.apply(weights_init_he)

    # Set to eval mode for test/visualize
    if run_mode in ('test', 'visualize'):
        gen.eval()

    # ========================================================================================
    # SECTION 6: INSTANTIATE LOSS FUNCTION
    # ========================================================================================
    base_criterion = config['sup_base_criterion']
    # Use SI or IS prefixed stats params based on training direction
    prefix = 'SI' if train_SI else 'IS'
    stats_criterion = config[f'{prefix}_stats_criterion']
    alpha_min = config[f'{prefix}_alpha_min']
    half_life_examples = config[f'{prefix}_half_life_examples']
    hybrid_loss = HybridLoss(
        base_loss=base_criterion,
        stats_loss=stats_criterion,
        alpha_min=alpha_min,
        half_life_examples=half_life_examples
    )

    # ========================================================================================
    # SECTION 7A: FILTER PATHS BY NETWORK TYPE (avoid loading unnecessary data)
    # ========================================================================================
    if network_type == 'ACT':
        # Activity-only: attenuation paths are not required
        paths['atten_image_path'] = None
        paths['atten_sino_path'] = None
    elif network_type == 'ATTEN':
        # Attenuation-only: activity paths are not required
        paths['act_image_path'] = None
        paths['act_sino_path'] = None

    # ========================================================================================
    # SECTION 7B: VALIDATE REQUIRED PATHS AND BUILD DATA LOADER
    # ========================================================================================
    # Enforce proper act_* and atten_* keys without legacy fallbacks
    # Determine required domains based on network type
    def require_path(key: str):
        if key not in paths or paths[key] is None:
            raise ValueError(f"Missing required path: '{key}'. Provide proper act_*/atten_* paths; legacy keys are not supported.")

    if network_type in ('ACT', 'GAN', 'CYCLEGAN'):
        # Require activity domain
        require_path('act_image_path')
        require_path('act_sino_path')
    if network_type in ('ATTEN', 'CONCAT'):
        # Require attenuation domain
        require_path('atten_image_path')
        require_path('atten_sino_path')
    if network_type == 'CONCAT':
        # CONCAT needs activity target and both sinograms
        require_path('act_image_path')

    dataloader = DataLoader(
        NpArrayDataSet(
            act_sino_path=paths['act_sino_path'],
            act_image_path=paths['act_image_path'],
            config=config,
            settings=settings,
            augment=augment,
            offset=offset,
            num_examples=num_examples,
            sample_division=sample_division,
            device=device,
            act_recon1_path=paths['act_recon1_path'],
            act_recon2_path=paths['act_recon2_path'],
            atten_image_path=paths['atten_image_path'],
            atten_sino_path=paths['atten_sino_path'],
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        collate_fn=collate_nested,
    )

    # ========================================================================================
    # SECTION 8: INITIALIZE RUNNING METRICS AND TIMERS
    # ========================================================================================
    mean_gen_loss = 0
    mean_CNN_SSIM = 0
    mean_CNN_MSE = 0
    report_num = 1  # First report to RayTune is report_num = 1

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
            act_sino_scaled, act_image_scaled = act_data
            atten_sino_scaled, atten_image_scaled = atten_data
            act_recon1, act_recon2 = recon_data
            recon1_output = act_recon1
            recon2_output = act_recon2
            
            # _____ SUBSECTION 10B: TIMING AND INPUT ROUTING _____
            _ = display_times('loader time', time_init_loader, show_times)
            time_init_full = display_times('FULL STEP TIME', time_init_full, show_times)
            
            batch_tensors = {
                'act_sino_scaled': act_sino_scaled,
                'act_image_scaled': act_image_scaled,
                'atten_sino_scaled': atten_sino_scaled,
                'atten_image_scaled': atten_image_scaled
            }
            input_, target = route_batch_inputs(train_SI, batch_tensors, network_type)

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
                test_dataframe, mean_CNN_MSE, mean_CNN_SSIM, mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM, recon1_output, recon2_output = \
                    compute_test_metrics(network_type, input_, CNN_output, target, act_image_scaled, test_dataframe, config, act_recon1, act_recon2)

            # Visualize: Generate reconstructions for display if necessary
            if run_mode == 'visualize':
                if network_type != 'ATTEN':
                    # Activity domain: generate recon comparisons
                    recon1_output, recon2_output = generate_reconstructions_for_visualization(recon1, recon2, input_, config)

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
                    batch_data['batch_step'] = batch_step
                    visualize_visualize(batch_data, batch_size, offset)

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
