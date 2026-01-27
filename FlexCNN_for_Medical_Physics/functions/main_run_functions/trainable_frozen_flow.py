import os
import time
import torch
import pandas as pd
import logging
from torch import nn
from torch.utils.data import DataLoader

from FlexCNN_for_Medical_Physics.classes.dataset_classes import NpArrayDataSet
from FlexCNN_for_Medical_Physics.classes.losses import HybridLoss
from FlexCNN_for_Medical_Physics.functions.helper.timing import display_times
from FlexCNN_for_Medical_Physics.functions.helper.metrics_wrappers import calculate_metric
from FlexCNN_for_Medical_Physics.functions.helper.metrics import SSIM, MSE
from FlexCNN_for_Medical_Physics.functions.helper.weights_init import weights_init_he
from FlexCNN_for_Medical_Physics.functions.helper.displays_and_reports import (
    compute_display_step,
    get_tune_session,
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
    route_batch_inputs,
    generate_reconstructions_for_visualization,
    compute_test_metrics,
    init_checkpoint_state,
)

from FlexCNN_for_Medical_Physics.functions.helper.generators_discriminator_setup import (
    instantiate_dual_generators,
    load_dual_generator_checkpoints,
    create_optimizer,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_trainable_frozen_flow(config, paths, settings):
    """
    Train, test, or visualize a network using a frozen attenuation network (backbone)
    guiding a trainable activity network. Handles coflow/counterflow feature injection.
    """
    # ========================================================================================
    # SECTION 1: EXTRACT CONFIGURATION VARIABLES
    # ========================================================================================
    run_mode = settings['run_mode']
    device = settings['device']
    tune_debug = settings.get('tune_debug', False)
    if tune_debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"[TUNE_DEBUG] Entering run_trainable_frozen_flow; device: {device}")

    # Determine flow mode and always use sino->image for activity network
    flow_mode = 'coflow' if config['network_type'] == 'FROZEN_COFLOW' else 'counterflow'
    train_SI_act = True  # Activity network is always sino->image

    # Data loading and augmentation settings
    augment = settings['augment']
    shuffle = settings['shuffle']
    offset = settings['offset']
    num_examples = settings['num_examples']
    sample_division = settings['sample_division']

    # Training and checkpoint settings
    num_epochs = settings['num_epochs']
    load_state = settings['load_state']
    save_state = settings['save_state']
    show_times = settings['show_times']

    # Tuning-specific settings
    if run_mode == 'tune':
        tune_dataframe_fraction = settings['tune_dataframe_fraction']
        tune_max_t = settings['tune_max_t']
        tune_restore = settings['tune_restore']
        tune_dataframe_path = paths['tune_dataframe_path']

    # Checkpoint path
    checkpoint_path = paths['checkpoint_path']

    # ========================================================================================
    # SECTION 2: COMPUTE BATCH SIZE AND RUNTIME PARAMETERS
    # ========================================================================================
    if 'batch_base2_exponent' in config and run_mode in ('tune', 'train'):
        config['batch_size'] = 2 ** config['batch_base2_exponent']
    batch_size = config['batch_size'] # Batch size for in train/test/visualize remains unchanged
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
    # SECTION 4: INSTANTIATE MODELS (FROZEN ATTENUATION + TRAINABLE ACTIVITY)
    # ========================================================================================
    # Instantiate both generators using helper function
    gen_atten, gen_act = instantiate_dual_generators(config, device, flow_mode)
    gen_act_opt = create_optimizer(gen_act, config)
    
    # ========================================================================================
    # SECTION 5: LOAD OR INITIALIZE CHECKPOINTS AND WEIGHTS
    # ========================================================================================
    act_ckpt = checkpoint_path + '-act' if checkpoint_path is not None else None
    atten_ckpt = checkpoint_path + '-atten' if checkpoint_path is not None else None

    gen_act, gen_atten = load_dual_generator_checkpoints(
        gen_act,
        gen_atten,
        act_ckpt,
        atten_ckpt,
        load_state,
        run_mode,
        device,
    )
    # Optimizer (activity only) is ready after checkpoint loading; feature scales are no longer used
    # Use settings['load_state'] directly, as it is set appropriately for each run_mode

    start_epoch, end_epoch, batch_step, gen_state_dict, gen_act_opt_state_dict = init_checkpoint_state(
        load_state, run_mode, act_ckpt, num_epochs, device
    )
    if gen_act_opt_state_dict:
        gen_act_opt.load_state_dict(gen_act_opt_state_dict)

    # ========================================================================================
    # SECTION 6: INSTANTIATE LOSS FUNCTION FOR ACTIVITY NETWORK
    # ========================================================================================
    base_criterion = config['sup_base_criterion']
    stats_criterion = config['SI_stats_criterion']
    alpha_min = config['SI_alpha_min']
    half_life_examples = config['SI_half_life_examples']
    hybrid_loss = HybridLoss(
        base_loss=base_criterion,
        stats_loss=stats_criterion,
        alpha_min=alpha_min,
        half_life_examples=half_life_examples,
    )

    # ========================================================================================
    # SECTION 7: VALIDATE REQUIRED PATHS AND BUILD DATA LOADER
    # ========================================================================================
    def require_path(key: str):
        if key not in paths or paths[key] is None:
            raise ValueError(f"Missing required path: '{key}'. Provide act_*/atten_* paths for frozen flow.")
    require_path('act_image_path')
    require_path('act_sino_path')
    require_path('atten_image_path')
    require_path('atten_sino_path')

    # Build dataloader for joint activity/attenuation batches
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
    report_num = 1

    time_init_full = time.time()
    time_init_loader = time.time()

    # ========================================================================================
    # SECTION 9: EPOCH LOOP
    # ========================================================================================
    # For test/visualize, run a single epoch; for train/tune, use full epoch range
    for epoch in range(start_epoch, end_epoch):
        # ========================================================================================
        # SECTION 10: BATCH LOOP - FORWARD/BACKWARD PASS
        # ========================================================================================
        for act_data, atten_data, recon_data in iter(dataloader):
            # ----- SUBSECTION 10A: UNPACK BATCH DATA -----
            act_sino_scaled, act_image_scaled = act_data
            atten_sino_scaled, atten_image_scaled = atten_data
            act_recon1, act_recon2 = recon_data
            recon1_output = act_recon1
            recon2_output = act_recon2

            # ----- SUBSECTION 10B: TIMING AND INPUT ROUTING -----
            _ = display_times('loader time', time_init_loader, show_times)
            time_init_full = display_times('FULL STEP TIME', time_init_full, show_times)

            # ----- SUBSECTION 10C: FROZEN ATTENUATION FORWARD PASS -----
            with torch.no_grad():
                # Use attenuation input based on flow mode
                atten_input = atten_sino_scaled if flow_mode == 'coflow' else atten_image_scaled
                result = gen_atten(atten_input, return_features=True)
                frozen_enc_feats = result['encoder']
                frozen_dec_feats = result['decoder']

            # ----- SUBSECTION 10D: ACTIVITY NETWORK FORWARD/BACKWARD PASS -----

            # Prepare batch tensors for routing
            batch_tensors = {
                'act_sino_scaled': act_sino_scaled,
                'act_image_scaled': act_image_scaled,
                'atten_sino_scaled': atten_sino_scaled,
                'atten_image_scaled': atten_image_scaled,
            }
            # Route input/target for activity network
            input_, target = route_batch_inputs(train_SI_act, batch_tensors, network_type='ACT') # input_ is sino, target is image

            if run_mode in ('tune', 'train'):
                time_init_train = time.time()

                gen_act_opt.zero_grad()
                CNN_output = gen_act(
                    input_,
                    frozen_encoder_features=frozen_enc_feats,
                    frozen_decoder_features=frozen_dec_feats,
                )

                gen_loss = hybrid_loss(CNN_output, target)
                gen_loss.backward()
                gen_act_opt.step()

                if tune_debug:
                    log_tune_debug(gen_act, epoch, batch_step, gen_loss, device)

                mean_gen_loss += gen_loss.item() / display_step
                _ = display_times('training time', time_init_train, show_times)
            else:
                # Test/visualize: forward only
                CNN_output = gen_act(
                    input_,
                    frozen_encoder_features=frozen_enc_feats,
                    frozen_decoder_features=frozen_dec_feats,
                ).detach()

            # _____ SUBSECTION 10E: INCREMENT BATCH COUNTER _____
            batch_step += 1

            # ========================================================================================
            # SECTION 11: RUN-TYPE SPECIFIC OPERATIONS
            # ========================================================================================
            time_init_metrics = time.time()

            # Tuning or Training: we only calculate the mean value of the metrics, but not dataframes or reconstructions. Mean values are used to calculate the optimization metrics #
            if run_mode in ('tune', 'train'):
                mean_CNN_SSIM += calculate_metric(target, CNN_output, SSIM) / display_step
                mean_CNN_MSE += calculate_metric(target, CNN_output, MSE) / display_step

            # Test: Calculate individual image metrics and store in dataframe
            if run_mode == 'test':
                test_dataframe, mean_CNN_MSE, mean_CNN_SSIM, mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM, recon1_output, recon2_output = \
                    compute_test_metrics('ACT', input_, CNN_output, target, act_image_scaled, test_dataframe, config, act_recon1, act_recon2)

            # Visualize: Generate reconstructions for display if necessary
            if run_mode == 'visualize':
                recon1_output, recon2_output = generate_reconstructions_for_visualization(recon1_output, recon2_output, input_, config)

            _ = display_times('metrics time', time_init_metrics, show_times)

            # ========================================================================================
            # SECTION 12: REPORTING AND VISUALIZATION (runs at display_step intervals)
            # ========================================================================================
            if batch_step % display_step == 0:
                time_init_visualization = time.time()
                example_num = batch_step * batch_size

                # Construct batch_data dict for visualization functions
                batch_data = {
                    'input': input_,
                    'target': target,
                    'CNN_output': CNN_output,
                    'recon1_output': recon1_output,
                    'recon2_output': recon2_output,
                }

                # Ray Tune reporting (tune mode)
                if run_mode == 'tune' and session is not None:
                    tune_dataframe = report_tune_metrics(
                        gen_act, paths, config, settings, tune_dataframe, tune_dataframe_path,
                        train_SI_act, tune_dataframe_fraction, tune_max_t, report_num,
                        example_num, batch_step, epoch, session, device,
                    )
                    report_num += 1

                # Training visualization
                if run_mode == 'train':
                    print('Flow mode: ', flow_mode)
                    visualize_train(batch_data, mean_gen_loss, mean_CNN_MSE, mean_CNN_SSIM, epoch, batch_step, example_num)

                # Test visualization
                if run_mode == 'test':
                    visualize_test(batch_data, mean_CNN_MSE, mean_CNN_SSIM, mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM)

                # Visualization mode (display input, output, recons)
                if run_mode == 'visualize':
                    batch_data['batch_step'] = batch_step
                    visualize_visualize(batch_data, batch_size, offset)

                # _____ STATE SAVING _____
                if save_state:
                    print('Saving model!')
                    checkpoint_dict = build_checkpoint_dict(gen_act, gen_act_opt, config, epoch, batch_step)
                    save_checkpoint(checkpoint_dict, checkpoint_path + '-act')

                # Reset running metrics after each display interval
                mean_gen_loss = 0
                mean_CNN_SSIM = 0
                mean_CNN_MSE = 0

                _ = display_times('visualization time', time_init_visualization, show_times)

            # End of batch: reset loader timer
            time_init_loader = time.time()

    # ========================================================================================
    # SECTION 13: FINAL STATE SAVING (after all epochs complete)
    # ========================================================================================
    if save_state:
        print('Saving model!')
        checkpoint_dict = build_checkpoint_dict(gen_act, gen_act_opt, config, epoch + 1, batch_step)
        save_checkpoint(checkpoint_dict, checkpoint_path + '-act')

    # ========================================================================================
    # SECTION 14: RETURN TEST DATAFRAME (if applicable)
    # ========================================================================================

    if run_mode == 'test':
        return test_dataframe
