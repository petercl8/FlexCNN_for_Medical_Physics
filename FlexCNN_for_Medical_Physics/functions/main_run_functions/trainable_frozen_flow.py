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
    create_generator,
    create_optimizer,
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
    apply_feature_injection,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_frozen_state_dict(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint.get('gen_state_dict', checkpoint)


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
    checkpoint_path = paths['checkpoint_path']

    # ========================================================================================
    # SECTION 2: COMPUTE BATCH SIZE AND RUNTIME PARAMETERS
    # ========================================================================================
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
    # SECTION 4: INSTANTIATE MODELS (FROZEN ATTENUATION + TRAINABLE ACTIVITY)
    # ========================================================================================
    # Create frozen attenuation generator (no injection, only feature extraction)
    atten_config = dict(config) # Copy config to modify
    atten_config['train_SI'] = True if flow_mode == 'coflow' else False # Guarantee network direction regardless of value set it notebook
    gen_atten = create_generator(atten_config, device, gen_skip_handling='1x1Conv', enc_inject_channels=None, dec_inject_channels=None)

    # Extract encoder/decoder channel tuples for injection into activity network
    enc_inject_ch = gen_atten.enc_stage_channels
    dec_inject_ch = gen_atten.dec_stage_channels

    # Create trainable activity generator with injection parameters
    act_config = dict(config)
    act_config['train_SI'] = True  # Activity network is always sino->image
    gen_act = create_generator(act_config, device, gen_skip_handling='1x1Conv', gen_flow_mode=flow_mode, enc_inject_channels=enc_inject_ch, dec_inject_channels=dec_inject_ch)

    gen_opt = create_optimizer(gen_act, config)
    
    # ========================================================================================
    # SECTION 5: DEFINE LOSS FUNCTION FOR ACTIVITY NETWORK
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
    # SECTION 6: VALIDATE REQUIRED PATHS AND BUILD DATA LOADER
    # ========================================================================================
    def require_path(key: str):
        if key not in paths or paths[key] is None:
            raise ValueError(f"Missing required path: '{key}'. Provide act_*/atten_* paths for frozen flow.")
    require_path('act_image_path')
    require_path('act_sino_path')
    require_path('atten_image_path')
    require_path('atten_sino_path')

    # Extract paths for activity and attenuation domains
    act_image_path = paths.get('act_image_path')
    act_sino_path = paths.get('act_sino_path')
    act_recon1_path = paths.get('act_recon1_path')
    act_recon2_path = paths.get('act_recon2_path')

    # Build dataloader for joint activity/attenuation batches
    dataloader = DataLoader(
        NpArrayDataSet(
            act_sino_path=act_sino_path,
            act_image_path=act_image_path,
            config=config,
            settings=settings,
            augment=augment,
            offset=offset,
            num_examples=num_examples,
            sample_division=sample_division,
            device=device,
            act_recon1_path=act_recon1_path,
            act_recon2_path=act_recon2_path,
            atten_image_path=paths['atten_image_path'],
            atten_sino_path=paths['atten_sino_path'],
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        collate_fn=collate_nested,
    )

    # ========================================================================================
    # SECTION 7: LOAD OR INITIALIZE CHECKPOINTS AND WEIGHTS
    # ========================================================================================
    batch_step = 0
    gen_opt_state_dict = None
    if run_mode == 'tune':
        # Tuning: only load frozen attenuation weights
        start_epoch, end_epoch, _, _, _ = init_checkpoint_state(False, run_mode, checkpoint_path + '-act', num_epochs)
        atten_ckpt = checkpoint_path + '-atten'
        if os.path.exists(atten_ckpt):
            gen_atten.load_state_dict(_load_frozen_state_dict(atten_ckpt, device))
        gen_atten.eval()
    else:
        # Training/testing: load both attenuation and activity weights if present
        start_epoch, end_epoch, batch_step, gen_state_dict, gen_opt_state_dict = init_checkpoint_state(
            load_state, run_mode, checkpoint_path + '-act', num_epochs
        )
        atten_ckpt = checkpoint_path + '-atten'
        if os.path.exists(atten_ckpt):
            gen_atten.load_state_dict(_load_frozen_state_dict(atten_ckpt, device))
        gen_atten.eval()
        if gen_state_dict:
            gen_act.load_state_dict(gen_state_dict)
        else:
            gen_act = gen_act.apply(weights_init_he)
    # Optimizer (activity only) will be created after feature scales are initialized
    gen_opt = None

    # Set eval mode for test/visualize
    if run_mode in ('test', 'visualize'):
        gen_act.eval()

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
    for epoch in range(0 if run_mode in ('test', 'visualize') else start_epoch, 1 if run_mode in ('test', 'visualize') else end_epoch):
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

            # Prepare batch tensors for routing
            batch_tensors = {
                'act_sino_scaled': act_sino_scaled,
                'act_image_scaled': act_image_scaled,
                'atten_sino_scaled': atten_sino_scaled,
                'atten_image_scaled': atten_image_scaled,
            }
            # Route input/target for activity network
            input_, target = route_batch_inputs(train_SI_act, batch_tensors, network_type='ACT')

            # ----- SUBSECTION 10D: ACTIVITY NETWORK FORWARD/BACKWARD PASS -----
            if run_mode in ('tune', 'train'):
                time_init_train = time.time()

                gen_opt.zero_grad()
                CNN_output = apply_feature_injection(
                    gen_act,
                    input_,
                    frozen_enc_feats,
                    frozen_dec_feats,
                    None,
                    flow_mode=flow_mode,
                    enable_inject_to_encoder=settings.get('enable_inject_to_encoder', True),
                    enable_inject_to_decoder=settings.get('enable_inject_to_decoder', True),
                )

                gen_loss = hybrid_loss(CNN_output, target)
                gen_loss.backward()
                gen_opt.step()

                if tune_debug:
                    log_tune_debug(gen_act, epoch, batch_step, gen_loss, device)

                mean_gen_loss += gen_loss.item() / display_step
                _ = display_times('training time', time_init_train, show_times)
            else:
                # Test/visualize: forward only
                CNN_output = apply_feature_injection(
                    gen_act,
                    input_,
                    frozen_enc_feats,
                    frozen_dec_feats,
                    None,
                    flow_mode=flow_mode,
                    enable_inject_to_encoder=settings.get('enable_inject_to_encoder', True),
                    enable_inject_to_decoder=settings.get('enable_inject_to_decoder', True),
                ).detach()

            # ----- SUBSECTION 10E: METRIC TRACKING AND RECONSTRUCTION -----
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

            _ = display_times('metrics time', time.time(), show_times)

            # ========================================================================================
            # SECTION 11: REPORTING AND VISUALIZATION (runs at display_step intervals)
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
                    print('-----------------------------------------------')
                    print('Flow mode: ', flow_mode)
                    print(f"Epoch {epoch+1}/{end_epoch}, Batch {batch_step}, Examples {example_num}: "
                          f"Mean Gen Loss: {mean_gen_loss:.6f}, ")
                    visualize_train(batch_data, mean_gen_loss, mean_CNN_MSE, mean_CNN_SSIM, epoch, batch_step, example_num)

                # Test visualization
                if run_mode == 'test':
                    visualize_test(batch_data, mean_CNN_MSE, mean_CNN_SSIM, mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM)

                # Visualization mode (display input, output, recons)
                if run_mode == 'visualize':
                    batch_data['batch_step'] = batch_step
                    visualize_visualize(batch_data, batch_size, offset)

                # Reset running metrics after each display interval
                mean_gen_loss = 0
                mean_CNN_SSIM = 0
                mean_CNN_MSE = 0

                _ = display_times('visualization time', time_init_visualization, show_times)

            # End of batch: reset loader timer
            time_init_loader = time.time()
            if run_mode not in ('test', 'visualize'):
                batch_step += 1

    # ========================================================================================
    # SECTION 12: FINAL STATE SAVING (after all epochs complete)
    # ========================================================================================
    if save_state and run_mode in ('train', 'tune'):
        checkpoint_dict = build_checkpoint_dict(gen_act, gen_opt, config, epoch + 1, batch_step)
        save_checkpoint(checkpoint_dict, checkpoint_path + '-act')

    # ========================================================================================
    # SECTION 13: RETURN TEST DATAFRAME (if applicable)
    # ========================================================================================

    if run_mode == 'test':
        return test_dataframe
