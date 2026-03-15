import os
import time
import torch
import pandas as pd
import logging
from torch import nn
from torch.utils.data import DataLoader

from FlexCNN_for_Medical_Physics.classes.dataset.dataset_classes import NpArrayDataSet
from FlexCNN_for_Medical_Physics.custom_criteria import HybridLoss
from FlexCNN_for_Medical_Physics.functions.helper.utilities.timing import display_times
from FlexCNN_for_Medical_Physics.functions.helper.metrics.metrics_wrappers import calculate_metric, append_train_learning_curve_row
from FlexCNN_for_Medical_Physics.functions.helper.metrics.metrics import SSIM, MSE
from FlexCNN_for_Medical_Physics.functions.helper.model_setup.weights_init import weights_init_he
from FlexCNN_for_Medical_Physics.functions.helper.utilities.displays_and_reports import (
    compute_display_step,
    get_tune_session,
)
from FlexCNN_for_Medical_Physics.functions.main_run_functions.train_utils import (
    collate_nested,
    build_checkpoint_dict,
    save_checkpoint,
    log_tune_debug,
    visualize_train_frozen,
    visualize_test,
    visualize_visualize,
    generate_reconstructions_for_visualization,
    compute_test_metrics,
    init_checkpoint_state,
    compute_and_validate_moment_weights,
    check_eval_paths_provided,
)

from FlexCNN_for_Medical_Physics.functions.main_run_functions.run_time_evaluation import report_cross_validation_metrics

from FlexCNN_for_Medical_Physics.functions.helper.model_setup.setup_generators_optimizer import (
    instantiate_dual_generators,
    load_dual_generator_checkpoints,
    create_optimizer,
    create_lr_scheduler,
)
from FlexCNN_for_Medical_Physics.functions.helper.model_setup.config_materialize import materialize_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_trainable_frozen_flow(config, paths, settings):
    """
    Train, test, or visualize a network using a frozen attenuation network (backbone)
    guiding a trainable activity network. Handles coflow/counterflow feature injection.
    """
    # Materialize config (convert string references to objects for consistency)
    config = materialize_config(config)
    
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
    elif run_mode == 'train':
        train_dataframe_path = paths.get('train_dataframe_path')

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
            if os.path.exists(tune_dataframe_path):
                # Preserve rows written by previous trials in the same Ray Tune run
                tune_dataframe = pd.read_csv(tune_dataframe_path)
            else:
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

    if run_mode == 'train':
        # Create or load training learning-curve dataframe (if train_val_* files provided)
        # When resuming (load_state=True), load existing CSV if present to append new rows
        if load_state and train_dataframe_path is not None and os.path.exists(train_dataframe_path):
            train_dataframe = pd.read_csv(train_dataframe_path)
            print(f'[TRAIN LEARNING CURVES] Loaded existing dataframe from {train_dataframe_path} ({len(train_dataframe)} rows)')
        else:
            train_dataframe = pd.DataFrame(columns=['epoch', 'batch_step', 'example_num', 'eval_split', 'MSE', 'SSIM', 'CUSTOM'])

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

    start_epoch, end_epoch, batch_step, gen_state_dict, gen_act_opt_state_dict, lr_scheduler_state_dict = init_checkpoint_state(
        load_state, run_mode, act_ckpt, num_epochs, device
    )
    if gen_act_opt_state_dict:
        gen_act_opt.load_state_dict(gen_act_opt_state_dict)

    # Epoch-based scheduler is active in train mode only (activity network).
    lr_scheduler = create_lr_scheduler(
        gen_act_opt,
        settings,
        total_epochs=num_epochs,
        resumed_epochs=start_epoch,
        advance_on_resume=(lr_scheduler_state_dict is None),
    )
    if lr_scheduler is not None and lr_scheduler_state_dict is not None:
        lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    if run_mode == 'train':
        start_lr = gen_act_opt.param_groups[0]['lr']
        print(f"[LR] Training start: {start_lr:.6e}")

    # ========================================================================================
    # SECTION 6: INSTANTIATE LOSS FUNCTION FOR ACTIVITY NETWORK
    # ========================================================================================
    base_criterion = config['sup_base_criterion']
    stats_criterion = config['SI_stats_criterion']
    alpha_min = config['SI_alpha_min']
    half_life_examples = config['SI_half_life_examples']
    
    # Compute and explicitly set moment weights if stats loss is enabled
    moment_weights = compute_and_validate_moment_weights(stats_criterion, config, 'SI')
    if moment_weights is not None:
        stats_criterion.moment_weights = moment_weights
    
    hybrid_loss = HybridLoss(
        base_loss=base_criterion,
        stats_loss=stats_criterion,
        alpha_min=alpha_min,
        half_life_examples=half_life_examples,
    )

    # ========================================================================================
    # SECTION 7A: FILTER PATHS BY NETWORK TYPE (avoid loading unnecessary data)
    # ========================================================================================
    # Set unneeded training data paths to None
    if config['network_type'] == 'FROZEN_COFLOW':
        # Coflow uses attenuation sinogram input only
        paths['atten_image_path'] = None
        paths['tune_val_atten_image_path'] = None
        paths['tune_qa_atten_image_path'] = None
    elif config['network_type'] == 'FROZEN_COUNTERFLOW':
        # Counterflow uses attenuation image input only
        paths['atten_sino_path'] = None
        paths['tune_val_atten_sino_path'] = None
        paths['tune_qa_atten_sino_path'] = None


    # ========================================================================================
    # SECTION 7B: VALIDATE REQUIRED PATHS AND BUILD DATA LOADER
    # ========================================================================================
    def require_path(key: str):
        if key not in paths or paths[key] is None:
            raise ValueError(f"Missing required path: '{key}'. Provide act_*/atten_* paths for frozen flow.")
    require_path('act_image_path')
    require_path('act_sino_path')
    if config['network_type'] == 'FROZEN_COFLOW':
        require_path('atten_sino_path')
    else:
        require_path('atten_image_path')

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
    report_num = 1

    time_init_full = time.time()
    time_init_loader = time.time()
    
    # Track best holdout performance for conditional saving (train mode only)
    best_holdout_metrics = {}

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
                atten_output = result['output']
                frozen_enc_feats = result['encoder']
                frozen_dec_feats = result['decoder']

            # ----- SUBSECTION 10D: ACTIVITY NETWORK FORWARD/BACKWARD PASS -----
            
            input_ = act_sino_scaled
            target = act_image_scaled

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
                    'atten_input': atten_input,
                    'atten_output': atten_output,
                }

                # Ray Tune reporting (tune mode)
                if run_mode == 'tune' and session is not None:
                    report_cross_validation_metrics(
                        (gen_act, gen_atten), paths, config, settings, tune_dataframe, tune_dataframe_path,
                        train_SI_act, tune_dataframe_fraction, tune_max_t, report_num,
                        example_num, batch_step, epoch, device
                    )
                    report_num += 1

                # Training visualization
                if run_mode == 'train':
                    print('Flow mode: ', flow_mode)
                    current_CNN_SSIM = calculate_metric(target, CNN_output, SSIM)
                    current_CNN_MSE = calculate_metric(target, CNN_output, MSE)
                    visualize_train_frozen(batch_data, mean_gen_loss, current_CNN_MSE, current_CNN_SSIM, epoch, batch_step, example_num)

                # Test visualization
                if run_mode == 'test':
                    visualize_test(batch_data, mean_CNN_MSE, mean_CNN_SSIM, mean_recon1_MSE, mean_recon1_SSIM, mean_recon2_MSE, mean_recon2_SSIM)

                # Visualization mode (display input, output, recons)
                if run_mode == 'visualize':
                    batch_data['batch_step'] = batch_step
                    visualize_visualize(batch_data, batch_size, offset)

                # Reset running metrics after each display interval
                mean_gen_loss = 0

                _ = display_times('visualization time', time_init_visualization, show_times)

            # End of batch: reset loader timer
            time_init_loader = time.time()

        # ========================================================================================
        # SECTION 12.5: EPOCH-END LEARNING CURVE EVALUATION (train mode only)
        # ========================================================================================
        if run_mode == 'train':
            # Evaluate on training, holdout, and optionally QA splits at epoch boundary
            from FlexCNN_for_Medical_Physics.functions.main_run_functions.run_time_evaluation import load_eval_batch, evaluate_metrics
            
            # Check which splits are available
            available = check_eval_paths_provided(paths, config['network_type'])
            
            gen_act.eval()
            try:
                # ===== TRAINING SPLIT (always available) =====
                train_batch = load_eval_batch('train', paths, config, settings)
                train_metrics = evaluate_metrics(
                    (gen_act, gen_atten), train_batch, device, train_SI_act, config['network_type'],
                    tune_metric='MSE',  # Metric doesn't matter for train split, we'll compute all standard metrics
                    evaluate_on='val',
                    run_mode='train',
                    compute_standard_metrics=True  # Always compute MSE/SSIM/CUSTOM for CSV consistency
                )
                train_dataframe = append_train_learning_curve_row(
                    train_dataframe, train_dataframe_path, train_metrics,
                    eval_split='training set', epoch=epoch+1, batch_step=batch_step, example_num=batch_step
                )
                
                # ===== HOLDOUT SPLIT (conditional) =====
                if available['holdout']:
                    holdout_batch = load_eval_batch('holdout', paths, config, settings)
                    holdout_metrics = evaluate_metrics(
                        (gen_act, gen_atten), holdout_batch, device, train_SI_act, config['network_type'],
                        tune_metric='MSE',
                        evaluate_on='val',
                        run_mode='train',
                        compute_standard_metrics=True  # Always compute standard metrics for CSV consistency
                    )
                    train_dataframe = append_train_learning_curve_row(
                        train_dataframe, train_dataframe_path, holdout_metrics,
                        eval_split='holdout set', epoch=epoch+1, batch_step=batch_step, example_num=batch_step
                    )
                
                # ===== QA SPLIT (conditional) =====
                if available['qa']:
                    qa_batch = load_eval_batch('qa', paths, config, settings, augment=('SI', True))
                    qa_metrics = evaluate_metrics(
                        (gen_act, gen_atten), qa_batch, device, train_SI_act, config['network_type'],
                        tune_metric='MSE',
                        evaluate_on='val',  # Force validation metrics for learning curves
                        run_mode='train',
                        compute_standard_metrics=True  # Always compute standard metrics for CSV consistency
                    )
                    train_dataframe = append_train_learning_curve_row(
                        train_dataframe, train_dataframe_path, qa_metrics,
                        eval_split='QA set', epoch=epoch+1, batch_step=batch_step, example_num=batch_step
                    )
                
                print(f"[TRAIN LEARNING CURVES] Epoch {epoch+1}: train={train_metrics[list(train_metrics.keys())[0]]:.4f}")
                if available['holdout']:
                    print(f"  holdout={holdout_metrics[list(holdout_metrics.keys())[0]]:.4f}")
                if available['qa']:
                    print(f"  qa={qa_metrics[list(qa_metrics.keys())[0]]:.4f}")
                
                # ===== CONDITIONAL SAVE BASED ON HOLDOUT PERFORMANCE =====
                if save_state and available['holdout']:
                    if settings['train_save_on'] == 'always':
                        should_save = True
                    else:
                        # Get the metric specified by train_save_on ('SSIM', 'MSE', or 'CUSTOM')
                        metric_value = holdout_metrics[settings['train_save_on']]
                        
                        if settings['train_save_on'] == 'SSIM':
                            # Maximize: save if new value is higher
                            should_save = metric_value > best_holdout_metrics.get(settings['train_save_on'], -float('inf'))
                        elif settings['train_save_on'] in ['MSE', 'CUSTOM']:
                            # Minimize: save if new value is lower
                            should_save = metric_value < best_holdout_metrics.get(settings['train_save_on'], float('inf'))
                    
                    if should_save:
                        best_holdout_metrics[settings['train_save_on']] = metric_value
                        save_lr = gen_act_opt.param_groups[0]['lr']
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print(f'💾 Saving model! New best {settings["train_save_on"]}: {metric_value:.6f}')
                        print(f'[LR] Save-time LR: {save_lr:.6e}')
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        checkpoint_dict = build_checkpoint_dict(gen_act, gen_act_opt, config, epoch+1, batch_step, scheduler=lr_scheduler)
                        save_checkpoint(checkpoint_dict, checkpoint_path + '-act')
                    
            except FileNotFoundError:
                raise
            except Exception as e:
                print(f"[TRAIN LEARNING CURVES] Warning: epoch {epoch+1} evaluation failed: {str(e)}")
                print("  Continuing training without learning curve logging for this epoch.")
            finally:
                gen_act.train()  # Restore training mode

            if lr_scheduler is not None:
                lr_scheduler.step()
                current_lr = gen_act_opt.param_groups[0]['lr']
                print(f"[LR] Epoch {epoch+1}: {current_lr:.6e}")

    # ========================================================================================
    # SECTION 13: RETURN DATAFRAMES (if applicable)
    # ========================================================================================

    if run_mode == 'test':
        return test_dataframe
