import os

def _prefix_config_keys(config, prefix):
    """
    Rename all keys in config by adding prefix.
    
    Args:
        config: Dictionary to prefix
        prefix: String prefix to add (e.g., 'FROZEN')
    
    Returns:
        New dictionary with all keys prefixed as 'prefix_originalkey'
    """
    return {f"{prefix}_{k}": v for k, v in config.items()}

def construct_config(
    run_mode,
    network_opts,
    tune_opts,
    test_opts,
    viz_opts,
    config_ACT_SI=None,
    config_ACT_IS=None,
    config_ATTEN_SI=None,
    config_ATTEN_IS=None,
    config_CONCAT=None,
    config_FROZEN_COFLOW=None,
    config_FROZEN_COUNTERFLOW=None,
    config_GAN_SI=None,
    config_GAN_IS=None,
    config_CYCLEGAN=None,
    config_RAY_SI=None,
    config_RAY_SI_learnScale=None,
    config_RAY_SI_fixedScale=None,
    config_RAY_IS=None,
    config_RAY_IS_learnScale=None,
    config_RAY_IS_fixedScale=None,
    config_RAY_SUP=None,
    config_RAY_SUP_FROZEN=None,
    config_RAY_GAN=None,
    config_RAY_GAN_CYCLE=None):

    """
    Combines configuration dictionaries based on run_mode, network_type, and train_SI.
    
    Args:
        run_mode: 'tune', 'train', 'test', or 'visualize'
        network_opts: dict with keys: network_type, train_SI, gen_image_size, gen_sino_size, gen_image_channels, gen_sino_channels
        tune_opts: dict with keys: tune_debug, etc.
        test_opts: dict with keys: test_batch_size, etc.
        viz_opts: dict with keys: visualize_batch_size, etc.
        config_ACT_SI, config_ACT_IS, config_ATTEN_SI, config_ATTEN_IS, config_CONCAT, config_FROZEN_COFLOW, config_FROZEN_COUNTERFLOW, etc.: Network configuration dictionaries
    
    Returns:
        config dict with all network hyperparameters and data dimensions
    """
    # Extract network options
    network_type = str(network_opts['network_type']).upper()  # Normalize strings
    train_SI = network_opts['train_SI']
    gen_image_size = network_opts['gen_image_size']
    gen_sino_size = network_opts['gen_sino_size']
    gen_image_channels = network_opts['gen_image_channels']
    gen_sino_channels = network_opts['gen_sino_channels']
    SI_normalize = network_opts['SI_normalize']
    IS_normalize = network_opts['IS_normalize']

    # If not tuning (or forcing tuning with a fixed config for debugging), choose config dictionary based on run_mode and network_type
    if run_mode in ['train', 'test', 'visualize', 'none'] or tune_opts.get('tune_force_fixed_config')==True:
        if network_type == 'ACT':
            config = config_ACT_SI if train_SI else config_ACT_IS
        elif network_type == 'ATTEN':
            config = config_ATTEN_SI if train_SI else config_ATTEN_IS
        elif network_type == 'CONCAT':
            config = config_CONCAT
        elif network_type == 'FROZEN_COFLOW':
            config = config_FROZEN_COFLOW
        elif network_type == 'FROZEN_COUNTERFLOW':
            config = config_FROZEN_COUNTERFLOW
        elif network_type == 'GAN':
            config = config_GAN_SI if train_SI else config_GAN_IS
        elif network_type == 'CYCLEGAN':
            config = config_CYCLEGAN
        else:
            raise ValueError(f"Unknown network_type '{network_type}'.")
        
        # Validate that user-specified dimensions match the loaded config
        # (can't change network architecture for already-trained networks)
        mismatches = []
        if config.get('gen_image_size') != gen_image_size:
            mismatches.append(f"gen_image_size: config has {config.get('gen_image_size')}, but network_opts specifies {gen_image_size}")
        if config.get('gen_sino_size') != gen_sino_size:
            mismatches.append(f"gen_sino_size: config has {config.get('gen_sino_size')}, but network_opts specifies {gen_sino_size}")
        if config.get('gen_image_channels') != gen_image_channels:
            mismatches.append(f"gen_image_channels: config has {config.get('gen_image_channels')}, but network_opts specifies {gen_image_channels}")
        if config.get('gen_sino_channels') != gen_sino_channels:
            mismatches.append(f"gen_sino_channels: config has {config.get('gen_sino_channels')}, but network_opts specifies {gen_sino_channels}")
        if config.get('network_type') != network_type:
            mismatches.append(f"network_type: config has {config.get('network_type')}, but network_opts specifies {network_type}")

        if mismatches:
            error_msg = f"\n❌ Network dimension or type mismatch detected!\n\nThe loaded configuration has different dimensions than your network_opts settings.\n"
            error_msg += f"This usually means the checkpoint was trained with different input/output sizes.\n\n"
            error_msg += "Mismatches found:\n" + "\n".join(f"  • {m}" for m in mismatches)
            error_msg += f"\n\nPlease update your network_opts to match the trained network, or use a different checkpoint."
            raise ValueError(error_msg)

    # If tuning, we need to construct the dictionary from smaller pieces
    elif run_mode == 'tune':
        # First, we add user normalization and scaling options. These must be added before combining dicts (below).
        if network_type == 'ACT':
            if train_SI:
                if SI_normalize:
                    config = {**config_RAY_SI, **config_RAY_SI_fixedScale ,**config_RAY_SUP}
                else:
                    config = {**config_RAY_SI, **config_RAY_SI_learnScale ,**config_RAY_SUP}
            else:
                if IS_normalize:
                    config = {**config_RAY_IS, **config_RAY_IS_fixedScale ,**config_RAY_SUP}
                else:
                    config = {**config_RAY_IS, **config_RAY_IS_learnScale ,**config_RAY_SUP}
        elif network_type == 'ATTEN':
            if train_SI:
                if SI_normalize:
                    config = {**config_RAY_SI, **config_RAY_SI_fixedScale ,**config_RAY_SUP}
                else:
                    config = {**config_RAY_SI, **config_RAY_SI_learnScale ,**config_RAY_SUP}
            else:
                if IS_normalize:
                    config = {**config_RAY_IS, **config_RAY_IS_fixedScale ,**config_RAY_SUP}
                else:
                    config = {**config_RAY_IS, **config_RAY_IS_learnScale ,**config_RAY_SUP}
        elif network_type == 'CONCAT':
            if SI_normalize:
                config = {**config_RAY_SI, **config_RAY_SI_fixedScale ,**config_RAY_SUP}
            else:
                config = {**config_RAY_SI, **config_RAY_SI_learnScale ,**config_RAY_SUP}
        elif network_type == 'FROZEN_COFLOW':
            if SI_normalize:
                config = {**_prefix_config_keys(config_ATTEN_SI, 'FROZEN'), **config_RAY_SI, **config_RAY_SI_fixedScale, **config_RAY_SUP, **config_RAY_SUP_FROZEN}
            else:
                config = {**_prefix_config_keys(config_ATTEN_SI, 'FROZEN'), **config_RAY_SI, **config_RAY_SI_learnScale, **config_RAY_SUP, **config_RAY_SUP_FROZEN}
        elif network_type == 'FROZEN_COUNTERFLOW':
            if SI_normalize:
                config = {**_prefix_config_keys(config_ATTEN_IS, 'FROZEN'), **config_RAY_SI, **config_RAY_SI_fixedScale, **config_RAY_SUP, **config_RAY_SUP_FROZEN}
            else:
                config = {**_prefix_config_keys(config_ATTEN_IS, 'FROZEN'), **config_RAY_SI, **config_RAY_SI_learnScale, **config_RAY_SUP, **config_RAY_SUP_FROZEN}
        elif network_type == 'GAN':
            if train_SI:
                if SI_normalize:
                    config = {**config_RAY_SI, **config_RAY_SI_fixedScale ,**config_RAY_GAN}
                else:
                    config = {**config_RAY_SI, **config_RAY_SI_learnScale ,**config_RAY_GAN}
            else:
                if IS_normalize:
                    config = {**config_RAY_IS, **config_RAY_IS_fixedScale ,**config_RAY_GAN}
                else:
                    config = {**config_RAY_IS, **config_RAY_IS_learnScale ,**config_RAY_GAN}
        elif network_type == 'CYCLEGAN':
            config = {**config_GAN_SI, **config_GAN_IS, **config_RAY_GAN_CYCLE}
        else:
            raise ValueError(f"Unknown network_type '{network_type}'.")

        # Add data dimensions to config. These are set by the user and not tuned.
        config['network_type'] = network_type # If config is being built from smaller configs (CYCLESUP, CYCLEGAN), then this overwrites any existing value.
        config['train_SI'] = train_SI # Only used for SUP and GAN networks but added here for consistency.
        config['gen_image_size'] = gen_image_size
        config['gen_sino_size'] = gen_sino_size
        config['gen_image_channels'] = gen_image_channels
        config['gen_sino_channels'] = gen_sino_channels
    else:
        raise ValueError(f"Unknown run_mode '{run_mode}'.")


    ## Overrides ##
    # Override batch size for test or visualize modes. Otherwise, when testing or visualizing, the batch size from training/tuning would be used.
    if run_mode == 'test':
        config['batch_size'] = test_opts['test_batch_size']
    elif run_mode == 'visualize':
        config['batch_size'] = viz_opts['visualize_batch_size']

    return config

def setup_paths(run_mode, base_dirs, data_files, mode_files, test_ops, viz_ops):
    """
    Build all path-related configuration.
    
    Args:
        base_dirs: dict with keys: project_dirPath, plot_dirName, checkpoint_dirName, tune_storage_dirName,
                   tune_dataframe_dirName, test_dataframe_dirName, data_dirPath (optional absolute path), 
                   data_dirName (fallback directory name)
        data_files: dict with keys:
            - tune_sino_file, tune_image_file, tune_recon1_file, tune_recon2_file,
              tune_atten_image_file, tune_atten_sino_file,
              tune_val_sino_file, tune_val_image_file, tune_val_atten_image_file, tune_val_atten_sino_file,
              tune_qa_sino_file, tune_qa_image_file, tune_qa_hotMask_file, tune_qa_hotBackgroundMask_file,
              tune_qa_coldMask_file, tune_qa_coldBackgroundMask_file, tune_qa_atten_image_file, tune_qa_atten_sino_file,
              train_sino_file, train_image_file, train_recon1_file, train_recon2_file, train_atten_image_file, train_atten_sino_file,
              test_sino_file, test_image_file, test_recon1_file, test_recon2_file, test_atten_image_file, test_atten_sino_file,
              visualize_sino_file, visualize_image_file, visualize_recon1_file, visualize_recon2_file, visualize_atten_image_file, visualize_atten_sino_file
        mode_files: dict with keys: train_checkpoint_file, test_checkpoint_file, visualize_checkpoint_file,
                    tune_csv_file, test_csv_file
        run_mode: 'tune', 'train', 'test', or 'visualize'
    
    Returns:
        paths dict with directory paths, mode-specific data paths, active sino/image paths, checkpoint_path,
        tune/test dataframe paths.
    """

    paths = {}
    
    # Base directories
    paths['plot_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['plot_dirName'])
    paths['checkpoint_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['checkpoint_dirName'])
    paths['tune_storage_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['tune_storage_dirName'])
    paths['tune_dataframe_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['tune_dataframe_dirName'])
    paths['test_dataframe_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['test_dataframe_dirName'])
    
    # Data directory: Use absolute path if provided, otherwise place in project directory
    if base_dirs.get('data_dirPath') is not None:
        paths['data_dirPath'] = base_dirs['data_dirPath']
    else:
        paths['data_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['data_dirName'])
    
    # Mode-specific data file paths (activity domain renamed to act_*)
    paths['tune_act_sino_path'] = os.path.join(paths['data_dirPath'], data_files['tune_act_sino_file'])
    paths['tune_act_image_path'] = os.path.join(paths['data_dirPath'], data_files['tune_act_image_file'])
    paths['tune_act_recon1_path'] = os.path.join(paths['data_dirPath'], data_files['tune_act_recon1_file']) if data_files['tune_act_recon1_file'] is not None else None
    paths['tune_act_recon2_path'] = os.path.join(paths['data_dirPath'], data_files['tune_act_recon2_file']) if data_files['tune_act_recon2_file'] is not None else None
    paths['tune_atten_image_path'] = os.path.join(paths['data_dirPath'], data_files['tune_atten_image_file']) if data_files['tune_atten_image_file'] is not None else None
    paths['tune_atten_sino_path'] = os.path.join(paths['data_dirPath'], data_files['tune_atten_sino_file']) if data_files['tune_atten_sino_file'] is not None else None
    paths['tune_val_act_sino_path'] = os.path.join(paths['data_dirPath'], data_files['tune_val_act_sino_file']) if data_files['tune_val_act_sino_file'] is not None else None
    paths['tune_val_act_image_path'] = os.path.join(paths['data_dirPath'], data_files['tune_val_act_image_file']) if data_files['tune_val_act_image_file'] is not None else None
    paths['tune_val_atten_image_path'] = os.path.join(paths['data_dirPath'], data_files['tune_val_atten_image_file']) if data_files['tune_val_atten_image_file'] is not None else None
    paths['tune_val_atten_sino_path'] = os.path.join(paths['data_dirPath'], data_files['tune_val_atten_sino_file']) if data_files['tune_val_atten_sino_file'] is not None else None
    paths['tune_qa_act_sino_path'] = os.path.join(paths['data_dirPath'], data_files['tune_qa_act_sino_file']) if data_files['tune_qa_act_sino_file'] is not None else None
    paths['tune_qa_act_image_path'] = os.path.join(paths['data_dirPath'], data_files['tune_qa_act_image_file']) if data_files['tune_qa_act_image_file'] is not None else None
    paths['tune_qa_backMask_path'] = os.path.join(paths['data_dirPath'], data_files['tune_qa_backMask_file']) if data_files['tune_qa_backMask_file'] is not None else None
    paths['tune_qa_hotMask_path'] = os.path.join(paths['data_dirPath'], data_files['tune_qa_hotMask_file']) if data_files['tune_qa_hotMask_file'] is not None else None
    paths['tune_qa_hotBackgroundMask_path'] = os.path.join(paths['data_dirPath'], data_files['tune_qa_hotBackgroundMask_file']) if data_files['tune_qa_hotBackgroundMask_file'] is not None else None
    paths['tune_qa_coldMask_path'] = os.path.join(paths['data_dirPath'], data_files['tune_qa_coldMask_file']) if data_files['tune_qa_coldMask_file'] is not None else None
    paths['tune_qa_coldBackgroundMask_path'] = os.path.join(paths['data_dirPath'], data_files['tune_qa_coldBackgroundMask_file']) if data_files['tune_qa_coldBackgroundMask_file'] is not None else None
    paths['tune_qa_atten_image_path'] = os.path.join(paths['data_dirPath'], data_files['tune_qa_atten_image_file']) if data_files['tune_qa_atten_image_file'] is not None else None
    paths['tune_qa_atten_sino_path'] = os.path.join(paths['data_dirPath'], data_files['tune_qa_atten_sino_file']) if data_files['tune_qa_atten_sino_file'] is not None else None
    paths['train_act_sino_path'] = os.path.join(paths['data_dirPath'], data_files['train_act_sino_file'])
    paths['train_act_image_path'] = os.path.join(paths['data_dirPath'], data_files['train_act_image_file'])
    paths['train_act_recon1_path'] = os.path.join(paths['data_dirPath'], data_files['train_act_recon1_file']) if data_files['train_act_recon1_file'] is not None else None
    paths['train_act_recon2_path'] = os.path.join(paths['data_dirPath'], data_files['train_act_recon2_file']) if data_files['train_act_recon2_file'] is not None else None
    paths['train_atten_image_path'] = os.path.join(paths['data_dirPath'], data_files['train_atten_image_file']) if data_files['train_atten_image_file'] is not None else None
    paths['train_atten_sino_path'] = os.path.join(paths['data_dirPath'], data_files['train_atten_sino_file']) if data_files['train_atten_sino_file'] is not None else None
    paths['test_act_sino_path'] = os.path.join(paths['data_dirPath'], data_files['test_act_sino_file'])
    paths['test_act_image_path'] = os.path.join(paths['data_dirPath'], data_files['test_act_image_file'])
    paths['test_act_recon1_path'] = os.path.join(paths['data_dirPath'], data_files['test_act_recon1_file']) if data_files['test_act_recon1_file'] is not None else None
    paths['test_act_recon2_path'] = os.path.join(paths['data_dirPath'], data_files['test_act_recon2_file']) if data_files['test_act_recon2_file'] is not None else None
    paths['test_atten_image_path'] = os.path.join(paths['data_dirPath'], data_files['test_atten_image_file']) if data_files['test_atten_image_file'] is not None else None
    paths['test_atten_sino_path'] = os.path.join(paths['data_dirPath'], data_files['test_atten_sino_file']) if data_files['test_atten_sino_file'] is not None else None
    paths['visualize_act_sino_path'] = os.path.join(paths['data_dirPath'], data_files['visualize_act_sino_file'])
    paths['visualize_act_image_path'] = os.path.join(paths['data_dirPath'], data_files['visualize_act_image_file'])
    paths['visualize_act_recon1_path'] = os.path.join(paths['data_dirPath'], data_files['visualize_act_recon1_file']) if data_files['visualize_act_recon1_file'] is not None else None
    paths['visualize_act_recon2_path'] = os.path.join(paths['data_dirPath'], data_files['visualize_act_recon2_file']) if data_files['visualize_act_recon2_file'] is not None else None
    paths['visualize_atten_image_path'] = os.path.join(paths['data_dirPath'], data_files['visualize_atten_image_file']) if data_files['visualize_atten_image_file'] is not None else None
    paths['visualize_atten_sino_path'] = os.path.join(paths['data_dirPath'], data_files['visualize_atten_sino_file']) if data_files['visualize_atten_sino_file'] is not None else None
    
    # Backward-compatible note: attenuation image/sinogram paths already assigned above for each mode
    
    # Active paths and checkpoint filename selection (act_* only; no legacy fallbacks)
    if run_mode == 'tune':
        paths['act_sino_path'] = paths['tune_act_sino_path']
        paths['act_image_path'] = paths['tune_act_image_path']
        paths['act_recon1_path'] = None # We do not use recon paths during tuning
        paths['act_recon2_path'] = None
        paths['atten_image_path'] = paths['tune_atten_image_path']
        paths['atten_sino_path'] = paths['tune_atten_sino_path']
        checkpoint_file = ''
    elif run_mode == 'train':
        paths['act_sino_path'] = paths['train_act_sino_path']
        paths['act_image_path'] = paths['train_act_image_path']
        paths['act_recon1_path'] = None # We do not use recon paths during training
        paths['act_recon2_path'] = None
        paths['atten_image_path'] = paths['train_atten_image_path']
        paths['atten_sino_path'] = paths['train_atten_sino_path']
        checkpoint_file = mode_files['train_checkpoint_file']
    elif run_mode == 'test':
        paths['act_sino_path'] = paths['test_act_sino_path']
        paths['act_image_path'] = paths['test_act_image_path']
        paths['act_recon1_path'] = paths['test_act_recon1_path']
        paths['act_recon2_path'] = paths['test_act_recon2_path']
        paths['atten_image_path'] = paths['test_atten_image_path']
        paths['atten_sino_path'] = paths['test_atten_sino_path']
        checkpoint_file = mode_files['test_checkpoint_file']
    elif run_mode in ['visualize', 'none']:
        paths['act_sino_path'] = paths['visualize_act_sino_path']
        paths['act_image_path'] = paths['visualize_act_image_path']
        paths['act_recon1_path'] = paths['visualize_act_recon1_path']
        paths['act_recon2_path'] = paths['visualize_act_recon2_path']
        paths['atten_image_path'] = paths['visualize_atten_image_path']
        paths['atten_sino_path'] = paths['visualize_atten_sino_path']
        checkpoint_file = mode_files['visualize_checkpoint_file']
    else:
        raise ValueError(f"Unknown run_mode: {run_mode}")
    
    # Checkpoint path
    paths['checkpoint_path'] = os.path.join(paths['checkpoint_dirPath'], checkpoint_file)
    
    # Dataframe paths (always constructed for clarity)
    paths['tune_dataframe_path'] = os.path.join(paths['tune_dataframe_dirPath'], f"{mode_files['tune_csv_file']}.csv")
    paths['test_dataframe_path'] = os.path.join(paths['test_dataframe_dirPath'], f"{mode_files['test_csv_file']}.csv")
    
    return paths

def setup_settings( run_mode, common_settings, tune_opts, train_opts, test_opts, viz_opts):
    """
    Build all non-path runtime settings.
    
    Args:
        common_settings: dict with keys: device, num_examples
        tune_opts: dict with keys: tune_augment, tune_batches_per_report, tune_examples_per_report,
                   tune_even_reporting, tune_max_t, tune_dataframe_fraction
        train_opts: dict with keys: train_augment, training_epochs, train_load_state, train_save_state,
                    train_show_times, train_sample_division, train_display_step
        test_opts: dict with keys: test_show_times, test_display_step, test_batch_size, test_chunk_size,
                   testset_size, test_begin_at, test_compute_MLEM, test_merge_dataframes,
                   test_shuffle, test_sample_division
        viz_opts: dict with keys: visualize_shuffle, visualize_offset, visualize_batch_size
        run_mode: 'tune', 'train', 'test', or 'visualize'
    
    Returns:
        settings dict containing runtime (non-path) configuration.
    """
    settings = {}
    
    # Common settings (now minimal)
    settings['run_mode'] = run_mode
    settings['device'] = common_settings['device']
    settings['num_examples'] = common_settings['num_examples']
    settings['act_sino_scale'] = common_settings['act_sino_scale']
    settings['act_recon1_scale'] = common_settings['act_recon1_scale']
    settings['act_recon2_scale'] = common_settings['act_recon2_scale']
    settings['act_image_scale'] = common_settings['act_image_scale']
    settings['atten_image_scale'] = common_settings['atten_image_scale']
    settings['atten_sino_scale'] = common_settings['atten_sino_scale']

    # Mode-specific
    if run_mode == 'tune':
        settings['augment'] = tune_opts['tune_augment']
        settings['shuffle'] = True
        settings['num_epochs'] = 1000  # Tuning is stopped when the iteration = tune_max_t (defined later). We set num_epochs to a large number so tuning doesn't terminate early.
        settings['load_state'] = False
        settings['save_state'] = False
        settings['offset'] = 0
        settings['show_times'] = False
        settings['sample_division'] = 1
        
        settings['tune_exp_name'] = tune_opts['tune_exp_name']
        settings['tune_scheduler'] = tune_opts['tune_scheduler']
        settings['tune_dataframe_fraction'] = tune_opts['tune_dataframe_fraction']
        settings['tune_restore'] = tune_opts['tune_restore']
        settings['tune_max_t'] = tune_opts['tune_max_t']
        settings['tune_minutes'] = tune_opts['tune_minutes']
        settings['tune_metric'] = tune_opts['tune_metric']
        settings['tune_even_reporting'] = tune_opts['tune_even_reporting']
        settings['tune_batches_per_report'] = tune_opts['tune_batches_per_report']
        settings['tune_examples_per_report'] = tune_opts['tune_examples_per_report']
        settings['tune_augment'] = tune_opts['tune_augment']
        settings['tune_qa_load_mode'] = tune_opts.get('tune_qa_load_mode', 'random')
        settings['tune_qa_slice_range'] = tune_opts.get('tune_qa_slice_range')
        settings['tune_debug'] = tune_opts['tune_debug']
        settings['tune_report_for'] = tune_opts['tune_report_for']
        settings['tune_eval_batch_size'] = tune_opts['tune_eval_batch_size']
        settings['tune_qa_hot_weight'] = tune_opts['tune_qa_hot_weight']

    elif run_mode == 'train':
        settings['augment'] = train_opts['train_augment']
        # DEBUG: Log what augmentation is being set
        print(f"[SETUP_SETTINGS DEBUG - TRAIN] Setting augment to: {train_opts['train_augment']}")
        settings['shuffle'] = train_opts['train_shuffle']
        settings['num_epochs'] = train_opts['training_epochs']
        settings['load_state'] = train_opts['train_load_state']
        settings['save_state'] = train_opts['train_save_state']
        settings['offset'] = 0
        settings['show_times'] = train_opts['train_show_times']
        settings['sample_division'] = train_opts['train_sample_division']
        settings['train_display_step'] = train_opts['train_display_step'] # Used in compute_display_step()
        settings['train_report_eval'] = train_opts['train_report_eval']
        
        # If train_report_eval is enabled, copy tune validation settings needed for evaluation
        if train_opts['train_report_eval']:
            settings['tune_report_for'] = tune_opts['tune_report_for']
            settings['tune_qa_load_mode'] = tune_opts.get('tune_qa_load_mode', 'random')
            settings['tune_qa_slice_range'] = tune_opts.get('tune_qa_slice_range')
            settings['tune_metric'] = tune_opts['tune_metric']
            settings['tune_eval_batch_size'] = tune_opts['tune_eval_batch_size']
            settings['tune_qa_hot_weight'] = tune_opts.get('tune_qa_hot_weight', 0.5)

    elif run_mode == 'test':
        settings['augment'] = (None, False) # If testing, do not augment
        settings['shuffle'] = False
        settings['num_epochs'] = 1
        settings['load_state'] = True
        settings['save_state'] = False
        settings['offset'] = 0
        settings['show_times'] = test_opts['test_show_times']
        settings['sample_division'] = test_opts['test_sample_division', 1]
        settings['test_display_step'] = test_opts['test_display_step'] # Used in compute_display_step()
        #settings['test_batch_size'] = test_opts['test_batch_size']
    
    elif run_mode in ['visualize', 'none']:
        settings['augment'] = (None, False) # If visualizing, do not augment
        settings['shuffle'] = viz_opts['visualize_shuffle']
        settings['num_epochs'] = 1
        settings['load_state'] = True
        settings['save_state'] = False
        settings['show_times'] = False
        settings['offset'] = viz_opts['visualize_offset']
        settings['sample_division'] = 1
        #settings['visualize_batch_size'] = viz_opts['visualize_batch_size']
    else:
        raise ValueError(f"Unknown run_mode: {run_mode}")
    
    return settings