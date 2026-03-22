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
    config_DENOISE_SI=None,
    config_RECON_SINO_SI=None,
    config_RECON_SINO_IS=None,
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
    recon_variant = network_opts.get('recon_variant', 1)
    frozen_variant = str(network_opts.get('frozen_variant', 'ATTEN')).upper()
    if frozen_variant not in ('ATTEN', 'RECON_SINO'):
        raise ValueError(f"Invalid frozen_variant='{network_opts.get('frozen_variant')}'. Expected 'atten'/'ATTEN' or 'recon_sino'/'RECON_SINO'.")

    # If not tuning (or forcing tuning with a fixed config for debugging), choose config dictionary based on run_mode and network_type
    if run_mode in ['train', 'test', 'visualize', 'none'] or tune_opts.get('tune_force_fixed_config')==True:
        if network_type == 'ACT':
            config = config_ACT_SI if train_SI else config_ACT_IS
        elif network_type == 'ATTEN':
            config = config_ATTEN_SI if train_SI else config_ATTEN_IS
        elif network_type == 'DENOISE':
            config = config_DENOISE_SI
        elif network_type == 'RECON_SINO':
            config = config_RECON_SINO_SI if train_SI else config_RECON_SINO_IS
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

        if network_type in ('FROZEN_COFLOW', 'FROZEN_COUNTERFLOW'):
            loaded_frozen_type = str(config.get('FROZEN_network_type')).upper()
            if loaded_frozen_type != frozen_variant:
                raise ValueError(
                    f"Frozen config mismatch: loaded FROZEN_network_type='{loaded_frozen_type}' "
                    f"but frozen_variant='{frozen_variant}'. Update frozen_variant or load matching frozen config."
                )

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
        elif network_type == 'DENOISE':
            if SI_normalize:
                config = {**config_RAY_SI, **config_RAY_SI_fixedScale, **config_RAY_SUP}
            else:
                config = {**config_RAY_SI, **config_RAY_SI_learnScale, **config_RAY_SUP}
        elif network_type == 'RECON_SINO':
            if train_SI:
                if SI_normalize:
                    config = {**config_RAY_SI, **config_RAY_SI_fixedScale, **config_RAY_SUP}
                else:
                    config = {**config_RAY_SI, **config_RAY_SI_learnScale, **config_RAY_SUP}
            else:
                if IS_normalize:
                    config = {**config_RAY_IS, **config_RAY_IS_fixedScale, **config_RAY_SUP}
                else:
                    config = {**config_RAY_IS, **config_RAY_IS_learnScale, **config_RAY_SUP}
        elif network_type == 'FROZEN_COFLOW':
            frozen_base_config = config_ATTEN_SI if frozen_variant == 'ATTEN' else config_RECON_SINO_SI
            if SI_normalize:
                config = {**_prefix_config_keys(frozen_base_config, 'FROZEN'), **config_RAY_SI, **config_RAY_SI_fixedScale, **config_RAY_SUP, **config_RAY_SUP_FROZEN}
            else:
                config = {**_prefix_config_keys(frozen_base_config, 'FROZEN'), **config_RAY_SI, **config_RAY_SI_learnScale, **config_RAY_SUP, **config_RAY_SUP_FROZEN}
        elif network_type == 'FROZEN_COUNTERFLOW':
            frozen_base_config = config_ATTEN_IS if frozen_variant == 'ATTEN' else config_RECON_SINO_IS
            if SI_normalize:
                config = {**_prefix_config_keys(frozen_base_config, 'FROZEN'), **config_RAY_SI, **config_RAY_SI_fixedScale, **config_RAY_SUP, **config_RAY_SUP_FROZEN}
            else:
                config = {**_prefix_config_keys(frozen_base_config, 'FROZEN'), **config_RAY_SI, **config_RAY_SI_learnScale, **config_RAY_SUP, **config_RAY_SUP_FROZEN}
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

    config['recon_variant'] = recon_variant
    config['frozen_variant'] = frozen_variant


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
        base_dirs: dict with keys: project_dirPath, plot_dirName, checkpoint_dirPath (optional absolute path),
                   checkpoint_dirName (fallback directory name), tune_storage_dirName,
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

    def join_if_present(root_path, file_name):
        return os.path.join(root_path, file_name) if file_name is not None else None
    
    # Base directories
    paths['plot_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['plot_dirName'])
    paths['tune_storage_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['tune_storage_dirName'])
    paths['tune_dataframe_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['tune_dataframe_dirName'])
    paths['train_dataframe_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['train_dataframe_dirName'])
    paths['test_dataframe_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['test_dataframe_dirName'])
    
    # Checkpoint directory: Use absolute path if provided, otherwise place in project directory
    if base_dirs.get('checkpoint_dirPath') is not None:
        paths['checkpoint_dirPath'] = base_dirs['checkpoint_dirPath']
    else:
        paths['checkpoint_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['checkpoint_dirName'])
    
    # Data directory: Use absolute path if provided, otherwise place in project directory
    if base_dirs.get('data_dirPath') is not None:
        paths['data_dirPath'] = base_dirs['data_dirPath']
    else:
        paths['data_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['data_dirName'])
    
    # Mode-specific data file paths (activity domain renamed to act_*)
    paths['tune_act_sino_path'] = join_if_present(paths['data_dirPath'], data_files['tune_act_sino_file'])
    paths['tune_act_image_path'] = join_if_present(paths['data_dirPath'], data_files['tune_act_image_file'])
    paths['tune_act_recon1_path'] = join_if_present(paths['data_dirPath'], data_files['tune_act_recon1_file'])
    paths['tune_act_recon2_path'] = join_if_present(paths['data_dirPath'], data_files['tune_act_recon2_file'])
    paths['tune_atten_image_path'] = join_if_present(paths['data_dirPath'], data_files['tune_atten_image_file'])
    paths['tune_atten_sino_path'] = join_if_present(paths['data_dirPath'], data_files['tune_atten_sino_file'])
    paths['tune_val_act_sino_path'] = join_if_present(paths['data_dirPath'], data_files['tune_val_act_sino_file'])
    paths['tune_val_act_image_path'] = join_if_present(paths['data_dirPath'], data_files['tune_val_act_image_file'])
    paths['tune_val_act_recon1_path'] = join_if_present(paths['data_dirPath'], data_files.get('tune_val_act_recon1_file'))
    paths['tune_val_act_recon2_path'] = join_if_present(paths['data_dirPath'], data_files.get('tune_val_act_recon2_file'))
    paths['tune_val_atten_image_path'] = join_if_present(paths['data_dirPath'], data_files['tune_val_atten_image_file'])
    paths['tune_val_atten_sino_path'] = join_if_present(paths['data_dirPath'], data_files['tune_val_atten_sino_file'])
    paths['qa_act_sino_path'] = join_if_present(paths['data_dirPath'], data_files['qa_act_sino_file'])
    paths['qa_act_image_path'] = join_if_present(paths['data_dirPath'], data_files['qa_act_image_file'])
    paths['qa_act_recon1_path'] = join_if_present(paths['data_dirPath'], data_files['qa_act_recon1_file'])
    paths['qa_act_recon2_path'] = join_if_present(paths['data_dirPath'], data_files['qa_act_recon2_file'])
    paths['qa_backMask_path'] = join_if_present(paths['data_dirPath'], data_files['qa_backMask_file'])
    paths['qa_hotMask_path'] = join_if_present(paths['data_dirPath'], data_files['qa_hotMask_file'])
    paths['qa_hotBackgroundMask_path'] = join_if_present(paths['data_dirPath'], data_files['qa_hotBackgroundMask_file'])
    paths['qa_coldMask_path'] = join_if_present(paths['data_dirPath'], data_files['qa_coldMask_file'])
    paths['qa_coldBackgroundMask_path'] = join_if_present(paths['data_dirPath'], data_files['qa_coldBackgroundMask_file'])
    paths['qa_atten_image_path'] = join_if_present(paths['data_dirPath'], data_files['qa_atten_image_file'])
    paths['qa_atten_sino_path'] = join_if_present(paths['data_dirPath'], data_files['qa_atten_sino_file'])
    paths['train_act_sino_path'] = join_if_present(paths['data_dirPath'], data_files['train_act_sino_file'])
    paths['train_act_image_path'] = join_if_present(paths['data_dirPath'], data_files['train_act_image_file'])
    paths['train_act_recon1_path'] = join_if_present(paths['data_dirPath'], data_files['train_act_recon1_file'])
    paths['train_act_recon2_path'] = join_if_present(paths['data_dirPath'], data_files['train_act_recon2_file'])
    paths['train_atten_image_path'] = join_if_present(paths['data_dirPath'], data_files['train_atten_image_file'])
    paths['train_atten_sino_path'] = join_if_present(paths['data_dirPath'], data_files['train_atten_sino_file'])
    paths['train_val_act_sino_path'] = join_if_present(paths['data_dirPath'], data_files['train_val_act_sino_file'])
    paths['train_val_act_image_path'] = join_if_present(paths['data_dirPath'], data_files['train_val_act_image_file'])
    paths['train_val_act_recon1_path'] = join_if_present(paths['data_dirPath'], data_files.get('train_val_act_recon1_file'))
    paths['train_val_act_recon2_path'] = join_if_present(paths['data_dirPath'], data_files.get('train_val_act_recon2_file'))
    paths['train_val_atten_image_path'] = join_if_present(paths['data_dirPath'], data_files['train_val_atten_image_file'])
    paths['train_val_atten_sino_path'] = join_if_present(paths['data_dirPath'], data_files['train_val_atten_sino_file'])
    paths['test_act_sino_path'] = join_if_present(paths['data_dirPath'], data_files['test_act_sino_file'])
    paths['test_act_image_path'] = join_if_present(paths['data_dirPath'], data_files['test_act_image_file'])
    paths['test_act_recon1_path'] = join_if_present(paths['data_dirPath'], data_files['test_act_recon1_file'])
    paths['test_act_recon2_path'] = join_if_present(paths['data_dirPath'], data_files['test_act_recon2_file'])
    paths['test_atten_image_path'] = join_if_present(paths['data_dirPath'], data_files['test_atten_image_file'])
    paths['test_atten_sino_path'] = join_if_present(paths['data_dirPath'], data_files['test_atten_sino_file'])
    paths['visualize_act_sino_path'] = join_if_present(paths['data_dirPath'], data_files['visualize_act_sino_file'])
    paths['visualize_act_image_path'] = join_if_present(paths['data_dirPath'], data_files['visualize_act_image_file'])
    paths['visualize_act_recon1_path'] = join_if_present(paths['data_dirPath'], data_files['visualize_act_recon1_file'])
    paths['visualize_act_recon2_path'] = join_if_present(paths['data_dirPath'], data_files['visualize_act_recon2_file'])
    paths['visualize_atten_image_path'] = join_if_present(paths['data_dirPath'], data_files['visualize_atten_image_file'])
    paths['visualize_atten_sino_path'] = join_if_present(paths['data_dirPath'], data_files['visualize_atten_sino_file'])
    
    # Backward-compatible note: attenuation image/sinogram paths already assigned above for each mode
    
    # Evaluation path aliases: unified naming for holdout and QA splits across tune/train modes
    # eval_holdout_* maps to tune_val_* (tune mode) or train_val_* (train mode)
    if run_mode == 'tune':
        paths['eval_holdout_act_sino_path'] = paths['tune_val_act_sino_path']
        paths['eval_holdout_act_image_path'] = paths['tune_val_act_image_path']
        paths['eval_holdout_act_recon1_path'] = paths['tune_val_act_recon1_path']
        paths['eval_holdout_act_recon2_path'] = paths['tune_val_act_recon2_path']
        paths['eval_holdout_atten_sino_path'] = paths['tune_val_atten_sino_path']
        paths['eval_holdout_atten_image_path'] = paths['tune_val_atten_image_path']
    elif run_mode == 'train':
        paths['eval_holdout_act_sino_path'] = paths['train_val_act_sino_path']
        paths['eval_holdout_act_image_path'] = paths['train_val_act_image_path']
        paths['eval_holdout_act_recon1_path'] = paths['train_val_act_recon1_path']
        paths['eval_holdout_act_recon2_path'] = paths['train_val_act_recon2_path']
        paths['eval_holdout_atten_sino_path'] = paths['train_val_atten_sino_path']
        paths['eval_holdout_atten_image_path'] = paths['train_val_atten_image_path']
    
    # eval_qa_* aliases for unified QA split naming (consistent across tune/train modes)
    paths['eval_qa_act_sino_path'] = paths['qa_act_sino_path']
    paths['eval_qa_act_image_path'] = paths['qa_act_image_path']
    paths['eval_qa_act_recon1_path'] = paths['qa_act_recon1_path']
    paths['eval_qa_act_recon2_path'] = paths['qa_act_recon2_path']
    paths['eval_qa_atten_sino_path'] = paths['qa_atten_sino_path']
    paths['eval_qa_atten_image_path'] = paths['qa_atten_image_path']
    paths['eval_qa_hotMask_path'] = paths['qa_hotMask_path']
    paths['eval_qa_hotBackgroundMask_path'] = paths['qa_hotBackgroundMask_path']
    paths['eval_qa_coldMask_path'] = paths['qa_coldMask_path']
    paths['eval_qa_coldBackgroundMask_path'] = paths['qa_coldBackgroundMask_path']
    paths['eval_qa_backMask_path'] = paths['qa_backMask_path']
    
    # Active paths and checkpoint filename selection (act_* only; no legacy fallbacks)
    if run_mode == 'tune':
        paths['act_sino_path'] = paths['tune_act_sino_path']
        paths['act_image_path'] = paths['tune_act_image_path']
        paths['act_recon1_path'] = paths['tune_act_recon1_path']
        paths['act_recon2_path'] = paths['tune_act_recon2_path']
        paths['atten_image_path'] = paths['tune_atten_image_path']
        paths['atten_sino_path'] = paths['tune_atten_sino_path']
        checkpoint_file = ''
    elif run_mode == 'train':
        paths['act_sino_path'] = paths['train_act_sino_path']
        paths['act_image_path'] = paths['train_act_image_path']
        paths['act_recon1_path'] = paths['train_act_recon1_path']
        paths['act_recon2_path'] = paths['train_act_recon2_path']
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
    if run_mode == 'train':
        paths['train_dataframe_path'] = os.path.join(paths['train_dataframe_dirPath'], f"{mode_files['train_csv_file']}.csv")
    else:
        # Explicitly set to None for non-train modes to prevent accidental logging to training CSV
        paths['train_dataframe_path'] = None
    paths['test_dataframe_path'] = os.path.join(paths['test_dataframe_dirPath'], f"{mode_files['test_csv_file']}.csv")
    
    return paths

def setup_settings( run_mode, common_settings, qa_opts, tune_opts, train_opts, test_opts, viz_opts):
    """
    Build all non-path runtime settings.
    
    Args:
        common_settings: dict with keys: device, num_examples
        qa_opts: dict with keys: qa_load_mode, qa_slice_range, qa_hot_weight (QA phantom settings, shared across modes)
        tune_opts: dict with keys: tune_augment, tune_batches_per_report, tune_examples_per_report,
                   tune_even_reporting, tune_max_t, tune_dataframe_fraction
        train_opts: dict with keys: train_augment, training_epochs, train_load_state, train_save_state,
                    train_show_times, train_sample_division, train_display_step,
                    train_lr_schedule_type, train_lr_min_factor
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
    settings['frozen_features_drop_max_prob'] = common_settings.get('frozen_features_drop_max_prob')
    settings['frozen_features_drop_min_prob'] = common_settings.get('frozen_features_drop_min_prob')
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
        settings['tune_debug'] = tune_opts['tune_debug']
        settings['evaluate_on'] = tune_opts['tune_report_for']
        settings['tune_eval_batch_size'] = tune_opts['tune_eval_batch_size']
        settings['eval_batch_size'] = tune_opts['tune_eval_batch_size']
        settings['tune_metric'] = tune_opts['tune_metric']
        
        # QA phantom settings (can be used for both tune and train modes)
        settings['qa_load_mode'] = qa_opts.get('qa_load_mode', 'random')
        settings['qa_slice_range'] = qa_opts.get('qa_slice_range')
        settings['qa_hot_weight'] = qa_opts.get('qa_hot_weight', 0.5)

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
        settings['train_eval_batch_size'] = train_opts['train_eval_batch_size']
        settings['eval_batch_size'] = train_opts['train_eval_batch_size']
        settings['train_lr_schedule_type'] = train_opts['train_lr_schedule_type']
        settings['train_lr_min_factor'] = train_opts['train_lr_min_factor']
        settings['train_save_on'] = train_opts['train_save_on']
        
        # Validate train_save_on setting
        if settings['train_save_on'] not in ['always', 'SSIM', 'MSE', 'CUSTOM']:
            raise ValueError(f"Invalid train_save_on='{settings['train_save_on']}'. Must be: 'always', 'SSIM', 'MSE', or 'CUSTOM'")
        
        # QA phantom settings (shared with tune mode via qa_opts)
        settings['qa_load_mode'] = qa_opts.get('qa_load_mode', 'random')
        settings['qa_slice_range'] = qa_opts.get('qa_slice_range')
        settings['qa_hot_weight'] = qa_opts.get('qa_hot_weight', 0.5)

    elif run_mode == 'test':
        settings['augment'] = (None, False) # If testing, do not augment
        settings['shuffle'] = False
        settings['num_epochs'] = 1
        settings['load_state'] = True
        settings['save_state'] = False
        settings['offset'] = 0
        settings['test_frozen_drop'] = bool(test_opts.get('test_frozen_drop'))
        settings['show_times'] = test_opts['test_show_times']
        settings['sample_division'] = test_opts['test_sample_division']
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
        settings['test_frozen_drop'] = False
        settings['sample_division'] = 1
        #settings['visualize_batch_size'] = viz_opts['visualize_batch_size']
    else:
        raise ValueError(f"Unknown run_mode: {run_mode}")
    
    # ===== VALIDATION: Check tune_metric vs evaluate_on compatibility (tune mode only) =====
    if run_mode == 'tune':
        # Both evaluate_on and tune_metric must be explicitly set (no defaults)
        if 'evaluate_on' not in settings:
            raise ValueError(
                f"evaluate_on must be explicitly set for tune mode. "
                f"Valid values: 'val' (for standard metrics) or 'qa' (for QA metrics)."
            )
        if 'tune_metric' not in settings:
            raise ValueError(
                f"tune_metric must be explicitly set for tune mode."
            )
        
        evaluate_on = settings['evaluate_on']
        tune_metric = settings['tune_metric']
        qa_metrics = ('qa-simple', 'qa-nema')
        
        # Validate: evaluate_on must be 'val' or 'qa'
        if evaluate_on not in ('val', 'qa'):
            raise ValueError(
                f"evaluate_on must be either 'val' or 'qa', got '{evaluate_on}'."
            )
        
        # Validate: QA metrics must use evaluate_on='qa'
        if tune_metric in qa_metrics and evaluate_on != 'qa':
            raise ValueError(
                f"tune_metric='{tune_metric}' requires evaluate_on='qa', "
                f"but got evaluate_on='{evaluate_on}'."
            )
        
        # Validate: evaluate_on='qa' requires QA metric
        if evaluate_on == 'qa' and tune_metric not in qa_metrics:
            raise ValueError(
                f"evaluate_on='qa' requires tune_metric in {qa_metrics}, "
                f"but got tune_metric='{tune_metric}'."
            )
    
    return settings