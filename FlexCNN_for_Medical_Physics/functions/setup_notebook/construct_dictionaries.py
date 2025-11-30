import os

def construct_config(
    run_mode,
    network_opts,
    tune_opts,
    test_opts,
    viz_opts,
    config_SUP_SI=None,
    config_SUP_IS=None,
    config_GAN_SI=None,
    config_GAN_IS=None,
    config_CYCLEGAN=None,
    config_CYCLESUP=None,
    config_RAY_SI=None,
    config_RAY_IS=None,
    config_RAY_SUP=None,
    config_RAY_GAN=None,
    config_SUP_RAY_cycle=None,
    config_GAN_RAY_cycle=None):

    """
    Combines configuration dictionaries based on run_mode, network_type, and train_SI.
    
    Args:
        run_mode: 'tune', 'train', 'test', or 'visualize'
        network_opts: dict with keys: network_type, train_SI, image_size, sino_size, image_channels, sino_channels
        tune_opts: dict with keys: tune_debug, etc.
        test_opts: dict with keys: test_batch_size, etc.
        viz_opts: dict with keys: visualize_batch_size, etc.
        config_SUP_SI, config_SUP_IS, etc.: Network configuration dictionaries
    
    Returns:
        config dict with all network hyperparameters and data dimensions
    """
    
    # Extract network options
    network_type = str(network_opts['network_type']).upper()  # Normalize strings
    train_SI = network_opts['train_SI']
    image_size = network_opts['image_size']
    sino_size = network_opts['sino_size']
    image_channels = network_opts['image_channels']
    sino_channels = network_opts['sino_channels']

    # Combine dictionaries based on run_mode and network_type
    if run_mode in ['train', 'test', 'visualize', 'none'] or tune_opts.get('tune_force_fixed_config')==True:
        if network_type == 'SUP':
            config = config_SUP_SI if train_SI else config_SUP_IS
        elif network_type == 'GAN':
            config = config_GAN_SI if train_SI else config_GAN_IS
        elif network_type == 'CYCLEGAN':
            config = config_CYCLEGAN
        elif network_type == 'CYCLESUP':
            config = config_CYCLESUP
        else:
            raise ValueError(f"Unknown network_type '{network_type}'.")

    elif run_mode == 'tune':
        if network_type == 'SUP':
            config = {**(config_RAY_SI if train_SI else config_RAY_IS), **config_RAY_SUP}
        elif network_type == 'GAN':
            config = {**(config_RAY_SI if train_SI else config_RAY_IS), **config_RAY_GAN}
        elif network_type == 'CYCLESUP':
            config = {**config_SUP_SI, **config_SUP_IS, **config_SUP_RAY_cycle}
        elif network_type == 'CYCLEGAN':
            config = {**config_GAN_SI, **config_GAN_IS, **config_GAN_RAY_cycle}
        else:
            raise ValueError(f"Unknown network_type '{network_type}'.")
        
        # Add data dimensions to config. These are set by the user and not tuned.
        config['network_type'] = network_type # If config is being built from smaller configs (CYCLESUP, CYCLEGAN), then this overwrites any existing value.
        config['train_SI'] = train_SI # Only used for SUP and GAN networks but added here for consistency.
        config['image_size'] = image_size
        config['sino_size'] = sino_size
        config['image_channels'] = image_channels
        config['sino_channels'] = sino_channels
    else:
        raise ValueError(f"Unknown run_mode '{run_mode}'.")

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
                   tune_dataframe_dirName, test_dataframe_dirName, data_dirName
        data_files: dict with keys: tune_sino_file, tune_image_file, train_sino_file, train_image_file,
                    test_sino_file, test_image_file
        mode_files: dict with keys: train_checkpoint_file, test_checkpoint_file, visualize_checkpoint_file,
                    tune_csv_file, test_csv_file
        run_mode: 'tune', 'train', 'test', or 'visualize'
        test_set_type: 'test' or 'train'
        visualize_type: 'test' or 'train'
    
    Returns:
        paths dict with directory paths, mode-specific data paths, active sino/image paths, checkpoint_path,
        tune/test dataframe paths.
    """
    test_set_type = test_ops.get('testset_type', 'test')
    visualize_type = viz_ops.get('visualize_type', 'test')

    paths = {}
    
    # Base directories
    paths['plot_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['plot_dirName'])
    paths['checkpoint_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['checkpoint_dirName'])
    paths['tune_storage_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['tune_storage_dirName'])
    paths['tune_dataframe_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['tune_dataframe_dirName'])
    paths['test_dataframe_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['test_dataframe_dirName'])
    paths['data_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['data_dirName'])
    
    # Mode-specific data file paths
    paths['tune_sino_path'] = os.path.join(paths['data_dirPath'], data_files['tune_sino_file'])
    paths['tune_image_path'] = os.path.join(paths['data_dirPath'], data_files['tune_image_file'])
    paths['train_sino_path'] = os.path.join(paths['data_dirPath'], data_files['train_sino_file'])
    paths['train_image_path'] = os.path.join(paths['data_dirPath'], data_files['train_image_file'])
    paths['test_sino_path'] = os.path.join(paths['data_dirPath'], data_files['test_sino_file'])
    paths['test_image_path'] = os.path.join(paths['data_dirPath'], data_files['test_image_file'])
    
    # Active paths and checkpoint filename selection
    if run_mode == 'tune':
        paths['sino_path'] = paths['tune_sino_path']
        paths['image_path'] = paths['tune_image_path']
        checkpoint_file = ''
    elif run_mode == 'train':
        paths['sino_path'] = paths['train_sino_path']
        paths['image_path'] = paths['train_image_path']
        checkpoint_file = mode_files['train_checkpoint_file']
    elif run_mode == 'test':
        if test_set_type == 'test':
            paths['sino_path'] = paths['test_sino_path']
            paths['image_path'] = paths['test_image_path']
        elif test_set_type == 'train':
            paths['sino_path'] = paths['train_sino_path']
            paths['image_path'] = paths['train_image_path']
        else:
            raise ValueError(f"Test_set_type not 'test' or 'train': {test_set_type}")
        checkpoint_file = mode_files['test_checkpoint_file']
    elif run_mode in ['visualize', 'none']:
        if visualize_type == 'test':
            paths['sino_path'] = paths['test_sino_path']
            paths['image_path'] = paths['test_image_path']
        elif visualize_type == 'train':
            paths['sino_path'] = paths['train_sino_path']
            paths['image_path'] = paths['train_image_path']
        else:
            raise ValueError(f"Visualize_type not 'test' or 'train': {visualize_type}")
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
                   testset_size, test_begin_at, test_compute_MLEM, testset_type, test_merge_dataframes,
                   test_shuffle, test_sample_division
        viz_opts: dict with keys: visualize_shuffle, visualize_offset, visualize_batch_size, visualize_type
        run_mode: 'tune', 'train', 'test', or 'visualize'
    
    Returns:
        settings dict containing runtime (non-path) configuration.
    """
    settings = {}
    
    # Common settings (now minimal)
    settings['run_mode'] = run_mode
    settings['device'] = common_settings['device']
    settings['num_examples'] = common_settings.get('num_examples', -1)
    
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
        
        settings['tune_exp_name']= tune_opts['tune_exp_name']
        settings['tune_scheduler'] = tune_opts['tune_scheduler']
        settings['tune_dataframe_fraction'] = tune_opts.get('tune_dataframe_fraction', 1.0)
        settings['tune_restore'] = tune_opts.get('tune_restore', False)
        settings['tune_max_t'] = tune_opts.get('tune_max_t', 100)
        settings['tune_minutes'] = tune_opts.get('tune_minutes', 180)
        settings['tune_for'] = tune_opts['tune_for']
        settings['tune_even_reporting'] = tune_opts.get('tune_even_reporting', False)
        settings['tune_batches_per_report'] = tune_opts.get('tune_batches_per_report')
        settings['tune_examples_per_report'] = tune_opts.get('tune_examples_per_report')
        settings['tune_augment'] = tune_opts['tune_augment']
        settings['tune_debug'] = tune_opts.get('tune_debug', False)

    elif run_mode == 'train':
        settings['augment'] = train_opts['train_augment']
        settings['shuffle'] = True
        settings['num_epochs'] = train_opts['training_epochs']
        settings['load_state'] = train_opts['train_load_state']
        settings['save_state'] = train_opts['train_save_state']
        settings['offset'] = 0
        settings['show_times'] = train_opts['train_show_times']
        settings['sample_division'] = train_opts['train_sample_division']
        settings['train_display_step'] = train_opts['train_display_step'] # Used in compute_display_step()
    
    elif run_mode == 'test':
        settings['augment'] = False
        settings['shuffle'] = False
        settings['num_epochs'] = 1
        settings['load_state'] = True
        settings['save_state'] = False
        settings['offset'] = 0
        settings['show_times'] = test_opts['test_show_times']
        settings['sample_division'] = test_opts.get('test_sample_division', 1)
        settings['test_display_step'] = test_opts['test_display_step'] # Used in compute_display_step()
        #settings['test_batch_size'] = test_opts['test_batch_size']
    
    elif run_mode in ['visualize', 'none']:
        settings['augment'] = False
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