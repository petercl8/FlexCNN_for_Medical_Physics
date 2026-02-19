# -*- coding: utf-8 -*-
"""
Dictionary Builder for FlexCNN Medical Physics Pipeline

This module builds configuration dictionaries from user parameters.
It organizes parameters into logical groups and constructs network configs.

Usage:
    from build_dicts import build_all_dicts
    from user_params import get_params
    
    params = get_params()
    all_dicts = build_all_dicts(params)
    
    config = all_dicts['config']
    paths = all_dicts['paths']
    settings = all_dicts['settings']
"""

from FlexCNN_for_Medical_Physics.functions.setup_environment.construct_dictionaries import (
    setup_paths, setup_settings, construct_config
)
from FlexCNN_for_Medical_Physics.config_net_dicts.tuned_ACT_SI import config_ACT_SI
from FlexCNN_for_Medical_Physics.config_net_dicts.tuned_ACT_IS import config_ACT_IS
from FlexCNN_for_Medical_Physics.config_net_dicts.tuned_ATTEN_SI import config_ATTEN_SI
from FlexCNN_for_Medical_Physics.config_net_dicts.tuned_ATTEN_IS import config_ATTEN_IS
from FlexCNN_for_Medical_Physics.config_net_dicts.tuned_CONCAT import config_CONCAT
from FlexCNN_for_Medical_Physics.config_net_dicts.tuned_FROZEN_COFLOW import config_FROZEN_COFLOW
from FlexCNN_for_Medical_Physics.config_net_dicts.tuned_FROZEN_COUNTERFLOW import config_FROZEN_COUNTERFLOW
from FlexCNN_for_Medical_Physics.config_net_dicts.search_spaces import (
    config_RAY_SI, config_RAY_SI_learnScale, config_RAY_SI_fixedScale,
    config_RAY_IS, config_RAY_IS_learnScale, config_RAY_IS_fixedScale,
    config_RAY_SUP, config_RAY_SUP_FROZEN
)


def build_all_dicts(params):
    """
    Build all configuration dictionaries from user parameters.
    
    Args:
        params (dict): Dictionary of user parameters from user_params.get_params()
    
    Returns:
        dict: Dictionary containing:
            - 'common_settings': Common settings across all modes
            - 'base_dirs': Base directory paths
            - 'data_files': All data file names
            - 'mode_files': Mode-specific files (checkpoints, csvs)
            - 'network_opts': Network configuration options
            - 'tune_opts': Tuning options
            - 'train_opts': Training options
            - 'test_opts': Testing options
            - 'viz_opts': Visualization options
            - 'paths': Full paths dictionary from setup_paths()
            - 'settings': Settings dictionary from setup_settings()
            - 'config': Network config dictionary from construct_config()
    """
    
    # Build grouped parameter dictionaries
    common_settings = {
        'run_mode': params['run_mode'],
        'device': params['device_opt'],  # Note: this should be set to actual device before passing
        'num_examples': params['num_examples'],
        'act_recon1_scale': params['act_recon1_scale'],
        'act_recon2_scale': params['act_recon2_scale'],
        'act_sino_scale': params['act_sino_scale'],
        'act_image_scale': params['act_image_scale'],
        'atten_image_scale': params['atten_image_scale'],
        'atten_sino_scale': params['atten_sino_scale'],
    }
    
    base_dirs = {
        'project_dirPath': params.get('project_dirPath'),  # Set by caller
        'plot_dirName': params['plot_dirName'],
        'checkpoint_dirName': params['checkpoint_dirName'],
        'tune_storage_dirName': params['tune_storage_dirName'],
        'tune_dataframe_dirName': params['tune_dataframe_dirName'],
        'test_dataframe_dirName': params['test_dataframe_dirName'],
        'data_dirPath': params.get('data_dirPath'),  # Absolute path to data directory (or None)
        'data_dirName': params['data_dirName']  # Fallback directory name if data_dirPath is None
    }
    
    data_files = {
        'tune_act_sino_file': params['tune_act_sino_file'],
        'tune_act_image_file': params['tune_act_image_file'],
        'tune_act_recon1_file': params['tune_act_recon1_file'],
        'tune_act_recon2_file': params['tune_act_recon2_file'],
        'tune_atten_image_file': params['tune_atten_image_file'],
        'tune_atten_sino_file': params['tune_atten_sino_file'],
        'tune_val_act_sino_file': params['tune_val_act_sino_file'],
        'tune_val_act_image_file': params['tune_val_act_image_file'],
        'tune_val_atten_image_file': params['tune_val_atten_image_file'],
        'tune_val_atten_sino_file': params['tune_val_atten_sino_file'],
        'tune_qa_act_sino_file': params['tune_qa_act_sino_file'],
        'tune_qa_act_image_file': params['tune_qa_act_image_file'],
        'tune_qa_hotMask_file': params['tune_qa_hotMask_file'],
        'tune_qa_hotBackgroundMask_file': params['tune_qa_hotBackgroundMask_file'],
        'tune_qa_coldMask_file': params['tune_qa_coldMask_file'],
        'tune_qa_coldBackgroundMask_file': params['tune_qa_coldBackgroundMask_file'],
        'tune_qa_backMask_file': None,  # Added to fix KeyError
        'tune_qa_atten_image_file': params['tune_qa_atten_image_file'],
        'tune_qa_atten_sino_file': params['tune_qa_atten_sino_file'],
        'train_act_sino_file': params['train_act_sino_file'],
        'train_act_image_file': params['train_act_image_file'],
        'train_act_recon1_file': params['train_act_recon1_file'],
        'train_act_recon2_file': params['train_act_recon2_file'],
        'train_atten_image_file': params['train_atten_image_file'],
        'train_atten_sino_file': params['train_atten_sino_file'],
        'test_act_sino_file': params['test_act_sino_file'],
        'test_act_image_file': params['test_act_image_file'],
        'test_act_recon1_file': params['test_act_recon1_file'],
        'test_act_recon2_file': params['test_act_recon2_file'],
        'test_atten_image_file': params['test_atten_image_file'],
        'test_atten_sino_file': params['test_atten_sino_file'],
        'visualize_act_sino_file': params['visualize_act_sino_file'],
        'visualize_act_image_file': params['visualize_act_image_file'],
        'visualize_act_recon1_file': params['visualize_act_recon1_file'],
        'visualize_act_recon2_file': params['visualize_act_recon2_file'],
        'visualize_atten_image_file': params['visualize_atten_image_file'],
        'visualize_atten_sino_file': params['visualize_atten_sino_file'],
    }
    
    mode_files = {
        'tune_csv_file': params['tune_csv_file'],
        'train_checkpoint_file': params['train_checkpoint_file'],
        'test_checkpoint_file': params['test_checkpoint_file'],
        'test_csv_file': params['test_csv_file'],
        'visualize_checkpoint_file': params['visualize_checkpoint_file']
    }
    
    network_opts = {
        'network_type': params['network_type'],
        'train_SI': params['train_SI'],
        'gen_image_size': params['gen_image_size'],
        'gen_sino_size': params['gen_sino_size'],
        'gen_image_channels': params['gen_image_channels'],
        'gen_sino_channels': params['gen_sino_channels'],
        'SI_normalize': params['SI_normalize'],
        'IS_normalize': params['IS_normalize'],
    }
    
    tune_opts = {
        'tune_exp_name': params['tune_exp_name'],
        'tune_scheduler': params['tune_scheduler'],
        'tune_dataframe_fraction': params['tune_dataframe_fraction'],
        'tune_restore': params['tune_restore'],
        'tune_max_t': params['tune_max_t'],
        'tune_minutes': params['tune_minutes'],
        'tune_metric': params['tune_metric'],
        'tune_even_reporting': params['tune_even_reporting'],
        'tune_batches_per_report': params['tune_batches_per_report'],
        'tune_examples_per_report': params['tune_examples_per_report'],
        'tune_augment': params['tune_augment'],
        'tune_qa_load_mode': params['tune_qa_load_mode'],
        'tune_qa_slice_range': params['tune_qa_slice_range'],
        'tune_grace_period': params['tune_grace_period'],
        'tune_debug': params['tune_debug'],
        'tune_force_fixed_config': params['tune_force_fixed_config'],
        'tune_report_for': params['tune_report_for'],
        'tune_qa_hot_weight': params['tune_qa_hot_weight'],
        'tune_eval_batch_size': params['tune_eval_batch_size'],
        'num_CPUs': params['num_CPUs'],
        'num_GPUs': params['num_GPUs'],
        'cpus_per_trial': params['CPUs_per_trial'],
        'gpus_per_trial': params['GPUs_per_trial'],
        'tune_search_alg': params['tune_search_alg'],
    }
    
    train_opts = {
        'train_load_state': params['train_load_state'],
        'train_save_state': params['train_save_state'],
        'training_epochs': params['train_epochs'],
        'train_augment': params['train_augment'],
        'train_shuffle': params['train_shuffle'],
        'train_display_step': params['train_display_step'],
        'train_sample_division': params['train_sample_division'],
        'train_show_times': params['train_show_times'],
        'train_report_eval': params['train_report_eval'],
    }
    
    test_opts = {
        'test_display_step': params['test_display_step'],
        'test_batch_size': params['test_batch_size'],
        'test_chunk_size': params['test_chunk_size'],
        'testset_size': params['testset_size'],
        'test_begin_at': params['test_begin_at'],
        'test_compute_MLEM': params['test_compute_MLEM'],
        'test_merge_dataframes': params['test_merge_dataframes'],
        'test_show_times': params['test_show_times'],
        'test_shuffle': params['test_shuffle'],
        'test_sample_division': params['test_sample_division']
    }
    
    viz_opts = {
        'visualize_batch_size': params['visualize_batch_size'],
        'visualize_offset': params['visualize_offset'],
        'visualize_shuffle': params['visualize_shuffle'],
    }
    
    # Build paths and settings using package functions
    paths = setup_paths(
        run_mode=params['run_mode'],
        base_dirs=base_dirs,
        data_files=data_files,
        mode_files=mode_files,
        test_ops=test_opts,
        viz_ops=viz_opts
    )
    
    settings = setup_settings(
        run_mode=params['run_mode'],
        common_settings=common_settings,
        tune_opts=tune_opts,
        train_opts=train_opts,
        test_opts=test_opts,
        viz_opts=viz_opts,
    )
    
    # DEBUG: Log augmentation settings
    print(f"\n[AUGMENT DEBUG] run_mode={params['run_mode']}, network_type={params['network_type']}")
    print(f"[AUGMENT DEBUG] settings['augment']={settings['augment']}")
    print(f"[AUGMENT DEBUG] Expected augment from params: train_augment={params.get('train_augment', 'NOT SET')}")
    print()
    
    # Build config dictionary
    config = construct_config(
        run_mode=params['run_mode'],
        network_opts=network_opts,
        tune_opts=tune_opts,
        test_opts=test_opts,
        viz_opts=viz_opts,
        config_ACT_SI=config_ACT_SI,
        config_ACT_IS=config_ACT_IS,
        config_ATTEN_SI=config_ATTEN_SI,
        config_ATTEN_IS=config_ATTEN_IS,
        config_CONCAT=config_CONCAT,
        config_FROZEN_COFLOW=config_FROZEN_COFLOW,
        config_FROZEN_COUNTERFLOW=config_FROZEN_COUNTERFLOW,
        config_GAN_SI=None,
        config_GAN_IS=None,
        config_RAY_SI=config_RAY_SI,
        config_RAY_SI_learnScale=config_RAY_SI_learnScale,
        config_RAY_SI_fixedScale=config_RAY_SI_fixedScale,
        config_RAY_IS=config_RAY_IS,
        config_RAY_IS_learnScale=config_RAY_IS_learnScale,
        config_RAY_IS_fixedScale=config_RAY_IS_fixedScale,
        config_RAY_SUP=config_RAY_SUP,
        config_RAY_SUP_FROZEN=config_RAY_SUP_FROZEN,
        config_RAY_GAN=None,
        config_RAY_GAN_CYCLE=None,
    )
    
    return {
        'common_settings': common_settings,
        'base_dirs': base_dirs,
        'data_files': data_files,
        'mode_files': mode_files,
        'network_opts': network_opts,
        'tune_opts': tune_opts,
        'train_opts': train_opts,
        'test_opts': test_opts,
        'viz_opts': viz_opts,
        'paths': paths,
        'settings': settings,
        'config': config,
    }
