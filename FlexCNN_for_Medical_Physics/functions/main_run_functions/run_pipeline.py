import time

from FlexCNN_for_Medical_Physics.functions.main_run_functions.trainable import run_trainable
from FlexCNN_for_Medical_Physics.functions.main_run_functions.trainable_frozen_flow import run_trainable_frozen_flow
from FlexCNN_for_Medical_Physics.functions.main_run_functions.tune import tune_networks
from FlexCNN_for_Medical_Physics.functions.main_run_functions.test_by_chunks import test_by_chunks

def run_pipeline(
    config,
    paths,
    settings,
    tune_opts=None,
    base_dirs=None,
    test_opts=None,
):
    """
    Dispatch function for tuning, training, visualization, or testing.
    
    Args:
        config: Network configuration dict (contains 'network_type')
        paths: Path configuration dict
        settings: Runtime settings dict (contains 'run_mode')
        tune_opts: Tuning options dict
        base_dirs: Base directories dict
        test_opts: Test options dict (contains test_begin_at, test_chunk_size, testset_size, etc.)
    """
    # Extract from dictionaries
    run_mode = settings['run_mode']
    network_type = config['network_type']

    allowed_types = ('ACT', 'ATTEN', 'CONCAT', 'FROZEN_COFLOW', 'FROZEN_COUNTERFLOW', 'CYCLEGAN', 'GAN')
    if network_type not in allowed_types:
        raise ValueError(f"Unknown network_type '{network_type}'.")

    if run_mode == 'tune':
        print('Tuning with trainable pipeline.')
        time.sleep(1)
        tune_networks(
            config, paths, settings, tune_opts, base_dirs
        )

    elif run_mode in ('train', 'visualize'):
        # Route frozen flow network types to specialized function
        if network_type in ('FROZEN_COFLOW', 'FROZEN_COUNTERFLOW'):
            run_trainable_frozen_flow(config, paths, settings)
        else:
            run_trainable(config, paths, settings)

    elif run_mode == 'test':
        test_by_chunks(
            config,
            paths,
            settings,
            test_begin_at=test_opts['test_begin_at'],
            test_chunk_size=test_opts['test_chunk_size'],
            testset_size=test_opts['testset_size'],
            sample_division=settings.get('sample_division', 1),
            part_name='batch_dataframe_part_',
            test_merge_dataframes=test_opts['test_merge_dataframes'],
            test_csv_file=test_opts['test_csv_file'],
        )

    elif run_mode == 'none':
        raise SystemExit
    else:
        raise ValueError(f"Unknown run_mode '{run_mode}'.")
    
    return
