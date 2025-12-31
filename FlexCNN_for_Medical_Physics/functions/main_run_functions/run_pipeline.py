import time

from FlexCNN_for_Medical_Physics.functions.main_run_functions.run_supervisory import run_SUP
from FlexCNN_for_Medical_Physics.functions.main_run_functions.run_generative_adversarial import run_GAN
from FlexCNN_for_Medical_Physics.functions.main_run_functions.run_cycle_consistency import run_CYCLE
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

    if run_mode == 'tune':
        if network_type == "SUP":
            print('Tuning w/ Supervisory Only!')
            time.sleep(3)
            tune_networks(
                config, paths, settings, tune_opts, base_dirs, trainable='SUP'
            )

        elif network_type == 'GAN':
            print('Tuning a GAN!')
            time.sleep(3)
            tune_networks(
                config, paths, settings, tune_opts, base_dirs, trainable='GAN'
            )

        elif network_type in ('CYCLESUP', 'CYCLEGAN'):
            print('Tuning a Cycle!')
            time.sleep(3)
            tune_networks(
                config, paths, settings, tune_opts, base_dirs, trainable='CYCLE'
            )

    elif run_mode in ('train', 'visualize'):
        if network_type == "SUP":
            run_SUP(config, paths, settings)

        elif network_type == 'GAN':
            run_GAN(config, paths, settings)

        elif network_type in ('CYCLESUP', 'CYCLEGAN'):
            run_CYCLE(config, paths, settings)

    elif run_mode == 'test':
        test_by_chunks(
            test_begin_at=test_opts['test_begin_at'],
            test_chunk_size=test_opts['test_chunk_size'],
            testset_size=test_opts['testset_size'],
            sample_division=settings.get('sample_division', 1),
            part_name='batch_dataframe_part_',
            test_merge_dataframes=test_opts['test_merge_dataframes'],
            test_csv_file=test_opts['test_csv_file'],
        )
        
    elif run_mode == 'none':
        break
