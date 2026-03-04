import os
import numpy as np
import pandas as pd

from FlexCNN_for_Medical_Physics.functions.main_run_functions.trainable_standard import run_trainable
from FlexCNN_for_Medical_Physics.functions.main_run_functions.trainable_frozen_flow import run_trainable_frozen_flow

def _resolve_testset_size(paths):
    candidate_keys = (
        'act_image_path',
        'act_sino_path',
        'atten_image_path',
        'atten_sino_path',
        'act_recon1_path',
        'act_recon2_path',
    )

    for key in candidate_keys:
        array_path = paths.get(key)
        if array_path is not None and os.path.exists(array_path):
            return np.load(array_path, mmap_mode='r').shape[0]

    raise ValueError('Could not resolve test set size from test data paths.')

def test_by_chunks(
    config,
    paths,
    settings,
    test_begin_at=0,
    test_chunk_size=5000,
    testset_size=35000,
    sample_division=1,
    part_name='batch_dataframe_part_',
    test_merge_dataframes=False,
    test_csv_file='combined_dataframe'
):
    '''
    Splits up testing the CNN (on a test set) into smaller chunks so that computer time-outs don't result in lost work.

    test_begin_at:      Where to begin the testing. You set this to >0 if the test terminates early and you need to pick up partway through the test set.
    test_chunk_size:    How many examples to test in each chunk
    testset_size:       Number of examples that you wish to test. This can be less than the number of examples in the dataset file but not more.
                        Set to -1 or None to test the full available dataset.
    sample_division:    To test every example, set to 1. To test every other example, set to 2, and so forth.
    part_name:          Roots of dataframe parts files (containing testing results) that will be saved. These will have a number appended to them when saved.
    test_merge_dataframes:  Set to True to merge the smaller parts dataframes into a larger dataframe once the smaller parts have finished calculating.
                            Otherwise, you can use the MergeTests function below at a later time.
    '''

    test_dataframe_dirPath = paths['test_dataframe_dirPath']
    os.makedirs(test_dataframe_dirPath, exist_ok=True)

    full_test_requested = testset_size in (-1, None)
    if full_test_requested:
        testset_size = _resolve_testset_size(paths)

    run_all_at_once = test_chunk_size is None or test_chunk_size >= testset_size
    if run_all_at_once:
        print('###############################################')
        print('################# Working on full test set')
        print(f'################# Starting at example: ', test_begin_at)
        print('###############################################')

        num_examples = -1 if full_test_requested else max(testset_size - test_begin_at, 0)
        chunk_settings = dict(settings)
        chunk_settings.update({
            'run_mode': 'test',
            'offset': test_begin_at,
            'num_examples': num_examples,
            'sample_division': sample_division,
        })

        if config['network_type'] in ('FROZEN_COFLOW', 'FROZEN_COUNTERFLOW'):
            chunk_dataframe = run_trainable_frozen_flow(config, paths, chunk_settings)
        else:
            chunk_dataframe = run_trainable(config, paths, chunk_settings)

        test_dataframe_path = os.path.join(test_dataframe_dirPath, f"{test_csv_file}.csv")
        chunk_dataframe.to_csv(test_dataframe_path, index=False)
        return

    label_num = test_begin_at // test_chunk_size # Which numbered dataframe parts file you start at.

    for index in range(test_begin_at, testset_size, test_chunk_size):

        save_filename = part_name+str(label_num)+'.csv'

        print('###############################################')
        print(f'################# Working on:', save_filename)
        print(f'################# Starting at example: ', index)
        print('###############################################')

        # Since run_mode=='test', the training function returns a test dataframe. #
        chunk_settings = dict(settings)
        current_chunk_size = min(test_chunk_size, testset_size - index)
        chunk_settings.update({
            'run_mode': 'test',
            'offset': index,
            'num_examples': current_chunk_size,
            'sample_division': sample_division,
        })

        # Route to appropriate trainable function based on network type
        if config['network_type'] in ('FROZEN_COFLOW', 'FROZEN_COUNTERFLOW'):
            chunk_dataframe = run_trainable_frozen_flow(config, paths, chunk_settings)
        else:
            chunk_dataframe = run_trainable(config, paths, chunk_settings)
        chunk_dataframe_path = os.path.join(test_dataframe_dirPath, save_filename)
        chunk_dataframe.to_csv(chunk_dataframe_path, index=False)
        label_num += 1

    if test_merge_dataframes==True:
        max_index = label_num - 1
        merge_test_chunks(
            max_index,
            test_dataframe_dirPath=test_dataframe_dirPath,
            test_dataframe_path=os.path.join(test_dataframe_dirPath, f"{test_csv_file}.csv"),
            part_name=part_name,
            test_csv_file=test_csv_file
        )


def merge_test_chunks(max_index, test_dataframe_dirPath, test_dataframe_path, part_name='batch_dataframe_part_', test_csv_file='combined_dataframe'):
    '''
    Function for merging smaller dataframes (which contain metrics for individual images) into a single larger dataframe.

    max_index:      number of largest index
    part_name:      root of part filenames (not including the numbers appended to the end)
    test_csv_file:  filename for the combined dataframe
    '''

    ## Build list of filenames ##
    names = []
    for i in range(0, max_index+1):
        save_filename = part_name+str(i)+'.csv'
        names.append(save_filename)

    ## Concatenate parts dataframes ##
    dataframes = []
    for name in names:
        add_path = os.path.join(test_dataframe_dirPath, name)
        print('Concatenating: ', add_path)
        add_frame = pd.read_csv(add_path)
        dataframes.append(add_frame)

    test_dataframe = pd.concat(dataframes, axis=0, ignore_index=True)

    ## Save Result ##
    test_dataframe.to_csv(test_dataframe_path, index=False)