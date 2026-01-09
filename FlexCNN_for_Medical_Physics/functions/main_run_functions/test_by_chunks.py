import os
import pandas as pd

from FlexCNN_for_Medical_Physics.functions.main_run_functions.trainable import run_trainable

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
    sample_division:    To test every example, set to 1. To test every other example, set to 2, and so forth.
    part_name:          Roots of dataframe parts files (containing testing results) that will be saved. These will have a number appended to them when saved.
    test_merge_dataframes:  Set to True to merge the smaller parts dataframes into a larger dataframe once the smaller parts have finished calculating.
                            Otherwise, you can use the MergeTests function below at a later time.
    '''

    label_num=int(test_begin_at/test_chunk_size) # Which numbered dataframe parts file you start at.

    test_dataframe_dirPath = paths['test_dataframe_dirPath']
    os.makedirs(test_dataframe_dirPath, exist_ok=True)

    for index in range(test_begin_at, testset_size, test_chunk_size):

        save_filename = part_name+str(label_num)+'.csv'

        print('###############################################')
        print(f'################# Working on:', save_filename)
        print(f'################# Starting at example: ', index)
        print('###############################################')

        # Since run_mode=='test', the training function returns a test dataframe. #
        chunk_settings = dict(settings)
        chunk_settings.update({
            'run_mode': 'test',
            'offset': index,
            'num_examples': test_chunk_size,
            'sample_division': sample_division,
        })

        chunk_dataframe = run_trainable(config, paths, chunk_settings)
        chunk_dataframe_path = os.path.join(test_dataframe_dirPath, save_filename)
        chunk_dataframe.to_csv(chunk_dataframe_path, index=False)
        label_num += 1

    if test_merge_dataframes==True:
        max_index = int(testset_size/test_chunk_size)-1
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
    first = True
    for name in names:
        add_path = os.path.join(test_dataframe_dirPath, name)
        print('Concatenating: ', add_path)
        add_frame = pd.read_csv(add_path)

        if first==True:
            test_dataframe = add_frame
            first=False
        else:
            test_dataframe = pd.concat([test_dataframe, add_frame], axis=0)

    ## Save Result ##
    test_dataframe.to_csv(test_dataframe_path, index=False)
    test_dataframe.describe()

#merge_test_chunks(34)