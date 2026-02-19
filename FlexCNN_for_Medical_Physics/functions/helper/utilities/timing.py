import time

def display_times(label_string, init_time, show_times):
    '''
    Function to display the time it takes to perform individual steps in the code. This can be helpful when trying to streamline things. Also returns the current time, which can be used to reset the timer in the code that calls the function.

    init_time:      initiation time when the process started
    label_string:   string to label the displayed time
    show_times:     show times or not
    '''
    current_time = time.time()

    if show_times == True:
        print(f'{label_string} (ms): {(current_time-init_time)*1000}')

    return current_time