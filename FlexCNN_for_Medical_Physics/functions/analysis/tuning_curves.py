import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ray import tune
from FlexCNN_for_Medical_Physics.functions.main_run_functions.trainable_standard import run_trainable
import os


def PlotFrame(paths, tune_exp_name, ax, x_ticks, x_label, y_ticks, y_label, xlim=None, ylim=None, logy=False, max_plot_num=-1, fontsize=12, ticksize=10):
    '''
    This function plots the dataframes for each tuning (experiment).

    tune_exp_name:      name of the tuning experiment
    ax:                 Matplotlib axis object to plot the dataframes
    xlim:               lower limit for the x-axis. Set to None to set no limit.
    ylim:               lower limit for the y-axis. Set to None to set no limit.
    logy:               use a logarithmic scale for the y-axis?
    max_plot_num        maximum number of dataframes to plot. Set to -1 to plot all dataframes.
    fontsize:           font size for axis labels
    ticksize:           font size for tick labels   
    '''

    experiment_path = os.path.join(paths['tune_storage_dirPath'], tune_exp_name)

    restored_tuner = tune.Tuner.restore(experiment_path,
                                        trainable = tune.with_resources(run_trainable, {"CPU":4,"GPU":1}))
    result_grid = restored_tuner.get_results()

    for i, result in enumerate(result_grid):
        #print(i)
        #label = f"lr={result.config['lr']:.3f}, momentum={result.config['momentum']}"
        try: # Keeps plotting even if there is an error with one of the plots
            result.metrics_dataframe.plot(x=x_ticks, y=y_ticks, ax=ax, label='test', legend=False, xlim=xlim, ylim=ylim,
                                          logy=logy, fontsize=ticksize)
        except:
            print('Error Plotting')
        if i==max_plot_num:
            break
    ax.set_ylabel(y_label, fontsize=fontsize) # 'fontsize' is a variable set outside of the function (see below)
    ax.set_xlabel(x_label, fontsize=fontsize)

    bestResult_logDir = result_grid.get_best_result("SSIM", mode="max")

    return result_grid, bestResult_logDir