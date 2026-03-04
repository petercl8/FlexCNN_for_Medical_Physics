import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curves(
    csv_dir_path,
    csv_file_name,
    metric_name,
    epoch_range=None,
    metric_range=None,
    train_label=None,
    holdout_label=None,
    qa_label=None,
    ax=None,
    title=None,
    fontsize=12,
    ticksize=10,
):
    """
    Plot training, holdout, and QA learning curves from a training CSV dataframe.

    Parameters
    ----------
    csv_dir_path : str
        Directory containing the training dataframe CSV file.
    csv_file_name : str
        CSV file name. If extension is omitted, '.csv' is appended.
    metric_name : str
        Metric column to plot (e.g., 'MSE', 'SSIM', 'CUSTOM').
    epoch_range : tuple(int, int) or None
        Optional x-axis limits as (min_epoch, max_epoch).
    metric_range : tuple(float, float) or None
        Optional y-axis limits as (min_metric, max_metric).
    train_label : str or None
        Optional legend label for training split. Defaults to 'training set'.
    holdout_label : str or None
        Optional legend label for holdout split. Defaults to 'holdout set'.
    qa_label : str or None
        Optional legend label for QA split. Defaults to 'QA set'.
    ax : matplotlib.axes.Axes or None
        Target axes to draw into. If None, creates a new figure and axes.
    title : str or None
        Optional plot title. If None, a default title is used.
    fontsize : int
        Axis label and title font size.
    ticksize : int
        Tick label font size.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes containing the plot.
    """
    file_name = csv_file_name if csv_file_name.lower().endswith('.csv') else f"{csv_file_name}.csv"
    csv_path = os.path.join(csv_dir_path, file_name)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training dataframe CSV not found: '{csv_path}'")

    dataframe = pd.read_csv(csv_path)

    required_columns = ['epoch', 'eval_split', metric_name]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            f"CSV is missing required column(s): {missing_columns}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    train_split_name = 'training set'
    holdout_split_name = 'holdout set'
    qa_split_name = 'QA set'

    train_df = dataframe[dataframe['eval_split'] == train_split_name].copy()
    holdout_df = dataframe[dataframe['eval_split'] == holdout_split_name].copy()
    qa_df = dataframe[dataframe['eval_split'] == qa_split_name].copy()

    if epoch_range is not None:
        train_df = train_df[(train_df['epoch'] >= epoch_range[0]) & (train_df['epoch'] <= epoch_range[1])]
        holdout_df = holdout_df[(holdout_df['epoch'] >= epoch_range[0]) & (holdout_df['epoch'] <= epoch_range[1])]
        qa_df = qa_df[(qa_df['epoch'] >= epoch_range[0]) & (qa_df['epoch'] <= epoch_range[1])]

    if metric_range is not None:
        train_df = train_df[(train_df[metric_name] >= metric_range[0]) & (train_df[metric_name] <= metric_range[1])]
        holdout_df = holdout_df[(holdout_df[metric_name] >= metric_range[0]) & (holdout_df[metric_name] <= metric_range[1])]
        qa_df = qa_df[(qa_df[metric_name] >= metric_range[0]) & (qa_df[metric_name] <= metric_range[1])]

    train_df = train_df.sort_values(by='epoch')
    holdout_df = holdout_df.sort_values(by='epoch')
    qa_df = qa_df.sort_values(by='epoch')

    train_curve_label = train_label if train_label is not None else train_split_name
    holdout_curve_label = holdout_label if holdout_label is not None else holdout_split_name
    qa_curve_label = qa_label if qa_label is not None else qa_split_name

    if not train_df.empty:
        train_df.plot(
            x='epoch',
            y=metric_name,
            ax=ax,
            label=train_curve_label,
            legend=True,
            fontsize=ticksize,
        )

    if not holdout_df.empty:
        holdout_df.plot(
            x='epoch',
            y=metric_name,
            ax=ax,
            label=holdout_curve_label,
            legend=True,
            fontsize=ticksize,
        )

    if not qa_df.empty:
        qa_df.plot(
            x='epoch',
            y=metric_name,
            ax=ax,
            label=qa_curve_label,
            legend=True,
            fontsize=ticksize,
        )

    if epoch_range is not None:
        ax.set_xlim(epoch_range)
    if metric_range is not None:
        ax.set_ylim(metric_range)

    default_title = f"Learning Curves: {metric_name}"
    ax.set_title(title if title is not None else default_title, fontsize=fontsize)
    ax.set_xlabel('Epoch', fontsize=fontsize)
    ax.set_ylabel(metric_name, fontsize=fontsize)

    return fig, ax
