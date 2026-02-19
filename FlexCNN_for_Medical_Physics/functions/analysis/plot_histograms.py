import pandas as pd
import matplotlib.pyplot as plt


def plot_hist_1D(ax, dataframe, title, x_label, y_label, column_1, column_2, xlim, ylim, 
                 bins=400, alpha=0.5, titlesize=13, fontsize=12,
                 column_1_label=None, column_2_label=None
):
    """
    Plot overlaid histograms for two dataframe columns within specified limits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw into.
    dataframe : pd.DataFrame
        Source data.
    title, x_label, y_label : str
        Text for plot labelling.
    column_1, column_2 : str
        Column names to plot.
    xlim, ylim : tuple(float, float)
        Axis limits for x and y.
    bins : int
        Number of histogram bins (default 400).
    alpha : float
        Bar transparency for overlaid histograms.
    titlesize : int
        Title font size (default 13).
    fontsize : int
        Axis label and tick font size (default 12).
    column_1_label, column_2_label : str or None
        Custom legend labels for the columns. If None, uses column names directly.

    Notes
    -----
    - Rows outside xlim are filtered out for both columns before plotting.
    - Mutates the provided `ax` with histogram artists.
    - Assumes columns contain numeric data.
    """
    df = dataframe.copy()
    df = df[df[column_1] > xlim[0]]
    df = df[df[column_1] < xlim[1]]
    df = df[df[column_2] > xlim[0]]
    df = df[df[column_2] < xlim[1]]

    # Create a dataframe with custom labels if provided
    if column_1_label is not None or column_2_label is not None:
        plot_df = df[[column_1, column_2]].copy()
        if column_1_label is not None:
            plot_df.rename(columns={column_1: column_1_label}, inplace=True)
        if column_2_label is not None:
            plot_df.rename(columns={column_2: column_2_label}, inplace=True)
    else:
        plot_df = df[[column_1, column_2]]

    plot_df.plot.hist(
        xlim=xlim,
        ylim=ylim,
        bins=bins,
        alpha=alpha,
        ax=ax,
        fontsize=fontsize
    )
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)


def plot_hist_2D(ax, dataframe, title, x_label, y_label,
    x_column, y_column,
    xlim=(0, 1), ylim=(0, 1), gridsize=None,
    titlesize=13, fontsize=12, ticksize=10,
    x_column_label=None, y_column_label=None
):
    """
    Plot a hexbin (2D histogram) for two dataframe columns and draw diagonal line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw into.
    dataframe : pd.DataFrame
        Source data.
    title, x_label, y_label : str
        Plot labelling.
    x_column, y_column : str
        Column names for the x and y axes.
    xlim, ylim : tuple(float, float)
        Axis limits (defaults (0,1)).
    gridsize : int or None
        Hexbin resolution; None lets pandas choose defaults.
    titlesize : int
        Title font size (default 13).
    fontsize : int
        Axis label font size (default 12).
    ticksize : int
        Tick label font size (default 10).
    x_column_label, y_column_label : str or None
        Custom legend labels for the columns. If None, uses column names directly.

    Notes
    -----
    - Filters rows outside provided limits before plotting.
    - Always draws a diagonal reference line (y = x mapped through limits).
    - Assumes numeric columns.
    """
    df = dataframe.copy()
    df = df[df[x_column] > xlim[0]]
    df = df[df[x_column] < xlim[1]]
    df = df[df[y_column] > ylim[0]]
    df = df[df[y_column] < ylim[1]]

    # Create a dataframe with custom labels if provided
    if x_column_label is not None or y_column_label is not None:
        plot_df = df[[x_column, y_column]].copy()
        if x_column_label is not None:
            plot_df.rename(columns={x_column: x_column_label}, inplace=True)
            x_col_for_plot = x_column_label
        else:
            x_col_for_plot = x_column
        if y_column_label is not None:
            plot_df.rename(columns={y_column: y_column_label}, inplace=True)
            y_col_for_plot = y_column_label
        else:
            y_col_for_plot = y_column
    else:
        plot_df = df[[x_column, y_column]]
        x_col_for_plot = x_column
        y_col_for_plot = y_column

    plot_df.plot.hexbin(
        ax=ax,
        x=x_col_for_plot,
        y=y_col_for_plot,
        xlim=xlim,
        ylim=ylim,
        gridsize=gridsize,
        fontsize=ticksize
    )
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.plot(xlim, ylim, linestyle='--', color='gray')