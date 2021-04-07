"""Saves a boxplot figure of estimated average treatment effects for a suite of experiments.
"""

import fnmatch
import os
import re


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def save_boxplot(figure_name:str,
                 load_dir:str,
                 save_dir:str=None,
                 xlabel:str=None,
                 file_name_contain:str=None):
    """Saves a boxplot of estimated average treatment effects for a suite of experiments.

    Each file is expected to contain ATE estimates in the first column with each row being a single iteration
    of an experiment.

    Parameters
    ----------
    figure_name : str
        Name of the file in which the figure is saved. For example, if you want to save the boxplot as
         'boxplot.png', then 'figure_name="boxplot"'.
    load_dir : str
        Directory to load results from.
    save_dir : str
        Directory to save the figure to. If None then the figure is saved to 'load_dir'.
    xlabel : str
        Name of the figure's x-axis.
    file_name_contain : str
        Loads only files containing 'file_name_contain' in the name of the file.

    Returns
    -------
    z : None
    """

    if save_dir is None:
        save_dir = load_dir

    if xlabel is None:
        xlabel = "Estimated Average Treatment Effect"

    df = _load_results_to_plot(load_dir=load_dir, file_name_contain=file_name_contain)
    _plot_and_save(df, xlabel=xlabel, save_dir=save_dir, figure_name=figure_name)

    return None

def _load_results_to_plot(load_dir, file_name_contain):
    """Function to load results required for making the figure.

    Parameters
    ----------
    load_dir : str
        Directory to load results from.
    file_name_contain : str
        Loads only files containing 'file_name_contain' in the name of the file.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame with columns: Method - str indicating the models used for ATE prediction and ATE - estimated ATE;
         and rows corresponding to a single iteration of an experiment.
    """

    # Initialize results
    df = pd.DataFrame([])

    if file_name_contain is None:
        match_str = 'results_*.csv'
    else:
        match_str = f'results_*{file_name_contain}*.csv'

    for result in os.listdir(load_dir):
        if fnmatch.fnmatch(result, match_str):
            print("Loading:", result)
            model_name = re.search('_(.+?)_', result).group(1)
            result_values = pd.read_csv(result, header=None)
            df[model_name] = result_values.iloc[:, 0].round(2)

    # Transform results
    df = df.melt()
    df.columns = ['Method', 'ATE']

    return df

def _plot_and_save(df, figure_name, save_dir, xlabel, show_points:bool=False):
    """Shows and saves the generated boxplot.

        Parameters
        ----------
        df : pd.DataFrame
            A pd.DataFrame with column 'Method' indicating which model as used for ATE prediction and a columns ATE
            indicating the value of the estimated treatment effect.
        figure_name : str
            Name of the file in which the figure is saved. For example, if you want to save the boxplot as
            'boxplot.png', then 'figure_name="boxplot"'.
        save_dir : str
            Directory to save the figure to. If None then the figure is saved to 'load_dir'.
        xlabel : str
            Name of the figure's x-axis.
        show_points : bool
            If True, shows points indicating individual ATE estimates.

        Returns
        -------
        z : None
        """
    f, ax = plt.subplots(figsize=(7, 6))

    sns.boxplot(x="ATE", y="Method", data=df, orient='h',
              whis=[2.5, 97.5], width=.6, palette="vlag",
              color=".2", showmeans=True,
              meanprops={"marker":"s",
                         "markerfacecolor":"dimgrey",
                         "markeredgecolor":"dimgrey",
                         "markersize":"5"})

    if show_points:
        sns.stripplot(x="ATE", y="Method", data=df_plot, size=3, color=".35", linewidth=0)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    ax.set(xlabel=xlabel)
    ax.axvline(x=15, ymin=0.02, ymax=0.98, color='dimgrey', linestyle='--')
    sns.despine(trim=True, left=True)
    plt.show()
    with os.chdir(save_dir):
        plt.savefig(f"{figure_name}.png")

    return None


