import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
import os


def get_averages(df, version, statement):
    """
    Calculates the average statement values per passage

    :param df: Dataframe containing data about annotated passages
    :param version: 'piper_2022' or 'universals'; Experiment version of dataframe
    :param statement: Statement type of experiment
    :return: List of average statement value per passage
    """
    if version == 'piper_2022':
        stat_cols = list(df.columns[df.columns.str.endswith(statement)])
    elif version == 'universals':
        stat_cols = list(df.columns[df.columns.str.startswith(statement)])
    df_avgs = pd.concat([df['FILENAME'], df[stat_cols].mean(axis=1)], axis=1, join="inner")
    return list(df_avgs.sort_values(by='FILENAME', ascending=True)[0].values)


def polynomial_plot(x_df, y_df, p_or_n):
    """
    Construct a polynomial plot

    :param x_df: DataFrame for X-Axis
    :param y_df: DataFrame for Y-Axis
    :param p_or_n: String depicting if working on Piper data or Universal data
    :return: Polynomial Plot
    """
    y_df_sel = y_df[y_df['NARRATIVITY'] == p_or_n]
    x_df_sel = x_df[x_df['FILENAME'].isin(y_df_sel['FILENAME'])]
    polyfit = np.polyfit(x_df_sel['Probability.Narrative'], y_df_sel['Probability.Narrative'], 2).round(3)

    legend_labels = []
    for x in range(len(polyfit)):
        t = ""
        if x != 0:
            if polyfit[x] < 0:
                t += '-'
            else:
                t += '+'
        t += '{0:.3f}'.format(abs(polyfit[x]))
        legend_labels.append(t)
    poly_plot = sns.regplot(x=x_df_sel['Probability.Narrative'],
                            y=y_df_sel['Probability.Narrative'],
                            color='green',
                            order=2,
                            scatter_kws={'alpha': 0.15},
                            line_kws={'label': r'$y={0}x^2{1}x{2}$'.format(legend_labels[0],
                                                                           legend_labels[1],
                                                                           legend_labels[2])})
    return poly_plot


def get_pearson(x, y):
    """
    Get Pearson's r
    :param x: Numpy array
    :param y: Numpy array
    :return: float
    """
    return round(np.corrcoef(x, y)[0][1], 5)


def get_kendall(x, y):
    """
    Get Kendall's Tau
    :param x: Numpy array
    :param y: Numpy array
    :return: float
    """
    return round(kendalltau(x, y).correlation, 5)


def get_plot(x, y, pearson):
    """
    Get regression plot of points with linear regression line (pearson)
    :param x: Numpy array
    :param y: Numpy array
    :param pearson: float
    :return: regression plot
    """
    return sns.regplot(x=x,
                       y=y,
                       color='green',
                       scatter_kws={'alpha': 0.15},
                       line_kws={'label': 'Pearson\'s r: {}'.format(pearson)})


def make_dir(path):
    """
    Create a directory
    :param path: String depicitng a path
    """
    if not os.path.exists(path):
        os.mkdir(path)


def save_and_close(path):
    """
    Save figure to path
    :param path: String depicting a path
    """
    plt.savefig(path)
    plt.close()


def main():
    # Make directories
    make_dir('plots_statements')
    make_dir('plots_NP')

    # Make Dataframes
    df_piper_2022 = pd.read_csv('../data/MinNarrative_ReaderData_Final_Selection.csv').sort_values('FILENAME')
    df_universals = pd.read_csv('../data/Universal_Annotation_Results_Selection.csv').sort_values('FILENAME')

    # Calculate Pearson and plot heatmap
    stats_piper_2022 = ['_agency', '_event', '_world']
    stats_universals = ['suspense_cmb_', 'curiosity_cmb_', 'surprise_cmb_']

    df_corr = pd.DataFrame()
    for stat_piper_2022 in stats_piper_2022:
        avgs_piper_2022 = get_averages(df_piper_2022, 'piper_2022', stat_piper_2022)
        for stat_universals in stats_universals:
            x_label = stat_piper_2022.split('_')[-1].capitalize()
            y_label = stat_universals.split('_')[0].capitalize()

            avgs_universals = get_averages(df_universals, 'universals', stat_universals)
            pearson = get_pearson(avgs_piper_2022, avgs_universals)
            df_corr.at[x_label, y_label] = pearson
            print('Pearson\'s r ({0} vs. {1}):\t{2}'.format(x_label, y_label, pearson))

    heatmap = sns.heatmap(df_corr, cmap='crest', fmt='.5f', annot=True, vmin=-1, vmax=1)
    heatmap.set(title='Correlation (Pearson\'s r) per Statement Pair')
    save_and_close('plots_statements/pearson_heatmap')
    print()

    # Calculate Pearson and plot scatterplot for averages of all statements
    pearson = get_pearson(df_piper_2022['avg_overall'].to_numpy(), df_universals['avg_overall'].to_numpy())
    x_label, x_label_title = "Average Annotation Values of Piper (2022)", "Annotation Piper (2022)"
    y_label, y_label_title = "Average Annotation Values of Universals", "Annotation Universals"

    scatterplot_annotation_path = 'plots_statements/scatterplot_annotation_{0}_{1}.png'.format('piper', 'universals')
    ax = get_plot(df_piper_2022['avg_overall'].to_numpy(), df_universals['avg_overall'].to_numpy(), pearson)
    ax.set(xlabel=x_label, ylabel=y_label, xlim=(1.0, 5.0), ylim=(1.0, 5.0))
    ax.legend()
    save_and_close(scatterplot_annotation_path)

    print('Pearson\'s r ({0} vs. {1}):\t{2}'.format(x_label_title, y_label_title, pearson))
    print('Scatterplot saved to {0}'.format(scatterplot_annotation_path))
    print()

    # Calculate Pearson, Kendall and plot scatterplot for Narrative Probabilities
    pearson = get_pearson(df_piper_2022['Probability.Narrative'].to_numpy(),
                          df_universals['Probability.Narrative'].to_numpy())
    df_piper_2022_files_sorted = df_piper_2022.sort_values('Probability.Narrative')['FILENAME'].to_numpy()
    df_universals_files_sorted = df_universals.sort_values('Probability.Narrative')['FILENAME'].to_numpy()
    kendall = get_kendall(df_piper_2022_files_sorted, df_universals_files_sorted)

    x_label = "Narrative Probabilities of Piper (2022)"
    y_label = "Narrative Probabilities of Universals"

    ax = get_plot(df_piper_2022['Probability.Narrative'].to_numpy(), df_universals['Probability.Narrative'].to_numpy(),
                  pearson)
    ax.set(xlabel=x_label, ylabel=y_label, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax.legend()
    save_and_close('plots_NP/scatterplot_np_{0}_{1}.png'.format('piper', 'universals'))

    # Get Polynomial Plot
    plt.rcParams["figure.figsize"] = (8, 4)
    x_start, x_end, x_step = 0.0, 1.0, 0.1

    poly_plot_pos = polynomial_plot(df_piper_2022, df_universals, "POS")
    poly_plot_pos.set(ylim=(0.5, 1), xticks=np.arange(x_start, x_end + x_step, x_step), xlabel=x_label, ylabel=y_label)
    poly_plot_pos.legend()
    save_and_close('plots_NP/poly_plot_pos.png')

    poly_plot_neg = polynomial_plot(df_piper_2022, df_universals, "NEG")
    poly_plot_neg.set(ylim=(0, 0.5), xticks=np.arange(x_start, x_end + x_step, x_step), xlabel=x_label, ylabel=y_label)
    poly_plot_neg.legend()
    save_and_close('plots_NP/poly_plot_neg.png')

    print('Pearson\'s r ({0} vs. {1}):\t{2}'.format('NP Piper (2022)', 'NP Universals', pearson))
    print('Kendall\'s т ({0} vs. {1}):\t{2}'.format('NP Piper (2022)', 'NP Universals', kendall))
    print('Scatterplot saved to {0}'.format('plots_NP/scatterplot_np_{0}_{1}.png'.format('piper', 'universals')))
    plt.show()


if __name__ == "__main__":
    main()
