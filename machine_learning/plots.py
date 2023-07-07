import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_dft_vs_predicted(target_flag, prediction_df, outname):

    '''
    Plots a graph of ground_truth DFT vs ML_predicted properties

    :param target_flag: 'HCS', 'CCS' or 'NCS' for acidites, nucleophilicities or donicities

    :param prediction_df: Output prediction dataframe from ML models. Must contain
                            ground truth and predicted values

    :param outname: filename for saved output. No file extension needed

    :return: None
    '''

    non_zero = pd.DataFrame()
    xlabel = ''
    ylabel = ''
    title = ''
    color = ''
    ylim = 0, 0
    xlim = 0, 0

    if target_flag == 'HCS':
        flag = prediction_df[prediction_df['typestr'] == 'H']
        non_zero = flag[flag['shift'] != 0]
        ylabel = r'DFT Calculated Acidities [Kcal mol$^{-1}$]'
        xlabel = r'Predicted Acidities [Kcal mol$^{-1}$]'
        title = 'DFT vs Predicted Acidities'
        color = 'red'


    elif target_flag == 'CCS':
        flag = prediction_df[prediction_df['typestr'] == 'C']
        non_zero = flag[flag['shift'] != 0]
        ylabel = r'DFT Calculated Electrophile Affinities [Kcal mol$^{-1}$]'
        xlabel = r'Predicted Electrophile Affinities [Kcal mol$^{-1}$]'
        title = 'DFT vs Predicted Electrophile Affinities'
        color = 'blue'

    elif target_flag == 'NCS':
        flag = prediction_df[prediction_df['typestr'] == 'N']
        non_zero = flag[flag['shift'] != 0]
        ylabel = r'DFT Calculated Donicities [Kcal mol$^{-1}$]'
        xlabel = r'Predicted Donicities [Kcal mol$^{-1}$]'
        title = 'DFT vs Predicted Donicity'
        color = 'cyan'

    truth = non_zero['shift'].values
    pred = non_zero['predicted_shift'].values

    x = np.linspace(min(non_zero['predicted_shift'].values) - 10, max(non_zero['predicted_shift'].values) + 10)

    plt.plot(x, x, color='black')
    plt.scatter(pred, truth, marker='x', color=color)
    plt.ylim(min(non_zero['predicted_shift'].values) - 10, max(non_zero['predicted_shift'].values) + 10)
    plt.xlim(min(non_zero['predicted_shift'].values) - 10, max(non_zero['predicted_shift'].values) + 10)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(outname + '_' + target_flag[0] + 'DFT_vs_Predicted.png', dpi=(600))

def plot_error_distribution(target_flag, error, outname, zoom=False):

    '''
    Plots mean absolute error distribution between ground truth DFT and ML predictions.

    :param target_flag:'HCS', 'CCS' or 'NCS' for acidites, nucleophilicities or donicities

    :param error: Mean Absolute Error for ML predictions

    :param outname: ilename for saved output. No file extension needed

    :param zoom: Bool, zooms in to x axis if True

    :return: None
    '''

    err_plot = sns.displot(error)
    err_plot.set_xlabel(r'Error [Kcal mol^${-1}]$')

    if zoom:
        err_plot.set(ylim=(0, 0.02))
        err_plot.set_title('Error distribution for ' + target_flag[0] + outname + '_zoom')
        err_plot.figure.savefig('Error_distribution_' + target_flag[0] + outname + '_zoom')

    else:
        err_plot.set_title('Error distribution for ' + target_flag[0] + outname)
        err_plot.figure.savefig('Error_distribution_' + target_flag[0] + outname)

