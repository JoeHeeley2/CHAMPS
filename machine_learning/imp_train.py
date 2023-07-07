from model.fchl_model import FCHLmodel
from model.features.fchl_input import add_fchl_to_atom_df
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import numpy as np
import seaborn as sns
from plots import plot_dft_vs_predicted
from top_n_accuracy import calculate_order_accuracy


def train_and_predict(targetflag, model_params, outname, train_atom_df, train_pair_df, test_atom_df, test_pair_df,
                predict=True, write=True, zoom=False, return_worst_structures=True):

    '''
    This function will train an IMPRESSSION gen I model on the inputs given. Predict and write default to true.
    This function also plots the error distrubution of the predictions by default using the outname if predict=True.
    outname should be in the form: {training_data}_{test_data}

    :param targetflag: Which property you want to optimise for ['HCS'] = acidity,
                        ['CCS'] = nucleophilicity and ['NCS'] = donicity.

    :param model_params: Dictionary of optimised hyperparameters for model training.
                            If empty, default hyperparams will be used.

    :param outname: Name to save all files as.

    :param train_atom_df: Atom dataframe containing training structures

    :param train_pair_df: Pair dataframe containing training structures

    :param test_atom_df: Atom dataframe containing test structures

    :param test_pair_df: Pair dataframe containing test structures

    :param predict: Bool, if True the trained model will predict properties of test structures

    :param write: Bool, if true all files will be saved.

    :param zoom: Bool, if true error distribution plots will be zoomed in.

    :param return_worst_structures: Bool, if True the structures with the worst predicitons will be printed.

    :return: None
    '''

    plt.clf()
    overlapped_structures = [x for x in train_atom_df['molecule_name'].unique() if x in test_atom_df['molecule_name'].unique()]
    if len(overlapped_structures) > 0:
        raise ValueError('Found same structures in test and training sets')

    train_inp = add_fchl_to_atom_df(train_atom_df)
    test_inp = add_fchl_to_atom_df(test_atom_df)

    imp = FCHLmodel(model_args={'targetflag': targetflag}, model_params=model_params)

    imp_input_training = imp.get_input(train_inp, train_pair_df)
    imp_input_test = imp.get_input(test_inp, test_pair_df)
    imp.train_x = imp_input_training[0]
    imp.train_y = imp_input_training[1]
    test_x = imp_input_test[0]
    test_y = imp_input_test[1]
    imp.train(label=outname + '_' + targetflag[0])

    if predict:
        pred = imp.predict(test_x=test_x)
        pred_assign = imp.assign_preds(pred_y=pred, atom_df=test_atom_df, pair_df=test_pair_df)

        eval = imp.mae(test_x, test_y)
        stdev = imp.stdev(test_x, test_y)
        error = imp.error(test_x, test_y)

        print('MAE:', eval)
        print('stdev:', stdev)

        if return_worst_structures:
            pred_assign[0]['error'] = pred_assign[0]['shift'] - pred_assign[0]['predicted_shift']
            targetflag_df = pred_assign[0].loc[pred_assign[0]['typestr'] == list(targetflag[0])[0]]

            print('------------------------------------')
            print('Structures with most positive error:')
            for a, b, c in zip(targetflag_df.nlargest(5, 'error')['molecule_name'].values,
                               targetflag_df.nlargest(5, 'error')['atom_index'].values,
                               targetflag_df.nlargest(5, 'error')['error'].values):
                print(a, b, c)

            print('------------------------------------')
            print('Structures with most negative error:')
            for x, y, z in zip(targetflag_df.nsmallest(5, 'error')['molecule_name'].values,
                               targetflag_df.nsmallest(5, 'error')['atom_index'].values,
                               targetflag_df.nsmallest(5, 'error')['error'].values):
                print(x, y, z)

        err_plot = sns.distplot(error)
        err_plot.set_xlabel('Error_kcalpermol')

        if write:
            with open(outname +
                      '_' + targetflag[0] + '_predictions_' + outname.split('/')[1] + '.csv', 'w') as f:
                print(pred_assign[0].to_csv(sep=','), file=f)

            if zoom:
                err_plot.set(ylim=(0, 0.02))
                err_plot.set_title('Error distribution for ' + targetflag[0] + outname.split('/')[1] + '_zoom')
                err_plot.figure.savefig(outname.split('/')[0] + '/Error distribution for ' + targetflag[0] + outname.split('/')[-1] + '_zoom')

            else:
                err_plot.set_title('Error distribution for ' + targetflag[0] + outname)
                err_plot.figure.savefig(outname + '_error_dist_' + targetflag[0])

            plt.clf()
            plot_dft_vs_predicted(target_flag=targetflag[0], prediction_df=pred_assign[0], outname=outname)
            calculate_order_accuracy(targetflag[0], pred_assign[0], n=1)


    #return eval, stdev


def load_and_predict(targetflag, model_filename, outname, test_atom_df, test_pair_df,
                     predict=True, write=True, zoom=False, return_worst_structures=False):

    '''
    This function will load a pre-trained IMPRESSION GenI model and predict the structures in test_atom_df.

    :param targetflag: Which property you want to optimise for ['HCS'] = acidity,
                        ['CCS'] = nucleophilicity and ['NCS'] = donicity.

    :param model_params: Dictionary of optimised hyperparameters for model training.
                            If empty, default hyperparams will be used.

    :param outname: Name to save all files as.

    :param test_atom_df: Atom dataframe containing test structures

    :param test_pair_df: Pair dataframe containing test structures

    :param predict: Bool, if True the trained model will predict properties of test structures

    :param write: Bool, if true all files will be saved.

    :param zoom: Bool, if true error distribution plots will be zoomed in.

    :param return_worst_structures: Bool, if True the structures with the worst predicitons will be printed.

    :return: None
    '''
    plt.clf()
    test_inp = add_fchl_to_atom_df(test_atom_df)

    imp = FCHLmodel(model_args={'targetflag': targetflag})

    imp_input_test = imp.get_input(test_inp, test_pair_df)
    test_x = imp_input_test[0]
    test_y = imp_input_test[1]

    imp.load_model(filename=model_filename)
    print(imp.params)

    if predict:
        pred = imp.predict(test_x=test_x)
        pred_assign = imp.assign_preds(pred_y=pred, atom_df=test_atom_df, pair_df=test_pair_df)

        eval = imp.mae(test_x, test_y)
        stdev = imp.stdev(test_x, test_y)
        error = imp.error(test_x, test_y)

        print('MAE:', eval)
        print('stdev:', stdev)

        if return_worst_structures:
            pred_assign[0]['error'] = pred_assign[0]['shift'] - pred_assign[0]['predicted_shift']
            targetflag_df = pred_assign[0].loc[pred_assign[0]['typestr'] == list(targetflag[0])[0]]

            print('------------------------------------')
            print('Structures with most positive error:')
            for a, b, c in zip(targetflag_df.nlargest(5, 'error')['molecule_name'].values,
                               targetflag_df.nlargest(5, 'error')['atom_index'].values,
                               targetflag_df.nlargest(5, 'error')['error'].values):
                print(a, b, c)

            print('------------------------------------')
            print('Structures with most negative error:')
            for x, y, z in zip(targetflag_df.nsmallest(5, 'error')['molecule_name'].values,
                               targetflag_df.nsmallest(5, 'error')['atom_index'].values,
                               targetflag_df.nsmallest(5, 'error')['error'].values):
                print(x, y, z)

        err_plot = sns.distplot(error)
        err_plot.set_xlabel('Error_kcalpermol')

        if write:
            with open(outname.split('/')[0] + '/' + targetflag[0] + '_predictions_' + outname.split('/')[1] + '.csv',
                      'w') as f:
                print(pred_assign[0].to_csv(sep=','), file=f)

        if zoom:
            err_plot.set(ylim=(0, 0.02))
            err_plot.set_title('Error distribution for ' + targetflag[0] + outname.split('/')[1] + '_zoom')
            err_plot.figure.savefig(
                outname.split('/')[0] + '/Error distribution for ' + targetflag[0] + outname.split('/')[1] + '_zoom')

        else:
            err_plot.set_title('Error distribution for ' + targetflag[0] + outname)
            err_plot.figure.savefig(
                outname.split('/')[0] + '/Error distribution for ' + targetflag[0] + outname.split('/')[1])

        plt.clf()
        plot_dft_vs_predicted(target_flag=targetflag, prediction_df=pred_assign[0], outname=outname)
