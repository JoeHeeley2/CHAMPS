import pandas as pd

def calculate_order_accuracy(targetflag, assign_preds_dataframe, n=0, print_output=True):
    '''
    Calculates Top n % accuracy for predicted molecules
    :param assign_preds_dataframe: prediction dataframe as output from ML models
    :param n: number of sites your interested in (n=1 for top 1 accuracy)
    :return:
    '''

    regids = assign_preds_dataframe['molecule_name'].unique()

    total_preds = []
    true_pred = []

    for i in regids:

        mol_df = assign_preds_dataframe[assign_preds_dataframe['molecule_name'] == i]

        target_flag_preds = mol_df[mol_df['typestr'] == targetflag[0]]

        if targetflag == 'CCS':
            ascending = True

        else:
            ascending = False

        g_truth = sorted(target_flag_preds['shift'].values, reverse=ascending)

        pred = sorted(target_flag_preds['predicted_shift'].values, reverse=ascending)

        for x, y in zip(g_truth[: n], pred[: n]):
            total_preds.append(1)

            if target_flag_preds.loc[target_flag_preds['shift'] == x].index.values[0] == target_flag_preds.loc[target_flag_preds['predicted_shift']== y].index.values[0]:
                true_pred.append(1)

            #else:
             #   print(i, target_flag_preds.loc[target_flag_preds['shift'] == x].index.values[0], target_flag_preds.loc[target_flag_preds['predicted_shift']== y].index.values[0])


    if print_output:
        print('Top ' + str(n) + ' = ', len(true_pred)/len(total_preds)*100)

    return len(true_pred)/len(total_preds)*100

