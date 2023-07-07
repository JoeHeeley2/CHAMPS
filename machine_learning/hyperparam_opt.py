import pandas as pd
from hyperparameter_tuning import HPS_gaussian as gauss
from model.fchl_model import FCHLmodel
from model.features.fchl_input import add_fchl_to_atom_df
from tqdm import tqdm

def optimise_hyperparams_cv(targetflag, train_atom_df, train_pair_df, outname, write=True):
    '''
    Function to optimise hyperparameters for IMPRESSION Gen I models with 5-fold cross validation
    :param targetflag: Which property you want to optimise for ['HCS'] = acidity,
                        ['CCS'] = nucleophilicity and ['NCS'] = donicity.
    :param train_atom_df: The atom dataframe containing training structures
    :param train_pair_df: The pair dataframe containing training structures
    :param outname: filename for writing optimisation points. No file extension needed.
    :param write: Bool, if True then each optimisation step is saved to a .csv file
    :return: dictionary of optimised hyperparameters
    '''

    param_ranges = {'sigma': {'min': -4, 'max': 2, 'log': True}, 'cutoff': {'min': 4.0, 'max': 10.0, 'log': False},
                    'lamda': {'min': -6, 'max': -2, 'log': True}}

    setup = gauss.setup_gaussian(param_ranges, kappa=1, xi=1)
    model = FCHLmodel(model_args={'targetflag': targetflag})

    columns = ['iteration', 'cross_validation_score', 'training_error', 'sigma', 'lamda', 'cutoff']
    optimisation_df = pd.DataFrame(columns=columns)

    for i in tqdm(range(100)):
        opt = gauss.gaussian_iteration(setup[0], setup[1], model, train_atom_df, train_pair_df, add_fchl_to_atom_df,
                                       param_ranges=param_ranges)
        data = {'iteration': i, 'cross_validation_score': opt[0], 'training_error': opt[1],
                'sigma': opt[2]['sigma'], 'cutoff': opt[2]['cutoff'], 'lamda': opt[2]['lamda']}
        optimisation_df = optimisation_df.append(data, ignore_index=True)

    optimised_hyperparams = {'sigma': optimisation_df.sort_values(['cross_validation_score'])['sigma'].values[0],
                             'cutoff': optimisation_df.sort_values(['cross_validation_score'])['cutoff'].values[0],
                             'lamda': optimisation_df.sort_values(['cross_validation_score'])['lamda'].values[0]}

    print('Optimised Hyperparameters:', optimised_hyperparams)

    if write:
        with open('optimisations/' + targetflag[0] + '_optimisation_' + outname + '.csv', 'w+') as f:
            print(optimisation_df.to_csv(sep=','), file=f)

    return optimised_hyperparams

