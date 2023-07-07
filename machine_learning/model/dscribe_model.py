import datetime

from top_n_accuracy import calculate_order_accuracy
import numpy as np
import pandas as pd
import pickle
from hyperparameter_tuning.HPS_gaussian import gaussian_iteration_for_lmtr
from tqdm import tqdm
import random
from hyperparameter_tuning import HPS_gaussian as gauss
from model.features.dscribe_inputs import add_acsf_to_atom_df
from model.features.dscribe_inputs import add_coulomb_matrix_to_atom_df
from model.features.dscribe_inputs import add_local_many_body_tensor_to_atom_df
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from skopt import BayesSearchCV
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score
from plots import plot_dft_vs_predicted

class DSCRIBEmodel(object):

    def __init__(self, id='krrmodel', target_flag=None, fulldata_atom_df=None, fulldata_pair_df=None):

        self.id = id

        self.trained = False
        self.mol_rep = None
        self.optimised_model_hyperparams = None
        self.optimised_rep_hyperparams = None
        self.optimised = False

        self.trainids = []
        self.valids = []
        self.testids = []

        self.fulldata_atom_df = fulldata_atom_df
        self.fulldata_pair_df = fulldata_pair_df

        if not target_flag:
            print('No targetflag provided, defaulting to HCS')
            self.targetflag = 'HCS'

        else:
            self.targetflag = target_flag

        self.train_atom_df = pd.DataFrame()
        self.train_pair_df = pd.DataFrame()

        self.validation_atom_df = pd.DataFrame()
        self.validation_pair_df = pd.DataFrame()

        self.test_atom_df = pd.DataFrame()
        self.test_atom_df = pd.DataFrame()

        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []

        self.model = None

    def train_validation_test_split(self, test=True):

        atom_df = self.fulldata_atom_df

        if test:
            train = 6
            val = 8

        else:
            train = 8
            val = 100

        molids = atom_df['molecule_name'].unique()

        random.shuffle(molids)

        tenpercent = round(len(molids)/10)

        self.trainids = molids[:(tenpercent * train)]

        self.valids = molids[(tenpercent * train):(tenpercent * val)]

        self.testids = molids[(tenpercent * val):]


        if len(self.testids) == 0:
            self.testids = self.test_atom_df['molecule_name'].unique()

    def train_validation_split(self):

        atom_df = self.train_atom_df

        molids = atom_df['molecule_name'].unique()

        random.shuffle(molids)

        tenpercent = round(len(molids) / 10)

        self.trainids = molids[:(tenpercent * 8)]

        self.valids = molids[(tenpercent * 8):(tenpercent * 10)]

        self.train_atom_df = self.train_atom_df[self.train_atom_df['molecule_name'].isin(self.trainids)]
        self.validation_atom_df = self.train_atom_df[self.train_atom_df['molecule_name'].isin(self.valids)]

    def train_test_split(self):

        atom_df = self.fulldata_atom_df

        molids = atom_df['molecule_name'].unique()

        random.shuffle(molids)

        tenpercent = round(len(molids) / 10)

        self.trainids = molids[:(tenpercent * 8)]

        self.testids = molids[(tenpercent * 8):(tenpercent * 10)]

        self.train_atom_df = self.fulldata_atom_df[self.fulldata_atom_df['molecule_name'].isin(self.trainids)]
        self.test_atom_df = self.fulldata_atom_df[~self.fulldata_atom_df['molecule_name'].isin(self.trainids)]

    def split_fulldata_df(self, trainids, valids, testids, test=True):

        self.train_atom_df = self.fulldata_atom_df[self.fulldata_atom_df['molecule_name'].isin(trainids)]
        self.train_pair_df = self.fulldata_pair_df[self.fulldata_pair_df['molecule_name'].isin(trainids)]

        self.validation_atom_df = self.fulldata_atom_df[self.fulldata_atom_df['molecule_name'].isin(valids)]
        self.validation_pair_df = self.fulldata_pair_df[self.fulldata_pair_df['molecule_name'].isin(valids)]

        self.test_atom_df = self.fulldata_atom_df[self.fulldata_atom_df['molecule_name'].isin(testids)]
        self.test_pair_df = self.fulldata_pair_df[self.fulldata_pair_df['molecule_name'].isin(testids)]

        print('df split: Train', len(self.train_atom_df['molecule_name'].unique()),
              'Test', len(self.test_atom_df['molecule_name'].unique()),
              'Val', len(self.validation_atom_df['molecule_name'].unique()))

    def add_acsf(self, test=False):
        atom_acsf = add_acsf_to_atom_df(self.fulldata_atom_df, self.fulldata_pair_df)

        self.fulldata_atom_df = atom_acsf

        self.mol_rep = 'ascf'

    def add_soap(self):
        print('write this function')

    def add_coulomb_matrix(self, test=False):

        atom_coulomb_matrix = add_coulomb_matrix_to_atom_df(self.fulldata_atom_df, self.fulldata_pair_df)

        self.fulldata_atom_df = atom_coulomb_matrix

        self.mol_rep = 'coulomb_matrix'

    def add_lmbt(self, amols=None, opt=False, test=True):
        '''
        :param opt:
        :param test: If test=True then the fulldata_df will be split 60:20:20 into train:test:val.
        :return:
        '''

        k2 = {'max': 2, 'n': 50, 'sigma': 0.1}

        k3 = {'max': 180, 'n': 50, 'sigma': 5}

        if not self.trained:
            if opt:
                opt_dict = self.optimise_all_params_for_lmbtr(self.fulldata_atom_df, self.fulldata_pair_df,
                                                         outname=self.id)

                self.optimised_rep_hyperparams = opt_dict

                self.fulldata_atom_df = add_local_many_body_tensor_to_atom_df(self.fulldata_atom_df, self.fulldata_pair_df,
                                                                              weight_func=opt_dict[2],
                                                                              k2_param_dict=opt_dict[0],
                                                                              k3_param_dict=opt_dict[1], amols=amols)

            else:

                self.fulldata_atom_df = add_local_many_body_tensor_to_atom_df(self.fulldata_atom_df, self.fulldata_pair_df,
                                                                              weight_func='unity', k2_param_dict=k2,
                                                                              k3_param_dict=k3, amols=amols)
        else:
            self.test_atom_df = add_local_many_body_tensor_to_atom_df(self.test_atom_df, self.test_pair_df,
                                                                      weight_func='unity', k2_param_dict=k2,
                                                                      k3_param_dict=k3, amols=amols
                                                                      )

        self.mol_rep = 'lmbtr'

    def optimise_all_params_for_lmbtr(self, fulldata_atom_df, fulldata_pair_df,
                                      outname):

        print('Optimising LMBTR Representation Hyper Parameters...', datetime.datetime.now())

        self.train_validation_split()

        param_ranges = {'k2_n': {'min': 10, 'max': 300},
                        'k2_sigma': {'min': 1e-3, 'max': 0.2},
                        'k2_max': {'min': 1, 'max': 10},
                        'k3_n': {'min': 10, 'max': 200},
                        'k3_sigma': {'min': 1e-3, 'max': 0.2},
                        'k3_max': {'min': 10, 'max': 300},
                        'alpha': {'min': 1e-6, 'max': 1e+1},
                        'gamma': {'min': 1e-6, 'max': 1e+1},
                        'degree': {'min': 1, 'max': 8}}

        #weight_func_dict = {1: 'exp', 2: 'unity'}
        #kernel_dict = {1: 'linear', 2: 'poly', 3: 'rbf'}

        setup = gauss.setup_gaussian(param_ranges, kappa=1, xi=1)

        optimisation_df = pd.DataFrame()

        for j in tqdm(range(100)):

            opt = gaussian_iteration_for_lmtr(setup[0], setup[1], param_ranges=param_ranges)

            k2_param_dict = {'max': opt[0]['k2_max'], 'n': opt[0]['k2_n'], 'sigma': opt[0]['k2_sigma']}
            k3_param_dict = {'max': opt[0]['k3_max'], 'n': opt[0]['k3_n'], 'sigma': opt[0]['k3_sigma']}

            self.fulldata_atom_df = add_local_many_body_tensor_to_atom_df(fulldata_atom_df, fulldata_pair_df,
                                                                          weight_func='unity',
                                                                        k2_param_dict=k2_param_dict,
                                                                        k3_param_dict=k3_param_dict)

            self.split_fulldata_df(self.trainids, self.valids, self.testids, test=False)

            train = self.krr_train(test_type='validation', opt=False, save_model=False, print_output=False)
            self.trained = False

            data = pd.DataFrame({'iteration': j, 'validation_MAE': train[0], 'validation_r2': train[2],
                                'train_MAE': train[1], 'train_r2': train[3],
                                'weight_func': 'unity',
                                'K2_max': opt[0]['k2_max'], 'K2_n': opt[0]['k2_n'], 'K2_sigma': opt[0]['k2_sigma'],
                                'K3_max': opt[0]['k3_max'], 'K3_n': opt[0]['k3_n'], 'K3_sigma': opt[0]['k3_sigma']}, index=[0])


            #'alpha': opt['alpha'], 'degree': opt['degree'], 'gamma': opt['gamma'],
            #'kernel': kernel}, index=[0])

            optimisation_df = pd.concat([optimisation_df, data])
            setup[0].register(params=opt[1], target=-train[0])

        opt_K2_dict = {'max': optimisation_df.sort_values(['validation_MAE'])['K2_max'].values[0],
                  'n':  optimisation_df.sort_values(['validation_MAE'])['K2_n'].values[0],
                   'sigma': optimisation_df.sort_values(['validation_MAE'])['K2_sigma'].values[0]}

        opt_K3_dict = {'max': optimisation_df.sort_values(['validation_MAE'])['K3_max'].values[0],
                  'n':  optimisation_df.sort_values(['validation_MAE'])['K3_n'].values[0],
                   'sigma': optimisation_df.sort_values(['validation_MAE'])['K3_sigma'].values[0]}

        opt_weight_func = optimisation_df.sort_values(['validation_MAE'])['weight_func'].values[0]

        with open(outname + '_' + self.targetflag + '_optimisation.csv', 'w') as f:
            print(optimisation_df.to_csv(), file=f)

        self.optimised_rep_hyperparams = {'weight_func': opt_weight_func, 'K2': opt_K2_dict, 'K3': opt_K3_dict}

        return opt_K2_dict, opt_K3_dict, opt_weight_func

    def krr_data_setup(self, type='', truth=True):

        if type == 'train':
            atom_df = self.train_atom_df

        elif type == 'validation':
            atom_df = self.validation_atom_df

        elif type == 'test':
            atom_df = self.test_atom_df

        elif type == 'all':
            atom_df = self.fulldata_atom_df

        else:
            raise TypeError('Type:', type,  'not Recognised')

        targetflag = self.targetflag

        type_df = atom_df[atom_df['typestr'] == targetflag[0]]

        x = atom_df[atom_df['typestr'] == targetflag[0]]['atomic_rep']

        self.targetflag = targetflag

        atomic_reps = []

        for i in x.values:
            atomic_reps.append(i)

        x = np.array(atomic_reps)

        # Get shifts for all rows where the typestr matches the first character in any of the targetflags
        if truth:
            y = atom_df[atom_df['typestr'] == targetflag[0]]['shift'].to_numpy()
        else:
            y = None

        return x, y, type_df

    def cross_validation_setup(self, train_x, train_y, cv_chunks=[], cv_range=5):

        chunk_size = int(len(train_y) / cv_range)
        idx = []
        for chunk in range(cv_range):
            if chunk in cv_chunks:
                idx.extend(range(chunk, chunk + chunk_size))

        cv_x = train_x[idx]
        cv_y = train_y[idx]

        return cv_x, cv_y

    def optimise_model_params(self, train_x, train_y, model_type):

        print('Optimising Model Hyperparameters...')
        opt = BayesSearchCV(model_type,
                            {'alpha': (1e-6, 1e+1, 'log-uniform'),
                            'gamma': (1e-6, 1e+1, 'log-uniform'),
                            'degree': (1, 8),
                            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
                              }, n_iter=100, cv=5)

        opt.fit(train_x, train_y)

        #print("best params: %s" % str(opt.best_params_))

        self.optimised_model_hyperparams = opt.best_params_
        self.optimised = True

        return opt.best_params_

    def krr_train(self, test_type='test', predict=True, opt=True, save_model=True, print_output=True, truth=True):

        train_x, train_y, train_df = self.krr_data_setup(type='train', truth=truth)
        test_x, test_y, test_df = self.krr_data_setup(type=test_type, truth=truth)


        if opt:
            for i in range(100):
                try:
                    opt_params = self.optimise_model_params(train_x, train_y, model_type=KernelRidge())

                except:
                    AttributeError()
                    print('Attribute Error! Restarting run...')
                    continue

                else:
                    break

            model = KernelRidge(alpha=opt_params['alpha'], degree=opt_params['degree'], gamma=opt_params['gamma'], kernel=opt_params['kernel'])

        else:
            if self.targetflag == 'HCS':
                model = KernelRidge()

            elif self.targetflag == 'CCS':
                model = KernelRidge()

            elif self.targetflag == 'NCS':
                model = KernelRidge()

        model.fit(train_x, train_y)

        self.trained = True
        self.model = model

        if save_model:
            name = self.id + '_' + self.targetflag + '_model.pkl'
            self.save_model(model=model, outname=name)

        if predict:
            predict = self.predict(train_x, train_y, test_x, test_y, test_df, print_output=print_output)

            return predict

    def predict(self, train_x, train_y, test_x, test_y, test_df, print_output=True):

        #test_df = self.data_setup(type=test_type)

        predictions = self.model.predict(test_x)

        test_df['predicted_shift'] = pd.Series(predictions).values


        metrics = self.run_model_metrics(self.model, train_y, train_x, test_y, test_x, test_df, print_output=print_output)

        plot_dft_vs_predicted(target_flag=self.targetflag, prediction_df=test_df,
                              outname=self.id + '_DFT_vs_predicted.png')

        return metrics, test_df

    def assign_preds_from_load(self, model, test_df, truth=True, write=True):

        test_x, test_y, test_df = self.krr_data_setup(type='test', truth=truth)

        idx = test_df['typestr'].index.values
        preds = model.predict(test_x)

        pred_df = self.test_atom_df

        if 'predicted_shift' not in pred_df.columns.values:
            pred_df['predicted_shift'] = np.zeros(len(self.test_atom_df))

        pred_df.loc[idx, 'predicted_shift'] = preds

        pred_df = pred_df.loc[:, ~pred_df.columns.str.contains('^Unnamed')]

        if write:
            pred_df = pred_df.loc[:, ~pred_df.columns.str.contains('mol_rep')]
            pred_df.to_csv(self.id + '_prediction_df.csv')

        return pred_df

    def save_model(self, model, outname):

        to_save = {'model': model,
                   'rep_params': self.optimised_rep_hyperparams,
                   'training_Set': self.trainids,
                   'validation_set': self.valids,
                    'test_set': self.testids}

        pickle.dump(to_save, open(outname, 'wb'))

    def load_model(self, model_filename):

        loaded = pickle.load(open(model_filename, 'rb'))
        self.trained = True
        self.model = loaded['model']
        self.optimised_rep_hyperparams = loaded['rep_params']
        self.testids = loaded['test_set']
        self.valids = loaded['validation_set']
        self.trainids = loaded['training_Set']

        return loaded

    def run_model_metrics(self, model, train_y, train_x, test_y, test_x, pred_df,
                          print_output=True, save_output=True):

        if self.trained == False:
            print('Parsed model not trained, Aborting...')

        else:

            if self.train_atom_df.empty:
                MAE_train = None

                r2_train = None

                RMSE_train = None

                MAXE_train = None

            else:

                MAE_train = (mean_absolute_error(train_y, model.predict(train_x)))

                r2_train = r2_score(train_y, model.predict(train_x))

                RMSE_train = (mean_squared_error(train_y, model.predict(train_x), squared=False))

                MAXE_train = (max_error(train_y, model.predict(train_x)))

            MAE_test = (mean_absolute_error(test_y, model.predict(test_x)))

            r2_test = r2_score(test_y, model.predict(test_x))

            RMSE_test = (mean_squared_error(test_y, model.predict(test_x), squared=False))

            MAXE_test = (max_error(test_y, model.predict(test_x)))

            Top_1_accuracy = calculate_order_accuracy(targetflag=self.targetflag, assign_preds_dataframe=pred_df,
                                                      n=1, print_output=print_output)

            if print_output:
                print('MAE_test: ', MAE_test)
                print('r2_test: ', r2_test)
                print('RMSE_test: ', RMSE_test)
                print('MAXE_test: ', MAXE_test)

            if save_output:
                with open(self.id + '_' + self.targetflag + '_summary.txt', 'w') as f:
                    print('atomic_rep:', self.mol_rep, file=f)
                    print('model params:', model, file=f)
                    print('rep_params:', self.optimised_rep_hyperparams, file=f)
                    print('MAE_train: ', MAE_train, file=f)
                    print('r2_train: ', r2_train, file=f)
                    print('RMSE_train: ', RMSE_train, file=f)
                    print('MAXE_train: ', MAXE_train, file=f)
                    print('\n', file=f)
                    print('Top 1:', Top_1_accuracy, file=f)
                    print('MAE_test: ', MAE_test, file=f)
                    print('r2_test: ', r2_test, file=f)
                    print('RMSE_test: ', RMSE_test, file=f)
                    print('MAXE_test: ', MAXE_test, file=f)

            return MAE_test, MAE_train, r2_test, r2_train, RMSE_test, RMSE_train, MAXE_test, MAXE_test
