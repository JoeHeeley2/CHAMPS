import datetime
import numpy as np
import pandas as pd
import pickle
import random
import matplotlib .pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from model.features.dscribe_inputs import add_many_body_tensor_to_class_df, add_coulomb_matrix_to_class_df, add_DFT_input
from model.features.dscribe_inputs import add_soap_to_class_df
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from modules.make_3d_graphs import make_graphs
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score, \
    precision_recall_curve, roc_curve
from torch_geometric.utils.convert import to_networkx
import networkx as nx

class ClassifierModel(object):

    def __init__(self, id='classifier_model', fulldata_atom_df=None, fulldata_pair_df=None, fulldata_class_df=None):
        self.id = id

        self.trained = False
        self.mol_rep = None
        self.model_type = None
        self.optimised_model_hyperparams = None
        self.optimised = False

        self.trainids = []
        self.valids = []
        self.testids = []

        self.fulldata_atom_df = fulldata_atom_df
        self.fulldata_pair_df = fulldata_pair_df
        self.fulldata_class_df = fulldata_class_df

        self.train_class_df = pd.DataFrame()

        self.validation_class_df = pd.DataFrame()

        self.test_class_df = pd.DataFrame()

        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []

        self.model = None

    def train_validation_test_split(self, val=True):

        atom_df = self.fulldata_atom_df

        if val:
            train = 6
            test = 8

        else:
            train = 8
            test = 100

        molids = atom_df['molecule_name'].unique()

        random.shuffle(molids)

        tenpercent = round(len(molids)/10)

        self.trainids = molids[:(tenpercent * train)]

        self.testids = molids[(tenpercent * train):(tenpercent * test)]

        self.valids = molids[(tenpercent * test):]

    def split_class_df(self, trainids, valids, testids, val=False):

        self.train_class_df = self.fulldata_class_df[self.fulldata_class_df['Regid'].isin(trainids)]

        self.validation_class_df = self.fulldata_class_df[self.fulldata_class_df['Regid'].isin(valids)]

        self.test_class_df = self.fulldata_class_df[self.fulldata_class_df['Regid'].isin(testids)]

        print('Class df split: Train', len(self.train_class_df['Regid']),
                                'Test', len(self.test_class_df['Regid']),
                                'Val', len(self.validation_class_df['Regid']))

    def stratified_test_train_split(self):

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, test_index in split.split(self.fulldata_class_df, self.fulldata_class_df['class']):
            strat_train = self.fulldata_class_df.loc[train_index]
            strat_test = self.fulldata_class_df.loc[test_index]

        self.trainids = strat_train['Regid'].unique()
        self.testids = strat_test['Regid'].unique()
        self.valids = []

        print('train distribution:\n', 'len:', len(strat_train), '\n', strat_train['class'].value_counts())
        print('test_distribution:\n', 'len:', len(strat_test), '\n', strat_test['class'].value_counts())

    def add_mbtr_to_class_df(self, atom_df, pair_df, class_df):

        mbtr = add_many_body_tensor_to_class_df(atom_df, pair_df, class_df)

        self.fulldata_class_df = mbtr
        self.mol_rep = 'mbtr'

        return mbtr

    def add_dft_to_class_df(self, atom_df, pair_df, class_df, type='shift'):

        dft = add_DFT_input(atom_df, class_df, type=type)

        self.fulldata_class_df = dft
        self.mol_rep = 'DFT_descriptors'

        return dft

    def add_coulomb_matrix_to_class_df(self, atom_df, pair_df, class_df):

        cm = add_coulomb_matrix_to_class_df(atom_df, pair_df, class_df)

        self.fulldata_class_df = cm
        self.mol_rep = 'coulomb_matrix'

        return cm

    def add_soap_to_class_df(self, atom_df, pair_df, class_df):

        soap = add_soap_to_class_df(atom_df, pair_df, class_df)

        self.fulldata_class_df = soap
        self.mol_rep = 'SOAP'

        return soap

    def add_graph_to_class_df(self, atom_df, class_df):

        sotred_ids = []
        graphs = []
        classes = []
        tr_graphs, tr_scl_dict, data, failed = make_graphs(atom_df, type='class')

        for i in tr_graphs:

            g = to_networkx(i, to_undirected=True)
            print(nx.to_numpy_matrix(g))


        self.train_x = graphs
        self.train_y = classes
        self.mol_rep = 'MOLECULAR_GRAPH'

    def data_setup(self, type=''):

        if type == 'train':
            atom_df = self.train_class_df

        elif type == 'validation':
            atom_df = self.validation_class_df

        elif type == 'test':
            atom_df = self.test_class_df

        elif type == 'all':
            atom_df = self.fulldata_class_df

        else:
            raise TypeError('Type:', type, 'not Recognised')

        x = np.array([i for i in atom_df['mol_rep'].values])

        y = atom_df['class'].to_numpy()

        return x, y

    def optimise_model_params(self, train_x, train_y):

        print('Optimising Model Hyperparameters...')
        opt = BayesSearchCV(SVC(),
                            {
                             'gamma': (1e-6, 1e+1, 'log-uniform'),
                             'degree': (1, 8),
                             'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
                             }, n_iter=100, cv=5)

        opt.fit(train_x, train_y)

        # print("best params: %s" % str(opt.best_params_))

        self.optimised_model_hyperparams = opt.best_params_
        self.optimised = True

        return opt.best_params_

    def train_model(self, model_type='', test_type='test', predict=True, opt=True, save_model=True, print_output=True):

        if self.mol_rep != 'MOLECULAR_GRAPH':
            train_x, train_y = self.data_setup(type='train')
            test_x, test_y = self.data_setup(type=test_type)

        else:
            train_x = self.train_x
            train_y = self.train_y

        if self.trained:
            model = self.model

        else:
            #print(train_y)
            model = None

            if model_type == 'SVC':
                model = SVC()
                self.model_type = 'SVC'

            elif model_type == 'RF':

                #opt = self.optimise_RF(train_x=train_x, train_y=train_y)

                model = RandomForestClassifier(random_state=42)
                self.model_type = 'RandomForest'

            elif model_type == 'Gauss':
                model = GaussianProcessClassifier()
                self.model_type = 'GaussianProcess'

            elif model_type == 'KNeighbors':
                model = KNeighborsClassifier()
                self.model_type = 'KNeighbors'

            elif model_type == 'DTree':
                model = DecisionTreeClassifier()
                self.model_type = 'DecisionTree'

            elif model_type == 'MLP':
                model = MLPClassifier()
                self.model_type = 'MLP'

            elif model_type == 'SGD':
                model = SGDClassifier()
                self.model_type = 'SGD'

            else:
                raise AttributeError(print('Model Type Not Recognised, model_type should be: SVC, RF, Gauss, KNeighbors, DTree, MLP'))

        model.fit(train_x, train_y)
        self.trained = True
        self.model = model

        self.run_model_metrics(model=model, test_y=test_y, test_x=test_x)
        self.assign_predictions(test_df=self.test_class_df, print_output=False)

        if save_model:
            name = self.id + '_' + self.model_type + '_' + self.mol_rep + '_classifier_model.pkl'
            self.save_model(model=model, outname=name)

    def train_binary(self, type=''):

        train_y = []
        train_x = np.array([i for i in self.train_class_df['mol_rep'].values])
        test_y = []
        test_x = np.array([i for i in self.test_class_df['mol_rep'].values])
        p = ''
        r = ''
        title = ''

        if type == 'nuc':
            train_y = [1 if c == 2 else 0 for c in self.train_class_df['class'].values]
            test_y = [1 if c == 2 else 0 for c in self.test_class_df['class'].values]
            p = 'b'
            r = 'c'
            title = 'Nucleophilicity Precision Recall Curve'
            #threshold = 0.35

        elif type == 'ac':
            train_y = [1 if c == 1 else 0 for c in self.train_class_df['class'].values]
            test_y = [1 if c == 1 else 0 for c in self.test_class_df['class'].values]
            p = 'r'
            r = 'pink'
            title = 'Acidity Precision Recall Curve'

        elif type == 'non':
            train_y = [1 if c == 0 else 0 for c in self.train_class_df['class'].values]
            test_y = [1 if c == 0 else 0 for c in self.test_class_df['class'].values]
            p = 'dimgrey'
            r = 'lightgrey'
            title = 'Unreactive Precision Recall Curve'
            #threshold = 0.45

        opt = self.optimise_RF(train_x=train_x, train_y=train_y)
        model = RandomForestClassifier(max_features=None,
                                       min_samples_leaf=opt['min_samples_leaf'],
                                       min_samples_split=opt['min_samples_split'],
                                       n_estimators=opt['n_estimators'],
                                       bootstrap=True,
                                       max_depth=opt['max_depth'],
                                       criterion=opt['criterion'],
                                       max_leaf_nodes=opt['max_leaf_nodes'],
                                       min_impurity_decrease=opt['min_impurity_decrease'],
                                       min_weight_fraction_leaf=opt['min_weight_fraction_leaf'],
                                       random_state=42)
        self.model_type = 'RF'

        model.fit(train_x, train_y)
        self.trained = True
        self.model = model

        self.run_model_metrics(model=model, train_y=train_y, train_x=train_x, test_y=test_y, test_x=test_x)
        self.binary_plots(model, train_x, train_y, p, r, title, type=type)

        name = self.id + '_' + self.model_type + '_' + self.mol_rep + type + '_binary_classifier_model.pkl'
        self.save_model(model=model, outname=name)

    def binary_plots(self, model, x, y, p, r, title, type=''):

        threshold = 8000

        x = np.array([i for i in self.train_class_df['mol_rep'].values])

        scores = cross_val_predict(model, x, y, cv=3, method='predict_proba')
        rf_scores = scores[:, 1]

        plt.clf()

        precisions, recalls, thresholds = precision_recall_curve(y, rf_scores)
        fpr, tpr, thresholds2 = roc_curve(y, rf_scores)

        plt.plot(thresholds, precisions[:-1], color=p, linestyle='--', label='Precision')
        plt.plot(thresholds, recalls[:-1], color=r, linestyle='-', label='Recall')
        plt.xlabel('Threshold')
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.savefig(self.id + '_' + type + '_precision_recall_curve.png')

        plt.clf()

        plt.plot(fpr, tpr, color=p, linestyle='--', label=self.model_type)
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid()
        plt.legend()
        plt.savefig(self.id + '_' + type + '_ROC_curve.png')

        plt.clf()

        plt.plot(recalls[:-1], precisions[:-1], color=p)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid()
        plt.savefig(self.id + '_' + type + '_precision_v_recall.png')

    def assign_predictions(self, test_df, type='test', print_output=True):

        test_x, test_y = self.data_setup(type=type)

        predictions = self.model.predict(test_x)

        test_df['predicted_class'] = pd.Series(predictions).values

        if print_output:
            self.run_model_metrics(model=self.model, test_x=test_x, test_y=test_y, print_output=print_output)

        test_df.to_csv(self.id + '_class_prediction_df.csv')

    def run_model_metrics(self, model, test_y, test_x,
                          print_output=True, save_output=True):

        if self.trained == False:
            print('Parsed model not trained, Aborting...')

        else:

            test_pred = model.predict(test_x)

            accuracy_test = (accuracy_score(test_y, test_pred))
            balanced_accuracy_test = balanced_accuracy_score(test_y, test_pred)
            precision_test = precision_score(test_y, test_pred, average=None)
            weighted_precision_test = precision_score(test_y, test_pred, average='weighted')
            recall_score_test = recall_score(test_y, test_pred, average=None)
            weighted_recall_score_test = recall_score(test_y, test_pred, average='weighted')
            f1_score_test = f1_score(test_y, test_pred, average=None)
            weighted_f1_score_test = f1_score(test_y, test_pred, average='weighted')

            if print_output:
                print('accuracy_test: ', accuracy_test)
                print('balanced_accuracy_test: ', balanced_accuracy_test)
                print('weighted_precision_test: ', weighted_precision_test)
                print('weighted_recall_score_test: ', weighted_recall_score_test)
                print('weighted_f1_score_test: ', weighted_f1_score_test)
                print('precision:', precision_test)
                print('recall:', recall_score_test)

            if save_output:
                with open(self.id + '_classifier_summary.txt', 'w') as f:
                    print('atomic_rep:', self.mol_rep, file=f)
                    print('model params:', model, file=f)
                    #print('rep_params:', self.optimised_rep_hyperparams, file=f)

                    print('\n', file=f)
                    print('accuracy_test: ', accuracy_test, file=f)
                    print('balanced_accuracy_test: ', balanced_accuracy_test, file=f)
                    print('precision_test: ', precision_test, file=f)
                    print('weighted_precision_test: ', weighted_precision_test, file=f)
                    print('recall_score_test: ', recall_score_test, file=f)
                    print('weighted_recall_score_test: ', weighted_precision_test, file=f)
                    print('f1_score_test: ', f1_score_test, file=f)
                    print('weighted_f1_score_test: ', weighted_f1_score_test, file=f)

            #return MAE_test, MAE_train, r2_test, r2_train, RMSE_test, RMSE_train, MAXE_test, MAXE_test


    def predict_from_load(self):

        train_x, train_y = self.data_setup(type='train')
        test_x, test_y = self.data_setup(type='test')
        self.run_model_metrics(model=self.model, train_y=train_y, train_x=train_x, test_y=test_y, test_x=test_x,
                               save_output=True)

    def save_model(self, model, outname):

        to_save = {'model': model,
                   'training_Set': self.trainids,
                   'validation_set': self.valids,
                    'test_set': self.testids}

        pickle.dump(to_save, open(outname, 'wb'))

    def load_model(self, model_filename):

        loaded = pickle.load(open(model_filename, 'rb'))
        self.trained = True
        self.model = loaded['model']
        self.testids = loaded['test_set']
        self.valids = loaded['validation_set']
        self.trainids = loaded['training_Set']

        #self.split_class_df(self.trainids, self.valids, self.testids)

        return loaded

    def optimise_RF(self, train_x, train_y):

        print('Optimising RF Model Hyperparameters...')
        opt = BayesSearchCV(RandomForestClassifier(bootstrap=True, random_state=42, max_features=None),
                            {
                             'max_leaf_nodes': [2, 4, 6, 8, 16, None],
                             'n_estimators': [2, 4, 8, 16, 32, 64, 128, 256, 512],
                             }, n_iter=50, cv=5)

        opt.fit(train_x, train_y)

        #print("best params: %s" % str(opt.best_params_))

        self.optimised_model_hyperparams = opt.best_params_
        self.optimised = True

        return opt.best_params_
