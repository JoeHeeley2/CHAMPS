# Copyright 2020 Will Gerrard
# This file is part of autoenrich.

# autoenrich is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# autoenrich is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with autoenrich.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
import pickle

import qml.fchl
import qml.math


class FCHLmodel(object):

    def __init__(self, id='krrmodel', model_args={}, model_params={}):
        '''
            model_args is anything needed to run the model, such as number of
            layers in a nn
        '''
        self.id = id
        self.args = model_args
        self.params = model_params
        self.trained = False

        self.alpha = []

        self.train_x = []
        self.train_y = []

        self.check_params()
        self.check_args()

    def check_args(self):

        if 'targetflag' not in self.args:
            print("No target supplied, using ['HCS']")
            self.args['targetflag'] = ['HCS']

        set_len = len(self.args['targetflag'][0])

        for targetflag in self.args['targetflag']:
            assert len(targetflag) == set_len, print('Can only predict atom or pair properties, not both in one model')

    def check_params(self):
        '''
            This function should go through the self.args dictionary and check
            whether all required arguments are present. If not a default value
            should be provided
        '''

        if not 'sigma' in self.params.keys():
            self.params['sigma'] = 0.1
        if not 'lamda' in self.params.keys():
            self.params['lamda'] = 0.0001
        if not 'cutoff' in self.params.keys():
            self.params['cutoff'] = 5.0
        if not 'two_body_scaling' in self.params:
            self.params['two_body_scaling'] = np.sqrt(8)
        if not 'three_body_scaling' in self.params:
            self.params['three_body_scaling'] = 1.6
        if not 'two_body_width' in self.params:
            self.params['two_body_width'] = 0.2
        if not 'three_body_width' in self.params:
            self.params['three_body_width'] = np.pi
        if not 'two_body_power' in self.params:
            self.params['two_body_power'] = 4.0
        if not 'three_body_power' in self.params:
            self.params['three_body_power'] = 2.0
        if not 'cut_start' in self.params:
            self.params['cut_start'] = 1.0
        if not 'alchemy_period_width' in self.params:
            self.params['alchemy_period_width'] = 1.6
        if not 'alchemy_group_width' in self.params:
            self.params['alchemy_group_width'] = 1.6

    def get_input(self, atom_df, pair_df, cv_chunks=[], cv_range=5):
        '''
            This function should obtain the direct input to train/predict from
            the dataframes produced by function in "prep"
        '''

        '''
            Not overly happy with this method of pulling stuff out, its messy
            but it does work
        '''

        if len(self.args['targetflag'][0]) == 3:
            # Get atomic reps for all rows where the typestr matches the first character in any of the targetflags
            x = [*atom_df.loc[
                [any(type == targetflag[0] for targetflag in self.args['targetflag']) for type in atom_df['typestr']]][
                'atomic_rep']]
            x = np.asarray(x)
            x = x.reshape((1, x.shape[0], x[0].shape[0], x[0].shape[1]))
            # Get shifts for all rows where the typestr matches the first character in any of the targetflags
            y = atom_df.loc[
                [any(type == targetflag[0] for targetflag in self.args['targetflag']) for type in atom_df['typestr']]][
                'shift'].to_numpy().squeeze()

        elif len(self.args['targetflag'][0]) == 4:
            # Get coupling for all rows where the targetflag matches any of the targetflags (just using self.args['targetflag'] doesnt work)
            y = pair_df.loc[
                [any(type == targetflag for targetflag in self.args['targetflag']) for type in pair_df['nmr_types']]][
                ['coupling']].to_numpy().squeeze()
            # Get atom pairs for all rows where the targetflag matches any of the targetflags
            atoms = pair_df.loc[
                [any(type == targetflag for targetflag in self.args['targetflag']) for type in pair_df['nmr_types']]][
                ['molecule_name', 'atom_index_0', 'atom_index_1']].to_numpy()
            # For above atom pairs, get two lists of atomic reps, one for indexes from atom_index_0, another for indexes from atom_index_1
            x1 = []
            x2 = []
            for molname, atom0, atom1 in atoms:
                x1.append(atom_df.loc[(atom_df['molecule_name'] == molname) & (atom_df['atom_index'] == atom0)][
                              'atomic_rep'].to_numpy()[0])
                x2.append(atom_df.loc[(atom_df['molecule_name'] == molname) & (atom_df['atom_index'] == atom0)][
                              'atomic_rep'].to_numpy()[0])
            x1 = np.asarray(x1)
            x2 = np.asarray(x2)
            x = np.asarray([x1, x2])

        assert x.shape[1] == y.shape[0], print(x.shape, y.shape)

        if len(cv_chunks) != 0:
            chunk_size = int(len(y) / cv_range)
            idx = []
            for chunk in range(cv_range):
                if chunk in cv_chunks:
                    idx.extend(range(chunk, chunk + chunk_size))
            x = x[:, idx]
            y = y[idx]
        return x, y

    def assign_preds(self, pred_y, atom_df=pd.DataFrame(), pair_df=pd.DataFrame(), assign_to='predicted_', clear=False,
                     variance=[]):
        '''
            If this matches the way the reps are pulled out of the dataframe above then there's no need to match up indices
        '''
        if len(self.args['targetflag'][0]) == 3:
            if clear or not str(assign_to) + 'shift' in atom_df.keys():
                atom_df[str(assign_to) + 'shift'] = np.zeros(len(atom_df), dtype=np.float64)
            atom_df.loc[[any(type == targetflag[0] for targetflag in self.args['targetflag']) for type in
                         atom_df['typestr']], str(assign_to) + 'shift'] = pred_y
        elif len(self.args['targetflag'][0]) == 4:
            if clear or not str(assign_to) + 'coupling' in pair_df.keys():
                pair_df[str(assign_to) + 'coupling'] = np.zeros(len(pair_df), dtype=np.float64)
            pair_df.loc[[any(type == targetflag for targetflag in self.args['targetflag']) for type in
                         pair_df['nmr_types']], str(assign_to) + 'coupling'] = pred_y

        if len(variance) != 0:
            if len(self.args['targetflag'][0]) == 3:
                if clear or not 'shift_var' in atom_df.keys():
                    atom_df['shift_var'] = np.zeros(len(atom_df), dtype=np.float64)
                atom_df.loc[[any(type == targetflag[0] for targetflag in self.args['targetflag']) for type in
                             atom_df['typestr']], 'shift_var'] = variance
            elif len(self.args['targetflag'][0]) == 4:
                if clear or not 'coupling_var' in pair_df.keys():
                    pair_df['coupling_var'] = np.zeros(len(pair_df), dtype=np.float64)
                pair_df.loc[[any(type == targetflag for targetflag in self.args['targetflag']) for type in
                             pair_df['nmr_types']], 'coupling_var'] = variance

        return atom_df, pair_df

    def train(self, label=None):
        '''
            This function should train the model, the result of the training
            should be the saving of a pickle file called <label>_model.pkl
            and self.trained = True
        '''

        if not label:
            label = self.id

        k = []
        # loop through representation sets (allows for multiple reps for couplings)
        for d in range(self.train_x.shape[0]):
            k.append(
                qml.fchl.get_atomic_symmetric_kernels(self.train_x[d], sigmas=[self.params['sigma']],
                                                      two_body_scaling=self.params['two_body_scaling'],
                                                      three_body_scaling=self.params['three_body_scaling'],
                                                      two_body_width=self.params['two_body_width'],
                                                      three_body_width=self.params['three_body_width'],
                                                      two_body_power=self.params['two_body_power'],
                                                      three_body_power=self.params['three_body_power'],
                                                      cut_start=self.params['cut_start'],
                                                      cut_distance=self.params['cutoff'], fourier_order=1,
                                                      alchemy="periodic-table",
                                                      alchemy_period_width=self.params['alchemy_period_width'],
                                                      alchemy_group_width=self.params['alchemy_group_width'])[0] +
                self.params['lamda'] * np.identity(self.train_x[d].shape[0]))

        K = k[0]
        if len(k) > 1:
            for i in range(1, len(k)):
                K = K * k[i]

        self.alpha = qml.math.cho_solve(K, self.train_y)
        self.trained = True

        outname = label + '_model.pkl'
        self.save_model(outname)

    def predict(self, test_x):
        '''
            This function should make predictions using a trained model
        '''

        assert self.trained == True
        assert test_x.shape[0] == self.train_x.shape[0], print('Bad array shapes test/train', test_x.shape,
                                                               self.train_x.shape)
        assert test_x.shape[2] == self.train_x.shape[2], print('Bad array shapes test/train', test_x.shape,
                                                               self.train_x.shape)

        ks = []
        for d in range(self.train_x.shape[0]):
            ks.append(
                qml.fchl.get_atomic_kernels(test_x[d], self.train_x[d], sigmas=[self.params['sigma']],
                                            two_body_scaling=self.params['two_body_scaling'],
                                            three_body_scaling=self.params['three_body_scaling'],
                                            two_body_width=self.params['two_body_width'],
                                            three_body_width=self.params['three_body_width'],
                                            two_body_power=self.params['two_body_power'],
                                            three_body_power=self.params['three_body_power'],
                                            cut_start=self.params['cut_start'], cut_distance=self.params['cutoff'],
                                            fourier_order=1, alchemy="periodic-table",
                                            alchemy_period_width=self.params['alchemy_period_width'],
                                            alchemy_group_width=self.params['alchemy_group_width'])[0])

        Ks = ks[0]
        if len(ks) > 1:
            for i in range(1, len(ks)):
                Ks = Ks * ks[i]

        pred_y = np.dot(Ks, self.alpha)

        return pred_y

    def mae(self, test_x, test_y):
        out = self.predict(test_x)
        return np.mean(np.absolute(test_y-out))

    def error(self, test_x, test_y):
        out = self.predict(test_x)
        return (test_y - out)

    def stdev(self, test_x, test_y):
        out = self.predict(test_x)
        return np.std(test_y - out)

    def save_model(self, outname):

        to_save = {'alpha': self.alpha,
                   'train_x': self.train_x,
                   'train_y': self.train_y,
                   'params': self.params,
                   'args': self.args,
                   'id': self.id,
                   'trained': self.trained}
        pickle.dump(to_save, open(outname, 'wb'))

    def load_model(self, filename):

        loaded = pickle.load(open(filename, 'rb'))

        self.alpha = loaded['alpha']
        self.params = loaded['params']
        self.train_x = loaded['train_x']
        self.train_y = loaded['train_y']
        self.args = loaded['args']
        self.id = loaded['id']
        self.trained = loaded['trained']



