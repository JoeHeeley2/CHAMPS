# Copyright 2020 Will Gerrard
#This file is part of autoenrich.

#autoenrich is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#autoenrich is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with autoenrich.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import qml.fchl

def add_fchl_to_atom_df(atom_df, params={}):

    if 'cutoff' not in params.keys():
        params['cutoff'] = 5.0
    if 'max_size' not in params.keys():
        params['max_size'] = 150

    atom_df['atomic_rep'] = np.empty((len(atom_df), 0)).tolist()

    for molid in atom_df['molecule_name'].unique():

        xyz = atom_df.loc[(atom_df.molecule_name == molid)][['x', 'y', 'z']].to_numpy()
        types = atom_df.loc[(atom_df.molecule_name == molid)][['typeint']].to_numpy().flatten()

        reps = qml.fchl.generate_representation(np.asarray(xyz), np.asarray(types), max_size=params['max_size'], cut_distance=params['cutoff'])

        for i in range(len(types)):
            idx = atom_df.loc[(atom_df['molecule_name'] == molid) & (atom_df['atom_index'] == i)].index[0]
            atom_df.at[idx, 'atomic_rep'] = np.asarray(reps[i])

    return atom_df
