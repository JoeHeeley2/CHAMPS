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
import copy

def random_iteration(model, tr_atom, tr_pair, rep_func, param_ranges={}, cv_range=5):
	params = {}
	for param in param_ranges.keys():
		diff = param_ranges[param]['max'] - param_ranges[param]['min']
		if param_ranges[param]['log']:
			params[param] = 10**(np.random.rand()*diff)+param_ranges[param]['min']
		else:
			params[param] = (np.random.rand()*diff)+param_ranges[param]['min']

	tr_atom = rep_func(tr_atom, copy.copy(params))
	model.params = copy.copy(params)
	model.check_params()
	scores = []
	for cv in range(cv_range):
		test_x, test_y = model.get_input(tr_atom, tr_pair, cv_chunks=[cv], cv_range=cv_range)
		train_x, train_y = model.get_input(tr_atom, tr_pair, cv_chunks=[x for x in range(cv_range) if x!=cv], cv_range=cv_range)
		model.train_x = train_x
		model.train_y = train_y
		model.train()
		scores.append(model.evaluate(test_x, test_y))

	return np.mean(scores), params
