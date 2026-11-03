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


def make_grid(size=3, param_ranges={}):

	grid = {}
	param_lists = {}
	for param in param_ranges.keys():
		if param_ranges[param]['log']:
			param_lists[param] = np.geomspace(param_ranges[param]['min'], param_ranges[param]['max'], size)
		else:
			param_lists[param] = np.linspace(param_ranges[param]['min'], param_ranges[param]['max'], size)
		grid[param] = np.zeros(size**len(param_ranges.keys()))

	for i in range(size ** len(param_ranges.keys())):
		ii = 0
		for param in param_ranges.keys():
			idx = int(i / size ** ii) % size
			grid[param][i] = param_lists[param][idx]
			ii += 1

	return grid


def grid_iteration(model, tr_atom, tr_pair, rep_func, grid, iter, cv_range=5):
	params = {}
	for param in grid.keys():
		params[param] = grid[param][iter]

	tr_atom = rep_func(tr_atom, params)
	model.params = params
	scores = []
	for cv in range(cv_range):
		test_x, test_y = model.get_input(tr_atom, tr_pair, cv=[cv])
		train_x, train_y = model.get_input(tr_atom, tr_pair, cv=[x for x in range(cv_range) if x!=cv])
		model.train_x = train_x
		model.train_y = train_y
		model.train()
		scores.append(model.evaluate(test_x, test_y))

	return np.mean(scores), params
