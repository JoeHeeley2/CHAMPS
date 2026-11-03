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
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

import numpy as np

import copy

def setup_gaussian(param_ranges, kappa, xi):

	pbounds = {}
	for param in param_ranges.keys():
		pbounds[param] = (0, 1)

	#print(pbounds)

	optimizer = BayesianOptimization(
		f=None,
		pbounds=pbounds,
		verbose=0, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
		random_state=None
	)
	utility = UtilityFunction(kind="ucb", kappa=kappa, xi=xi)

	return optimizer, utility


# Bug, don't know where else to put this,
# with scikit-learn != 0.22.x it will complain about NaN values in an array when running opt.suggest(util)
# This is due to scikit-learn trying to normalise the data and so dividing y values by the std of y, which
# when there is only 1 y value in the optmiser creates a divide by zero, and hence the NaN
# downgrading to 0.22.x works
def gaussian_iteration(opt, util, model, tr_atom, tr_pair, rep_func, param_ranges={}, cv_range=5):

	next_point_to_probe = opt.suggest(util)
	params = {}
	for param in param_ranges.keys():
		diff = param_ranges[param]['max'] - param_ranges[param]['min']
		params[param] = (next_point_to_probe[param]*diff) + param_ranges[param]['min']
		
		if param_ranges[param]['log']:
			params[param] = 10**params[param] 

	tr_atom = rep_func(tr_atom, copy.copy(params))
	model.params = copy.copy(params)
	model.check_params()

	cv_scores = []
	train_scores = []
	for cv in range(cv_range):
		test_x, test_y = model.get_input(tr_atom, tr_pair, cv_chunks=[cv], cv_range=cv_range)
		train_x, train_y = model.get_input(tr_atom, tr_pair, cv_chunks=[x for x in range(cv_range) if x!=cv], cv_range=cv_range)
		model.train_x = train_x
		model.train_y = train_y
		model.train()
		cv_scores.append(model.mae(test_x, test_y))
		train_scores.append(model.mae(train_x, train_y))

	cv_score = np.mean(cv_scores)
	train_score = np.mean(train_scores)
	if cv_score > 1000 or np.isnan(cv_score):
		cv_score = np.random.rand() + 9999.9
	opt.register(params=next_point_to_probe, target=-cv_score)

	return cv_score, train_score, params, opt

def gaussian_iteration_validation(opt, util, model, tr_atom, tr_pair, valid_atom, valid_pair,
								  rep_func, param_ranges={}, cv_range=5):

	next_point_to_probe = opt.suggest(util)
	params = {}

	for param in param_ranges.keys():
		diff = param_ranges[param]['max'] - param_ranges[param]['min']
		params[param] = (next_point_to_probe[param] * diff) + param_ranges[param]['min']

		if param_ranges[param]['log']:
			params[param] = 10 ** params[param]

	tr_atom = rep_func(tr_atom, copy.copy(params))
	val_atom = rep_func(valid_atom, copy.copy(params))
	model.params = copy.copy(params)
	model.check_params()

	train_x, train_y = model.get_input(tr_atom, tr_pair)
	test_x, test_y = model.get_input(valid_atom, valid_pair)
	model.train_x = train_x
	model.train_y = train_y
	model.train()
	validation_score = model.mae(test_x, test_y)
	training_score = model.mae(train_x, train_y)

	opt.register(params=next_point_to_probe, target=-validation_score)

	return validation_score, training_score, params, opt

def gaussian_iteration_for_lmtr(opt, util, param_ranges={}):

	next_point_to_probe = opt.suggest(util)
	params = {}
	for param in param_ranges.keys():
		diff = param_ranges[param]['max'] - param_ranges[param]['min']
		params[param] = (next_point_to_probe[param] * diff) + param_ranges[param]['min']

	return params, next_point_to_probe
