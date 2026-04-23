import numpy as np
from scipy.sparse import csr_matrix, diags

import pickle
from tqdm import tqdm

from implicit.gpu import bpr, als
from implicit.evaluation import ranking_metrics_at_k

from loaders import movielens, toy
import utils

import argparse

'''
Write a util function that provides the BPR loss. 
* plot the training and validation loss curves
* plot the validation ranking metrics curves.

Given a label group, create a base set of users. 

Create an augmentation group of the same size as the base group.
The proportion of each group is specified by a random vector p.
For a group i, sample with replacement n * p_i users from gorup i for the augmentation set

Calculate the validation loss and validation ranking metrics for the base group

Store output as: 
[
(p, {metric: value}) * n_trials=1000
]

Assess regression fit in analysis. 
* Q1: how well does a regression fit the validation loss?
* Q2: how well does the validaiton loss correlate with the validation ranking metrics?
* Q3: how well does the validation loss generalize to the test loss?

'''

###### CONFIGS ######


EXP_NAME = "default"

UPDATE_P_FREQUENCY = 50

BASE_SIZE = -1 # size of the base group. if -1, the entire group is included.
AUGMENTATION_SIZE = -1 # size of the augmentation group. if -1, size = n - BASE_SIZE
AUGMENTATION_SAMPLE_WITH_REPLACEMENT = True
AUGMENTATION_TRIALS = 100
MODEL_NAME = "" # als or bpr

N_AUG = -1
FILTER_GROUPS = False
TEST_ALL = False

SEED = 0
MIN_USERS = 200

def _parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--exp_name",
		type=str,
	)

	parser.add_argument(
		"--filter_groups",
		type=int
	)

	parser.add_argument(
		"--model_name",
		type=str
	)

	parser.add_argument(
		"--base_size",
		type=int,
	)

	parser.add_argument(
		"--aug_size",
		type=int,
	)

	parser.add_argument(
		"--n_aug",
		type=int
	)

	parser.add_argument(
		"--aug_w_replacement",
		type=bool,
	)

	parser.add_argument(
		"--aug_trials",
		type=int,
	)

	parser.add_argument(
		"--test_all_p",
		type=int
	)

	parser.add_argument(
		"--min_users",
		type=int,
		default=200,
	)

	global EXP_NAME, FILTER_GROUPS, MODEL_NAME, BASE_SIZE, AUGMENTATION_SIZE, N_AUG, AUGMENTATION_SAMPLE_WITH_REPLACEMENT, AUGMENTATION_TRIALS, TEST_ALL, MIN_USERS
	args = parser.parse_args()
	EXP_NAME = args.exp_name
	BASE_SIZE = args.base_size
	AUGMENTATION_SIZE = args.aug_size
	AUGMENTATION_SAMPLE_WITH_REPLACEMENT = args.aug_w_replacement > 0
	AUGMENTATION_TRIALS = args.aug_trials
	MIN_USERS = args.min_users

	assert args.model_name in ["bpr", "als"]
	MODEL_NAME = args.model_name

	if args.n_aug == -1:
		N_AUG = AUGMENTATION_SIZE
	else:
		N_AUG = args.n_aug

	FILTER_GROUPS = args.filter_groups > 0
	TEST_ALL = args.test_all_p > 0

def get_als_user_coefs(label_to_idxs, base_label, p):
	'''
	returns a list of ALS coefficients for each user in the mixed training dataset. the coefficient value for a 
	user is the expected number of occurances of the user in the mixed dataset.
	'''
	coefs_by_group = []

	# base group
	coefs_by_group.append(np.ones(len(label_to_idxs[base_label])))

	n = np.sum([len(idxs) for idxs in label_to_idxs.values()])
	base_size = len(label_to_idxs[base_label])

	n_aug = N_AUG if N_AUG > 0 else n - base_size

	for label, idxs in label_to_idxs.items():
		if p[label] == 0:
			continue

		n_g = len(idxs)
		p_g = p[label]
		group_coef = (n_aug * p_g) / n_g
		coefs_by_group.append(
			group_coef * np.ones(n_g)
			)

	return np.concatenate(coefs_by_group)

def construct_mixed_dataset(label_to_idxs, base_label, p, aug_seed=0):
	'''
	Construct a single mixed dataset. The samples for the base block are fixed. 
	The samples for the augmentation block depend on the input seed.
	Inputs
		label_to_idxs maps group labels to lists of row indices 
		base_label is the base group label
		p is a mixing vector mapping groups to mixing ratios. sum(p) == 1.
	Return
		The mixed data matrix
	'''

	rng = np.random.default_rng(seed=SEED)
	reordered_idxs = []
	n = np.sum([len(idxs) for idxs in label_to_idxs.values()])

	# base group
	base_size = BASE_SIZE if BASE_SIZE > 0 else len(label_to_idxs[base_label])
	base_idxs = rng.choice(
		label_to_idxs[base_label],
		size=base_size,
		replace=False)
	reordered_idxs.extend(base_idxs)

	# augmentation groups
	rng = np.random.default_rng(seed=aug_seed)
	aug_size = AUGMENTATION_SIZE if AUGMENTATION_SIZE > 0 else n - base_size

	for label, idxs in label_to_idxs.items():
		if p[label] == 0:
			continue

		idxs = np.array(idxs)
		num_idxs = int(aug_size * p[label])

		if MODEL_NAME == "als":
			num_idxs = len(idxs)

		if not AUGMENTATION_SAMPLE_WITH_REPLACEMENT:
			if label == base_label:
				idxs = idxs[~np.isin(idxs, np.fromiter(base_idxs, dtype=idxs.dtype))]
				assert num_idxs <= len(idxs), "Not enough remaining base indices for augmentation"
			else:
				assert num_idxs <= len(idxs)

		reordered_idxs.extend(rng.choice(
			idxs,
			size=num_idxs,
			replace=AUGMENTATION_SAMPLE_WITH_REPLACEMENT))

	if np.any([ratio > 0 for ratio in p.values()]):
		assert abs(base_size + aug_size - len(reordered_idxs)) <= len(label_to_idxs)

	return reordered_idxs

def get_base_group_metrics(X_train_mixed, X_test_mixed, base_size, r, seed=0):
	'''
	Train a BPR model on the mixed training data and then evaluate on the test data for the base group.
	'''
	ranking_model = None

	if MODEL_NAME == "als":
		ranking_model = als.AlternatingLeastSquares(factors=r, iterations=100, random_state=seed)
	elif MODEL_NAME == "bpr":
		ranking_model = bpr.BayesianPersonalizedRanking(factors=r, iterations=100, random_state=seed)
	
	ranking_model.fit(X_train_mixed, show_progress=False)

	ranking_metrics = ranking_metrics_at_k(ranking_model, 
										   X_train_mixed, 
										   X_test_mixed[:base_size], 
										   K=20,
										   show_progress=False)

	if MODEL_NAME == "bpr":
		ranking_metrics["test loss"] = utils.get_bpr_loss(ranking_model, 
			X_train_mixed[:base_size],
			X_test_mixed[:base_size])
	elif MODEL_NAME == "als":
		ranking_metrics["test loss"] = utils.get_als_loss(ranking_model, 
			X_train_mixed[:base_size],
			X_test_mixed[:base_size])

	return ranking_metrics

def normalize_p(p):
	total_weight =  np.sum(list(p.values()))
	if total_weight > 0:
		for label in p:
			p[label] /= total_weight

def random_data_mixing(X_train, X_val, X_test, label_to_idxs, base_label, r=64, trials=100):
	n, m = X_train.shape
	g = len(label_to_idxs)
	base_size = BASE_SIZE if BASE_SIZE > 0 else len(label_to_idxs[base_label])

	rng = np.random.default_rng(seed=SEED)
	
	val_ps, test_ps = [], []
	val_metrics, test_metrics = [], []

	best_p = {}
	highest_validation_metric = -np.inf

	# random mix
	dirichlet = rng.dirichlet(alpha=0.5 * np.ones(g))
	p = {
		label: dirichlet[idx] if label != base_label else 0 
		for idx, label in enumerate(label_to_idxs.keys())
	}
	normalize_p(p)

	# validation trials
	for trial_num in tqdm(range(trials)):

		if trial_num == 3 or trial_num % UPDATE_P_FREQUENCY == 0:
			# random mix
			dirichlet = rng.dirichlet(alpha=0.5 * np.ones(g))
			p = {
				label: dirichlet[idx] if label != base_label else 0 
				for idx, label in enumerate(label_to_idxs.keys())
			}
			normalize_p(p)

		if trial_num == 0:
			p = {label: 0.0 for label in label_to_idxs}
		elif trial_num == 1:
			p = {label: len(idxs) if label != base_label else 0 for label, idxs in label_to_idxs.items()}
			normalize_p(p)
		elif trial_num == 2:
			p = {label: 1 if label != base_label else 0 for label in label_to_idxs}
			normalize_p(p)

		if TEST_ALL:
			test_ps.append(p.copy())
		else:
			if trial_num <= 2:
				test_ps.append(p.copy())
				
		reordered_idxs = construct_mixed_dataset(
			label_to_idxs,
			base_label,
			p,
			aug_seed=trial_num) 

		X_train_mixed = X_train[reordered_idxs]
		X_val_mixed = X_val[reordered_idxs]

		###### scale values if neded 

		user_coefs = get_als_user_coefs(label_to_idxs, base_label, p)
		# print(f"num coefs: {len(user_coefs)} num users: {len(reordered_idxs)}")

		X_train_mixed = diags(user_coefs) @ X_train_mixed

		val_ps.append(p.copy()) #verify if this copy is needed
		ranking_metrics = get_base_group_metrics(X_train_mixed, X_val_mixed, base_size, r, seed=trial_num)
		val_metrics.append(ranking_metrics)

		if ranking_metrics["precision"] > highest_validation_metric:
			highest_validation_metric = ranking_metrics["precision"]
			best_p = p.copy()

	test_ps.append(best_p.copy())
	for p in test_ps:
		reordered_idxs = construct_mixed_dataset(
			label_to_idxs,
			base_label,
			p,
			aug_seed=0) 

		X_train_mixed = (X_train + X_val)[reordered_idxs]
		X_test_mixed = X_test[reordered_idxs]

		###### scale values if neded 
		user_coefs = get_als_user_coefs(label_to_idxs, base_label, p)
		X_train_mixed = diags(user_coefs) @ X_train_mixed

		ranking_metrics = get_base_group_metrics(X_train_mixed, X_test_mixed, base_size, r, seed=SEED)
		test_metrics.append(ranking_metrics)		

	return val_ps, val_metrics, test_ps, test_metrics

if __name__ == "__main__":

	_parse_args()

	# load data
	movielens_obj = movielens.movielens(
		min_ratings = 1,
		min_users = MIN_USERS,
		binary=True, 
		data_dir="data/")

	label_to_idxs = movielens_obj.get_user_labels("Age")

	if FILTER_GROUPS:
		label_to_idxs = {label: idxs for label, idxs in label_to_idxs.items() if label in ["18", "25", "35"]}
	
	# Analyze all groups
	# block_size_by_label = {label: 100 if len(idxs) < 1000 else 200 for label, idxs in label_to_idxs.items()}

	# Analyze two groups
	# label_to_idxs = {label: idxs for label, idxs in label_to_idxs.items() if label in ["18", "25", "35"]}
	# block_size_by_label = {label: 500 for label in label_to_idxs}

	for label, idxs in label_to_idxs.items():
		print(f"{label}: {len(idxs)}")
		
	# create a 70/10/20 split
	X = movielens_obj.get_X()
	X_train, X_test = utils.get_train_test_X(X, p=0.8, seed=SEED)
	X_train, X_val = utils.get_train_test_X(X_train.toarray(), p=7/8, seed=SEED)

	assert np.sum(X > 0) == np.sum(X_train > 0) + np.sum(X_val > 0) + np.sum(X_test > 0)

	mixing_results_by_group = {}
	try:
		with open(F"results/mixing_results_by_group_{EXP_NAME}.pickle", "rb") as pickleFile:
			mixing_results_by_group = pickle.load(pickleFile)
	except:
		print("no existing pickle file")

	# loop over groups
	for label_idx, label in enumerate(label_to_idxs):
		
		if label not in ["1", "45"]:
			continue
				
		val_ps, val_metrics, test_ps, test_metrics = random_data_mixing(
			X_train,
			X_val,
			X_test,
			label_to_idxs,
			base_label=label,
			r=64,
			trials=AUGMENTATION_TRIALS)
		mixing_results_by_group[label] = {
			"validation ps": val_ps, 
			"validation metrics": val_metrics, 
			"test ps": test_ps,
			"test metrics": test_metrics
		}
		with open(F"results/mixing_results_by_group_{EXP_NAME}.pickle", "wb") as pickleFile:
			pickle.dump(mixing_results_by_group, pickleFile)
