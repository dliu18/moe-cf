import numpy as np
from scipy.sparse import csr_matrix

import pickle
from tqdm import tqdm

from implicit.gpu import bpr
from implicit.evaluation import ranking_metrics_at_k

from loaders import movielens, toy
import utils


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

Note: for now rows cannot be sampled twice. Each new data point is unique

Assess regression fit in analysis. 
* Q1: how well does a regression fit the validation loss?
* Q2: how well does the validaiton loss correlate with the validation ranking metrics?
* Q3: how well does the validation loss generalize to the test loss?


'''
SEED = 0

def construct_mixed_dataset(label_to_idxs, base_label, p, block_size, seed=0):
	'''
	Construct a single mixed dataset. 
	Inputs
		label_to_idxs maps group labels to lists of row indices 
		base_label is the base group label
		p is a mixing vector mapping groups to mixing ratios. sum(p) == 1.
		block size is the size of the base group and the augmentation group.
	Return
		The mixed data matrix
	'''

	rng = np.random.default_rng(seed=seed)
	reordered_idxs = []

	# base group
	num_base_idxs = block_size + int(block_size * p[base_label])
	assert num_base_idxs <= len(label_to_idxs[base_label])
	reordered_idxs.extend(rng.choice(
		label_to_idxs[base_label],
		size=num_base_idxs,
		replace=False))

	# augmentation groups
	for label, idxs in label_to_idxs.items():
		if label == base_label:
			continue

		num_idxs = int(block_size * p[label])
		assert num_idxs <= len(label_to_idxs[label])
		reordered_idxs.extend(rng.choice(
			label_to_idxs[label],
			size=num_idxs,
			replace=False))

	assert abs(len(reordered_idxs) - 2 * block_size) <= len(label_to_idxs)
	return reordered_idxs

def random_data_mixing(X_train, X_test, label_to_idxs, base_label, block_size, r=64, trials=100, seed=0):
	n, m = X_train.shape
	rng = np.random.default_rng(seed=seed)
	
	ps = []
	metrics = []

	# uniform mixing to start
	p = {label: 1 for label in label_to_idxs}
	total_weight =  np.sum(list(p.values()))
	for label in p:
		p[label] /= total_weight
	
	for trial_num in tqdm(range(trials)):
		reordered_idxs = construct_mixed_dataset(
			label_to_idxs,
			base_label,
			p,
			block_size,
			seed=trial_num) 

		X_train_mixed = X_train[reordered_idxs]

		bpr_model = bpr.BayesianPersonalizedRanking(factors=r, iterations=100)
		bpr_model.fit(X_train_mixed, show_progress=False)

		ranking_metrics = ranking_metrics_at_k(bpr_model, 
											   X_train_mixed, 
											   X_test[reordered_idxs[:block_size]], 
											   K=20,
											   show_progress=False)
		ranking_metrics["test loss"] = utils.get_bpr_loss(bpr_model, 
			X_train_mixed[:block_size],
			X_test[reordered_idxs][:block_size])

		ps.append(p.copy()) #verify if this copy is needed
		metrics.append(ranking_metrics)

		p = {label: rng.random() for label in label_to_idxs}
		total_weight =  np.sum(list(p.values()))
		for label in p:
			p[label] /= total_weight

	return ps, metrics

if __name__ == "__main__":

	# load data
	movielens_obj = movielens.movielens(
		min_ratings = 1,
		min_users = 200,
		binary=True, 
		data_dir="data/")

	label_to_idxs = movielens_obj.get_user_labels("Age")
	block_size_by_label = {label: 100 if len(idxs) < 1000 else 200 for label, idxs in label_to_idxs.items()}

	for label, idxs in label_to_idxs.items():
		print(f"{label}: {len(idxs)}")
		
	# create a 70/10/20 split
	X = movielens_obj.get_X()
	X_train, X_test = utils.get_train_test_X(X, p=0.8, seed=SEED)
	X_train, X_val = utils.get_train_test_X(X_train.toarray(), p=7/8, seed=SEED)

	assert np.sum(X > 0) == np.sum(X_train > 0) + np.sum(X_val > 0) + np.sum(X_test > 0)

	mixing_results_by_group = {}

	# loop over groups
	for label in label_to_idxs:
		ps, metrics = random_data_mixing(
			X_train,
			X_val,
			label_to_idxs,
			base_label=label,
			block_size=block_size_by_label[label],
			r=64,
			trials=1000,
			seed=SEED)
		mixing_results_by_group[label] = {"ps": ps, "metrics": metrics}

		with open("results/mixing_results_by_group.pickle", "wb") as pickleFile:
			pickle.dump(mixing_results_by_group, pickleFile)

