import numpy as np
from scipy.sparse import csr_matrix, diags

import matplotlib.pyplot as plt

import pickle
from tqdm import tqdm

from implicit.gpu import bpr, als
from implicit.evaluation import ranking_metrics_at_k, precision_at_k

import utils

import time

from size_dependent_ratio import create_X


if __name__ == "__main__":

	ALPHA_STEP = 0.1

	n_0 = 100
	aug_ratio = 5

	m = 100
	k = 3

	p = 0.1
	q = 0.01

	r_0 = np.array([1/2, 1/3, 1/6])
	r_1 = r_0
	r_2 = np.array([1/6, 1/3, 1/2])

	num_trials = 200

	lam = 0.1

	alphas = np.arange(0, 1.1, ALPHA_STEP)
	alpha_to_precs = {alpha: [] for alpha in alphas}

	for seed in tqdm(range(num_trials)):

		# Construct X
		X_0_true = create_X(
			n = n_0,
			r = r_0,
			m = m,
			k = k,
			p = 1.0,
			q = 0.0, 
			seed = seed)
		X_0_true = csr_matrix(X_0_true)


		X_0 = create_X(
			n = n_0,
			r = r_0,
			m = m,
			k = k,
			p = p,
			q = q, 
			seed = seed)

		X_1 = create_X(
			n = aug_ratio * n_0,
			r = r_1,
			m = m,
			k = k,
			p = p,
			q = q, 
			seed = seed)

		X_2 = create_X(
			n = aug_ratio * n_0,
			r = r_2,
			m = m,
			k = k,
			p = p,
			q = q, 
			seed = seed)

		for alpha in alphas:

			X_train_mixed = np.concatenate((
				(1-lam) * X_0,
				lam * alpha * X_1,
				lam * (1 - alpha) * X_2
				))
			X_train_mixed = csr_matrix(X_train_mixed)

			ranking_model = als.AlternatingLeastSquares(factors=3, iterations=15, random_state=0)
			ranking_model.fit(X_train_mixed, show_progress=False)

			precision = precision_at_k(ranking_model, 
											X_train_mixed, 
											X_0_true, 
											K=m,
											show_progress=False,
											num_threads=1)

			alpha_to_precs[alpha].append(precision)


	fig, ax = plt.subplots()

	ax.plot(
		[alpha for alpha in alpha_to_precs],
		[np.mean(precs) for precs in alpha_to_precs.values()]
	)

	# for alpha, precs in alpha_to_precs.items():
	# 	ax.scatter(alpha * np.ones(len(precs)), precs, alpha=0.1)


	ax.set_xlabel("alpha")
	ax.set_ylabel("Precision@K")
	ax.set_title(f"Lambda = {lam}")

	fig.savefig("plots/size_dependent_ratio_single_lambda.pdf", bbox_inches="tight")






