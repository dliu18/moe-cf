import numpy as np
from scipy.sparse import csr_matrix, diags

import matplotlib.pyplot as plt

import pickle
from tqdm import tqdm

from implicit.gpu import bpr, als
from implicit.evaluation import ranking_metrics_at_k, precision_at_k

import utils

import time

def create_X(n, r, m, k, p, q, seed=0):
	'''
	Output matrix is n x (mk)
	'''

	X = np.zeros((n, m * k))
	current_group = 0
	rng = np.random.default_rng(seed=seed)

	for i in range(n):

		if i > int(n * np.sum(r[:current_group + 1])):
			current_group += 1

		for j in range(m * k):

			prob = q
			if j >= current_group * m and j < (current_group + 1) * m:
				prob = p

			X[i,j] = rng.binomial(n=1, p=prob)

	return X 

if __name__ == "__main__":

	# # Unit test for create_X
	# X = create_X(
	# 	n=500,
	# 	r = np.ones(5) / 5,
	# 	m = 100, 
	# 	k = 5,
	# 	p = 0.1,
	# 	q=0.01,
	# 	seed=0)
	# plt.imshow(X, cmap="binary")
	# plt.savefig("debug.pdf", bbox_inches="tight")

	ALPHA_STEP = 0.05

	n_0 = 100
	aug_ratio = 5

	m = 100
	k = 3

	p = 0.02
	q = 0.01

	r_0 = np.array([1/2, 1/3, 1/6])
	r_1 = r_0
	r_2 = np.array([1/6, 1/3, 1/2])

	num_trials = 125
	lams = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0])

	best_alphas = np.zeros((len(lams), num_trials))

	for lam_idx, lam in enumerate(lams):

		for seed in tqdm(range(num_trials)):

			start = time.time()
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

			# print(f"Dataset creation time: {round(time.time() - start, 3)}")

			# Loop over alphas
			best_alpha = -1
			highest_precision = -1 
			for alpha in np.arange(0, 1.0, ALPHA_STEP):
				start = time.time()

				X_train_mixed = np.concatenate((
					(1-lam) * X_0,
					lam * alpha * X_1,
					lam * (1 - alpha) * X_2
					))
				X_train_mixed = csr_matrix(X_train_mixed)
				# print(f"Mixing time: {round(time.time() - start, 3)}")

				# # Debug iterations needed
				# if alpha == 0.5:
				# 	training_losses = []
				# 	for iterations in range(1, 30):
				# 		ranking_model = als.AlternatingLeastSquares(factors=3, iterations=iterations, random_state=0)
				# 		ranking_model.fit(X_train_mixed, show_progress=False)
				# 		loss = utils.get_als_loss(ranking_model,
				# 			csr_matrix(X_train_mixed.shape),
				# 			X_train_mixed)
				# 		training_losses.append(loss)

				# 	fig, ax = plt.subplots()

				# 	ax.plot(range(1, 30), training_losses)
				# 	ax.set_xlabel("Iterations")
				# 	ax.set_ylabel("Training Loss")

				# 	fig.savefig("plots/unit_test_loss.pdf", bbox_inches="tight")

				start = time.time()
				ranking_model = als.AlternatingLeastSquares(factors=3, iterations=15, random_state=0)
				ranking_model.fit(X_train_mixed, show_progress=False)
				# print(f"Fitting time: {round(time.time() - start, 3)}")

				start = time.time()
				precision = precision_at_k(ranking_model, 
												X_train_mixed, 
												X_0_true, 
												K=m,
												show_progress=False,
												num_threads=1)
				# print(f"Eval time: {round(time.time() - start, 3)}")

				if precision > highest_precision:
					best_alpha = alpha 
					highest_precision = precision
			# Save best alpha
			best_alphas[lam_idx, seed] = best_alpha


		fig, ax = plt.subplots()

		mean_alpha = np.mean(best_alphas, axis=1)
		std_alpha  = np.std(best_alphas, axis=1)

		ax.plot(lams, mean_alpha)

		stderr = std_alpha / np.sqrt(num_trials)

		ax.fill_between(
		    lams,
		    mean_alpha - stderr,
		    mean_alpha + stderr,
		    alpha=0.3
		)

		ax.set_xscale("log")

		ax.set_xlabel("lambda")
		ax.set_ylabel("Best Alpha")

		fig.savefig("plots/size_dependent_ratio.pdf", bbox_inches="tight")






