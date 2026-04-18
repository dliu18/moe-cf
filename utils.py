
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from implicit.gpu import bpr, als
from implicit.evaluation import ranking_metrics_at_k
from scipy.sparse import csr_matrix

from loaders import movielens, toy

def _sigmoid(X):
    return 1 / (1 + np.exp(-X))

####### DATA SPLITTING
    
def get_train_test_X(X, p, seed=None):
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")

    X = np.asarray(X)
    rng = np.random.default_rng(seed)

    X_train = np.zeros_like(X)
    X_test = np.zeros_like(X)

    n_rows, n_cols = X.shape

    for i in range(n_rows):
        # Indices of non-zero entries in row i
        nz_cols = np.flatnonzero(X[i, :])

        # If the row is all zeros, skip
        if nz_cols.size == 0:
            continue

        # Number of non-zeros to put in train for this row
        n_train = int(np.round(p * nz_cols.size))

        # Randomly permute the non-zero column indices
        perm = rng.permutation(nz_cols)

        train_cols = perm[:n_train]
        test_cols = perm[n_train:]

        # Assign values
        X_train[i, train_cols] = X[i, train_cols]
        X_test[i, test_cols] = X[i, test_cols]

    return csr_matrix(X_train), csr_matrix(X_test)

##### BPR LOSS FUNCTION 

def get_combined_positives(X_train, X_test):
    excluded_items_by_user = {}

    n, _ = X_train.shape
    for i in range(n):
        excluded_item_idxs = X_test[i].nonzero()[1]
        if X_train[i].nnz > 0:
            excluded_item_idxs = np.concatenate((X_train[i].nonzero()[1], excluded_item_idxs))
        excluded_items_by_user[i] = set(excluded_item_idxs)
    return excluded_items_by_user

def get_als_loss(als_model, X_train, X_test):
    '''
    Calculate the l2 reconstruction error on X_test. positive pairs from X_train are excluded.

    X_train and X_test are csr matrices containing interaction data for the first n users in als_model.
    '''
    assert X_train.shape == X_test.shape
    n, m = X_test.shape

    U = als_model.user_factors.to_numpy()
    U = U[:n]

    V = als_model.item_factors.to_numpy()

    X_hat = U @ V.T
    assert X_hat.shape == X_test.shape

    mask = X_train.toarray() == 0
    return np.linalg.norm(mask * (X_hat - X_test.toarray()))

def get_bpr_loss(bpr_model, X_train, X_test, 
    excluded_items_by_user=None,
    k=5, seed=0, sampling_ratio=1.0):
    '''
    Evaluate the non-regularized bpr loss using the current user and item embeddings in bpr_model, 
    which is an implicit.gpu.bpr.BayesianPersonalizedRanking object.
    
    X_train and X_test are n x m user-item matrices and sparse csr_matrix
    non-zero entries in X_train are excluded as negatives for evaluation
    k is a parameter controlling the number of negatives to sample per positive.
    '''

    U = bpr_model.user_factors.to_numpy()
    V = bpr_model.item_factors.to_numpy()

    # get positive indices from X_test
    n, m = X_train.shape
    rng = np.random.default_rng(seed=seed)

    user_idx, item_idx = X_test.nonzero()
    num_pairs = 0
    loss = 0.0

    if excluded_items_by_user is None:
        excluded_items_by_user = get_combined_positives(X_train, X_test)


    for i, j in zip(user_idx, item_idx):
        if rng.random() > sampling_ratio:
            continue

        pos = np.dot(U[i], V[j])

        num_neg_processed = 0
        while num_neg_processed < k:
            j_prime = rng.integers(m)
            if j_prime in excluded_items_by_user[i]:
                continue

            neg = np.dot(U[i], V[j_prime])
            loss += -np.log(_sigmoid(pos - neg))
            num_neg_processed += 1
        num_pairs += k

    return loss / num_pairs


if __name__ == "__main__":
    movielens_obj = movielens.movielens(
        min_ratings = 1,
        min_users = 200,
        binary=True, 
        data_dir="data/")

    X = movielens_obj.get_X()
    X_train, X_test = get_train_test_X(X, p=0.8)
    X_train, X_val = get_train_test_X(X_train.toarray(), p=7/8)

    assert np.sum(X > 0) == np.sum(X_train > 0) + np.sum(X_val > 0) + np.sum(X_test > 0)

    excluded_items_by_user = get_combined_positives(X_train + X_val, X_test)

    iters = 20
    training_loss_values = []
    test_loss_values = []
    model_name = "als"

    for iter_num in tqdm(range(1, iters)):
        ranking_model = None

        if model_name == "bpr":
            ranking_model = bpr.BayesianPersonalizedRanking(factors=64, iterations=iter_num, random_state=0, regularization=0.0)
        elif model_name == "als":
            ranking_model = als.AlternatingLeastSquares(factors=32, iterations=iter_num, random_state=0, regularization=0.001)

        ranking_model.fit(X_train + X_val, show_progress=False)

        if model_name == "bpr":
            training_loss_values.append(
                get_bpr_loss(ranking_model, 
                    csr_matrix(X_train.shape), X_train + X_val, 
                    seed=iter_num, sampling_ratio=0.2))
            training_loss_values.append(
                get_bpr_loss(ranking_model, 
                    X_train + X_val, X_test,
                    seed=iter_num, sampling_ratio=0.2))
        elif model_name == "als":
            training_loss_values.append(
                get_als_loss(ranking_model, 
                    csr_matrix(X_train.shape), X_train + X_val))
            test_loss_values.append(
                get_als_loss(ranking_model, 
                    X_train + X_val, X_test))
        # loss_values.append(
        #   get_bpr_loss(bpr_model, X_train + X_val, X_test, seed=iter_num))

    training_loss_values = np.array(training_loss_values)
    test_loss_values = np.array(test_loss_values)

    fig, ax = plt.subplots()
    ax.plot(range(1, iters), training_loss_values, label="training loss")
    ax.plot(range(1, iters), test_loss_values / 0.2, label="test loss")

    ax.legend()

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    fig.savefig("plots/unit_test_loss.pdf", bbox_inches="tight")
