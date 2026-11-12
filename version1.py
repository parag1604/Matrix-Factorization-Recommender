"""
version1.py: remove user and movie biases before the SVD.

What is NEW compared to version0.py
- Much of a rating is not "taste" but bias: some users rate everything high,
  some movies are liked by everyone. The baseline r_ui ~ mu + b_u + b_i
  captures this (see fit_biases in utils.py) and already beats the global mean.
- Filling the missing entries with the global mean (version0) pretends every
  unseen movie is average for every user. Here we instead take the residuals
  r_ui - (mu + b_u + b_i), fill the missing ones with 0 (= "no information
  beyond the baseline"), and let the truncated SVD model only what is left.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import get_data, evaluate, clip_ratings, add_common_args, sorted_eigh_desc, fit_biases


def main():
    parser = argparse.ArgumentParser(description="Bias baseline + truncated SVD of the residuals")
    add_common_args(parser)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=25)
    parser.add_argument("--reg", type=float, default=10.0, help="shrinkage of the biases")
    args = parser.parse_args()
    np.random.seed(args.seed)

    ratings_matrix, train_set, test_set, _ = get_data(args.data_dir, args.test_size)
    observed = ratings_matrix > 0

    mu, b_u, b_i = fit_biases(ratings_matrix, reg=args.reg)
    baseline = mu + b_u[:, None] + b_i[None, :]
    print("Bias baseline (mu + b_u + b_i): Test RMSE, MAE =", evaluate(clip_ratings(baseline), test_set))

    residual = np.where(observed, ratings_matrix - baseline, 0.0)
    A = residual.T  # (n_movies, n_users)
    eigenvalues, eigenvectors = sorted_eigh_desc(A.T @ A, args.k_max)

    ks, test_errors = list(range(args.k_min, args.k_max + 1)), []
    for k in ks:
        vs = eigenvectors[:, :k]
        s = np.sqrt(np.maximum(eigenvalues[:k], 1e-12))
        us = A @ vs / s
        predictions = clip_ratings(baseline + ((us * s) @ vs.T).T)
        train_error, train_abs_error = evaluate(predictions, train_set)
        test_error, test_abs_error = evaluate(predictions, test_set)
        test_errors.append(test_error)
        print(f"k: {k}, Train RMSE: {train_error}, Train MAE: {train_abs_error}, "
              f"Test RMSE: {test_error}, Test MAE: {test_abs_error}")

    plt.plot(ks, test_errors, marker="o", label="Test RMSE (biases + SVD)")
    plt.xlabel("rank k")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bias_svd_rmse.png")
    print("Saved bias_svd_rmse.png")


if __name__ == "__main__":
    main()
