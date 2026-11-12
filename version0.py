"""
version0.py: a recommender from a truncated SVD of the (filled-in) ratings matrix.

What this file teaches
- Ratings form a users x movies matrix that is mostly empty (~98% missing for
  MovieLens). The low-rank assumption: tastes are driven by a few hidden
  factors, so the full matrix should be close to rank k.
- A plain SVD needs a complete matrix, so we fill every missing entry with the
  global mean rating, compute the top-k singular triplets, and read the
  predictions off the rank-k reconstruction (the best rank-k approximation,
  by Eckart-Young).
- We get the top-k right singular vectors as eigenvectors of A^T A (a small
  users x users matrix), then u_i = A v_i / sigma_i.
- We compare against the simplest baseline: predict the global mean.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import get_data, evaluate, clip_ratings, add_common_args, sorted_eigh_desc


def SVD(A, k, eigenvalues, eigenvectors):
    # A is (n_movies, n_users); the top-k right singular vectors are eigenvectors of A^T A
    vs = eigenvectors[:, :k]
    s = np.sqrt(eigenvalues[:k])
    us = A @ vs / s
    return clip_ratings((us * s) @ vs.T)  # rank-k reconstruction, (n_movies, n_users)


def main():
    parser = argparse.ArgumentParser(description="Truncated SVD recommender")
    add_common_args(parser)
    parser.add_argument("--k-min", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=25)
    args = parser.parse_args()
    np.random.seed(args.seed)

    ratings_matrix, train_set, test_set, _ = get_data(args.data_dir, args.test_size)
    observed = ratings_matrix > 0
    print(f"{ratings_matrix.shape[0]} users x {ratings_matrix.shape[1]} movies, "
          f"{100 * observed.mean():.2f}% of the entries observed")

    # Baseline: always predict the mean training rating
    mean_rating = ratings_matrix[observed].mean()
    print("Global mean baseline: Test RMSE, MAE =", evaluate(np.full(ratings_matrix.shape, mean_rating), test_set))

    # Fill the missing entries with the mean rating
    filled = ratings_matrix.copy()
    filled[~observed] = mean_rating
    A = filled.T  # (n_movies, n_users)

    # Get the top k eigenvalues and eigenvectors of A^T A, in descending order
    eigenvalues, eigenvectors = sorted_eigh_desc(A.T @ A, args.k_max)

    ks, test_errors = list(range(args.k_min, args.k_max + 1)), []
    for k in ks:
        predictions = SVD(A, k, eigenvalues, eigenvectors).T  # back to (n_users, n_movies)
        train_error, train_abs_error = evaluate(predictions, train_set)
        test_error, test_abs_error = evaluate(predictions, test_set)
        test_errors.append(test_error)
        print(f"k: {k}, Train RMSE: {train_error}, Train MAE: {train_abs_error}, "
              f"Test RMSE: {test_error}, Test MAE: {test_abs_error}")

    plt.plot(ks, test_errors, marker="o", label="Test RMSE")
    plt.xlabel("rank k")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("svd_rmse.png")
    print("Saved svd_rmse.png")


if __name__ == "__main__":
    main()
