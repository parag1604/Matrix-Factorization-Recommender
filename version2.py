"""
version2.py: matrix factorization fitted on the observed ratings only (ALS).

What is NEW compared to version1.py
- The SVD needs a complete matrix, so versions 0 and 1 had to invent values
  for the ~98% missing entries. Matrix factorization skips that: model
      r_ui ~ mu + b_u + b_i + p_u . q_i
  with user factors p_u and movie factors q_i in R^k, and fit them on the
  OBSERVED entries only, with L2 regularization against overfitting:
      min sum_{(u,i) observed} (r_ui - prediction)^2 + lam (|p_u|^2 + |q_i|^2).
- Alternating least squares (ALS): with Q fixed, each p_u is a small ridge
  regression with a closed form, and vice versa. Each half-step cannot increase
  the loss.
- Finally we use the model: the top recommendations for one user (titles are
  read from movies.csv if it is present).
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import get_data, evaluate, clip_ratings, add_common_args, fit_biases, load_titles


def ridge_solve(F, r, lam):
    # argmin_x |F x - r|^2 + lam |x|^2 = (F^T F + lam I)^{-1} F^T r
    k = F.shape[1]
    return np.linalg.solve(F.T @ F + lam * np.eye(k), F.T @ r)


def als(residual, observed, k, lam, iters, seed):
    n_users, n_movies = residual.shape
    rng = np.random.default_rng(seed)
    P = 0.1 * rng.standard_normal((n_users, k))
    Q = 0.1 * rng.standard_normal((n_movies, k))
    user_items = [np.nonzero(observed[u])[0] for u in range(n_users)]
    item_users = [np.nonzero(observed[:, i])[0] for i in range(n_movies)]
    losses = []
    for _ in range(iters):
        for u, items in enumerate(user_items):          # fix Q, solve every p_u
            P[u] = ridge_solve(Q[items], residual[u, items], lam)
        for i, users in enumerate(item_users):           # fix P, solve every q_i
            if len(users) > 0:
                Q[i] = ridge_solve(P[users], residual[users, i], lam)
        err = np.where(observed, residual - P @ Q.T, 0.0)
        losses.append((err ** 2).sum() + lam * ((P ** 2).sum() + (Q ** 2).sum()))
    return P, Q, losses


def main():
    parser = argparse.ArgumentParser(description="Matrix factorization with ALS")
    add_common_args(parser)
    parser.add_argument("--k", type=int, default=10, help="number of latent factors")
    parser.add_argument("--lam", type=float, default=10.0, help="L2 regularization")
    parser.add_argument("--iters", type=int, default=10, help="ALS sweeps")
    parser.add_argument("--user", type=int, default=0, help="row index of the user to recommend for")
    args = parser.parse_args()
    np.random.seed(args.seed)

    ratings_matrix, train_set, test_set, movie_ids = get_data(args.data_dir, args.test_size)
    observed = ratings_matrix > 0

    mu, b_u, b_i = fit_biases(ratings_matrix)
    baseline = mu + b_u[:, None] + b_i[None, :]
    print("Bias baseline: Test RMSE, MAE =", evaluate(clip_ratings(baseline), test_set))

    residual = np.where(observed, ratings_matrix - baseline, 0.0)
    P, Q, losses = als(residual, observed, args.k, args.lam, args.iters, args.seed)
    predictions = clip_ratings(baseline + P @ Q.T)
    print(f"ALS (k = {args.k}, lam = {args.lam}): Train RMSE, MAE = {evaluate(predictions, train_set)}, "
          f"Test RMSE, MAE = {evaluate(predictions, test_set)}")

    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("ALS sweep")
    plt.ylabel("regularized training loss")
    plt.tight_layout()
    plt.savefig("als_loss.png")
    print("Saved als_loss.png")

    # Recommend: highest predicted ratings among the movies the user has not rated
    titles = load_titles(args.data_dir)
    scores = np.where(observed[args.user], -np.inf, predictions[args.user])
    print(f"\nTop 5 recommendations for user index {args.user}:")
    for i in np.argsort(scores)[::-1][:5]:
        print(f"  {scores[i]:.2f}  {titles.get(int(movie_ids[i]), f'movieId {movie_ids[i]}')}")


if __name__ == "__main__":
    main()
