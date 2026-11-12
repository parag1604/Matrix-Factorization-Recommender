import csv
import os

import numpy as np


def get_data(data_dir: str = "data", size_of_test_set: int = 250):
    """Load MovieLens ratings and hold out one rating for some users.

    Returns
    - ratings_matrix: (n_users, n_movies) array, 0 where the rating is unknown
      (or held out); MovieLens ratings are 0.5 ... 5 in half stars, so 0 is free.
    - train_set, test_set: arrays of rows (user_idx, movie_idx, rating).
    - movie_ids: the MovieLens movieId of every column.
    """
    path = os.path.join(data_dir, "ratings.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; see the Data section of the README")
    # columns: userId, movieId, rating, timestamp
    ratings = np.loadtxt(path, delimiter=",", skiprows=1)[:, :3]
    # ids are integers, but ratings are half stars (3.5): do not cast them to int
    user_ids, movie_ids = ratings[:, 0].astype(int), ratings[:, 1].astype(int)
    # Get unique movies and users, and map ids to row/column indices
    users, user_idx = np.unique(user_ids, return_inverse=True)
    movies, movie_idx = np.unique(movie_ids, return_inverse=True)
    # Create and fill a matrix of shape (n_users, n_movies)
    ratings_matrix = np.zeros((len(users), len(movies)))
    ratings_matrix[user_idx, movie_idx] = ratings[:, 2]

    # Create a test set by removing one rating from randomly chosen users, such
    # that every chosen user keeps at least one rating in the train set
    nonzero_idxs = ratings_matrix > 0
    users_with_at_least_two_ratings = np.where(nonzero_idxs.sum(axis=1) >= 2)[0]
    size_of_test_set = min(size_of_test_set, len(users_with_at_least_two_ratings))
    test_users = np.random.choice(users_with_at_least_two_ratings, size_of_test_set, replace=False)
    test_set = []
    for user in test_users:
        random_rating_idx = np.random.choice(np.where(nonzero_idxs[user])[0])
        test_set.append([user, random_rating_idx, ratings_matrix[user, random_rating_idx]])
        ratings_matrix[user, random_rating_idx] = 0

    rows, cols = np.nonzero(ratings_matrix)
    train_set = np.column_stack([rows, cols, ratings_matrix[rows, cols]])
    return ratings_matrix, train_set, np.array(test_set), movies


def evaluate(predictions: np.ndarray, data_set: np.ndarray):
    """RMSE and MAE of a (n_users, n_movies) prediction matrix on (user, movie, rating) rows."""
    users, movies = data_set[:, 0].astype(int), data_set[:, 1].astype(int)
    err = data_set[:, 2] - predictions[users, movies]
    return np.round(np.sqrt(np.mean(err ** 2)), 3), np.round(np.mean(np.abs(err)), 3)


def clip_ratings(predictions: np.ndarray) -> np.ndarray:
    # MovieLens ratings lie between 0.5 and 5 stars
    return np.clip(predictions, 0.5, 5.0)


def load_titles(data_dir: str = "data"):
    """movieId -> title, from movies.csv (titles contain commas, so use csv)."""
    path = os.path.join(data_dir, "movies.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {int(row["movieId"]): row["title"] for row in csv.DictReader(f)}


def add_common_args(parser):
    parser.add_argument("--data-dir", default="data", help="folder containing ratings.csv")
    parser.add_argument("--test-size", type=int, default=250, help="number of held-out ratings")
    parser.add_argument("--seed", type=int, default=2023)


def sorted_eigh_desc(S: np.ndarray, k: int):
    """Top-k eigenpairs of a symmetric matrix, eigenvalues in descending order."""
    import scipy.linalg as la
    n = S.shape[0]
    k = min(k, n)
    eigenvalues, eigenvectors = la.eigh(S, subset_by_index=(n - k, n - 1))
    idxs = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idxs], eigenvectors[:, idxs]


def fit_biases(ratings_matrix: np.ndarray, reg: float = 10.0, iters: int = 5):
    """Baseline predictor r_ui ~ mu + b_u + b_i, fitted on the observed entries.

    Alternating, regularized averages: b_i = sum_u (r - mu - b_u) / (reg + n_i), and
    the same for b_u. The regularizer shrinks the bias of a movie with very few
    ratings towards 0 instead of trusting one or two votes.
    """
    observed = ratings_matrix > 0
    mu = ratings_matrix[observed].mean()
    b_u = np.zeros(ratings_matrix.shape[0])
    b_i = np.zeros(ratings_matrix.shape[1])
    residual = np.where(observed, ratings_matrix - mu, 0.0)
    for _ in range(iters):
        b_i = (residual - b_u[:, None] * observed).sum(axis=0) / (reg + observed.sum(axis=0))
        b_u = (residual - b_i[None, :] * observed).sum(axis=1) / (reg + observed.sum(axis=1))
    return mu, b_u, b_i
