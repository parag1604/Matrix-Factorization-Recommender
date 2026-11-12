
# Matrix Factorization Recommender

Code (from scratch, in numpy/scipy) for a movie recommender: truncated SVD of the ratings matrix, a bias baseline, and matrix factorization with alternating least squares.

---

## Theory

Please note that the theory corresponding to the codes is hosted on this [page](https://parag1604.github.io/notes/Machine-Learning/Matrix-Factorization-and-Dimensionality-Reduction) (see the recommender-system section)

## Python Environment Setup

Use python >= 3.10 and install the requirements (numpy, scipy and matplotlib):

```bash
pip install -r requirements.txt
```

## Data

The code uses the MovieLens "latest-small" dataset (100,000 ratings by 610 users on about 9,700 movies), from GroupLens: https://grouplens.org/datasets/movielens/latest/

Download [ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) and unzip its contents into `./data`, so that the folder looks like:

```
data/
├── links.csv
├── movies.csv
├── ratings.csv
├── tags.csv
└── README.txt
```

Only `ratings.csv` is required (`movies.csv` is used for printing titles in `version2.py`). Use `--data-dir` to point somewhere else. Please read the GroupLens README for the dataset's terms of use.

Every script holds out one rating from each of `--test-size` (default 250) randomly chosen users as a test set, and reports RMSE and MAE on the train and test ratings.

## Running

### 0. Truncated SVD of the mean-filled matrix

```bash
python version0.py
```

The missing ratings are filled with the global mean, and the rank-k reconstruction of the matrix (computed from the top eigenvectors of the small users x users matrix $A^\top A$) gives the predictions. The script prints the global-mean baseline and the errors for k = 5 ... 25, and saves `svd_rmse.png` (test RMSE vs k).

### 1. Bias baseline + SVD of the residuals

```bash
python version1.py
```

Fits the baseline $r_{ui}\approx\mu+b_u+b_i$ (regularized user and movie biases), then applies the truncated SVD to the residuals, with missing residuals set to 0 instead of inventing ratings. Compare the test RMSE with `version0.py`. Saves `bias_svd_rmse.png`.

### 2. Matrix factorization with ALS

```bash
python version2.py --k 10 --lam 10 --iters 10
```

Fits $r_{ui}\approx\mu+b_u+b_i+\mathbf{p}_u^\top\mathbf{q}_i$ on the observed ratings only, alternating closed-form ridge regressions for the user factors and the movie factors. The regularized training loss decreases every sweep (`als_loss.png`). Finally, prints the top-5 recommendations for one user (`--user`). Try different `--k` and `--lam` and watch the gap between train and test error.
