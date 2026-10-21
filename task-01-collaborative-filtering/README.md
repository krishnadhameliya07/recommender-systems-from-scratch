# Task 01 — Collaborative Filtering

## Overview

This project implements and compares two recommendation algorithms
using the MovieLens 100K movie-rating dataset:

1. User-based collaborative filtering
2. Matrix factorization trained using stochastic gradient descent

A user-mean baseline provides a simple reference point. The goal is to
understand how recommendations can be generated from similar users and
how learned user/movie representations provide an alternative approach.

## Dataset

Dataset: MovieLens 100K, provided by GroupLens.

- Original ratings: 100,000
- Unique users: 943
- Unique movies: 1,682
- Rating scale: 1–5
- Each interaction records a user, movie, rating, and timestamp.

Official dataset:
https://grouplens.org/datasets/movielens/100k/

The dataset is stored locally and excluded from version control.
The repository contains the implementation and experiment outputs,
not a committed copy of the dataset.

## Data Preparation

The interaction records were divided into three disjoint subsets:

| Split | Records | Purpose |
|---|---:|---|
| Training | 80,000 | Fit the models |
| Validation | 10,000 | Select model settings and checkpoints |
| Test | 10,000 | Final held-out evaluation |

The split used a fixed random seed of 42 and stratification by rating.

The user–movie matrix was constructed from training data only.
Movies without any training ratings therefore do not appear as columns.

The training matrix contains:

- 943 users
- 1,652 movies
- 80,000 recorded ratings
- 1,557,836 possible user–movie combinations
- 94.86% sparsity

Missing ratings are treated as unobserved interactions, not as zero-star
ratings.

## Method 1: User-Based Collaborative Filtering

### Approach

Each user's observed ratings are centered by subtracting that user's
mean training rating. Cosine similarity is computed between the resulting
user vectors, with missing centered entries filled with zero for the
similarity calculation.

For a target user and movie, the prediction is:

    predicted rating =
        target user's mean rating
        + weighted average of neighbours' centered ratings

Only positive-similarity neighbours who rated the target movie in the
training set contribute to a normal neighbour-based prediction.

If no suitable neighbours are available, the implementation falls back
to the target user's training mean. Unknown users use the global
training mean.

Predictions are clipped to the 1–5 rating range.

### Neighbourhood-size experiment

The number of neighbours, k, was selected using validation RMSE.

| k | Validation MAE | Validation RMSE |
|---:|---:|---:|
| 5 | 0.7668 | 0.9849 |
| 10 | 0.7458 | 0.9556 |
| 20 | 0.7383 | 0.9451 |
| 40 | 0.7387 | 0.9450 |
| 80 | 0.7409 | 0.9473 |

The selected setting was k = 40 because it achieved the lowest
validation RMSE among the tested settings. The difference between
k = 20 and k = 40 was extremely small, and k = 20 had marginally
lower MAE.

## Method 2: Matrix Factorization

### Model

The model approximates ratings using global, user, movie, and latent
factor terms:

    predicted rating =
        global mean
        + user bias
        + movie bias
        + dot(user factors, movie factors)

User and movie factors are initialized with small random values and
learned using stochastic gradient descent (SGD).

The implementation uses the original user/movie factor vectors when
calculating the two factor updates for each interaction.

### Configuration

| Parameter | Value |
|---|---|
| Latent factors | 32 |
| Learning rate | 0.01 |
| Regularization | 0.02 |
| Maximum epochs | 30 |
| Early-stopping patience | 4 epochs |
| Random seed | 42 |
| Optimization | SGD |

Validation RMSE determines the best checkpoint. The model is restored
to the checkpoint with the lowest observed validation RMSE instead of
automatically using the last epoch.

The best checkpoint occurred at epoch 13, with validation RMSE 0.9386
and validation MAE 0.7377.

The falling training error and subsequently rising validation error
demonstrated the importance of monitoring overfitting.

## Evaluation Metrics

### Mean Absolute Error (MAE)

MAE is the mean absolute difference between actual and predicted ratings.

Lower MAE indicates smaller average absolute errors.

### Root Mean Squared Error (RMSE)

RMSE is the square root of the mean squared prediction error. It gives
greater weight to larger errors than MAE does.

Lower RMSE indicates better performance under this metric.

Model settings were selected using validation data. The test set was
reserved for the final comparison.

## Final Test Results

The following results were obtained on the 10,000 held-out test ratings.

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| User-mean baseline | 0.8348 | 1.0412 |
| User-based CF (k = 40) | 0.7219 | 0.9254 |
| Matrix factorization (epoch 13) | 0.7290 | 0.9265 |

Both collaborative-filtering methods performed better than the simple
user-mean baseline.

User-based collaborative filtering achieved marginally lower test MAE
and RMSE than matrix factorization in this experiment. The difference
between these two models was small, so the result should not be taken
as evidence that one approach is always superior.

The test set was not used to select the neighbourhood size or the
matrix-factorization checkpoint.

## Top-N Recommendations

Both models were used to rank unrated candidate movies for User 1.
Recommendations were saved with their predicted ratings and movie titles.

An additional recommendation audit examined the number of training
ratings supporting each candidate.

The filtered recommendation lists require at least 20 training ratings
per movie. This is a display-time support threshold; it does not change
the trained models or the test metrics.

The recommendation results are illustrative rankings for one user,
not proof that the user will like every recommended movie.

## Limitations

- Similarities can be unreliable when users have few overlapping ratings.
- Filling missing centered ratings with zero is a simplifying baseline.
- Sparse or unpopular movies may have limited evidence.
- User-based collaborative filtering can become computationally costly
  as the number of users increases.
- Matrix factorization is sensitive to hyperparameters and initialization.
- The experiment uses a random interaction split rather than a
  time-based evaluation.
- New users and movies require fallback handling.
- The experiment does not implement production-scale serving,
  online learning, or distributed training.

## Reproducibility

1. Set up the Python environment and install the required libraries.
2. Download MovieLens 100K from the official GroupLens source.
3. Place and extract the dataset under
   `task-01-collaborative-filtering/data/`.
4. Open `notebooks/00_start_here.ipynb` and execute the cells in order.
5. Inspect the generated results and figures.

The implementation uses a fixed data-split seed and fixed model
initialization seeds. Package versions and the local Python environment
should also be recorded when preparing a final reproducibility run.

## Results and Figures

The `results/` directory contains experiment tables, training history,
test metrics, and recommendation outputs.

The `figures/` directory contains visualizations such as the
neighbourhood-size comparison and matrix-factorization learning curves.

## What I Learned

This experiment demonstrates the difference between memory-based and
model-based collaborative filtering.

User-based collaborative filtering exploits similarities between
observed rating patterns. Matrix factorization learns compact latent
representations of users and movies.

The experiment also illustrates why train/validation/test separation,
baseline comparisons, early stopping, and careful evaluation are
necessary when developing recommendation algorithms.
