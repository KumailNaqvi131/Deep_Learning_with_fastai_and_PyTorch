"""
Chapter 8 — Collaborative Filtering Deep Dive
Topic 1: A first look at the data; learning the latent factors (book sections of the same names)

MovieLens: 100,000 ratings (1-5 stars) of 1,682 movies by 943 users. Most user/movie pairs have
no rating: the task is to fill in the blanks, to recommend movies people will like.

The idea: describe each movie by a few numbers (how much sci-fi, how much action, how old, ...)
and each user by how much they like each of these; a rating is then the dot product:

    last_skywalker = np.array([0.98,0.9,-0.9])     # very sci-fi, very action, not old
    user1 = np.array([0.9,0.8,-0.6])               # likes sci-fi and action, not old movies
    (user1*last_skywalker).sum()                   # 2.14 -> a match

We don't know these "latent factors": we will *learn* them with SGD.

Run:
    python 01_data_and_latent_factors.py         # add --real for the real MovieLens 100k (5 MB)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from movielens_data import ratings  # noqa: E402

print(ratings[["user", "movie", "rating", "timestamp"]].head(), "\n")
print(f"{ratings.user.nunique()} users, {ratings.movie.nunique()} movies, {len(ratings)} ratings "
      f"-> {len(ratings) / (ratings.user.nunique() * ratings.movie.nunique()):.1%} of the table is filled")

# A slice of the user x movie table: most cells are empty
top_u = ratings.user.value_counts().index[:8]
top_m = ratings.movie.value_counts().index[:10]
crosstab = ratings[ratings.user.isin(top_u) & ratings.movie.isin(top_m)].pivot_table(
    index="user", columns="movie", values="rating")
print("\nRatings of the most active users for the most rated movies (NaN = not rated):")
print(crosstab.to_string())

# ---- Latent factors by hand ---------------------------------------------------------------------------
last_skywalker = np.array([0.98, 0.9, -0.9])      # science-fiction, action, old
user1 = np.array([0.9, 0.8, -0.6])
casablanca = np.array([-0.99, -0.3, 0.8])
print(f"\nuser1 . last_skywalker = {(user1 * last_skywalker).sum():.2f}  (likely to like it)")
print(f"user1 . casablanca     = {(user1 * casablanca).sum():.2f} (unlikely)")
print("\nNext: learn these factors for every user and movie (02_dataloaders_and_embeddings.py).")
print(ratings[["user", "movie", "title", "rating"]].head(3).to_string())
