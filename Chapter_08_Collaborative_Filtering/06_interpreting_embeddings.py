"""
Chapter 8 — Collaborative Filtering Deep Dive
Topic 6: Interpreting embeddings and biases; using fastai.collab; embedding distance (book
sections "Interpreting Embeddings and Biases", "Using fastai.collab", "Embedding Distance")

After training:
  * movie biases: low bias = movies people dislike *even when they usually like that kind of movie*;
    high bias = movies everyone likes, even people who don't usually like that genre;
  * movie factors: too many dimensions to look at, so reduce them to 3 with PCA and plot the
    first two: movies of a similar kind end up close to each other;
  * embedding distance: the movies closest (cosine similarity) to a given movie are similar movies.

    learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))
    learn.fit_one_cycle(5, 5e-3, wd=0.1)
    movie_bias = learn.model.i_bias.weight.squeeze()

(The synthetic MovieLens data was generated from 7 movie "styles" whose words appear in the titles:
Star/Galaxy = sci-fi, Love/Kiss = romance, Blood/Scream = horror, ... see if the model found them.)

Run:
    python 06_interpreting_embeddings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from common import outputs_dir, set_seed, try_import_matplotlib  # noqa: E402
from common.collab import collab_learner  # noqa: E402
from movielens_data import get_dls, ratings  # noqa: E402

OUT = outputs_dir(__file__)
set_seed(42)
dls = get_dls()
learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))
learn.fit_one_cycle(5, 5e-3, wd=0.1)
print(learn.model)
titles = dls.classes["title"]

movie_bias = learn.model.i_bias.weight.squeeze().detach().cpu()
print("\nLowest movie biases (disliked even by people who like that kind of movie):")
for i in movie_bias.argsort()[:5]:
    print("  ", titles[i])
print("Highest movie biases:")
for i in movie_bias.argsort(descending=True)[:5]:
    print("  ", titles[i])

# ---- PCA on the 1000 most rated movies -------------------------------------------------------------
g = ratings.groupby("title")["rating"].count()
top_movies = list(g.sort_values(ascending=False).index[:1000])
o2i = {t: i for i, t in enumerate(titles)}
top_idxs = torch.tensor([o2i[m] for m in top_movies])
movie_w = learn.model.i_weight.weight[top_idxs].detach().cpu()
movie_w = movie_w - movie_w.mean(0)
U, S, V = torch.pca_lowrank(movie_w, q=3)
movie_pca = movie_w @ V                                      # [1000, 3]
fac0, fac1 = movie_pca[:, 0], movie_pca[:, 1]
print("\nMovies at both ends of the first principal component:")
print("   +", [top_movies[int(i)] for i in fac0.argsort(descending=True)[:4]])
print("   -", [top_movies[int(i)] for i in fac0.argsort()[:4]])
print("Movies at both ends of the second principal component:")
print("   +", [top_movies[int(i)] for i in fac1.argsort(descending=True)[:4]])
print("   -", [top_movies[int(i)] for i in fac1.argsort()[:4]])

plt = try_import_matplotlib()
if plt is not None:
    idxs = list(range(50))
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(fac0[idxs], fac1[idxs])
    for i in idxs:
        ax.text(float(fac0[i]), float(fac1[i]), top_movies[i], fontsize=8)
    fig.savefig(OUT / "movie_pca.png", bbox_inches="tight")
    print(f"saved {OUT / 'movie_pca.png'}")

# ---- Embedding distance -----------------------------------------------------------------------------------
movie_factors = learn.model.i_weight.weight.detach()
query = top_movies[0]
distances = nn.CosineSimilarity(dim=1)(movie_factors, movie_factors[o2i[query]][None])
closest = distances.argsort(descending=True)[1:6]
print(f"\nMovies most similar to '{query}':", [titles[i] for i in closest])
