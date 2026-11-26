# Chapter 8: Collaborative Filtering Deep Dive

Recommending movies from ratings alone: latent factors, embeddings, a dot-product model built from scratch, weight decay, interpreting what the model learned, and a neural-network version.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | The ratings table; latent factors and dot products by hand | A First Look at the Data, Learning the Latent Factors | `01_data_and_latent_factors.py` |
| 2 | `CollabDataLoaders`; one-hot @ matrix == indexing == `nn.Embedding` | Creating the DataLoaders | `02_dataloaders_and_embeddings.py` |
| 3 | `DotProduct`, `y_range` with sigmoid_range, `DotProductBias` | Collaborative Filtering from Scratch | `03_collab_from_scratch.py` |
| 4 | Weight decay (L2 regularization) | Weight Decay | `04_weight_decay.py` |
| 5 | `nn.Parameter` and our own embedding | Creating Our Own Embedding Module | `05_own_embedding_module.py` |
| 6 | Movie biases, PCA of the movie factors, `collab_learner`, cosine similarity | Interpreting Embeddings and Biases, Using fastai.collab, Embedding Distance | `06_interpreting_embeddings.py` |
| 7 | The cold-start problem; `CollabNN` and `collab_learner(use_nn=True)` | Bootstrapping a Collaborative Filtering Model, Deep Learning for Collaborative Filtering | `07_bootstrapping_and_deep_learning_collab.py` |

`movielens_data.py` loads the ratings merged with the movie titles.

## Run

```bash
cd Chapter_08_Collaborative_Filtering
python 01_data_and_latent_factors.py
python 02_dataloaders_and_embeddings.py
python 03_collab_from_scratch.py
python 04_weight_decay.py
python 05_own_embedding_module.py
python 06_interpreting_embeddings.py
python 07_bootstrapping_and_deep_learning_collab.py
```

## Data
Synthetic MovieLens 100k (`u.data`, `u.item` in the real formats; `--real` downloads the real 5 MB file). Movies belong to 7 hidden "styles" (sci-fi, action, romance, drama, horror, kids, noir) whose words appear in the titles ("Galaxy", "Kiss", "Scream", "Puppy", "Detective", ...). Ratings come from user and movie taste vectors plus biases. So `06` can check that the model rediscovered them: the PCA axes separate, for example, kids' movies from horror/noir and drama from action/sci-fi.

## Notes
The scripts use `common.Embedding`, fastai's `Embedding`: an `nn.Embedding` initialised with small values (std 0.01). PyTorch's default init (std 1) makes the dot products of 50 factors huge at the start.

## Key takeaways
- An embedding is a lookup in a matrix of parameters, with the gradient of a one-hot matrix product.
- Biases capture "this movie is good / this user is generous". Factors capture tastes.
- Weight decay keeps the weights small, which reduces overfitting: the training loss goes up and the validation loss goes down.
- Learned embeddings are interpretable (PCA, distances). The neural version can easily take extra information about users and items.
