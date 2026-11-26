# Chapter 9: Tabular Modeling Deep Dive

Predicting bulldozer auction prices with decision trees, random forests, and neural networks with categorical embeddings: data preparation, model interpretation, extrapolation and out-of-domain data, ensembling and boosting.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | Entity embeddings: learned embeddings capture structure (days of the week) | Categorical Embeddings | `01_categorical_embeddings.py` |
| 2 | The Kaggle dataset, ordinal categories, log price, `add_datepart` | Beyond Deep Learning, The Dataset, Look at the Data, Handling Dates | `02_look_at_the_data.py` |
| 3 | `TabularPandas`, `Categorify`, `FillMissing`, a time-based split | Using TabularPandas and TabularProc | `03_tabular_pandas_and_procs.py` |
| 4 | `DecisionTreeRegressor`, reading a tree, overfitting, `min_samples_leaf` | Decision Trees, Creating the Decision Tree, Categorical Variables | `04_decision_trees.py` |
| 5 | Bagging, `RandomForestRegressor`, error vs number of trees, OOB error | Random Forests, Out-of-Bag Error | `05_random_forests.py` |
| 6 | Tree variance, feature importance, removing low-importance and redundant columns | Model Interpretation (first four sections) | `06_feature_importance_and_redundancy.py` |
| 7 | Partial dependence, data leakage, a tree interpreter (computed from the trees) | Partial Dependence, Data Leakage, Tree Interpreter | `07_partial_dependence_and_tree_interpreter.py` |
| 8 | Forests can't extrapolate; finding out-of-domain columns; recent data only | The Extrapolation Problem, Finding Out-of-Domain Data | `08_extrapolation_and_out_of_domain.py` |
| 9 | `tabular_learner` with embeddings, `y_range=(8,12)` | Using a Neural Network | `09_tabular_neural_net.py` |
| 10 | Ensembling the forest and the net; gradient boosting by hand; embeddings as features | Ensembling, Boosting, Combining Embeddings with Other Methods | `10_ensembling_boosting_and_embeddings.py` |

`bulldozers_data.py` holds the shared preparation (`load_df`, splits, `get_to`, `rf`, `r_mse`, ...).

## Run

```bash
pip install scikit-learn scipy        # trees and forests (scripts 4-8 and 10), and the column dendrogram
cd Chapter_09_Tabular_Modeling
python 01_categorical_embeddings.py
python 02_look_at_the_data.py
python 03_tabular_pandas_and_procs.py
python 04_decision_trees.py
python 05_random_forests.py
python 06_feature_importance_and_redundancy.py      # run before 07-10: saves outputs/final_columns.txt
python 07_partial_dependence_and_tree_interpreter.py
python 08_extrapolation_and_out_of_domain.py
python 09_tabular_neural_net.py
python 10_ensembling_boosting_and_embeddings.py
```

## Data
The real data is a Kaggle competition download: `kaggle competitions download -c bluebook-for-bulldozers`, unzipped into `data/real/bluebook-for-bulldozers/`, then add `--real`. By default a synthetic `TrainAndValid.csv` / `Test.csv` (60,000 sales, 1989-2012) is generated with the real column names and planted structure:
- the price depends mostly on the model, `ProductSize`, age (`YearMade`), `Enclosure`, `Coupler_System` and `Hydraulics_Flow`;
- there are redundant pairs (`ProductGroup`/`ProductGroupDesc`, `fiModelDesc`/`fiBaseModel`, ...) and columns that are missing together;
- `YearMade` uses a placeholder of 1000 for unknown years;
- prices rise over time, and IDs grow with time (`SalesID`, `MachineID`).

The analyses find these, much like the book does with the real data. A random forest gets an RMSE of about 0.23 on the log price (the book: 0.23), and the neural net about 0.20.

## Notes and changes from the book
- The book's `treeinterpreter` and `waterfallcharts` packages are replaced by a short function that walks each tree's decision path. `plot_partial_dependence` (removed from recent scikit-learn) is replaced by the computation it does (set a column to each value, average the predictions).
- The column clusters use scipy if installed. Otherwise the most correlated pairs are printed.

## Key takeaways
- Start with a random forest: hard to get wrong, fast, and very interpretable.
- Feature importance, redundancy checks, partial dependence and tree interpreters tell you what drives predictions, and expose leakage.
- Trees can't extrapolate. Check whether your validation/test data is out of domain, and consider a neural net (or an ensemble) when there's a trend.
- The embeddings a neural net learns can improve other models.
