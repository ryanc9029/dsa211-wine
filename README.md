# DSA211 Project: Identifying high-quality white Vinho Verde wine

DSA211 Statistical Learning (SMU), Topic 2: Classification System Evaluation.

## Research question

**Can physicochemical properties identify high-quality white Vinho Verde wine?**

A wine is *good* if its expert quality score is 7 or above. The application is a producer pre-screening batches before paying for an expert tasting panel: a false positive wastes one tasting ($c_{FP} = 1$), and a false negative sells a good wine at the standard price ($c_{FN} = 3$ in the base case, varied from 1/3 to 10 in the sensitivity analysis). We value a good batch's premium at about 4 tasting fees; a miss loses the premium but saves one fee, so it costs about 3 fees. The cost-based threshold analysis (notebook 04) is the project's one extension.

## Data

UCI Machine Learning Repository, *Wine Quality* (P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis, 2009, *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems 47(4)). Licence: CC BY 4.0. <https://archive.ics.uci.edu/dataset/186/wine+quality>

**White wine only.** White has 4,898 rows (3,961 unique) with 21.6% good wines; red has 1,599 rows (1,359 unique) with 13.6% good wines, i.e. 825 vs 184 good wines after deduplication. Red's rare class is too small for stable 10-fold estimates (about 15 good wines per validation fold), and pooling the two would break LDA's equal-covariance assumption because their chemistry differs strongly. The red file is read only to compute these counts.

## Notebooks (run in this order)

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | `wine_white_data_split.ipynb` | Data quality checks, white-only justification, duplicate analysis (duplicates dropped), one stratified 80/20 split on unique wines (`random_state=211`). |
| 2 | `EDA.ipynb` | Training-set EDA of all 11 predictors; chooses the log-transformed predictors (`wine_utils.LOG_FEATURES`). |
| 3 | `03_model_comparison.ipynb` | Chooses each classifier's own predictor subset (best subset of each size, number of predictors by the one-SE rule, all under nested 10-fold CV), then compares logistic regression, LDA and Gaussian NB. Selection: best mean PR-AUC, one-SE rule, then lowest expected cost at $t^* = 0.25$. Writes `results/selected_model.json`, `cv_fold_scores.csv`, `oof_probabilities.csv`, `predictor_count_curve.csv`. |
| 4 | `04_cost_threshold.ipynb` | The single extension: cost-based threshold analysis on out-of-fold probabilities, sensitivity across cost ratios, saving over the no-model policies, break-even range, error analysis by quality score. Writes `results/operating_point.json`. |
| 5 | `05_test_and_refit.ipynb` | Loads the test set once, scores the frozen model, subset and threshold on the 793 test wines next to the CV estimates, then refits on all 3,961 wines. |

Shared definitions (features, transforms, folds, costs, candidate models, subset search) are in `wine_utils.py`.

## Duplicates: one row per wine

The white file has 937 exact duplicate rows (772 wines recorded 2-8 times). Copies match on all 11 predictors **and** quality, and no set of identical measurements has conflicting quality scores, so we treat them as the same wine measured more than once and **drop them before the split**. Every notebook uses one row per wine: 3,961 wines, split once into 3,168 train and 793 test (`datasets/winequality-white-{train,test}.csv`, never regenerated or edited). Because no wine appears twice, ordinary `StratifiedKFold(n_splits=10, shuffle=True, random_state=211)` cannot put a wine on both sides of a fold, and no wine counts twice. (An earlier version kept the duplicates and used `StratifiedGroupKFold` by wine.)

## Predictors: a separate subset for each classifier

All 11 predictors are examined in EDA. Each classifier then gets its own subset, trading a little accuracy for simplicity (notebook 03):

1. On a given set of training wines, all 2,047 subsets are scored by 5-fold CV PR-AUC, and the best subset of each size $k$ is kept.
2. Nested CV gives an honest PR-AUC for each $k$: step 1 runs inside each of the 10 outer folds, and the chosen subset is scored on the held-out fold.
3. The number of predictors $k^*$ is the smallest $k$ within one SE of the best $k$ (the one-SE rule).
4. The final subset is the best size-$k^*$ subset on all training wines.

| Classifier | Best $k$ (PR-AUC) | Chosen $k^*$ (PR-AUC) | Predictors |
|---|---|---|---|
| Logistic (selected) | 9 (0.556) | 6 (0.542) | fixed acidity, log residual sugar, free sulfur dioxide, density, pH, log sulphates |
| LDA | 9 (0.551) | 6 (0.537) | same 6 as logistic |
| Gaussian NB | 7 (0.530) | 4 (0.529) | alcohol, pH, log chlorides, citric acid |

The selected logistic model gives up 0.014 ± 0.008 PR-AUC against its best size (paired, nested CV) to use 6 predictors instead of 9. It leaves out alcohol, whose information is carried by density and residual sugar. Each pipeline selects its columns as its first step, which learns nothing from the data and so cannot leak. A log-vs-raw check on the final subsets shows the log transforms are neutral to slightly helpful; they were chosen for distribution shape, not performance.

## Reproducing the results

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# open the notebooks in Jupyter / VS Code and run 1-4 in order, top to bottom
```

Notebooks 1-4 run top to bottom from a fresh kernel. Re-running notebook 1 rewrites the split CSVs with identical contents (fixed seed). Notebook 05 must be run only once, after the model and threshold are frozen.

## Files

- `datasets/`: raw UCI files and the train/test split
- `results/`: CV fold scores, out-of-fold probabilities, the PR-AUC-by-number-of-predictors curve, the selected model and the operating point
- `results/subset_search/`: cached inner-CV scores of every predictor subset (lets notebook 03 rerun in about a minute)
- `AI_DECLARATION.md`: declaration of AI tool use
