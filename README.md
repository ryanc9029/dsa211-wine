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
| 1 | `wine_white_data_split.ipynb` | Data quality checks, white-only justification, duplicate analysis, one stratified 80/20 split on unique wines (`random_state=211`). |
| 2 | `EDA.ipynb` | Training-set EDA of all 11 predictors; chooses the log-transformed predictors (`wine_utils.LOG_FEATURES`). |
| 3 | `03_model_comparison.ipynb` | Fixes the predictor set (drops citric acid), then compares logistic regression, LDA and Gaussian NB on the 10 remaining predictors under grouped 10-fold CV. Selection: best mean PR-AUC, one-SE rule, then lowest expected cost at $t^* = 0.25$. Writes `results/selected_model.json`, `cv_fold_scores.csv`, `oof_probabilities.csv`. |
| 4 | `04_cost_threshold.ipynb` | The single extension: cost-based threshold analysis on out-of-fold probabilities, sensitivity across cost ratios, saving over the no-model policies, break-even range. Writes `results/operating_point.json`. |
| 5 | `05_test_audit_refit.ipynb` | **Run once.** Scores the frozen model and threshold on the test set, then refits on all 4,898 rows. |

Shared definitions (features, transforms, folds, costs, candidate models) are in `wine_utils.py`.

## Predictors

The data have 11 physicochemical predictors (`wine_utils.ALL_FEATURES`). All 11 are examined in EDA, but the classifiers use 10 (`wine_utils.MODEL_FEATURES`): **citric acid is dropped**. Best-subset selection for logistic regression, repeated inside each grouped CV fold in notebook 03, dropped citric acid in all 10 folds under AIC with no change in PR-AUC (paired difference 0.000 ± 0.001); BIC's smaller subsets (6-8 predictors) lowered PR-AUC by 0.017 ± 0.006. Citric acid is also the predictor least correlated with being a good wine in the training set ($r = -0.02$). The predictor set was fixed before the test set was loaded. Each pipeline selects its columns as its first step, which learns nothing from the data and so cannot leak.

## Duplicates and grouped CV

The white file has 937 exact duplicate rows (772 wines recorded 2-8 times). Copies match on all 11 predictors **and** quality, and no set of identical measurements has conflicting quality scores, so we treat them as the same wine measured more than once.

1. The split is made once, on the 3,961 unique wines, so each wine is entirely in train or entirely in test. `datasets/winequality-white-{train,test}.csv` are never regenerated or edited.
2. `wine_utils.load_train()` / `load_test()` give each unique wine a `wine_id` and merge the raw file back on all 12 columns, re-attaching the duplicates (train 3,926 rows, test 972 rows). Each returns `(X, y, groups)`.
3. All CV uses `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=211)` with `groups = wine_id`, so every copy of a wine stays in one fold. Notebook 03 asserts that no `wine_id` appears on both sides of any fold.

## Checks on the design decisions

The last section of `03_model_comparison.ipynb` re-runs CV on the final 10-predictor setup with one decision changed at a time, after the winner is saved, and ends with a table of every decision, its reason and where the evidence is.

- **Grouped vs ordinary folds:** ordinary folds put copies of about a third of validation wines in training, but PR-AUC changes by at most 0.010, so the conclusions do not depend on it. Grouping is kept because it is the correct design.
- **Log vs raw predictors:** the logs help Gaussian NB (+0.011 ± 0.006) and are neutral for logistic regression and LDA. They were chosen for distribution shape, not performance.
- **Dropping citric acid for LDA and NB:** no change for LDA (0.000 ± 0.001); Gaussian NB loses 0.006 ± 0.004, within noise, and would stay outside the one-SE cut-off with all 11, so the selection is unchanged.

## Reproducing the results

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# open the notebooks in Jupyter / VS Code and run 1-4 in order, top to bottom
```

Notebooks 1-4 run top to bottom from a fresh kernel. Re-running notebook 1 rewrites the split CSVs with identical contents (fixed seed). Notebook 05 must be run only once, after the model and threshold are frozen.

## Files

- `datasets/`: raw UCI files and the train/test split
- `results/`: CV fold scores, out-of-fold probabilities, the selected model and the operating point
- `AI_DECLARATION.md`: declaration of AI tool use
