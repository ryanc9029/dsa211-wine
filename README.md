# DSA211 Project: Identifying high-quality white Vinho Verde wine

DSA211 Statistical Learning (SMU), Topic 2: Classification System Evaluation.

## Research question

**Can physicochemical properties identify high-quality white Vinho Verde wine?**

A wine is *good* if its expert quality score is 7 or above (justified in the split notebook: at 6 or above two-thirds of wines would be "good", at 8 or above only about 11 good wines would fall in each CV fold). The application is a producer pre-screening batches before paying for an expert tasting panel: a false positive wastes one tasting ($c_{FP} = 1$), and a false negative sells a good wine at the standard price ($c_{FN} = 3$ in the base case, varied from 1/3 to 10 in the sensitivity analysis). We value a good batch's premium at about 4 tasting fees; a miss loses the premium but saves one fee, so it costs about 3 fees. The cost-based threshold analysis (notebook 04) is the project's one extension.

## Data

UCI Machine Learning Repository, *Wine Quality* (P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis, 2009, *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems 47(4)). Licence: CC BY 4.0. <https://archive.ics.uci.edu/dataset/186/wine+quality>

**White wine only.** White has 4,898 rows (3,961 unique) and red has 1,599 rows (1,359 unique); after deduplication there are 825 good white wines (20.8%) vs 184 good red wines (13.5%). Red's rare class is too small for stable 10-fold estimates (about 15 good wines per validation fold), and pooling the two would break LDA's equal-covariance assumption because their chemistry differs strongly. The red file is read only to compute these counts.

**One row per wine.** The white file has 937 repeat rows (772 wines recorded 2-8 times), identical on all 11 predictors and quality. We treat them as the same wine measured more than once and remove them before the split, leaving 3,961 unique wines: 3,168 in training and 793 in test (`datasets/winequality-white-{train,test}.csv`, stratified 80/20, `random_state=211`). Because every wine appears once, all CV uses ordinary `StratifiedKFold(n_splits=10, shuffle=True, random_state=211)`: each wine sits in exactly one fold, so no copy can leak between training and validation.

## Notebooks (run in this order)

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | `wine_white_data_split.ipynb` | Data quality checks, white-only justification, duplicate analysis (repeat rows removed), one stratified 80/20 split, and the quality-cut-off justification on the training set (`results/quality_distribution.png`). |
| 2 | `EDA.ipynb` | Training-set EDA of all 11 predictors; chooses the four log-transformed predictors (`wine_utils.LOG_FEATURES`). |
| 3 | `03_model_comparison.ipynb` | Per-model best-subset selection with the one-SE rule, evaluated by nested CV; compares logistic regression, LDA and Gaussian NB and selects one with the same rule. Writes `results/selected_model.json`, the nested out-of-fold probabilities and `results/subset_tradeoff.png`. |
| 4 | `04_cost_threshold.ipynb` | The single extension: cost-based threshold analysis on the nested out-of-fold probabilities, sensitivity across cost ratios, savings over the no-model policies, break-even range, bootstrap interval and error analysis by quality score. Writes `results/operating_point.json`. |
| 5 | `05_test_and_refit.ipynb` | Loads the test set once, scores the frozen model, subset and threshold next to the CV estimates, then refits on all 3,961 wines (odds ratios per SD with 95% CIs). |

Shared definitions (features, transforms, folds, costs, candidate models, the one-SE rule and the subset search) are in `wine_utils.py`.

## The one-SE rule (`wine_utils.one_se_choice`)

Used for every choice made by CV. For each candidate, take the mean CV PR-AUC and its SE (SD of the fold scores / √number of folds). The cut-off is the best mean minus the best candidate's SE. Among candidates at or above the cut-off, choose the simplest:
- **predictor subsets:** the fewest predictors, ties broken by the higher mean;
- **models:** the lowest expected cost per wine at $t^* = 0.25$, ties broken by the higher mean PR-AUC.

## Predictors: a separate subset for each classifier

All 11 predictors (four of them logged) are candidates, giving 2,047 subsets per classifier.
- **Final subset:** every subset is scored on the 10 CV folds of the full training set, and `one_se_choice` picks one.
- **Honest performance:** choosing among 2,047 subsets inflates the winner's CV score, so performance is estimated by nested CV. The same selection runs inside each of the 10 outer folds (inner 5-fold CV), and the chosen subset is scored on the held-out fold.

| Classifier | Final subset | Nested PR-AUC | All 11 predictors | Subset − all 11 (paired) |
|---|---|---|---|---|
| **Logistic (selected)** | 6: fixed acidity, log residual sugar, free sulfur dioxide, density, pH, log sulphates | 0.535 ± 0.018 | 0.551 ± 0.013 | −0.016 ± 0.009 |
| LDA | 6: the same as logistic | 0.526 ± 0.018 | 0.550 ± 0.016 | −0.024 ± 0.008 |
| Gaussian NB | 3: citric acid, pH, alcohol | 0.510 ± 0.020 | 0.514 ± 0.015 | −0.005 ± 0.012 |

Selection gives up a little PR-AUC for a much simpler model; the paired differences show how much. The logistic and LDA subsets leave out alcohol, whose information is carried by density and residual sugar. Each pipeline selects its columns and then standardises them inside the pipeline, so nothing is learned from the validation fold.

## Results in brief

- **Model:** logistic regression on 6 predictors, chosen over LDA (also within one SE) because it has the lower expected cost at $t^* = 0.25$.
- **Threshold:** 0.20 (bootstrap 95% interval 0.20-0.30). Per 1,000 wines: about 389 tasted, 162 of 208 good wines caught, a 41.6% saving over tasting nothing. The model beats both no-model policies for $1.40 < c_{FN}/c_{FP} < 12.26$.
- **Test set (793 wines):** PR-AUC 0.544 (CV 0.535 ± 0.018), ROC-AUC 0.820 (CV 0.802 ± 0.011), sensitivity 0.788, specificity 0.717, cost 0.357 per wine.

Every number used in the slides is listed, with its source cell, in `results/NUMBERS_FOR_SLIDES.md`. A plain-English list of the changes in this revision is in `results/CHANGES.md`.

## Reproducing the results

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# open the notebooks in Jupyter / VS Code and run them in the order above, top to bottom
```

Each notebook runs top to bottom from a fresh kernel. Re-running notebook 1 rewrites the split CSVs with identical contents (fixed seed). The first run of notebook 03 takes about 10 minutes; its subset scores are cached in `results/subsets/` (not in git), so later runs take about a minute. The test set is read only in notebook 05.

## Files

- `datasets/`: raw UCI files and the train/test split
- `results/`: the selected model and operating point (JSON), CV and test tables (CSV), figures, `NUMBERS_FOR_SLIDES.md` and `CHANGES.md`
- `AI_DECLARATION.md`: declaration of AI tool use
