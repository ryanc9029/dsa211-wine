# Numbers for the slides

Generated from the saved results after running all notebooks top to bottom on branch `revision/v2`. Each section names the notebook cell it comes from (cells numbered from 1 in notebook order). CV figures are mean ± SE over 10 folds unless stated.

## Quality distribution and cut-off (training set, 3,168 wines)
Source: `wine_white_data_split.ipynb`, cell 35 and `wine_white_data_split.ipynb`, cell 36

| Quality | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| Wines | 16 | 122 | 940 | 1430 | 551 | 105 | 4 |

| Good if quality ≥ | Good wines | Share good | Good wines per CV fold |
|---|---|---|---|
| 6 | 2090 | 66.0% | 209.0 |
| 7 | 660 | 20.8% | 66.0 |
| 8 | 109 | 3.4% | 10.9 |

## Class balance
Source: `wine_white_data_split.ipynb`, cell 5 (raw file), `03_model_comparison.ipynb`, cell 5 (folds), `05_test_and_refit.ipynb`, cell 6 (test)

- Raw white file: 4,898 rows, 21.6% good; after removing 937 repeat rows: 3,961 unique wines.
- Training: 3168 wines, 660 good (20.8%); every CV fold has 66 good wines (316-317 wines per fold).
- Test: 793 wines, share good 20.8%.

## Chosen predictor subset for each model
Source: `03_model_comparison.ipynb`, cell 11 (selection) and `03_model_comparison.ipynb`, cell 13 (trade-off graph, `results/subset_tradeoff.png`)

| Model | Size | Predictors | Best size (CV PR-AUC) | One-SE cut-off | Chosen subset CV PR-AUC (optimistic) |
|---|---|---|---|---|---|
| Logistic | 6 | fixed acidity, log residual sugar, free sulfur dioxide, density, pH, log sulphates | 8 (0.556) | 0.539 | 0.545 |
| LDA | 6 | fixed acidity, log residual sugar, free sulfur dioxide, density, pH, log sulphates | 9 (0.557) | 0.541 | 0.542 |
| Gaussian NB | 3 | citric acid, pH, alcohol | 7 (0.532) | 0.516 | 0.518 |

## Nested CV performance and the all-11 reference
Source: `03_model_comparison.ipynb`, cell 18

| Model | Subset sizes in outer folds | Nested PR-AUC | All-11 PR-AUC | Subset − all 11 (paired) | Nested ROC-AUC | All-11 ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic | 4-7 | 0.535 ± 0.018 | 0.551 ± 0.013 | -0.016 ± 0.009 | 0.802 ± 0.011 | 0.819 ± 0.009 |
| LDA | 4-7 | 0.526 ± 0.018 | 0.550 ± 0.016 | -0.024 ± 0.008 | 0.798 ± 0.011 | 0.815 ± 0.010 |
| Gaussian NB | 2-5 | 0.510 ± 0.020 | 0.514 ± 0.015 | -0.005 ± 0.012 | 0.792 ± 0.010 | 0.796 ± 0.009 |

## Model selection (one-SE rule)
Source: `03_model_comparison.ipynb`, cell 29

- Best nested PR-AUC: Logistic (0.535 ± 0.018); one-SE cut-off **0.516**.
- Within the cut-off: Logistic and LDA (Gaussian NB, 0.510, is just below).
- **Selected: Logistic with 6 predictors** (lowest expected cost at t* = 0.25 among eligible models).

## Each model at t* = 0.25 (nested out-of-fold)
Source: `03_model_comparison.ipynb`, cell 23 (table) and `03_model_comparison.ipynb`, cell 27 (confusion matrices, pooled over 3,168 wines)

| Model | Sensitivity @0.25 | Specificity @0.25 | Cost per wine @0.25 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| Logistic | 0.676 ± 0.021 | 0.79 ± 0.01 | 0.369 ± 0.016 | 1982 | 526 | 214 | 446 |
| LDA | 0.671 ± 0.015 | 0.787 ± 0.01 | 0.374 ± 0.015 | 1974 | 534 | 217 | 443 |
| Gaussian NB | 0.721 ± 0.018 | 0.738 ± 0.01 | 0.381 ± 0.013 | 1852 | 656 | 184 | 476 |

## Operating threshold (selected model, r = c_FN/c_FP = 3)
Source: `04_cost_threshold.ipynb`, cell 13 and `04_cost_threshold.ipynb`, cell 23

- Optimal threshold: **0.20** (theoretical t* = 0.25); 95% bootstrap interval **0.20-0.30** (2,000 resamples of wines).
- Sensitivity 0.779, specificity 0.714, precision 0.417, cost 0.365 per wine.

## Per 1,000 wines at the operating threshold
Source: `04_cost_threshold.ipynb`, cell 13

- Sent to the panel: 388.9
- of which good (caught): 162.2
- of which not good (wasted tastings): 226.6
- Good wines not sent (missed): 46.1
- Good wines in total: 208.3
- Precision of panel referrals (%): 41.7
- Base rate of good wines (%): 20.8
- Saving over the better no-model policy (taste none): 41.6% (0.260 per wine)

## No-model costs (per wine, r = 3)
Source: `04_cost_threshold.ipynb`, cell 26

- Taste none: 0.625; taste all: 0.792; model: 0.365.

## Break-even cost ratios (threshold frozen at the operating point)
Source: `04_cost_threshold.ipynb`, cell 19

- Counts at threshold 0.20: TP = 514, FP = 718, TN = 1790, FN = 146.
- Model beats both no-model policies for FP/TP = 1.40 < r < TN/FN = 12.26.

## Cost-ratio sensitivity table
Source: `04_cost_threshold.ipynb`, cell 9

| r | t* theory | t optimal | cost @ t optimal | taste none | taste all | better no-model | saving (abs) | saving (%) |
|---|---|---|---|---|---|---|---|---|
| 1/3 | 0.750 | 0.67 | 0.069 | 0.069 | 0.792 | taste none | 0.000 | 0.2 |
| 1/2 | 0.667 | 0.67 | 0.101 | 0.104 | 0.792 | taste none | 0.003 | 3.3 |
| 1 | 0.500 | 0.50 | 0.182 | 0.208 | 0.792 | taste none | 0.027 | 12.7 |
| 2 | 0.333 | 0.30 | 0.290 | 0.417 | 0.792 | taste none | 0.126 | 30.3 |
| 3 | 0.250 | 0.20 | 0.365 | 0.625 | 0.792 | taste none | 0.260 | 41.6 |
| 5 | 0.167 | 0.20 | 0.457 | 1.042 | 0.792 | taste all | 0.335 | 42.3 |
| 10 | 0.091 | 0.11 | 0.601 | 2.083 | 0.792 | taste all | 0.191 | 24.1 |

## Error analysis at the operating threshold
Source: `04_cost_threshold.ipynb`, cell 16

| Quality | Wines | Share sent to panel | False positives | % of all FPs | False negatives | % of all FNs |
|---|---|---|---|---|---|---|
| 3 | 16 | 37.5% | 6 | 0.8% | 0 | 0.0% |
| 4 | 122 | 9.8% | 12 | 1.7% | 0 | 0.0% |
| 5 | 940 | 11.7% | 110 | 15.3% | 0 | 0.0% |
| 6 | 1430 | 41.3% | 590 | 82.2% | 0 | 0.0% |
| 7 | 551 | 76.2% | 0 | 0.0% | 131 | 89.7% |
| 8 | 105 | 85.7% | 0 | 0.0% | 15 | 10.3% |
| 9 | 4 | 100.0% | 0 | 0.0% | 0 | 0.0% |

## Test set (793 wines) next to CV
Source: `05_test_and_refit.ipynb`, cell 8

| Metric | CV mean ± SE | Test | Absolute difference |
|---|---|---|---|
| PR-AUC | 0.535 ± 0.018 | 0.544 | 0.010 |
| ROC-AUC | 0.802 ± 0.011 | 0.820 | 0.018 |
| Sensitivity @0.2 | 0.779 ± 0.017 | 0.788 | 0.009 |
| Specificity @0.2 | 0.714 ± 0.009 | 0.717 | 0.003 |
| Cost per wine @0.2 | 0.365 ± 0.011 | 0.357 | 0.008 |

## Refit on all 3,961 wines: odds ratios per SD
Source: `05_test_and_refit.ipynb`, cell 11

| Predictor | Odds ratio per SD | 95% CI | p-value |
|---|---|---|---|
| log residual sugar | 3.12 | 2.69-3.62 | < 0.001 |
| pH | 1.90 | 1.70-2.11 | < 0.001 |
| fixed acidity | 1.73 | 1.54-1.94 | < 0.001 |
| free sulfur dioxide | 1.24 | 1.13-1.37 | < 0.001 |
| log sulphates | 1.23 | 1.13-1.34 | < 0.001 |
| density | 0.12 | 0.10-0.14 | < 0.001 |

Density largely stands in for alcohol (for fixed sugar, lower density means more alcohol), so read the odds ratios together.
