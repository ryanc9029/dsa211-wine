# What changed in revision v2, and why

A plain-English summary for group members who don't work in the code. All numbers are in `NUMBERS_FOR_SLIDES.md`.

## 1. Each wine is counted once
**What:** The raw data had 937 rows that were exact repeats of another row (the same wine measured more than once). We now remove them and keep one row per wine: 3,961 wines, of which 3,168 are used for building the model and 793 are kept aside for the final test.
**Why:** A repeated wine would count twice, and its copies could end up on both sides of a cross-validation split, which makes the model look better than it is. With one row per wine, the standard stratified 10-fold cross-validation is safe.

## 2. One rule for every choice: "the simplest option that is about as good as the best"
**What:** Whenever we choose something using cross-validation (which predictors, which model), we use the same rule. Find the best score, allow a margin of one standard error below it, and among everything inside that margin pick the simplest option: the fewest predictors, or, for models, the lowest expected cost.
**Why:** Small differences in score are often just noise. When options score about the same, the simpler one is easier to explain and less likely to be fitting noise.

## 3. Each model chooses its own predictors
**What:** Before, all three models used the same predictors. Now each model (logistic regression, LDA, Naive Bayes) tries every possible combination of the 11 measurements (2,047 combinations) and keeps the simplest good one. Logistic regression and LDA both chose the same 6 measurements; Naive Bayes chose 3.
**Why:** We want simple models, and we show the trade-off openly. The selected logistic model uses 6 measurements instead of 11 and loses a little accuracy for it: PR-AUC is 0.535 instead of 0.551. The graph `subset_tradeoff.png` shows that accuracy levels off after about 6 predictors.
**How we keep it honest:** Picking the best of 2,047 combinations flatters the winner's score, so the reported performance comes from "nested" cross-validation. The whole selection is repeated without ever looking at the wines being scored.

## 4. Logistic regression is still the chosen model
It had the best score. LDA was close enough to count as a tie, and logistic regression won the tie because it costs less per wine at our cost settings.

## 5. The "audit" is gone; the test set is simply reported
**What:** The final notebook is renamed `05_test_and_refit.ipynb`. It no longer has a pass/fail check. It uses the test set once and shows the test results next to the cross-validation estimates, with the difference for each one.
**Result:** The test results are close to the cross-validation estimates. PR-AUC is 0.544 on test vs 0.535 in CV, and ROC-AUC is 0.820 vs 0.802.

## 6. Why "good" means a score of 7 or more
**What:** We added a chart of quality scores and a table comparing cut-offs of 6, 7 and 8 (`quality_distribution.png`).
**Why:** At 6 or more, two-thirds of wines would be "good", so it would no longer mean premium. At 8 or more, only about 11 good wines would fall in each cross-validation fold, which is too few. At 7 or more, 20.8% of wines are good: rare, but enough to work with. We could not access the original paper to check whether it uses 7 as a cut-off, so we don't claim that it does.

## 7. Where the model makes mistakes
**What:** A new error analysis in notebook 04, by true quality score.
**Finding:** Almost all mistakes are borderline wines. 82% of the wines wrongly sent to the panel scored 6, and 90% of the good wines that were missed scored 7. The model struggles mainly at the 6/7 boundary, where expert scores are themselves least certain.

## 8. The cost results (our one extension) were rerun
The best threshold is now 0.20. Per 1,000 wines, the panel tastes about 389 and catches 162 of the 208 good ones, which is 42% cheaper than tasting nothing. This holds as long as a missed good wine costs between about 1.4 and 12 tastings.

## 9. Tidy-ups after revision v2
No results changed. The two first notebooks are now numbered like the rest: `wine_white_data_split.ipynb` is `01_data_split.ipynb` and `EDA.ipynb` is `02_eda.ipynb`. The EDA notebook now prints each predictor's median for good and not-good wines, so the medians quoted in the report can be checked. The share of good wines is quoted as 20.8% everywhere, and white and red wines are compared on unique wines only (20.8% vs 13.5% good).

## Things to know before presenting
- **Alcohol is not one of the 6 predictors**, even though it is the single strongest one. Density and residual sugar together carry almost the same information, because a wine's density depends on its sugar and alcohol. Expect a question on this. The large density effect in the final model is mostly the alcohol effect.
- **The best threshold (0.20) sits at the low end of its bootstrap range (0.20-0.30).** Any threshold from about 0.18 to 0.31 gives almost the same cost, so this does not change the conclusions.
