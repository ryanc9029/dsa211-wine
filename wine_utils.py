"""Shared definitions for the white wine quality project.

Every notebook imports from here so the features, transformations, folds and
cost assumptions are defined once.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 2026

FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

# Right-skewed predictors that get a natural log (chosen in EDA.ipynb on the
# training set: raw skew > 0.75 and the log reduces |skew|). A log learns
# nothing from the data, so it is applied outside the CV pipeline.
LOG_FEATURES = ['volatile acidity', 'residual sugar', 'chlorides', 'sulphates']

# Application: a producer pre-screens batches before paying for an expert
# tasting panel. FP = one wasted tasting; FN = a good wine sold at standard
# price. Costs are in units of one tasting (c_FP = 1).
C_FP = 1
C_FN_BASE = 3
COST_RATIOS = [1/3, 1/2, 1, 2, 3, 5, 10]   # c_FN / c_FP

MIN_PREDICTORS = 5


def load_train(path='datasets/winequality-white-train.csv'):
    """Return (X, y) for the training set with y = 1 if quality >= 7."""
    df = pd.read_csv(path)
    return transform(df[FEATURES]), (df['quality'] >= 7).astype(int)


def transform(X):
    """Apply the fixed log transforms and rename the logged columns."""
    X = X.copy()
    for col in LOG_FEATURES:
        X[col] = np.log(X[col])
    return X.rename(columns={c: f'log {c}' for c in LOG_FEATURES})


def make_cv(k=10):
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)


def expected_cost(y_true, prob, threshold, c_fn, c_fp=C_FP):
    """Average cost per wine when wines with prob >= threshold are flagged."""
    y_true = np.asarray(y_true)
    pred = np.asarray(prob) >= threshold
    fp = np.sum(pred & (y_true == 0))
    fn = np.sum(~pred & (y_true == 1))
    return (c_fp * fp + c_fn * fn) / len(y_true)


def lasso_pipeline(degree, C):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(l1_ratio=1, C=C, solver='liblinear',
                                     max_iter=5000)),
    ])


def predictors_used(pipe, feature_names):
    """Original predictors with at least one nonzero term in a fitted lasso."""
    poly = pipe.named_steps['poly']
    coef = pipe.named_steps['model'].coef_.ravel()
    powers = poly.powers_[coef != 0]
    return [f for j, f in enumerate(feature_names) if powers[:, j].any()]


class OneSELassoLogit(BaseEstimator, ClassifierMixin):
    """L1 logistic regression whose degree and C are chosen by inner CV.

    Selection rule (fixed before looking at results):
      1. Grid over degree x C, scored by PR-AUC with stratified inner CV.
      2. One-SE rule: keep candidates within one SE of the best mean.
         Prefer degree 1 over degree 2, then the smallest C (strongest
         penalty, fewest terms).
      3. If the chosen model uses fewer than MIN_PREDICTORS of the original
         predictors, step to the next larger C at the same degree until it
         uses at least MIN_PREDICTORS.

    Wrapping the whole procedure as one estimator lets the outer CV score
    the tuning procedure itself (nested CV), so its score is comparable with
    the untuned models.
    """

    def __init__(self, degrees=(1, 2), Cs=tuple(np.logspace(-3, 1, 13)),
                 inner_k=5):
        self.degrees = degrees
        self.Cs = Cs
        self.inner_k = inner_k

    def fit(self, X, y):
        grid = GridSearchCV(
            lasso_pipeline(1, 1.0),
            {'poly__degree': list(self.degrees), 'model__C': list(self.Cs)},
            scoring='average_precision', cv=make_cv(self.inner_k),
            refit=False, n_jobs=-1,
        )
        grid.fit(X, y)
        res = pd.DataFrame({
            'degree': grid.cv_results_['param_poly__degree'].astype(int),
            'C': grid.cv_results_['param_model__C'].astype(float),
            'mean': grid.cv_results_['mean_test_score'],
            'se': grid.cv_results_['std_test_score'] / np.sqrt(self.inner_k),
        })
        best = res.loc[res['mean'].idxmax()]
        ok = res[res['mean'] >= best['mean'] - best['se']]
        pick = ok.sort_values(['degree', 'C']).iloc[0]

        names = list(X.columns)
        Cs = sorted(res.loc[res['degree'] == pick['degree'], 'C'])
        for C in Cs[Cs.index(pick['C']):]:
            pipe = lasso_pipeline(int(pick['degree']), C).fit(X, y)
            used = predictors_used(pipe, names)
            if len(used) >= MIN_PREDICTORS:
                break

        self.cv_results_ = res
        self.one_se_pick_ = (int(pick['degree']), float(pick['C']))
        self.degree_, self.C_ = int(pick['degree']), float(C)
        self.predictors_used_ = used
        self.pipeline_ = pipe
        self.classes_ = pipe.classes_
        return self

    def predict_proba(self, X):
        return self.pipeline_.predict_proba(X)

    def predict(self, X):
        return self.pipeline_.predict(X)


def candidate_models():
    """The candidates compared in 03_model_comparison.ipynb."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    return {
        'Logistic (all 11)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=np.inf, max_iter=5000)),
        ]),
        'L1 logistic (CV-tuned)': OneSELassoLogit(),
        'LDA': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearDiscriminantAnalysis()),
        ]),
        'Gaussian NB': Pipeline([
            ('scaler', StandardScaler()),
            ('model', GaussianNB()),
        ]),
    }


def build_selected(spec):
    """Rebuild the frozen specification saved by 03_model_comparison.ipynb."""
    if spec['name'] == 'L1 logistic (CV-tuned)':
        return lasso_pipeline(spec['degree'], spec['C'])
    return candidate_models()[spec['name']]
