"""Shared definitions for the white wine quality project.

Every notebook imports from here so the features, transformations, folds and
cost assumptions are defined once.
"""

import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Same seed as the train/test split in wine_white_data_split.ipynb.
SEED = 211

# All 11 measured predictors. Each classifier uses its own subset of these,
# chosen in 03_model_comparison.ipynb and saved in results/selected_model.json.
ALL_FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]
COLUMNS = ALL_FEATURES + ['quality']

RAW_PATH = 'datasets/winequality-white.csv'
TRAIN_PATH = 'datasets/winequality-white-train.csv'
TEST_PATH = 'datasets/winequality-white-test.csv'
# The split CSVs hold one row per unique wine (duplicates were dropped before
# the split in wine_white_data_split.ipynb).
N_TRAIN, N_TEST = 3168, 793

# Right-skewed predictors that get a natural log (chosen in EDA.ipynb on the
# training set: raw skew > 0.75 and the log reduces |skew|). A log learns
# nothing from the data, so it is applied outside the CV pipeline.
LOG_FEATURES = ['volatile acidity', 'residual sugar', 'chlorides', 'sulphates']


def log_name(col):
    """Column name of a predictor after transform()."""
    return f'log {col}' if col in LOG_FEATURES else col


# All 11 predictors as named after transform().
ALL_COLUMNS = [log_name(c) for c in ALL_FEATURES]

# Application: a producer pre-screens batches before paying for an expert
# tasting panel. FP = one wasted tasting; FN = a good wine sold at standard
# price. Costs are in units of one tasting (c_FP = 1).
C_FP = 1
C_FN_BASE = 3
COST_RATIOS = [1/3, 1/2, 1, 2, 3, 5, 10]   # c_FN / c_FP


def load_split(split):
    """One row per unique wine of one split ('train' or 'test'), with quality."""
    path, n = {'train': (TRAIN_PATH, N_TRAIN), 'test': (TEST_PATH, N_TEST)}[split]
    df = pd.read_csv(path)
    assert len(df) == n and not df.duplicated().any()
    return df


def load_all():
    """All 3,961 unique wines (train + test), for the final refit only."""
    df = pd.concat([load_split('train'), load_split('test')], ignore_index=True)
    assert len(df) == N_TRAIN + N_TEST and not df.duplicated().any()
    return df


def _split_xy(df):
    df = df.reset_index(drop=True)
    return transform(df[ALL_FEATURES]), (df['quality'] >= 7).astype(int)


def load_train():
    """Return (X, y) for the training set, y = 1 if quality >= 7.

    X holds all 11 transformed predictors; each pipeline from
    candidate_models() selects its own columns.
    """
    return _split_xy(load_split('train'))


def load_test():
    """Return (X, y) for the test set. Load it once, in notebook 05."""
    return _split_xy(load_split('test'))


def transform(X):
    """Apply the fixed log transforms and rename the logged columns."""
    X = X.copy()
    for col in LOG_FEATURES:
        X[col] = np.log(X[col])
    return X.rename(columns={c: f'log {c}' for c in LOG_FEATURES})


def make_cv(k=10):
    """Stratified k-fold CV. Each row is a different wine, so folds cannot leak."""
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)


def expected_cost(y_true, prob, threshold, c_fn, c_fp=C_FP):
    """Average cost per wine when wines with prob >= threshold are flagged."""
    y_true = np.asarray(y_true)
    pred = np.asarray(prob) >= threshold
    fp = np.sum(pred & (y_true == 0))
    fn = np.sum(~pred & (y_true == 1))
    return (c_fp * fp + c_fn * fn) / len(y_true)


def no_model_costs(share_good, c_fn, c_fp=C_FP):
    """Cost per wine of the two policies that use no model."""
    return {'taste none': c_fn * share_good,
            'taste all': c_fp * (1 - share_good)}


def select_columns(columns):
    """Pipeline step that keeps the named columns and drops the rest.

    It learns nothing from the data (only which columns to keep), so it
    cannot leak information between folds.
    """
    return ColumnTransformer([('keep', 'passthrough', list(columns))],
                             remainder='drop', verbose_feature_names_out=False)


def candidate_models(columns=ALL_COLUMNS):
    """The three classifiers, each starting by selecting `columns`.

    Pipelines can therefore always be fitted on a frame holding all 11
    transformed predictors.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    models = {
        'Logistic': LogisticRegression(C=np.inf, max_iter=5000),
        'LDA': LinearDiscriminantAnalysis(),
        'Gaussian NB': GaussianNB(),
    }
    return {name: Pipeline([('select', select_columns(columns)),
                            ('scaler', StandardScaler()),
                            ('model', model)])
            for name, model in models.items()}


def build_selected(spec):
    """Rebuild the frozen specification saved by 03_model_comparison.ipynb."""
    return candidate_models(spec['predictors'])[spec['name']]


def _score_subsets(name, subsets, X, y, n_splits):
    """Inner-CV PR-AUC of each subset for one model; warnings are counted, not shown."""
    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(X, y))
    out = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for cols in subsets:
            model = candidate_models(cols)[name]
            s = [average_precision_score(y.iloc[va], model.fit(X.iloc[tr], y.iloc[tr])
                                         .predict_proba(X.iloc[va])[:, 1]) for tr, va in splits]
            out.append(('|'.join(cols), len(cols), np.mean(s), np.std(s, ddof=1) / np.sqrt(n_splits)))
    n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return out, n_conv, len(caught) - n_conv


def subset_cv_scores(name, X, y, n_splits=5, n_jobs=-1):
    """Inner stratified-CV PR-AUC (mean, SE) of every non-empty subset of X's columns.

    Returns a DataFrame with one row per subset (2,047 for 11 columns) and the
    number of warnings suppressed. Runs in parallel over subsets.
    """
    from joblib import Parallel, delayed, parallel_config, cpu_count
    # Worker processes must be able to import this module.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in os.environ.get('PYTHONPATH', '').split(os.pathsep):
        os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, [here, os.environ.get('PYTHONPATH')]))

    subsets = [list(c) for k in range(1, X.shape[1] + 1) for c in combinations(X.columns, k)]
    n_chunks = 4 * cpu_count()
    with parallel_config(backend='loky', inner_max_num_threads=1):
        res = Parallel(n_jobs=n_jobs)(delayed(_score_subsets)(name, subsets[i::n_chunks], X, y, n_splits)
                                      for i in range(n_chunks))
    scores = pd.DataFrame([r for rows, _, _ in res for r in rows], columns=['subset', 'size', 'mean', 'se'])
    warn = {'convergence': sum(r[1] for r in res), 'other': sum(r[2] for r in res)}
    return scores, warn


def best_of_size(scores, k):
    """Columns of the subset with the highest inner-CV mean among subsets of size k."""
    return scores[scores['size'] == k].sort_values('mean').iloc[-1]['subset'].split('|')
