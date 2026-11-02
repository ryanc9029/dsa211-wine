"""Shared definitions for the white wine quality project.

Every notebook imports from here so the features, transformations, folds and
cost assumptions are defined once.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Same seed as the train/test split in wine_white_data_split.ipynb.
SEED = 211

FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]
COLUMNS = FEATURES + ['quality']

RAW_PATH = 'datasets/winequality-white.csv'
TRAIN_PATH = 'datasets/winequality-white-train.csv'
TEST_PATH = 'datasets/winequality-white-test.csv'
N_TRAIN_WINES, N_TEST_WINES = 3168, 793
N_TRAIN_ROWS, N_TEST_ROWS = 3926, 972

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


def _attach_duplicates(split):
    """Raw rows (duplicates included) belonging to one split, with wine_id.

    The split CSVs hold one row per unique wine. Each unique wine gets a
    wine_id (train 0..3167, test 3168..3960), then the raw file is merged
    back on all 12 columns so every repeat measurement of a wine carries
    that wine's id. Only the CSV of the requested split is read.
    """
    raw = pd.read_csv(RAW_PATH, sep=';')
    # Repeat rows must be exact copies: matching on the 11 features alone
    # gives the same duplicates as matching on all 12 columns, so no wine
    # has conflicting quality scores.
    assert raw.duplicated(subset=FEATURES).sum() == raw.duplicated().sum()

    path, first_id, n_rows = {'train': (TRAIN_PATH, 0, N_TRAIN_ROWS),
                              'test': (TEST_PATH, N_TRAIN_WINES, N_TEST_ROWS)}[split]
    wines = pd.read_csv(path)
    wines['wine_id'] = first_id + np.arange(len(wines))
    df = raw.merge(wines, on=COLUMNS, how='inner', validate='many_to_one')
    assert len(df) == n_rows
    return df.assign(split=split)


def load_all():
    """All 4,898 raw rows with wine_id and split, for the final refit only."""
    df = pd.concat([_attach_duplicates('train'), _attach_duplicates('test')],
                   ignore_index=True)
    assert len(df) == len(pd.read_csv(RAW_PATH, sep=';')), 'unmatched raw rows'
    assert df['wine_id'].nunique() == N_TRAIN_WINES + N_TEST_WINES
    return df


def _split_xyg(df):
    df = df.reset_index(drop=True)
    return (transform(df[FEATURES]), (df['quality'] >= 7).astype(int),
            df['wine_id'])


def load_train():
    """Return (X, y, groups) for the training set, y = 1 if quality >= 7."""
    return _split_xyg(_attach_duplicates('train'))


def load_test():
    """Return (X, y, groups) for the test set. Load it once, in notebook 05."""
    return _split_xyg(_attach_duplicates('test'))


def transform(X):
    """Apply the fixed log transforms and rename the logged columns."""
    X = X.copy()
    for col in LOG_FEATURES:
        X[col] = np.log(X[col])
    return X.rename(columns={c: f'log {c}' for c in LOG_FEATURES})


def make_cv(k=10):
    """Stratified folds that keep every copy of a wine in the same fold.

    Always call cv.split(X, y, groups) with the wine_id groups.
    """
    return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)


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


def candidate_models():
    """The candidates compared in 03_model_comparison.ipynb."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    return {
        'Logistic': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=np.inf, max_iter=5000)),
        ]),
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
    return candidate_models()[spec['name']]
