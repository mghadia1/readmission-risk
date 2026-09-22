"""Splits, models, and metrics.

Three rules run through this module:

1. **Split by patient, never by row.** About 16,000 patients have more than one
   encounter. A row split puts the same person in train and test, and the model
   is scored partly on people it has already seen.
2. **The test split is scored once**, after the model and the threshold are
   chosen on validation.
3. **Probabilities are reported, not just ranks.** A risk score used to
   allocate follow-up calls has to mean what it says, so calibration is
   measured alongside ranking metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "readmitted_30d"
GROUP = "patient_nbr"
NUMERIC = [
    "age_mid", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
# race is deliberately absent: it is used to audit the model, not to drive it.
CATEGORICAL = [
    "gender", "admission_type", "discharge_disposition", "admission_source", "max_glu_serum",
    "a1c_result", "insulin", "metformin", "med_change", "diabetes_med", "medical_specialty",
    "payer_code", "diag_1_group", "diag_2_group", "diag_3_group",
]
FEATURES = NUMERIC + CATEGORICAL


def _sql(name: str) -> str:
    return resources.files("readmission").joinpath("sql", f"{name}.sql").read_text()


def connect(encounters: Path | pd.DataFrame):
    con = duckdb.connect()
    if isinstance(encounters, pd.DataFrame):
        con.register("raw_encounters", encounters)
    else:
        # The file marks missing values with "?". Diagnosis codes must stay text
        # ("250.83", "V57"), or type sniffing would turn some into numbers.
        path = str(encounters).replace("'", "''")
        con.execute(f"""
            CREATE VIEW raw_encounters AS
            SELECT * FROM read_csv('{path}', nullstr = '?',
                types = {{'diag_1': 'VARCHAR', 'diag_2': 'VARCHAR', 'diag_3': 'VARCHAR',
                          'payer_code': 'VARCHAR', 'medical_specialty': 'VARCHAR'}})
        """)
    con.execute(_sql("cohort"))
    return con


def load_cohort(con) -> pd.DataFrame:
    return con.execute("SELECT * FROM cohort").df()


def cohort_summary(con) -> pd.DataFrame:
    return con.execute(_sql("cohort_summary")).df()


# ---- splits -----------------------------------------------------------------

@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def grouped_split(frame: pd.DataFrame, seed: int = 0) -> Split:
    """60/20/20 by patient, stratified on the outcome.

    Five StratifiedGroupKFold folds: fold 0 is test, fold 1 validation, the
    other three training. Every encounter of a patient lands in one split.
    """
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_of = np.empty(len(frame), dtype=int)
    for k, (_, idx) in enumerate(folds.split(frame, frame[TARGET], frame[GROUP])):
        fold_of[idx] = k
    return Split(np.flatnonzero(fold_of >= 2), np.flatnonzero(fold_of == 1), np.flatnonzero(fold_of == 0))


def row_split(frame: pd.DataFrame, seed: int = 0) -> Split:
    """The WRONG split, kept only to measure how much it flatters the model."""
    order = np.random.default_rng(seed).permutation(len(frame))
    n = len(frame)
    return Split(order[: int(0.6 * n)], order[int(0.6 * n): int(0.8 * n)], order[int(0.8 * n):])


def shared_patients(frame: pd.DataFrame, split: Split) -> int:
    """Patients who appear in both train and test."""
    return len(set(frame[GROUP].iloc[split.train]) & set(frame[GROUP].iloc[split.test]))


# ---- models -----------------------------------------------------------------

def logistic_model(c: float = 1.0) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=c, max_iter=2000))])


def boosting_model(learning_rate: float = 0.05, max_leaf_nodes: int = 31, seed: int = 0) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), CATEGORICAL),
    ], verbose_feature_names_out=False)
    categorical_mask = [False] * len(NUMERIC) + [True] * len(CATEGORICAL)
    clf = HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, max_iter=500,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
        categorical_features=categorical_mask, random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def candidates() -> dict[str, Pipeline]:
    return {
        "logistic C=0.1": logistic_model(0.1),
        "logistic C=1": logistic_model(1.0),
        "boosting lr=0.05 leaves=15": boosting_model(0.05, 15),
        "boosting lr=0.05 leaves=31": boosting_model(0.05, 31),
        "boosting lr=0.1 leaves=31": boosting_model(0.1, 31),
    }


# ---- metrics ----------------------------------------------------------------

def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Mean |observed rate - mean predicted| over equal-count bins, weighted by size."""
    order = np.argsort(p)
    total = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            total += len(chunk) * abs(y[chunk].mean() - p[chunk].mean())
    return float(total / len(y))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    order = np.argsort(p)
    rows = [{"mean_predicted": p[c].mean(), "observed_rate": y[c].mean(), "n": len(c)}
            for c in np.array_split(order, bins) if len(c)]
    return pd.DataFrame(rows)


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "prevalence": float(y.mean()),
    }


def capacity_threshold(p_val: np.ndarray, capacity: float) -> float:
    """Score above which the top `capacity` share of validation encounters fall.

    The threshold answers an operational question ("we can call 10% of
    discharged patients; whom?") rather than maximising a metric.
    """
    return float(np.quantile(p_val, 1 - capacity))


def at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    flagged = p >= threshold
    tp = int((flagged & (y == 1)).sum())
    precision = tp / flagged.sum() if flagged.sum() else float("nan")
    return {
        "flagged_pct": float(100 * flagged.mean()),
        "precision": float(precision),
        "recall": float(tp / (y == 1).sum()) if (y == 1).sum() else float("nan"),
        "lift": float(precision / y.mean()) if y.mean() else float("nan"),
    }


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_boot: int = 1000,
                     seed: int = 0) -> tuple[float, float]:
    """95% CI for ROC-AUC, resampling patients (not encounters) with replacement."""
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    values = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(unique), len(unique))
        idx = np.concatenate([members[g] for g in pick])
        if 0 < y[idx].sum() < len(idx):
            values.append(roc_auc_score(y[idx], p[idx]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def leakage_check(frame: pd.DataFrame, make_model, seeds=range(5)) -> dict:
    """Test ROC-AUC under grouped and row splits, repeated over several seeds.

    The row split lets a patient's other encounters sit in training while one
    is scored in test. Whether that inflates the score depends on how much a
    patient's encounters resemble each other in the features; it is measured
    here rather than assumed.
    """
    X, y = frame[FEATURES], frame[TARGET].to_numpy()
    results = {"grouped": [], "row": []}
    shared = []
    for seed in seeds:
        for kind, splitter in (("grouped", grouped_split), ("row", row_split)):
            split = splitter(frame, seed=seed)
            model = make_model()
            model.fit(X.iloc[split.train], y[split.train])
            results[kind].append(roc_auc_score(y[split.test], model.predict_proba(X.iloc[split.test])[:, 1]))
            if kind == "row":
                shared.append(shared_patients(frame, split))
    g, r = np.array(results["grouped"]), np.array(results["row"])
    return {
        "seeds": len(g),
        "grouped_roc_auc_mean": float(g.mean()), "grouped_roc_auc_sd": float(g.std(ddof=1)),
        "row_roc_auc_mean": float(r.mean()), "row_roc_auc_sd": float(r.std(ddof=1)),
        "row_minus_grouped": float(r.mean() - g.mean()),
        "row_split_shared_patients_mean": float(np.mean(shared)),
        "grouped_split_shared_patients": 0,
    }
