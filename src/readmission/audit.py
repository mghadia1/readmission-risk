"""Subgroup audit: does the model rank, flag, and calibrate equally across groups?

Three questions per group, each with a different fairness meaning:

- **AUC**: does the model separate readmitted from not-readmitted patients as
  well inside this group? (ranking quality)
- **Recall at the capacity threshold**: of this group's readmissions, what share
  would get a follow-up call? (who benefits from the intervention)
- **Calibration gap**: mean predicted risk minus observed rate. A positive gap
  over-states the group's risk, and a negative one under-states it.

These can disagree, and no model can equalise all of them when base rates
differ. The audit reports them; it does not declare the model "fair".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from readmission.modeling import bootstrap_auc_ci
from sklearn.metrics import roc_auc_score

MIN_POSITIVES = 30   # below this, AUC and recall are too noisy to report


def age_group(age_mid: pd.Series) -> pd.Series:
    return pd.cut(age_mid, bins=[0, 50, 70, 200], right=False, labels=["under 50", "50-69", "70+"]).astype(str)


def subgroup_table(frame: pd.DataFrame, y: np.ndarray, p: np.ndarray, threshold: float,
                   columns: dict[str, pd.Series], n_boot: int = 500) -> pd.DataFrame:
    rows = []
    flagged = p >= threshold
    for attribute, values in columns.items():
        values = np.asarray(values)
        for group in sorted(pd.unique(values)):
            mask = values == group
            positives = int(y[mask].sum())
            row = {
                "attribute": attribute, "group": group, "encounters": int(mask.sum()),
                "readmitted": positives,
                "observed_rate": float(y[mask].mean()),
                "mean_predicted": float(p[mask].mean()),
                "calibration_gap": float(p[mask].mean() - y[mask].mean()),
                "flagged_pct": float(100 * flagged[mask].mean()),
            }
            if positives >= MIN_POSITIVES and positives < mask.sum():
                row["roc_auc"] = float(roc_auc_score(y[mask], p[mask]))
                low, high = bootstrap_auc_ci(y[mask], p[mask], frame["patient_nbr"].to_numpy()[mask], n_boot=n_boot)
                row["roc_auc_ci_low"], row["roc_auc_ci_high"] = low, high
                row["recall_at_threshold"] = float((flagged[mask] & (y[mask] == 1)).sum() / positives)
            else:
                row["note"] = f"fewer than {MIN_POSITIVES} readmissions; not reported"
            rows.append(row)
    return pd.DataFrame(rows)
