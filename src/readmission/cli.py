"""Command line: `readmission download | run`.

`run` does everything in one order that cannot be rearranged by accident:
select a model on validation, set the threshold on validation, then score the
test split exactly once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from readmission import audit, charts, data, modeling as m

CAPACITY = 0.10   # follow-up calls for the highest-risk 10% of discharges


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="readmission")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="fetch and checksum the UCI files (~19 MB)")
    dl.add_argument("--destination", type=Path, default=Path("data/raw"))
    rn = sub.add_parser("run", help="select, threshold, test once, audit; write outputs/")
    rn.add_argument("--raw", type=Path, default=Path("data/raw"))
    rn.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)

    if args.command == "download":
        for path in data.download(args.destination):
            print(path)
    else:
        print(json.dumps(run(args.raw, args.out), indent=2, default=float))


def run(raw: Path, out: Path) -> dict:
    for name, expected in data.FILES.items():
        data.verify(raw / name, expected)
    out.mkdir(parents=True, exist_ok=True)

    con = m.connect(raw / "diabetic_data.csv")
    cohort_steps = m.cohort_summary(con)
    frame = m.load_cohort(con)
    X, y = frame[m.FEATURES], frame[m.TARGET].to_numpy()
    split = m.grouped_split(frame)

    # 1. model selection on validation (PR-AUC: the positive class is 11%)
    selection, fitted = [], {}
    for name, model in m.candidates().items():
        model.fit(X.iloc[split.train], y[split.train])
        p_val = model.predict_proba(X.iloc[split.val])[:, 1]
        selection.append({"model": name, **m.scores(y[split.val], p_val)})
        fitted[name] = model
    selection = pd.DataFrame(selection).sort_values("pr_auc", ascending=False)
    best_name = selection.iloc[0]["model"]
    best = fitted[best_name]
    baseline_name = selection[selection["model"].str.startswith("logistic")].iloc[0]["model"]

    # 2. threshold from validation, for a fixed follow-up capacity
    p_val = best.predict_proba(X.iloc[split.val])[:, 1]
    threshold = m.capacity_threshold(p_val, CAPACITY)

    # 3. the single test evaluation
    p_test = best.predict_proba(X.iloc[split.test])[:, 1]
    p_test_baseline = fitted[baseline_name].predict_proba(X.iloc[split.test])[:, 1]
    y_test = y[split.test]
    test_frame = frame.iloc[split.test].reset_index(drop=True)
    test_scores = m.scores(y_test, p_test)
    test_scores["roc_auc_ci"] = m.bootstrap_auc_ci(y_test, p_test, test_frame["patient_nbr"].to_numpy())
    operating_point = m.at_threshold(y_test, p_test, threshold)

    curve = pd.DataFrame([
        {"capacity_pct": c, **m.at_threshold(y_test, p_test, np.quantile(p_test, 1 - c / 100))}
        for c in range(1, 101)
    ])

    # 4. does a row split flatter the model? One seed per design is not enough
    # to tell (the first version compared one of each and the split-to-split
    # noise was as large as the difference), so each design is repeated.
    leak = m.leakage_check(frame, lambda: _same_model(best_name), seeds=range(5))

    # 5. subgroup audit on test
    table = audit.subgroup_table(
        test_frame, y_test, p_test, threshold,
        {"race": test_frame["race"], "gender": test_frame["gender"], "age": audit.age_group(test_frame["age_mid"])},
    )

    # 6. permutation importance on validation (never on test)
    importance = permutation_importance(best, X.iloc[split.val], y[split.val], scoring="average_precision",
                                        n_repeats=5, random_state=0)
    importance = pd.DataFrame({"feature": m.FEATURES, "mean_drop_pr_auc": importance.importances_mean,
                               "std": importance.importances_std}).sort_values("mean_drop_pr_auc", ascending=False)

    reliability = m.reliability(y_test, p_test)
    for name, table_ in {"cohort_steps": cohort_steps, "model_selection": selection, "subgroups": table,
                         "capacity_curve": curve, "reliability_test": reliability,
                         "permutation_importance": importance}.items():
        table_.to_csv(out / f"{name}.csv", index=False)
    charts.reliability(reliability, out / "calibration.png")
    charts.capacity_curve(curve, CAPACITY, out / "capacity_curve.png")
    charts.subgroup_auc(table, out / "subgroup_auc.png")

    summary = {
        "cohort": {"encounters": int(len(frame)), "patients": int(frame["patient_nbr"].nunique()),
                   "readmit_30d_rate": float(y.mean())},
        "split_sizes": {k: int(len(getattr(split, k))) for k in ("train", "val", "test")},
        "selected_model": best_name,
        "baseline_model": baseline_name,
        "validation": selection.round(4).to_dict("records"),
        "test": {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in test_scores.items()},
        "test_baseline": {k: round(v, 4) for k, v in m.scores(y_test, p_test_baseline).items()},
        "capacity": CAPACITY,
        "threshold_from_validation": round(threshold, 4),
        "test_at_capacity": {k: round(v, 4) for k, v in operating_point.items()},
        "leakage_check": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in leak.items()},
        "top_features": importance.head(8).round(4).to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def _same_model(name: str):
    """A fresh, unfitted copy of the selected configuration."""
    return m.candidates()[name]


if __name__ == "__main__":
    main()
