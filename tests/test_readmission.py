"""Offline tests on hand-built encounters and synthetic data with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from readmission import audit, data, modeling as m

RAW_DEFAULTS = {
    "race": "Caucasian", "gender": "Female", "age": "[60-70)", "weight": None,
    "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
    "time_in_hospital": 3, "payer_code": None, "medical_specialty": None,
    "num_lab_procedures": 40, "num_procedures": 1, "num_medications": 15,
    "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0,
    "diag_1": "428", "diag_2": "250.01", "diag_3": "401", "number_diagnoses": 7,
    "max_glu_serum": None, "A1Cresult": None, "metformin": "No", "insulin": "Steady",
    "change": "No", "diabetesMed": "Yes", "readmitted": "NO",
}


def raw(encounter_id, patient, **overrides):
    return {"encounter_id": encounter_id, "patient_nbr": patient, **RAW_DEFAULTS, **overrides}


@pytest.fixture
def con():
    rows = [
        raw(1, 100, readmitted="<30"),
        raw(2, 100, readmitted=">30"),                       # later readmission: not a positive
        raw(3, 200),
        raw(4, 300, discharge_disposition_id=11),            # expired
        raw(5, 400, discharge_disposition_id=13),            # hospice / home
        raw(6, 500, gender="Unknown/Invalid"),
        raw(7, 600, race=None, max_glu_serum=">300", A1Cresult=">8", medical_specialty="Cardiology"),
    ]
    return m.connect(pd.DataFrame(rows))


def test_cohort_exclusions_and_target(con):
    cohort = m.load_cohort(con).set_index("encounter_id")
    assert list(cohort.index) == [1, 2, 3, 7]
    assert cohort.loc[1, "readmitted_30d"] == 1
    assert cohort.loc[2, "readmitted_30d"] == 0          # ">30" is not a 30-day readmission


def test_missing_values_become_explicit_categories(con):
    cohort = m.load_cohort(con).set_index("encounter_id")
    assert cohort.loc[3, "a1c_result"] == "Not measured"
    assert cohort.loc[7, "a1c_result"] == ">8"
    assert cohort.loc[7, "race"] == "Unknown"
    assert cohort.loc[3, "medical_specialty"] == "Missing"
    assert cohort.loc[7, "medical_specialty"] == "Cardiology"
    assert cohort.loc[3, "age_mid"] == 65


def test_cohort_summary_counts(con):
    steps = m.cohort_summary(con).set_index("step")["encounters"]
    assert steps.tolist() == [7, 5, 4]


@pytest.mark.parametrize("code, group", [
    ("250.83", "Diabetes"), ("428", "Circulatory"), ("785", "Circulatory"), ("486", "Respiratory"),
    ("530", "Digestive"), ("996", "Injury"), ("715", "Musculoskeletal"), ("599", "Genitourinary"),
    ("174", "Neoplasms"), ("V57", "Other"), ("E909", "Other"), ("300", "Other"), (None, "Missing"),
])
def test_icd9_grouping(con, code, group):
    assert con.execute("SELECT icd9_group(?::VARCHAR)", [code]).fetchone()[0] == group


def test_race_is_not_a_feature():
    assert "race" not in m.FEATURES
    assert m.GROUP not in m.FEATURES and "encounter_id" not in m.FEATURES


# ---- synthetic modelling data -----------------------------------------------------

def synthetic(n_patients: int = 600, per_patient: int = 3, seed: int = 0, fingerprint: bool = False) -> pd.DataFrame:
    """Encounters with all model features. With fingerprint=True every patient has a
    unique, constant feature value and a patient-level outcome, which is the
    situation where a row split leaks."""
    rng = np.random.default_rng(seed)
    rows = []
    for patient in range(n_patients):
        risk = rng.random()
        outcome = int(rng.random() < 0.3)
        for k in range(per_patient):
            row = {f: 0 for f in m.NUMERIC} | {f: "a" for f in m.CATEGORICAL}
            row.update(patient_nbr=patient, encounter_id=patient * 10 + k, race="Caucasian", gender="Female")
            row["number_inpatient"] = int(rng.poisson(1 + 3 * risk))
            row["num_lab_procedures"] = patient if fingerprint else int(rng.integers(1, 80))
            row["age_mid"] = int(rng.choice([35, 55, 75]))
            row[m.TARGET] = outcome if fingerprint else int(rng.random() < 0.05 + 0.3 * risk)
            rows.append(row)
    return pd.DataFrame(rows)


def test_grouped_split_keeps_patients_together_and_stratifies():
    frame = synthetic()
    split = m.grouped_split(frame)
    assert m.shared_patients(frame, split) == 0
    val_patients = set(frame["patient_nbr"].iloc[split.val])
    assert not val_patients & set(frame["patient_nbr"].iloc[split.test])
    assert len(split.train) + len(split.val) + len(split.test) == len(frame)
    rate = frame[m.TARGET].mean()
    for part in (split.train, split.val, split.test):
        assert abs(frame[m.TARGET].iloc[part].mean() - rate) < 0.05


def test_row_split_does_share_patients():
    frame = synthetic()
    assert m.shared_patients(frame, m.row_split(frame)) > 50


def test_leakage_check_detects_leakage_when_it_exists():
    # patient-level outcome + a feature that identifies the patient: a row split
    # lets the model memorise patients, a grouped split cannot. Patients need
    # many encounters for this: with 3 each (the first version of this test),
    # the booster's 20-sample minimum leaf size stops it isolating a patient and
    # the row split scored only 0.65 - which is also why the real data, where
    # most patients have one or two encounters, shows so little leakage.
    frame = synthetic(n_patients=120, per_patient=25, fingerprint=True)
    result = m.leakage_check(frame, lambda: m.boosting_model(0.1, 31), seeds=range(2))
    assert result["grouped_roc_auc_mean"] < 0.6
    assert result["row_roc_auc_mean"] > 0.8


def test_models_fit_and_output_probabilities():
    frame = synthetic()
    for model in (m.logistic_model(), m.boosting_model()):
        model.fit(frame[m.FEATURES], frame[m.TARGET])
        unseen = frame[m.FEATURES].head(3).assign(payer_code="never seen")
        p = model.predict_proba(unseen)[:, 1]
        assert ((p > 0) & (p < 1)).all()


def test_expected_calibration_error():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.4, 200_000)
    y = (rng.random(len(p)) < p).astype(int)
    assert m.expected_calibration_error(y, p) < 0.005               # calibrated by construction
    assert m.expected_calibration_error(y, np.clip(p + 0.1, 0, 1)) == pytest.approx(0.1, abs=0.005)


def test_capacity_threshold_and_operating_point():
    p = np.linspace(0, 1, 1000, endpoint=False)
    y = (p >= 0.95).astype(int)                                    # all positives in the top 5%
    threshold = m.capacity_threshold(p, 0.10)
    point = m.at_threshold(y, p, threshold)
    assert point["flagged_pct"] == pytest.approx(10.0, abs=0.2)
    assert point["recall"] == pytest.approx(1.0)
    assert point["precision"] == pytest.approx(0.5, abs=0.01)
    assert point["lift"] == pytest.approx(10.0, abs=0.2)


def test_bootstrap_auc_ci_brackets_auc_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.2 + rng.normal(0.4, 0.2, 2000), 0, 1)
    groups = np.arange(2000) // 2
    ci = m.bootstrap_auc_ci(y, p, groups, n_boot=300)
    assert ci[0] < roc_auc_score(y, p) < ci[1]
    assert ci == m.bootstrap_auc_ci(y, p, groups, n_boot=300)


def test_subgroup_table():
    frame = pd.DataFrame({"patient_nbr": np.arange(400)})
    rng = np.random.default_rng(2)
    y = np.r_[rng.integers(0, 2, 300), np.zeros(95, int), np.ones(5, int)]
    p = np.r_[np.full(300, 0.5), np.full(100, 0.2)]
    groups = np.r_[np.full(300, "big"), np.full(100, "small")]
    table = audit.subgroup_table(frame, y, p, 0.4, {"g": groups}, n_boot=50).set_index("group")
    assert table.loc["small", "calibration_gap"] == pytest.approx(0.2 - 0.05)
    assert table.loc["small", "note"].startswith("fewer than")          # 5 positives: not reported
    assert table.loc["big", "flagged_pct"] == pytest.approx(100.0)
    assert table.loc["big", "recall_at_threshold"] == pytest.approx(1.0)


def test_age_group_bins():
    assert audit.age_group(pd.Series([45, 50, 69, 70, 95])).tolist() == ["under 50", "50-69", "50-69", "70+", "70+"]


def test_checksum_mismatch(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.verify(path, data.FILES["diabetic_data.csv"])
