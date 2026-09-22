# How it works

```
UCI zip ──download──▶ diabetic_data.csv + IDS_mapping.csv (SHA-256)
                            │  '?' → NULL, diagnosis codes kept as text
                            ▼
DuckDB  encounters ─▶ cohort (exclusions, ICD-9 chapters, explicit "Not measured")
                            ▼
split by patient (StratifiedGroupKFold): train 60 / validation 20 / test 20
                            ▼
5 candidates fit on train ─▶ pick on validation PR-AUC ─▶ threshold on validation (10% budget)
                            ▼
test scored ONCE ─▶ calibration, capacity curve, subgroup audit
validation ─▶ permutation importance      5-seed row-vs-patient split comparison
```

## Choices I would defend

| Choice | Why |
|---|---|
| **Exclude deaths and hospice discharges** | These patients cannot be readmitted. Keeping them labels some of the sickest patients as "not readmitted" |
| **Split by patient** | A patient's encounters must not be in both training and test. This is correct even though, measured here, the row split inflates ROC-AUC by only 0.004 |
| **Race out of the model, into the audit** | The model should not use race to set risk, but whether it works equally well across races still has to be checked |
| **Missing lab results as a category ("Not measured")** | Whether a test was ordered is itself information. 83% of A1C results are missing, so dropping those rows would lose most of the data |
| **ICD-9 codes grouped into 9 chapters** | There are 715 distinct primary codes. Chapters follow the grouping in the paper that published the dataset |
| **Select on PR-AUC** | With an 11% positive class, ROC-AUC can look fine while the flagged list is mostly false alarms |
| **Threshold from a follow-up budget** | "We can call 10% of discharges" is a real constraint. A threshold chosen to maximise F1 answers no one's question |
| **Test scored once, after every choice** | Selection and threshold both use validation, so the test number is not tuned to the test set |
| **Bootstrap patients, not encounters** | Encounters of the same patient are correlated, and resampling them separately would narrow the interval falsely |

## Models

- **Logistic regression:** numbers standardised, categories one-hot encoded (categories with fewer than 20 rows pooled). This is the baseline, and it must be beaten.
- **Histogram gradient boosting** (scikit-learn): categories used natively, early stopping on an internal 15% of training data, and a 20-sample minimum leaf size (the default). That minimum is also why the model cannot memorise individual patients, which the leakage check found out.

## Metrics

- **ROC-AUC:** how often a random readmitted patient is ranked above a random non-readmitted one.
- **PR-AUC:** precision averaged over recall levels. Chance level equals the base rate (0.114).
- **Brier score:** mean squared error of the probabilities.
- **Calibration error:** sort predictions into 10 equal-size bins, take |observed rate − mean prediction| in each, and average weighted by size.
- **Lift:** precision at the threshold divided by the base rate.

## Tests (27)

The tests cover:
- **Cohort SQL:** exclusions, the `<30` vs `>30` target, missing values made explicit, and 13 ICD-9 grouping cases.
- **Race is never a feature.**
- **Splits:** the patient split shares no patients and keeps the outcome rate balanced; the row split does share patients.
- **Leakage check:** it detects leakage when patients can be memorised.
- **Models:** both handle a category they have never seen.
- **Metrics:** calibration error on synthetic data calibrated by construction (≈ 0) and on data shifted by 0.1 (≈ 0.1); the capacity threshold and precision/recall/lift arithmetic; bootstrap reproducibility.
- **Subgroup table:** the calibration gap, and suppression of small groups.
- **Checksum rejection.**

Two first attempts failed and are recorded:
- **The leakage test.** With 3 encounters per synthetic patient it found no leakage. That was the minimum leaf size at work, and it explains the real-data result.
- **The one-seed leakage comparison.** It was replaced by 5 seeds, because a single split's noise was as large as the effect being measured.

## Things to be able to explain

- Why exclude patients who died, and what would happen if you didn't?
- Why split by patient, and why did it matter less than expected here?
- Why is ROC-AUC 0.67 a reasonable result, and what would you need to do better?
- Why is the threshold a budget, and how would you pick the budget?
- What does the 70+ subgroup result mean, and what would you do about it?
- Why is race in the audit but not in the model?
