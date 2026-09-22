-- The modelling cohort and its features, built once in SQL.
-- Input: raw_encounters (diabetic_data.csv read with '?' as NULL).

CREATE OR REPLACE VIEW encounters AS
SELECT
    *,
    -- Patients who died or went to hospice cannot be readmitted; keeping them
    -- would teach the model that "sickest" means "safest".
    discharge_disposition_id IN (11, 13, 14, 19, 20, 21) AS is_expired_or_hospice,
    gender = 'Unknown/Invalid'                           AS is_unknown_gender
FROM raw_encounters;

-- Group ICD-9 codes into the broad chapters used in the paper that published
-- this dataset (Strack et al., 2014). 715 distinct primary codes are too many
-- categories for 70,000 patients; chapters keep the clinical signal.
CREATE OR REPLACE MACRO icd9_group(code) AS
    CASE
        WHEN code IS NULL                                            THEN 'Missing'
        WHEN code LIKE 'V%' OR code LIKE 'E%'                        THEN 'Other'
        WHEN floor(TRY_CAST(code AS DOUBLE)) = 250                   THEN 'Diabetes'
        WHEN TRY_CAST(code AS DOUBLE) BETWEEN 390 AND 459.99
          OR floor(TRY_CAST(code AS DOUBLE)) = 785                   THEN 'Circulatory'
        WHEN TRY_CAST(code AS DOUBLE) BETWEEN 460 AND 519.99
          OR floor(TRY_CAST(code AS DOUBLE)) = 786                   THEN 'Respiratory'
        WHEN TRY_CAST(code AS DOUBLE) BETWEEN 520 AND 579.99
          OR floor(TRY_CAST(code AS DOUBLE)) = 787                   THEN 'Digestive'
        WHEN TRY_CAST(code AS DOUBLE) BETWEEN 800 AND 999.99         THEN 'Injury'
        WHEN TRY_CAST(code AS DOUBLE) BETWEEN 710 AND 739.99         THEN 'Musculoskeletal'
        WHEN TRY_CAST(code AS DOUBLE) BETWEEN 580 AND 629.99
          OR floor(TRY_CAST(code AS DOUBLE)) = 788                   THEN 'Genitourinary'
        WHEN TRY_CAST(code AS DOUBLE) BETWEEN 140 AND 239.99         THEN 'Neoplasms'
        ELSE 'Other'
    END;

CREATE OR REPLACE VIEW cohort AS
WITH specialty_rank AS (
    SELECT medical_specialty, row_number() OVER (ORDER BY count(*) DESC) AS rk
    FROM encounters WHERE medical_specialty IS NOT NULL GROUP BY medical_specialty
)
SELECT
    e.encounter_id,
    e.patient_nbr,
    (e.readmitted = '<30')::INTEGER                           AS readmitted_30d,
    -- audit-only columns (race is NOT a model feature; see docs/how-it-works.md)
    coalesce(e.race, 'Unknown')                               AS race,
    e.gender,
    e.age                                                     AS age_band,
    -- numeric features
    TRY_CAST(regexp_extract(e.age, '\[(\d+)-', 1) AS INTEGER) + 5 AS age_mid,
    e.time_in_hospital, e.num_lab_procedures, e.num_procedures, e.num_medications,
    e.number_outpatient, e.number_emergency, e.number_inpatient, e.number_diagnoses,
    -- categorical features
    e.admission_type_id::VARCHAR                              AS admission_type,
    e.discharge_disposition_id::VARCHAR                       AS discharge_disposition,
    e.admission_source_id::VARCHAR                            AS admission_source,
    coalesce(e.max_glu_serum, 'Not measured')                 AS max_glu_serum,
    coalesce(e.A1Cresult, 'Not measured')                     AS a1c_result,
    e.insulin, e.metformin, e."change" AS med_change, e.diabetesMed AS diabetes_med,
    CASE WHEN e.medical_specialty IS NULL THEN 'Missing'
         WHEN s.rk <= 15 THEN e.medical_specialty ELSE 'Other' END AS medical_specialty,
    coalesce(e.payer_code, 'Missing')                         AS payer_code,
    icd9_group(e.diag_1) AS diag_1_group,
    icd9_group(e.diag_2) AS diag_2_group,
    icd9_group(e.diag_3) AS diag_3_group
FROM encounters AS e
LEFT JOIN specialty_rank AS s USING (medical_specialty)
WHERE NOT e.is_expired_or_hospice
  AND NOT e.is_unknown_gender
ORDER BY e.encounter_id;
