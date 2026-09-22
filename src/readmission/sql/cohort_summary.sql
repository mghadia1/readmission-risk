-- What the exclusions removed, and the shape of what is left.
SELECT 'all encounters' AS step, count(*) AS encounters, count(DISTINCT patient_nbr) AS patients,
       round(100.0 * avg((readmitted = '<30')::INTEGER), 2) AS readmit_30d_pct
FROM encounters
UNION ALL
SELECT 'minus expired / hospice discharges', count(*), count(DISTINCT patient_nbr),
       round(100.0 * avg((readmitted = '<30')::INTEGER), 2)
FROM encounters WHERE NOT is_expired_or_hospice
UNION ALL
SELECT 'minus unknown gender (modelling cohort)', count(*), count(DISTINCT patient_nbr),
       round(100.0 * avg(readmitted_30d), 2)
FROM cohort;
