-- ============================================================
-- Hospital Readmissions Analytics — Optum-Style SQL Reporting
-- Dataset: Hospital Readmissions Reduction Program (130K records)
-- Purpose: Identify service opportunities, patterns, and trends
--          to support performance guarantee reporting and
--          client-facing analytics deliverables
-- Author: Akanksha Patel
-- ============================================================


-- ============================================================
-- SECTION 1: DATA VALIDATION QUERIES
-- Verify accuracy of incoming data before report generation
-- ============================================================

-- 1a. Check for null values in critical fields
SELECT
    COUNT(*)                                        AS total_records,
    SUM(CASE WHEN patient_id IS NULL THEN 1 END)    AS null_patient_id,
    SUM(CASE WHEN age IS NULL THEN 1 END)            AS null_age,
    SUM(CASE WHEN primary_diagnosis IS NULL THEN 1 END) AS null_diagnosis,
    SUM(CASE WHEN readmitted IS NULL THEN 1 END)    AS null_readmitted,
    SUM(CASE WHEN discharge_disposition IS NULL THEN 1 END) AS null_discharge,
    ROUND(
        SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                AS readmission_rate_pct
FROM hospital_readmissions;

-- 1b. Validate age ranges and catch data entry errors
SELECT
    age_group,
    COUNT(*)            AS record_count,
    MIN(time_in_hospital) AS min_los,
    MAX(time_in_hospital) AS max_los,
    AVG(time_in_hospital) AS avg_los
FROM hospital_readmissions
WHERE age NOT BETWEEN 0 AND 120       -- flag impossible ages
GROUP BY age_group
ORDER BY age_group;

-- 1c. Check for duplicate patient encounters
SELECT
    encounter_id,
    COUNT(*) AS duplicate_count
FROM hospital_readmissions
GROUP BY encounter_id
HAVING COUNT(*) > 1;


-- ============================================================
-- SECTION 2: DESCRIPTIVE ANALYTICS — EXECUTIVE DASHBOARD
-- Performance metrics for contractual reporting
-- ============================================================

-- 2a. Overall performance summary (KPI report)
SELECT
    COUNT(DISTINCT patient_id)                          AS unique_patients,
    COUNT(*)                                            AS total_encounters,
    ROUND(AVG(time_in_hospital), 2)                     AS avg_length_of_stay,
    ROUND(AVG(num_procedures), 2)                       AS avg_procedures_per_visit,
    ROUND(AVG(num_medications), 2)                      AS avg_medications,
    ROUND(AVG(num_lab_procedures), 2)                   AS avg_lab_procedures,
    SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) AS readmissions_30day,
    SUM(CASE WHEN readmitted = '>30' THEN 1 ELSE 0 END) AS readmissions_after_30,
    SUM(CASE WHEN readmitted = 'NO' THEN 1 ELSE 0 END)  AS no_readmission,
    ROUND(
        SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS readmission_rate_30day_pct
FROM hospital_readmissions;

-- 2b. Readmission trends by quarter (time series for dashboard)
SELECT
    EXTRACT(YEAR FROM admission_date)           AS year,
    EXTRACT(QUARTER FROM admission_date)        AS quarter,
    COUNT(*)                                    AS total_admissions,
    SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) AS readmissions_30d,
    ROUND(
        SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                           AS readmission_rate_pct,
    ROUND(AVG(time_in_hospital), 2)             AS avg_los
FROM hospital_readmissions
GROUP BY 1, 2
ORDER BY 1, 2;


-- ============================================================
-- SECTION 3: DATA MINING — IDENTIFY SERVICE OPPORTUNITIES
-- Pattern recognition to find high-risk cohorts
-- ============================================================

-- 3a. High-risk cohort analysis: diagnoses with highest readmission rates
WITH diagnosis_stats AS (
    SELECT
        primary_diagnosis,
        COUNT(*)                                            AS total_cases,
        SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) AS readmissions_30d,
        ROUND(AVG(time_in_hospital), 2)                     AS avg_los,
        ROUND(AVG(num_medications), 2)                      AS avg_medications
    FROM hospital_readmissions
    GROUP BY primary_diagnosis
    HAVING COUNT(*) >= 100   -- sufficient sample size
)
SELECT
    primary_diagnosis,
    total_cases,
    readmissions_30d,
    ROUND(readmissions_30d * 100.0 / total_cases, 2)        AS readmission_rate_pct,
    avg_los,
    avg_medications,
    CASE
        WHEN readmissions_30d * 100.0 / total_cases >= 20 THEN 'HIGH RISK — intervention needed'
        WHEN readmissions_30d * 100.0 / total_cases >= 12 THEN 'MODERATE RISK — monitor closely'
        ELSE 'LOW RISK — standard care'
    END                                                     AS risk_flag
FROM diagnosis_stats
ORDER BY readmission_rate_pct DESC
LIMIT 20;

-- 3b. Age cohort performance analysis (recognizing trends across populations)
SELECT
    age_group,
    COUNT(*)                                            AS total_patients,
    ROUND(AVG(time_in_hospital), 2)                     AS avg_los,
    ROUND(AVG(num_medications), 2)                      AS avg_medications,
    ROUND(AVG(num_lab_procedures), 2)                   AS avg_lab_procedures,
    SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) AS readmissions_30d,
    ROUND(
        SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS readmission_rate_pct,
    -- Window function: rank age groups by readmission rate
    RANK() OVER (ORDER BY
        SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) DESC
    )                                                   AS risk_rank
FROM hospital_readmissions
GROUP BY age_group
ORDER BY readmission_rate_pct DESC;

-- 3c. Service opportunity: patients with multiple medications + high readmission
-- (Polypharmacy as readmission risk factor)
WITH medication_cohorts AS (
    SELECT
        CASE
            WHEN num_medications < 5  THEN '1-4 medications'
            WHEN num_medications < 10 THEN '5-9 medications'
            WHEN num_medications < 15 THEN '10-14 medications'
            WHEN num_medications < 20 THEN '15-19 medications'
            ELSE '20+ medications'
        END                                             AS medication_tier,
        num_medications,
        readmitted,
        time_in_hospital,
        num_diagnoses
    FROM hospital_readmissions
)
SELECT
    medication_tier,
    COUNT(*)                                            AS patient_count,
    ROUND(AVG(time_in_hospital), 2)                     AS avg_los,
    ROUND(AVG(num_diagnoses), 2)                        AS avg_diagnoses,
    SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) AS readmissions_30d,
    ROUND(
        SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS readmission_rate_pct
FROM medication_cohorts
GROUP BY medication_tier
ORDER BY AVG(num_medications);


-- ============================================================
-- SECTION 4: COMPLEX JOINS — MULTI-SOURCE REPORT
-- Combining patient data with provider and admission data
-- ============================================================

-- 4a. Discharge disposition impact on readmission rates
-- (Identifies where patients go after discharge and readmission outcomes)
SELECT
    h.discharge_disposition,
    COUNT(*)                                            AS total_discharges,
    SUM(CASE WHEN h.readmitted = '<30' THEN 1 ELSE 0 END) AS readmissions_30d,
    ROUND(
        SUM(CASE WHEN h.readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS readmission_rate_pct,
    ROUND(AVG(h.time_in_hospital), 2)                   AS avg_los,
    -- Running total using window function
    SUM(COUNT(*)) OVER (
        ORDER BY SUM(CASE WHEN h.readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) DESC
        ROWS UNBOUNDED PRECEDING
    )                                                   AS cumulative_discharges
FROM hospital_readmissions h
LEFT JOIN discharge_lookup dl
    ON h.discharge_disposition_id = dl.disposition_id
GROUP BY h.discharge_disposition
ORDER BY readmission_rate_pct DESC;

-- 4b. 30-day readmission rate by specialty + admission source
-- Complex multi-table join for client-facing performance report
SELECT
    h.medical_specialty,
    h.admission_source,
    COUNT(DISTINCT h.patient_id)                        AS unique_patients,
    COUNT(*)                                            AS encounters,
    ROUND(AVG(h.time_in_hospital), 2)                   AS avg_los,
    SUM(CASE WHEN h.readmitted = '<30' THEN 1 ELSE 0 END) AS readmissions_30d,
    ROUND(
        SUM(CASE WHEN h.readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS readmission_rate_pct,
    -- Benchmark comparison using window function
    ROUND(AVG(
        SUM(CASE WHEN h.readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
    ) OVER (), 2)                                       AS overall_avg_rate,
    ROUND(
        SUM(CASE WHEN h.readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
        - AVG(SUM(CASE WHEN h.readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) OVER (), 2
    )                                                   AS variance_from_benchmark
FROM hospital_readmissions h
LEFT JOIN specialty_lookup sl  ON h.medical_specialty_id = sl.specialty_id
LEFT JOIN admission_lookup al  ON h.admission_source_id = al.source_id
WHERE h.medical_specialty IS NOT NULL
GROUP BY h.medical_specialty, h.admission_source
HAVING COUNT(*) >= 50
ORDER BY readmission_rate_pct DESC;


-- ============================================================
-- SECTION 5: REGRESSION-BASED ANALYTICS SUPPORT
-- SQL feature preparation for statistical modeling
-- ============================================================

-- 5a. Feature engineering for readmission prediction model
-- Prepares structured dataset for Python regression analysis
SELECT
    patient_id,
    encounter_id,
    -- Numeric features
    age_numeric,
    time_in_hospital,
    num_lab_procedures,
    num_procedures,
    num_medications,
    num_diagnoses,
    -- Derived ratio features
    ROUND(num_lab_procedures * 1.0 / NULLIF(time_in_hospital, 0), 2) AS labs_per_day,
    ROUND(num_medications * 1.0 / NULLIF(num_diagnoses, 0), 2)        AS meds_per_diagnosis,
    -- Binary encoded categorical features
    CASE WHEN gender = 'Female' THEN 1 ELSE 0 END       AS is_female,
    CASE WHEN a1c_result IN ('Norm', '>7', '>8') THEN 1 ELSE 0 END AS has_a1c_result,
    CASE WHEN change = 'Ch' THEN 1 ELSE 0 END           AS medication_changed,
    CASE WHEN diabetesMed = 'Yes' THEN 1 ELSE 0 END     AS on_diabetes_med,
    -- Comorbidity flags
    CASE WHEN num_diagnoses >= 5 THEN 1 ELSE 0 END      AS complex_patient,
    CASE WHEN num_medications >= 15 THEN 1 ELSE 0 END   AS polypharmacy_flag,
    -- Target variable
    CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END      AS readmitted_30d
FROM hospital_readmissions
WHERE patient_id IS NOT NULL
  AND encounter_id IS NOT NULL;


-- ============================================================
-- SECTION 6: PERFORMANCE GUARANTEE REPORT
-- Contractual metrics for client-facing delivery
-- ============================================================

-- 6a. Monthly performance scorecard
WITH monthly_metrics AS (
    SELECT
        DATE_TRUNC('month', admission_date)             AS report_month,
        COUNT(*)                                        AS total_admissions,
        SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) AS readmissions_30d,
        ROUND(AVG(time_in_hospital), 2)                 AS avg_los,
        ROUND(AVG(num_medications), 2)                  AS avg_medications,
        COUNT(DISTINCT patient_id)                      AS unique_patients
    FROM hospital_readmissions
    GROUP BY 1
),
performance_targets AS (
    SELECT
        report_month,
        total_admissions,
        readmissions_30d,
        ROUND(readmissions_30d * 100.0 / total_admissions, 2) AS readmission_rate_pct,
        avg_los,
        avg_medications,
        unique_patients,
        -- Month-over-month change using LAG window function
        LAG(readmissions_30d * 100.0 / total_admissions) OVER (ORDER BY report_month) AS prev_month_rate,
        ROUND(
            readmissions_30d * 100.0 / total_admissions
            - LAG(readmissions_30d * 100.0 / total_admissions) OVER (ORDER BY report_month), 2
        )                                               AS rate_change_mom,
        -- Rolling 3-month average
        ROUND(AVG(readmissions_30d * 100.0 / total_admissions)
            OVER (ORDER BY report_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2
        )                                               AS rolling_3mo_avg,
        -- Performance vs 11% contractual target
        CASE
            WHEN readmissions_30d * 100.0 / total_admissions <= 11.0 THEN 'MEETS TARGET ✓'
            WHEN readmissions_30d * 100.0 / total_admissions <= 13.0 THEN 'NEAR THRESHOLD ⚠'
            ELSE 'EXCEEDS TARGET ✗'
        END                                             AS performance_status
    FROM monthly_metrics
)
SELECT * FROM performance_targets
ORDER BY report_month;
