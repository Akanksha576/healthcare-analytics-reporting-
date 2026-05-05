# Healthcare Analytics — Hospital Readmissions Reporting Platform

An Optum-style healthcare analytics project analyzing 130,000+ hospital encounters to identify service opportunities, generate performance guarantee reports, and deliver client-facing insights using SQL, Python, and data visualization.

## Business Context

Mirrors real-world Optum / UnitedHealth Group analytics work:
- **Contractual reporting**: Monthly performance guarantee scorecards with 30-day readmission rate vs target
- **Service opportunity identification**: Data mining to flag high-risk patient cohorts for intervention
- **Client-facing analytics**: Executive dashboards and regression-based insights for health plan clients
- **Data governance**: Validation queries, null checks, and documentation per data governance requirements

---

## Architecture

```
data/
└── diabetic_data.csv                ← 130K hospital encounters (Kaggle public dataset)

sql/
└── hospital_readmissions_analytics.sql
    ├── Section 1: Data Validation Queries
    ├── Section 2: Descriptive Analytics (KPIs, trends)
    ├── Section 3: Data Mining (service opportunities)
    ├── Section 4: Complex Multi-Table Joins
    ├── Section 5: Regression Feature Engineering
    └── Section 6: Performance Guarantee Report

analysis/
└── healthcare_analytics.py
    ├── Step 1: Data normalization + validation
    ├── Step 2: Descriptive analytics + executive dashboard
    ├── Step 3: Pattern recognition + correlation analysis
    ├── Step 4: Regression-based analytics (logistic regression)
    └── Step 5: Performance guarantee report generation

docs/
├── executive_dashboard.png          ← 4-panel KPI dashboard
├── correlation_heatmap.png          ← Feature correlation matrix
├── regression_feature_importance.png← Readmission risk drivers
├── business_requirements.md         ← This document
└── data_governance_notes.md         ← Data governance documentation
```

---

## Business Requirements

### Report 1: Executive Dashboard (client-facing)
| Requirement | Specification |
|------------|---------------|
| Scope | All encounters in reporting period |
| Metrics | 30-day readmission rate, avg LOS, avg medications |
| Segments | By age group, readmission category, medication tier |
| Refresh | Monthly |
| Audience | Health plan client executives |
| Data governance | Share minimally necessary data — no PII in dashboard |

### Report 2: Service Opportunity Analysis
| Requirement | Specification |
|------------|---------------|
| Purpose | Identify high-risk patient cohorts for care management |
| Method | Data mining by diagnosis code, age, discharge disposition |
| Threshold | Flag diagnoses with 30-day readmission rate ≥ 15% |
| Output | Ranked list of intervention opportunities |
| Audience | Clinical operations team |

### Report 3: Performance Guarantee Report
| Requirement | Specification |
|------------|---------------|
| Contractual target | 30-day readmission rate ≤ 11% |
| Reporting period | Monthly + rolling 3-month average |
| Trend analysis | Month-over-month change using LAG window functions |
| Status flag | Meets target / Near threshold / Exceeds target |
| Audience | Client — contractual delivery |

---

## Data Governance Notes

- **Minimum necessary data**: Patient IDs anonymized in all output reports
- **Validation**: 8 data quality checks run before every report generation
- **Null handling**: Documented thresholds for acceptable null rates per field
- **Accuracy verification**: Validation queries run on incoming data before automated report generation
- **Retention**: Raw data retained per HIPAA guidelines, aggregated reports shared with clients

---

## Key Findings (Sample Insights)

### Service Opportunities Identified:
1. **Polypharmacy risk**: Patients on 20+ medications show **2.3x higher** 30-day readmission rates — opportunity for medication reconciliation programs
2. **High-risk diagnoses**: Top 5 diagnosis codes (circulatory, respiratory) account for 40%+ of all 30-day readmissions
3. **Discharge planning gap**: Patients discharged to home without follow-up show significantly higher readmission rates than those with home health services
4. **Age cohort pattern**: 70-90 age group has highest readmission rates — opportunity for senior care management programs

### Regression Analysis:
- **Model**: Logistic Regression on 13 clinical features
- **Performance**: ROC-AUC ~0.65 (clinically meaningful for readmission prediction)
- **Top risk drivers**: Number of diagnoses, length of stay, polypharmacy flag
- **Actionable**: Risk scores generated per patient for care management prioritization

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| SQL Analytics | PostgreSQL / DuckDB |
| Data Processing | Python 3.10, pandas, NumPy |
| Statistical Analysis | scipy, scikit-learn |
| Visualization | matplotlib, seaborn |
| BI Reporting | Tableau / Power BI compatible output |
| Version Control | Git / GitHub |
| Documentation | Markdown (business requirements, process flow) |

---

## Running the Project

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# 2. Download dataset from Kaggle
# Dataset: "Diabetes 130-US hospitals for years 1999-2008"
# Place as: data/diabetic_data.csv

# 3. Run SQL analytics (in any SQL client)
# Open: sql/hospital_readmissions_analytics.sql

# 4. Run Python analysis
python analysis/healthcare_analytics.py
```

---

## Dataset

**Source**: Kaggle — "Diabetes 130-US Hospitals for Years 1999-2008"
**Records**: 130,000+ hospital encounters
**Features**: 50 clinical variables (demographics, diagnoses, procedures, medications, outcomes)
**Target**: 30-day hospital readmission flag
**License**: Public domain (UCI ML Repository)
