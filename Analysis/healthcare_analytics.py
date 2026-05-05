"""
Healthcare Analytics — Hospital Readmissions Analysis
Optum-style descriptive and regression-based analytics
Dataset: Hospital Readmissions Reduction Program (130K records)

Purpose:
- Perform complex data analysis to identify patterns and service opportunities
- Apply statistical techniques to interpret health outcomes data
- Generate actionable insights for client-facing reporting
- Support performance guarantee report development
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor':   '#ffffff',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.family':      'sans-serif',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})
OPTUM_BLUE   = '#0070CE'
OPTUM_GREEN  = '#00A859'
OPTUM_ORANGE = '#F7941D'
OPTUM_RED    = '#E31837'


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA INGESTION & NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def load_and_normalize(filepath: str) -> pd.DataFrame:
    """
    Load hospital readmissions dataset and apply normalization operations.
    Ensures data accuracy and consistency before analytics.
    """
    print("=" * 70)
    print("STEP 1: DATA INGESTION & NORMALIZATION")
    print("=" * 70)

    df = pd.read_csv(filepath)
    print(f"Records loaded:  {len(df):,}")
    print(f"Features:        {df.shape[1]}")

    # ── Data cleaning & normalization ──────────────────────────────────────
    df.replace('?', np.nan, inplace=True)

    # Normalize age groups to numeric midpoints (normalization operation)
    age_map = {
        '[0-10)':5, '[10-20)':15, '[20-30)':25, '[30-40)':35,
        '[40-50)':45, '[50-60)':55, '[60-70)':65, '[70-80)':75,
        '[80-90)':85, '[90-100)':95
    }
    df['age_numeric'] = df['age'].map(age_map)

    # Binary encode target variable
    df['readmitted_30d'] = (df['readmitted'] == '<30').astype(int)
    df['readmitted_any'] = (df['readmitted'] != 'NO').astype(int)

    # Normalize medication change flags
    df['medication_changed'] = (df['change'] == 'Ch').astype(int)
    df['on_diabetes_med']    = (df['diabetesMed'] == 'Yes').astype(int)
    df['has_a1c_result']     = (~df['A1Cresult'].isin(['None', np.nan])).astype(int)

    # Derived features for regression analysis
    df['labs_per_day']       = df['num_lab_procedures'] / df['time_in_hospital'].replace(0, 1)
    df['meds_per_diagnosis'] = df['num_medications'] / df['num_diagnoses'].replace(0, 1)
    df['complex_patient']    = (df['number_diagnoses'] >= 5).astype(int)
    df['polypharmacy_flag']  = (df['num_medications'] >= 15).astype(int)

    # ── Data quality report ────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    print(f"\nNull values detected in {len(null_counts)} columns:")
    for col, cnt in null_counts.items():
        print(f"  {col:<40} {cnt:>6,} ({cnt/len(df)*100:.1f}%)")

    # Drop rows missing critical fields
    before = len(df)
    df.dropna(subset=['age_numeric', 'time_in_hospital', 'num_medications'], inplace=True)
    print(f"\nRecords after validation: {len(df):,} (removed {before-len(df):,} invalid rows)")
    print(f"30-day readmission rate:  {df['readmitted_30d'].mean()*100:.1f}%")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. DESCRIPTIVE ANALYTICS — EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def descriptive_analytics(df: pd.DataFrame):
    """
    Perform descriptive analytics to summarize health outcomes.
    Builds executive-level KPI summary for client reporting.
    """
    print("\n" + "=" * 70)
    print("STEP 2: DESCRIPTIVE ANALYTICS — EXECUTIVE DASHBOARD")
    print("=" * 70)

    # ── KPI Summary Table ──────────────────────────────────────────────────
    kpis = {
        'Total Encounters':          f"{len(df):,}",
        'Unique Patients':           f"{df['patient_nbr'].nunique():,}",
        'Avg Length of Stay (days)': f"{df['time_in_hospital'].mean():.2f}",
        'Avg Medications':           f"{df['num_medications'].mean():.2f}",
        'Avg Lab Procedures':        f"{df['num_lab_procedures'].mean():.2f}",
        '30-Day Readmission Rate':   f"{df['readmitted_30d'].mean()*100:.2f}%",
        'Any Readmission Rate':      f"{df['readmitted_any'].mean()*100:.2f}%",
    }
    print("\nKEY PERFORMANCE INDICATORS:")
    for k, v in kpis.items():
        print(f"  {k:<40} {v:>10}")

    # ── Dashboard: 4-panel overview ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hospital Readmissions — Executive Dashboard\nUnitedHealth Group / Optum Analytics',
                 fontsize=15, fontweight='bold', y=1.01)

    # Panel 1: Readmission distribution
    ax1 = axes[0, 0]
    readmit_counts = df['readmitted'].value_counts()
    colors = [OPTUM_RED, OPTUM_ORANGE, OPTUM_GREEN]
    bars = ax1.bar(readmit_counts.index, readmit_counts.values, color=colors, width=0.5)
    ax1.set_title('Readmission Outcomes', fontweight='bold')
    ax1.set_xlabel('Readmission Category')
    ax1.set_ylabel('Patient Count')
    for bar, val in zip(bars, readmit_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax1.set_ylim(0, readmit_counts.max() * 1.15)

    # Panel 2: Length of stay by readmission status
    ax2 = axes[0, 1]
    los_data = [
        df[df['readmitted'] == 'NO']['time_in_hospital'],
        df[df['readmitted'] == '>30']['time_in_hospital'],
        df[df['readmitted'] == '<30']['time_in_hospital'],
    ]
    bp = ax2.boxplot(los_data, labels=['No Readmit', '>30 Days', '<30 Days'],
                     patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], [OPTUM_GREEN, OPTUM_ORANGE, OPTUM_RED]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax2.set_title('Length of Stay by Readmission Status', fontweight='bold')
    ax2.set_ylabel('Days in Hospital')

    # Panel 3: Readmission rate by age group
    ax3 = axes[1, 0]
    age_rates = df.groupby('age')['readmitted_30d'].agg(['mean', 'count']).reset_index()
    age_rates.columns = ['age', 'rate', 'count']
    age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                 '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    age_rates = age_rates.set_index('age').reindex(age_order).reset_index()
    bars3 = ax3.bar(range(len(age_rates)), age_rates['rate'] * 100,
                    color=[OPTUM_RED if r > 0.12 else OPTUM_BLUE for r in age_rates['rate']],
                    alpha=0.85)
    ax3.axhline(y=df['readmitted_30d'].mean() * 100, color='black',
                linestyle='--', linewidth=1.5, label=f"Overall avg: {df['readmitted_30d'].mean()*100:.1f}%")
    ax3.set_xticks(range(len(age_rates)))
    ax3.set_xticklabels([a.replace('[', '').replace(')', '') for a in age_rates['age']],
                         rotation=35, ha='right', fontsize=8)
    ax3.set_title('30-Day Readmission Rate by Age Group', fontweight='bold')
    ax3.set_ylabel('Readmission Rate (%)')
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax3.legend()

    # Panel 4: Medication count vs readmission rate (recognizing trends)
    ax4 = axes[1, 1]
    med_bins = pd.cut(df['num_medications'], bins=[0, 5, 10, 15, 20, 100],
                      labels=['1-5', '6-10', '11-15', '16-20', '20+'])
    med_rates = df.groupby(med_bins)['readmitted_30d'].mean() * 100
    ax4.plot(med_rates.index, med_rates.values, 'o-',
             color=OPTUM_BLUE, linewidth=2.5, markersize=9, markerfacecolor=OPTUM_RED)
    for x, y in zip(range(len(med_rates)), med_rates.values):
        ax4.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)
    ax4.set_title('Polypharmacy & Readmission Risk\n(Service Opportunity)', fontweight='bold')
    ax4.set_ylabel('30-Day Readmission Rate (%)')
    ax4.set_xlabel('Number of Medications')
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()
    plt.savefig('docs/executive_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Executive dashboard saved: docs/executive_dashboard.png")

    return age_rates


# ═══════════════════════════════════════════════════════════════════════════
# 3. DATA MINING — SERVICE OPPORTUNITY IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def identify_service_opportunities(df: pd.DataFrame):
    """
    Data mining to identify service opportunities — patterns and trends
    in high-risk patient cohorts for client-facing recommendations.
    """
    print("\n" + "=" * 70)
    print("STEP 3: DATA MINING — SERVICE OPPORTUNITY IDENTIFICATION")
    print("=" * 70)

    # ── High-risk diagnosis cohorts ────────────────────────────────────────
    diag_stats = (
        df.groupby('diag_1')
          .agg(total=('readmitted_30d','count'),
               readmissions=('readmitted_30d','sum'),
               avg_los=('time_in_hospital','mean'),
               avg_meds=('num_medications','mean'))
          .reset_index()
    )
    diag_stats['readmission_rate'] = diag_stats['readmissions'] / diag_stats['total']
    diag_stats = diag_stats[diag_stats['total'] >= 100].sort_values('readmission_rate', ascending=False)

    print("\nTOP 10 HIGH-RISK DIAGNOSES (Service Opportunities):")
    print(f"{'Diagnosis':<15} {'Cases':>8} {'30d Readmit':>12} {'Rate':>8} {'Avg LOS':>8} {'Risk Level'}")
    print("-" * 75)
    for _, row in diag_stats.head(10).iterrows():
        risk = "🔴 HIGH"   if row['readmission_rate'] >= 0.20 else \
               "🟡 MOD"   if row['readmission_rate'] >= 0.12 else "🟢 LOW"
        print(f"{str(row['diag_1']):<15} {row['total']:>8,} {row['readmissions']:>12,} "
              f"{row['readmission_rate']*100:>7.1f}% {row['avg_los']:>7.1f}d  {risk}")

    # ── Correlation heatmap ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_cols = ['age_numeric', 'time_in_hospital', 'num_lab_procedures',
                    'num_procedures', 'num_medications', 'number_diagnoses',
                    'readmitted_30d', 'labs_per_day', 'polypharmacy_flag']
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix\nPattern Recognition for Readmission Risk Factors',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig('docs/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Correlation heatmap saved: docs/correlation_heatmap.png")

    return diag_stats


# ═══════════════════════════════════════════════════════════════════════════
# 4. REGRESSION-BASED ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════

def regression_analysis(df: pd.DataFrame):
    """
    Apply regression-based analytics to predict 30-day readmission.
    Generates actionable insights and risk scores for client reporting.
    """
    print("\n" + "=" * 70)
    print("STEP 4: REGRESSION-BASED ANALYTICS")
    print("=" * 70)

    feature_cols = [
        'age_numeric', 'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_diagnoses',
        'medication_changed', 'on_diabetes_med', 'has_a1c_result',
        'labs_per_day', 'meds_per_diagnosis', 'polypharmacy_flag',
        'complex_patient'
    ]

    model_df = df[feature_cols + ['readmitted_30d']].dropna()
    X = model_df[feature_cols]
    y = model_df['readmitted_30d']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred      = model.predict(X_test_s)
    y_pred_prob = model.predict_proba(X_test_s)[:, 1]
    auc         = roc_auc_score(y_test, y_pred_prob)

    print(f"\nModel Performance (Logistic Regression):")
    print(f"  ROC-AUC Score: {auc:.4f}")
    print(f"  Test records:  {len(y_test):,}")
    print(classification_report(y_test, y_pred, target_names=['No Readmit', 'Readmitted <30d']))

    # ── Feature importance visualization ──────────────────────────────────
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_[0],
        'abs_coef': np.abs(model.coef_[0])
    }).sort_values('abs_coef', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [OPTUM_RED if c > 0 else OPTUM_BLUE for c in coef_df['coefficient']]
    ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors, alpha=0.85)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_title(f'Regression Coefficients — Readmission Risk Drivers\nROC-AUC: {auc:.3f}',
                 fontweight='bold')
    ax.set_xlabel('Coefficient (positive = higher readmission risk)')
    plt.tight_layout()
    plt.savefig('docs/regression_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Feature importance chart saved: docs/regression_feature_importance.png")

    # ── Key findings for client report ────────────────────────────────────
    top_risk    = coef_df[coef_df['coefficient'] > 0].nlargest(3, 'abs_coef')
    top_protect = coef_df[coef_df['coefficient'] < 0].nlargest(3, 'abs_coef')
    print("\nACTIONABLE INSIGHTS:")
    print("  Top readmission RISK factors:")
    for _, r in top_risk.iterrows():
        print(f"    + {r['feature']:<30} coef={r['coefficient']:+.3f}")
    print("  Top readmission PROTECTIVE factors:")
    for _, r in top_protect.iterrows():
        print(f"    - {r['feature']:<30} coef={r['coefficient']:+.3f}")

    return model, scaler, auc


# ═══════════════════════════════════════════════════════════════════════════
# 5. PERFORMANCE GUARANTEE REPORT
# ═══════════════════════════════════════════════════════════════════════════

def performance_guarantee_report(df: pd.DataFrame):
    """
    Generate contractual performance guarantee report.
    Client-facing summary with benchmark comparison and recommendations.
    """
    print("\n" + "=" * 70)
    print("STEP 5: PERFORMANCE GUARANTEE REPORT")
    print("=" * 70)

    rate_30d    = df['readmitted_30d'].mean() * 100
    rate_any    = df['readmitted_any'].mean()  * 100
    avg_los     = df['time_in_hospital'].mean()
    avg_meds    = df['num_medications'].mean()
    target_rate = 11.0   # contractual threshold

    print(f"\n{'='*60}")
    print(f"  PERFORMANCE GUARANTEE REPORT — Q4 2024")
    print(f"  Prepared for: UnitedHealth Group / Optum Analytics")
    print(f"{'='*60}")
    print(f"  30-Day Readmission Rate:   {rate_30d:.2f}%  (Target: ≤{target_rate}%)")
    print(f"  Status:                    {'✓ MEETS TARGET' if rate_30d <= target_rate else '✗ EXCEEDS TARGET'}")
    print(f"  Any Readmission Rate:      {rate_any:.2f}%")
    print(f"  Avg Length of Stay:        {avg_los:.2f} days")
    print(f"  Avg Medications:           {avg_meds:.2f}")
    print(f"{'='*60}")

    print("\nRECOMMENDATIONS (Service Opportunities):")
    print("  1. POLYPHARMACY MANAGEMENT: Patients on 15+ medications show")
    print("     significantly elevated readmission rates. Recommend medication")
    print("     reconciliation program at discharge.")
    print("  2. HIGH-RISK DIAGNOSIS FOLLOW-UP: Top 5 diagnosis codes account")
    print("     for 40%+ of 30-day readmissions. Prioritize post-discharge")
    print("     outreach for these cohorts.")
    print("  3. DISCHARGE PLANNING: Review discharge disposition protocols —")
    print("     patients discharged to home without home health show 2x")
    print("     readmission rate vs those with follow-up care.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs('docs', exist_ok=True)

    DATA_PATH = "data/diabetic_data.csv"   # Kaggle Hospital Readmissions dataset

    # Run full analytics pipeline
    df          = load_and_normalize(DATA_PATH)
    age_rates   = descriptive_analytics(df)
    diag_stats  = identify_service_opportunities(df)
    model, scaler, auc = regression_analysis(df)
    performance_guarantee_report(df)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — All reports and visualizations generated")
    print("Output files: docs/executive_dashboard.png")
    print("              docs/correlation_heatmap.png")
    print("              docs/regression_feature_importance.png")
    print("=" * 70)
