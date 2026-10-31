"""
Training Pipeline
Run this script to generate data, train all models, and save artifacts.

Usage
-----
    python train.py
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure src is on path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_generator import generate_placement_data
from src.data_preprocessing import PlacementPreprocessor, prepare_salary_data
from src.feature_engineering import add_engineered_features, encode_performance_tier
from src.models import get_classifiers, get_regressors
from src.evaluation import (
    classification_report_dict, regression_report_dict,
    compare_classifiers, compare_regressors,
    plot_roc_curves, plot_feature_importance,
    plot_confusion_matrix, plot_actual_vs_predicted
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'placement_data.csv')
MODEL_DIR   = os.path.join(BASE_DIR, 'models')
PLOTS_DIR   = os.path.join(BASE_DIR, 'plots')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("  CAMPUS PLACEMENT PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    # ── Step 1: Generate / Load Data ──────────────────────────────
    if not os.path.exists(DATA_PATH):
        print("\n[1/6] Generating synthetic dataset...")
        df = generate_placement_data(n_students=1000)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"      Saved: {DATA_PATH}")
    else:
        print("\n[1/6] Loading existing dataset...")
        df = pd.read_csv(DATA_PATH)

    print(f"      Records : {len(df):,}")
    print(f"      Placed  : {df['placement_status'].sum():,} ({df['placement_status'].mean():.1%})")

    # ── Step 2: Feature Engineering ───────────────────────────────
    print("\n[2/6] Engineering features...")
    df_feat = add_engineered_features(df)
    df_feat = encode_performance_tier(df_feat)
    print(f"      Features after engineering: {df_feat.shape[1]}")

    # ── Step 3: Preprocessing for Classification ──────────────────
    print("\n[3/6] Preprocessing (classification)...")
    preprocessor = PlacementPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df_feat)
    preprocessor.save(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
    print(f"      Train: {X_train.shape}  Test: {X_test.shape}")

    # ── Step 4: Train & Evaluate Classifiers ──────────────────────
    print("\n[4/6] Training classifiers...")
    classifiers = get_classifiers()
    clf_results  = {}
    roc_data     = {}
    best_clf_score = 0
    best_clf_name  = ''
    best_clf       = None

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None

        metrics = classification_report_dict(y_test, y_pred, y_prob)
        clf_results[name] = metrics
        roc_data[name] = {'y_true': y_test, 'y_prob': y_prob}

        print(f"      {name:<25}  Acc={metrics['Accuracy']:.4f}  "
              f"F1={metrics['F1 Score']:.4f}  AUC={metrics.get('ROC-AUC', 'N/A')}")

        plot_confusion_matrix(y_test, y_pred, name, save_dir=PLOTS_DIR)
        plot_feature_importance(clf, preprocessor.feature_columns, name, save_dir=PLOTS_DIR)

        if metrics['F1 Score'] > best_clf_score:
            best_clf_score = metrics['F1 Score']
            best_clf_name  = name
            best_clf       = clf

        joblib.dump(clf, os.path.join(MODEL_DIR, f'clf_{name.replace(" ", "_")}.pkl'))

    print(f"\n      Best classifier: {best_clf_name} (F1={best_clf_score:.4f})")
    joblib.dump(best_clf, os.path.join(MODEL_DIR, 'best_classifier.pkl'))

    roc_data_clean = {k: v for k, v in roc_data.items() if v['y_prob'] is not None}
    plot_roc_curves(roc_data_clean, save_dir=PLOTS_DIR)

    # Save comparison table
    clf_df = compare_classifiers(clf_results)
    clf_df.to_csv(os.path.join(MODEL_DIR, 'classifier_results.csv'))
    print("\n      Classifier Comparison:")
    print(clf_df.to_string())

    # ── Step 5: Train & Evaluate Regressors ───────────────────────
    print("\n[5/6] Training salary regressors (placed students only)...")
    (X_s_train, X_s_test, y_s_train, y_s_test), salary_scaler = prepare_salary_data(
        df_feat, preprocessor
    )
    joblib.dump(salary_scaler, os.path.join(MODEL_DIR, 'salary_scaler.pkl'))

    regressors  = get_regressors()
    reg_results = {}
    best_reg_score = -np.inf
    best_reg_name  = ''
    best_reg       = None

    for name, reg in regressors.items():
        reg.fit(X_s_train, y_s_train)
        y_pred = reg.predict(X_s_test)
        metrics = regression_report_dict(y_s_test, y_pred)
        reg_results[name] = metrics

        print(f"      {name:<25}  R²={metrics['R²']:.4f}  "
              f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}")

        plot_actual_vs_predicted(y_s_test, y_pred, name, save_dir=PLOTS_DIR)

        if metrics['R²'] > best_reg_score:
            best_reg_score = metrics['R²']
            best_reg_name  = name
            best_reg       = reg

        joblib.dump(reg, os.path.join(MODEL_DIR, f'reg_{name.replace(" ", "_")}.pkl'))

    print(f"\n      Best regressor: {best_reg_name} (R²={best_reg_score:.4f})")
    joblib.dump(best_reg, os.path.join(MODEL_DIR, 'best_regressor.pkl'))

    reg_df = compare_regressors(reg_results)
    reg_df.to_csv(os.path.join(MODEL_DIR, 'regressor_results.csv'))
    print("\n      Regressor Comparison:")
    print(reg_df.to_string())

    # ── Step 6: Done ──────────────────────────────────────────────
    print("\n[6/6] Training complete!")
    print(f"      Models saved to : {MODEL_DIR}")
    print(f"      Plots saved to  : {PLOTS_DIR}")
    print("\n  Run the dashboard with:")
    print("      streamlit run app.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
