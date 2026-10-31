"""
Feature Engineering Module
Creates derived features that improve model performance.
"""

import pandas as pd
import numpy as np


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the placement dataset.

    New Features
    ------------
    academic_composite    : Weighted average of all academic scores
    academic_consistency  : Std dev of academic scores (lower = more consistent)
    performance_tier      : Low / Mid / High based on composite score
    degree_to_mba_growth  : Improvement from degree to MBA
    exp_fin_interaction   : Work experience × Finance specialization flag
    top_performer         : Binary flag for top quartile students
    """
    df = df.copy()

    academic_cols = ['ssc_percentage', 'hsc_percentage', 'degree_percentage', 'mba_percentage']

    # Weighted composite (recent education weighted more)
    weights = np.array([0.15, 0.20, 0.30, 0.35])
    df['academic_composite'] = (df[academic_cols].values * weights).sum(axis=1)

    # Consistency: lower std = more consistent performer
    df['academic_consistency'] = df[academic_cols].std(axis=1)

    # Degree to MBA growth
    df['degree_to_mba_growth'] = df['mba_percentage'] - df['degree_percentage']

    # Performance tier — fixed thresholds so single-row inference never errors
    def _tier(score):
        if score < 60:
            return 'Low'
        elif score < 72:
            return 'Mid'
        return 'High'
    df['performance_tier'] = df['academic_composite'].apply(_tier)

    # Work experience × Finance specialization interaction
    exp_flag = (df['work_experience'] == 'Yes').astype(int)
    fin_flag = (df['mba_specialization'] == 'Mkt&Fin').astype(int)
    df['exp_fin_interaction'] = exp_flag * fin_flag

    # Top performer flag (top 25% composite + above median etest)
    composite_q75 = df['academic_composite'].quantile(0.75)
    etest_median  = df['etest_percentage'].median()
    df['top_performer'] = (
        (df['academic_composite'] >= composite_q75) &
        (df['etest_percentage'] >= etest_median)
    ).astype(int)

    return df


def encode_performance_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode the performance_tier column."""
    tier_map = {'Low': 0, 'Mid': 1, 'High': 2}
    if 'performance_tier' in df.columns:
        df = df.copy()
        df['performance_tier'] = df['performance_tier'].map(tier_map).fillna(1).astype(int)
    return df
