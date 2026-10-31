"""
Synthetic Campus Placement Dataset Generator
Generates realistic student placement data with statistically plausible distributions.
"""

import numpy as np
import pandas as pd
import os

def generate_placement_data(n_students: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic campus placement dataset.

    Parameters
    ----------
    n_students : int
        Number of student records to generate
    random_state : int
        Seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with student features and placement outcomes
    """
    rng = np.random.default_rng(random_state)

    # ── Demographics ──────────────────────────────────────────────────────────
    gender = rng.choice(['Male', 'Female'], size=n_students, p=[0.65, 0.35])

    ssc_board = rng.choice(['Central', 'Others'], size=n_students, p=[0.55, 0.45])
    hsc_board = rng.choice(['Central', 'Others'], size=n_students, p=[0.50, 0.50])
    hsc_subject = rng.choice(['Commerce', 'Science', 'Arts'], size=n_students, p=[0.45, 0.40, 0.15])
    degree_type = rng.choice(['Comm&Mgmt', 'Sci&Tech', 'Others'], size=n_students, p=[0.50, 0.38, 0.12])
    work_experience = rng.choice(['Yes', 'No'], size=n_students, p=[0.35, 0.65])
    mba_specialization = rng.choice(['Mkt&HR', 'Mkt&Fin'], size=n_students, p=[0.52, 0.48])

    # ── Academic Scores (correlated) ──────────────────────────────────────────
    # Base ability drives all scores
    base_ability = rng.normal(0, 1, n_students)

    ssc_percentage  = np.clip(60 + 10 * base_ability + rng.normal(0, 5, n_students), 40, 98)
    hsc_percentage  = np.clip(60 + 9  * base_ability + rng.normal(0, 6, n_students), 40, 98)
    degree_percentage = np.clip(62 + 8 * base_ability + rng.normal(0, 6, n_students), 40, 98)
    mba_percentage  = np.clip(60 + 7  * base_ability + rng.normal(0, 7, n_students), 40, 98)
    etest_percentage = np.clip(55 + 10 * base_ability + rng.normal(0, 8, n_students), 30, 99)

    # ── Placement Logic ───────────────────────────────────────────────────────
    # Logistic placement probability based on multiple factors
    log_odds = (
        0.3                                     # tuned for ~65% placement rate
        + 0.04 * (degree_percentage - 60)
        + 0.03 * (mba_percentage - 60)
        + 0.03 * (etest_percentage - 55)
        + 0.02 * (ssc_percentage - 60)
        + 0.5  * (work_experience == 'Yes').astype(float)
        + 0.3  * (mba_specialization == 'Mkt&Fin').astype(float)
        + 0.2  * (gender == 'Male').astype(float)
        + 0.15 * (degree_type == 'Sci&Tech').astype(float)
    )
    prob_placed = 1 / (1 + np.exp(-log_odds))
    placement_status = (rng.uniform(size=n_students) < prob_placed).astype(int)

    # ── Salary (only for placed students) ────────────────────────────────────
    salary = np.full(n_students, np.nan)
    placed_idx = placement_status == 1

    base_salary = (
        2.5
        + 0.05 * (mba_percentage[placed_idx] - 60)
        + 0.04 * (degree_percentage[placed_idx] - 60)
        + 0.03 * (etest_percentage[placed_idx] - 55)
        + 0.8  * (work_experience[placed_idx] == 'Yes').astype(float)
        + 0.6  * (mba_specialization[placed_idx] == 'Mkt&Fin').astype(float)
        + 0.4  * (degree_type[placed_idx] == 'Sci&Tech').astype(float)
    )
    salary[placed_idx] = np.clip(
        base_salary + rng.normal(0, 0.8, placed_idx.sum()), 2.0, 12.0
    )

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        'gender'             : gender,
        'ssc_percentage'     : np.round(ssc_percentage, 2),
        'ssc_board'          : ssc_board,
        'hsc_percentage'     : np.round(hsc_percentage, 2),
        'hsc_board'          : hsc_board,
        'hsc_subject'        : hsc_subject,
        'degree_percentage'  : np.round(degree_percentage, 2),
        'degree_type'        : degree_type,
        'work_experience'    : work_experience,
        'etest_percentage'   : np.round(etest_percentage, 2),
        'mba_specialization' : mba_specialization,
        'mba_percentage'     : np.round(mba_percentage, 2),
        'placement_status'   : placement_status,
        'salary'             : np.round(salary, 2),
    })

    return df


if __name__ == '__main__':
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'placement_data.csv')
    df = generate_placement_data(n_students=1000)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated: {len(df)} records")
    print(f"Placement rate   : {df['placement_status'].mean():.1%}")
    print(f"Avg salary (placed): {df['salary'].mean():.2f} LPA")
    print(f"Saved to: {os.path.abspath(output_path)}")
