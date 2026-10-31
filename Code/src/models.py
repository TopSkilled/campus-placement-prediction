"""
ML Model Definitions
Provides factory functions for classification and regression models.
"""

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor


def get_classifiers() -> dict:
    """Return a dict of classification models keyed by name."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, verbosity=0
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        ),
        'SVM': SVC(
            kernel='rbf', probability=True, random_state=42, C=1.0
        ),
    }


def get_regressors() -> dict:
    """Return a dict of regression models keyed by name."""
    return {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42, verbosity=0
        ),
    }
