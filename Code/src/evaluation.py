"""
Model Evaluation Module
Metrics, confusion matrices, ROC curves, and feature importance plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
import os


# ── Classification ────────────────────────────────────────────────────────────

def classification_report_dict(y_true, y_pred, y_prob=None) -> dict:
    """Return classification metrics as a dictionary."""
    metrics = {
        'Accuracy' : round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall'   : round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1 Score' : round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        metrics['ROC-AUC'] = round(roc_auc_score(y_true, y_prob), 4)
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir: str = None):
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Not Placed', 'Placed'],
        yticklabels=['Not Placed', 'Placed']
    )
    ax.set_title(f'Confusion Matrix — {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'cm_{model_name.replace(" ", "_")}.png'), dpi=150)
    plt.close(fig)
    return fig


def plot_roc_curves(models_results: dict, save_dir: str = None):
    """
    Plot ROC curves for multiple models.

    Parameters
    ----------
    models_results : dict
        {model_name: {'y_true': ..., 'y_prob': ...}}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in models_results.items():
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
        auc = roc_auc_score(res['y_true'], res['y_prob'])
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Classifiers')
    ax.legend(loc='lower right')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=150)
    plt.close(fig)
    return fig


def plot_feature_importance(model, feature_names: list, model_name: str, save_dir: str = None):
    """Plot feature importance for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        return None

    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    top_n = min(15, len(feature_names))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        range(top_n),
        importance[indices[:top_n]][::-1],
        color=sns.color_palette('viridis', top_n)
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[:top_n]][::-1])
    ax.set_title(f'Feature Importance — {model_name}')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'fi_{model_name.replace(" ", "_")}.png'), dpi=150)
    plt.close(fig)
    return fig


def compare_classifiers(results: dict) -> pd.DataFrame:
    """Build a comparison DataFrame from classification results."""
    rows = []
    for name, metrics in results.items():
        row = {'Model': name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).set_index('Model').sort_values('F1 Score', ascending=False)


# ── Regression ────────────────────────────────────────────────────────────────

def regression_report_dict(y_true, y_pred) -> dict:
    """Return regression metrics as a dictionary."""
    return {
        'MAE' : round(mean_absolute_error(y_true, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        'R²'  : round(r2_score(y_true, y_pred), 4),
    }


def compare_regressors(results: dict) -> pd.DataFrame:
    """Build a comparison DataFrame from regression results."""
    rows = []
    for name, metrics in results.items():
        row = {'Model': name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).set_index('Model').sort_values('R²', ascending=False)


def plot_actual_vs_predicted(y_true, y_pred, model_name: str, save_dir: str = None):
    """Scatter plot of actual vs predicted salary."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidths=0.3, color='steelblue')
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect Fit')
    ax.set_xlabel('Actual Salary (LPA)')
    ax.set_ylabel('Predicted Salary (LPA)')
    ax.set_title(f'Actual vs Predicted — {model_name}')
    ax.legend()
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'avp_{model_name.replace(" ", "_")}.png'), dpi=150)
    plt.close(fig)
    return fig
