# ============================================================
#  explainability.py
#  Generates SHAP and LIME explanations for model predictions.
#  - SHAP: global feature importance summary + per-sample bars
#  - LIME: local explanation for a single prediction
# ============================================================

import io
import base64
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shap
import lime
import lime.lime_tabular

from data_preprocessing import LABEL_MAP

warnings.filterwarnings('ignore')

CLASS_NAMES = {v: k for k, v in LABEL_MAP.items()}

# ─── Helper: matplotlib figure → base64 PNG string ───────────
def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#0D1B2A')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img


# ════════════════════════════════════════════════════════════
#  SHAP EXPLANATIONS
# ════════════════════════════════════════════════════════════

def shap_bar_for_sample(model, input_scaled: pd.DataFrame,
                         pred_class_idx: int = 1) -> str:
    """
    Compute SHAP values for ONE sample and return a bar-chart
    image (base64 PNG) showing top-10 feature contributions.
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    # Extract SHAP values for the predicted class
    sv = _extract_shap(shap_values, index=0, class_idx=pred_class_idx)
    sv = np.array(sv, dtype=float).flatten()

    top_idx    = np.argsort(np.abs(sv))[-10:][::-1]
    feat_names = [input_scaled.columns[i] for i in top_idx]
    feat_vals  = [float(sv[i]) for i in top_idx]
    colors     = ['#FF5252' if v > 0 else '#00E676' for v in feat_vals]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0D1B2A')
    ax.set_facecolor('#0D1B2A')
    ax.barh(range(len(feat_names)), feat_vals, color=colors, edgecolor='none')
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, color='#B0BEC5', fontsize=9)
    ax.set_xlabel('SHAP Value  (red=pushes toward attack, green=pushes toward normal)',
                  color='#B0BEC5', fontsize=8)
    ax.set_title('SHAP — Feature Contributions', color='#00C8FF',
                 fontweight='bold', fontsize=12)
    ax.tick_params(colors='#B0BEC5')
    for spine in ax.spines.values():
        spine.set_color('#2A3A4A')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(0, color='#2A3A4A', linewidth=1)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig_to_b64(fig), sv, top_idx


def shap_summary_plot(model, X_sample: pd.DataFrame,
                       max_samples: int = 200) -> str:
    """
    Generate a SHAP summary (beeswarm) plot for up to max_samples rows.
    Returns base64 PNG.
    """
    sample = X_sample.sample(min(max_samples, len(X_sample)), random_state=42)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # For multi-class, take the class with highest average |SHAP|
    if isinstance(shap_values, list):
        # shap_values is list of arrays, one per class
        sv_abs = [np.abs(np.array(sv)).mean() for sv in shap_values]
        best   = int(np.argmax(sv_abs))
        sv     = np.array(shap_values[best])
    else:
        sv = np.array(shap_values.values) if hasattr(shap_values, 'values') else np.array(shap_values)
        if sv.ndim == 3:
            sv = sv[:, :, 1]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('#0D1B2A')
    ax.set_facecolor('#0D1B2A')

    # Mean |SHAP| bar chart (simpler than beeswarm, always works)
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-15:]
    top_names = [sample.columns[i] for i in top_idx]
    top_vals  = mean_abs[top_idx]

    ax.barh(range(len(top_names)), top_vals, color='#00C8FF', edgecolor='none', alpha=0.8)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, color='#B0BEC5', fontsize=8)
    ax.set_xlabel('Mean |SHAP Value|', color='#B0BEC5', fontsize=9)
    ax.set_title('SHAP Summary — Top 15 Features (Global)', color='#00C8FF',
                 fontweight='bold', fontsize=13)
    ax.tick_params(colors='#B0BEC5')
    for spine in ax.spines.values():
        spine.set_color('#2A3A4A')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig_to_b64(fig)


def feature_importance_plot(model, feature_names: list) -> str:
    """
    Bar chart of model.feature_importances_ (works for RF and DT).
    Returns base64 PNG.
    """
    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:15]
    top_names   = [feature_names[i] for i in top_idx]
    top_vals    = importances[top_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0D1B2A')
    ax.set_facecolor('#0D1B2A')

    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_vals)))
    ax.bar(range(len(top_names)), top_vals, color=colors, edgecolor='none')
    ax.set_xticks(range(len(top_names)))
    ax.set_xticklabels(top_names, rotation=45, ha='right',
                       color='#B0BEC5', fontsize=8)
    ax.set_ylabel('Importance', color='#B0BEC5', fontsize=9)
    ax.set_title('Feature Importance (Model-level)', color='#FF9800',
                 fontweight='bold', fontsize=13)
    ax.tick_params(colors='#B0BEC5')
    for spine in ax.spines.values():
        spine.set_color('#2A3A4A')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig_to_b64(fig)


# ════════════════════════════════════════════════════════════
#  LIME EXPLANATION
# ════════════════════════════════════════════════════════════

def lime_explanation(model, input_scaled: pd.DataFrame,
                      X_train_scaled: pd.DataFrame,
                      class_names: list = None) -> tuple:
    """
    Explain one prediction with LIME.
    Returns (base64 PNG, list of (feature, weight) tuples).
    """
    if class_names is None:
        class_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data   = X_train_scaled.values,
        feature_names   = X_train_scaled.columns.tolist(),
        class_names     = class_names,
        mode            = 'classification',
        random_state    = 42
    )

    exp = explainer.explain_instance(
        input_scaled.values[0],
        model.predict_proba,
        num_features = 10,
        num_samples  = 300,
    )

    lime_list = exp.as_list()
    lf = [x[0] for x in lime_list]
    lv = [float(x[1]) for x in lime_list]
    lc = ['#FF5252' if v > 0 else '#00E676' for v in lv]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0D1B2A')
    ax.set_facecolor('#0D1B2A')
    ax.barh(range(len(lf)), lv, color=lc, edgecolor='none')
    ax.set_yticks(range(len(lf)))
    ax.set_yticklabels(lf, color='#B0BEC5', fontsize=8)
    ax.set_xlabel('LIME Weight  (red=toward attack class, green=toward normal)',
                  color='#B0BEC5', fontsize=8)
    ax.set_title('LIME — Local Feature Explanation', color='#FF9800',
                 fontweight='bold', fontsize=12)
    ax.tick_params(colors='#B0BEC5')
    for spine in ax.spines.values():
        spine.set_color('#2A3A4A')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(0, color='#2A3A4A', linewidth=1)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig_to_b64(fig), lime_list


# ─── Internal helper ─────────────────────────────────────────
def _extract_shap(shap_values, index=0, class_idx=1):
    """Safely extract SHAP values for one sample and one class."""
    if isinstance(shap_values, list):
        # Multi-class list: shap_values[class_idx][sample_idx]
        ci = min(class_idx, len(shap_values) - 1)
        sv = np.array(shap_values[ci])
        return sv[index] if sv.ndim == 2 else sv
    if hasattr(shap_values, 'values'):
        sv = np.array(shap_values.values)
        if sv.ndim == 3:
            return sv[index, :, class_idx] if sv.shape[2] > class_idx else sv[index, :, -1]
        elif sv.ndim == 2:
            return sv[index]
        return sv
    sv = np.array(shap_values)
    if sv.ndim == 3:
        return sv[index, :, class_idx]
    elif sv.ndim == 2:
        return sv[index]
    return sv
