# ============================================================
#  model.py
#  Trains Random Forest and Decision Tree classifiers on the
#  NSL-KDD dataset (multi-class: Normal / DoS / Probe / R2L / U2R).
#  Also computes evaluation metrics and confusion matrices.
# ============================================================

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from data_preprocessing import prepare_datasets, LABEL_MAP

# Reverse map: number → class name (for display)
CLASS_NAMES = {v: k for k, v in LABEL_MAP.items()}
CLASS_LIST  = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]

# ─── Paths ───────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, 'data')
MODEL_DIR  = os.path.join(BASE, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

RF_PATH    = os.path.join(MODEL_DIR, 'random_forest.pkl')
DT_PATH    = os.path.join(MODEL_DIR, 'decision_tree.pkl')
SCALER_PATH= os.path.join(MODEL_DIR, 'scaler.pkl')


# ─── Train ───────────────────────────────────────────────────

def train_models(X_train, y_train, X_test, y_test):
    """
    Train both classifiers and return them with their metrics.
    """
    print("[Model] Training Random Forest (n_estimators=100) ...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'   # handles class imbalance
    )
    rf.fit(X_train, y_train)

    print("[Model] Training Decision Tree ...")
    dt = DecisionTreeClassifier(
        max_depth=15,
        random_state=42,
        class_weight='balanced'
    )
    dt.fit(X_train, y_train)

    # Save models and scaler to disk for later reuse
    print("[Model] Saving models to disk ...")
    with open(RF_PATH,  'wb') as f: pickle.dump(rf, f)
    with open(DT_PATH,  'wb') as f: pickle.dump(dt, f)

    # Compute metrics
    rf_metrics = _compute_metrics(rf, X_test, y_test, 'Random Forest')
    dt_metrics = _compute_metrics(dt, X_test, y_test, 'Decision Tree')

    print("\n[Model] Random Forest:")
    print(f"  Accuracy: {rf_metrics['accuracy']:.2f}%")
    print("[Model] Decision Tree:")
    print(f"  Accuracy: {dt_metrics['accuracy']:.2f}%")

    return rf, dt, rf_metrics, dt_metrics


def _compute_metrics(model, X_test, y_test, name: str) -> dict:
    """Calculate accuracy, precision, recall, F1, confusion matrix."""
    y_pred = model.predict(X_test)
    labels = sorted(y_test.unique())

    acc  = round(accuracy_score(y_test, y_pred)  * 100, 2)
    prec = round(precision_score(y_test, y_pred, average='weighted', labels=labels, zero_division=0) * 100, 2)
    rec  = round(recall_score(y_test, y_pred,    average='weighted', labels=labels, zero_division=0) * 100, 2)
    f1   = round(f1_score(y_test, y_pred,        average='weighted', labels=labels, zero_division=0) * 100, 2)
    cm   = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    report = classification_report(y_test, y_pred,
                                   target_names=[CLASS_NAMES.get(l, str(l)) for l in labels],
                                   zero_division=0)
    print(f"\n[{name}] Classification Report:\n{report}")

    return {
        'name':      name,
        'accuracy':  acc,
        'precision': prec,
        'recall':    rec,
        'f1':        f1,
        'cm':        cm,
        'labels':    labels,
    }


def save_confusion_matrix_plot(metrics: dict, filename: str):
    """
    Save a styled confusion matrix heatmap as a PNG.
    """
    cm     = np.array(metrics['cm'])
    labels = [CLASS_NAMES.get(l, str(l)) for l in metrics['labels']]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0D1B2A')
    ax.set_facecolor('#0D1B2A')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor='#1A2A3A',
                annot_kws={'size': 11, 'color': 'white'}, ax=ax)

    ax.set_xlabel('Predicted', color='#B0BEC5', fontsize=11)
    ax.set_ylabel('Actual',    color='#B0BEC5', fontsize=11)
    ax.set_title(f'{metrics["name"]} — Confusion Matrix',
                 color='#00C8FF', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='#B0BEC5', labelsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight', facecolor='#0D1B2A')
    plt.close()
    print(f"[Model] Confusion matrix saved → {filename}")


def load_models():
    """Load saved models from disk. Returns (rf, dt) or (None, None)."""
    if os.path.exists(RF_PATH) and os.path.exists(DT_PATH):
        with open(RF_PATH, 'rb') as f: rf = pickle.load(f)
        with open(DT_PATH, 'rb') as f: dt = pickle.load(f)
        print("[Model] Loaded saved models from disk.")
        return rf, dt
    return None, None


# ── Self-test / standalone training ──────────────────────────
if __name__ == '__main__':
    data = prepare_datasets(
        os.path.join(DATA_DIR, 'KDDTrain+.txt'),
        os.path.join(DATA_DIR, 'KDDTest+.txt'),
        multiclass=True
    )
    rf, dt, rf_m, dt_m = train_models(
        data['X_train'], data['y_train'],
        data['X_test'],  data['y_test']
    )
    # Save confusion matrix plots
    save_confusion_matrix_plot(rf_m, os.path.join(MODEL_DIR, 'rf_cm.png'))
    save_confusion_matrix_plot(dt_m, os.path.join(MODEL_DIR, 'dt_cm.png'))
    print("\nmodel.py self-test complete.")
