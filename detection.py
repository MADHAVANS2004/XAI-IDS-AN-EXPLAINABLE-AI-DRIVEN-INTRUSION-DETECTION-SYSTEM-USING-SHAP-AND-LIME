# ============================================================
#  detection.py
#  Real-time detection engine.
#  Feeds simulated packets into the trained model and returns
#  predictions with confidence and top feature importances.
# ============================================================

import datetime
import logging
import os
import numpy as np
import pandas as pd

from data_preprocessing import LABEL_MAP
from simulation import packet_to_df, FEATURE_COLS

# Reverse map: number → class name
CLASS_NAMES = {v: k for k, v in LABEL_MAP.items()}

# ─── Alert Logger ─────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
LOG_DIR   = os.path.join(BASE, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE  = os.path.join(LOG_DIR, 'alerts.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
    ]
)
alert_logger = logging.getLogger('IDS_ALERTS')


def get_severity(attack_cat: str, confidence: float) -> str:
    """Map attack category + confidence to a severity string."""
    if attack_cat == 'Normal':
        return 'INFO'
    if attack_cat == 'U2R':
        return 'CRITICAL'          # Privilege escalation = always critical
    if confidence >= 90:
        return 'CRITICAL'
    if confidence >= 70:
        return 'HIGH'
    if confidence >= 50:
        return 'MEDIUM'
    return 'LOW'


def detect(pkt: dict, model, scaler, feature_names: list) -> dict:
    """
    Run detection on one simulated packet.

    Parameters
    ----------
    pkt          : dict from simulation.generate_packet()
    model        : trained sklearn classifier (RF or DT)
    scaler       : fitted StandardScaler
    feature_names: list of feature column names (from preprocessing)

    Returns
    -------
    dict with: prediction, label, confidence, severity,
               top_features, src_ip, dst_ip, timestamp
    """
    # 1. Build feature DataFrame
    input_df = packet_to_df(pkt)

    # Ensure column order matches what the model was trained on
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # 2. Scale
    input_scaled = pd.DataFrame(
        scaler.transform(input_df), columns=feature_names)

    # 3. Predict
    prediction  = int(model.predict(input_scaled)[0])
    proba       = model.predict_proba(input_scaled)[0]

    # Map class index to class name
    classes     = model.classes_.tolist()
    pred_idx    = classes.index(prediction)
    confidence  = round(float(proba[pred_idx]) * 100, 2)

    attack_cat  = CLASS_NAMES.get(prediction, 'Unknown')
    severity    = get_severity(attack_cat, confidence)

    # 4. Top feature importances (tree-based, no per-sample SHAP here)
    importances  = model.feature_importances_
    top_idx      = np.argsort(importances)[::-1][:5]
    top_features = [
        {'name': feature_names[i], 'importance': round(float(importances[i]), 4)}
        for i in top_idx
    ]

    result = {
        'timestamp':      datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'src_ip':         pkt.get('src_ip', 'N/A'),
        'dst_ip':         pkt.get('dst_ip', 'N/A'),
        'protocol':       pkt.get('proto_name', 'N/A'),
        'service':        pkt.get('service_name', 'N/A'),
        'prediction':     prediction,
        'label':          attack_cat,
        'confidence':     confidence,
        'severity':       severity,
        'top_features':   top_features,
        'src_bytes':      pkt.get('src_bytes', 0),
        'dst_bytes':      pkt.get('dst_bytes', 0),
        'duration':       pkt.get('duration', 0),
    }

    # 5. Log alerts (non-normal traffic)
    if attack_cat != 'Normal':
        _log_alert(result)

    return result


def _log_alert(result: dict):
    """Append a formatted alert line to alerts.log."""
    msg = (
        f"[{result['severity']}] {result['label']} detected | "
        f"src={result['src_ip']} dst={result['dst_ip']} | "
        f"protocol={result['protocol']} service={result['service']} | "
        f"confidence={result['confidence']}% | "
        f"top_feature={result['top_features'][0]['name'] if result['top_features'] else 'N/A'}"
    )
    if result['severity'] == 'CRITICAL':
        alert_logger.critical(msg)
    elif result['severity'] == 'HIGH':
        alert_logger.warning(msg)
    else:
        alert_logger.info(msg)


def get_recent_log_lines(n: int = 20) -> list:
    """Return the last n lines of the alerts log as a list of strings."""
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-n:] if l.strip()]


# ── Standalone demo ───────────────────────────────────────────
if __name__ == '__main__':
    import sys, pickle
    from data_preprocessing import prepare_datasets

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RF_PATH  = os.path.join(BASE_DIR, 'models', 'random_forest.pkl')

    if not os.path.exists(RF_PATH):
        print("Train models first: python model.py")
        sys.exit(1)

    with open(RF_PATH, 'rb') as f:
        model = pickle.load(f)

    data = prepare_datasets(
        os.path.join(BASE_DIR, 'data', 'KDDTrain+.txt'),
        os.path.join(BASE_DIR, 'data', 'KDDTest+.txt'),
    )
    scaler       = data['scaler']
    feature_names= data['feature_names']

    from simulation import generate_packet
    print("Running 5 real-time detections:\n")
    for i in range(5):
        pkt    = generate_packet('auto')
        result = detect(pkt, model, scaler, feature_names)
        print(f"  [{result['timestamp']}] {result['src_ip']} → {result['dst_ip']} "
              f"| {result['label']} ({result['confidence']}%) [{result['severity']}]")
