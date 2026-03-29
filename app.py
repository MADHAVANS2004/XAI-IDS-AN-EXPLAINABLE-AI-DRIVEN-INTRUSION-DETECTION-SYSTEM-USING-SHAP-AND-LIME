# ============================================================
#  app.py  —  XAI-Driven Intrusion Detection System (Upgraded)
#  Adds: multi-class detection, simulation module, real-time
#        streaming, SHAP/LIME per sample, alerts.log
#
#  Run:  python app.py
#  Open: http://127.0.0.1:5000   (login: admin / ids2025)
# ============================================================

import os, io, base64, warnings, json, hashlib, datetime, pickle
from functools import wraps

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, flash, Response, stream_with_context)

from sklearn.metrics import confusion_matrix
import shap
import lime.lime_tabular

# ─── Our modules ──────────────────────────────────────────────
from data_preprocessing import (
    prepare_datasets, PROTOCOL_MAP, FLAG_MAP, SERVICE_MAP,
    SERVICE_LIST, LABEL_MAP, COL_NAMES
)
from model import train_models, CLASS_NAMES, CLASS_LIST
from simulation import generate_packet, packet_to_df, FEATURE_COLS
from detection import detect, get_recent_log_lines
from explainability import (
    shap_bar_for_sample, shap_summary_plot,
    feature_importance_plot, lime_explanation, fig_to_b64
)

warnings.filterwarnings('ignore')

# ─── Flask setup ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__)
app.secret_key = 'xai-ids-secret-key-2025'

# ─── File paths ───────────────────────────────────────────────
USERS_FILE   = os.path.join(BASE, 'users.json')
ALERTS_FILE  = os.path.join(BASE, 'alerts.json')
LOG_FILE     = os.path.join(BASE, 'logs', 'alerts.log')
DATA_DIR     = os.path.join(BASE, 'data')
MODEL_DIR    = os.path.join(BASE, 'models')
os.makedirs(os.path.join(BASE, 'logs'),   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Globals ──────────────────────────────────────────────────
rf_model       = None
dt_model       = None
scaler         = None
X_train_scaled = None
X_test_scaled  = None
y_test         = None
rf_metrics     = {}
dt_metrics     = {}
feature_names  = []
is_trained     = False

# ════════════════════════════════════════════════════════════
#  USER MANAGEMENT
# ════════════════════════════════════════════════════════════
def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()

def _load_users():
    base = {'admin': _hash('ids2025')}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            base.update(json.load(f))
    return base

def _save_user(username, password):
    users = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            users = json.load(f)
    users[username] = _hash(password)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

# ════════════════════════════════════════════════════════════
#  ALERT MANAGEMENT (JSON + .log file)
# ════════════════════════════════════════════════════════════
def _load_alerts():
    if not os.path.exists(ALERTS_FILE): return []
    with open(ALERTS_FILE) as f: return json.load(f)

def _save_alert(alert: dict):
    alerts = _load_alerts()
    alerts.append(alert)
    with open(ALERTS_FILE, 'w') as f:
        json.dump(alerts, f, indent=2)
    # Also write to alerts.log
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as lf:
            sev  = alert.get('severity', 'INFO')
            cat  = alert.get('label', 'UNKNOWN')
            src  = alert.get('src_ip', 'N/A')
            dst  = alert.get('dst_ip', 'N/A')
            conf = alert.get('confidence', 0)
            ts   = alert.get('timestamp', '')
            feat = alert.get('top_feature', 'N/A')
            lf.write(
                f"{ts} [{sev}] {cat} | src={src} dst={dst} | "
                f"confidence={conf}% | top_feature={feat}\n"
            )
    except Exception:
        pass

def _get_severity(cat: str, conf: float) -> str:
    if cat == 'Normal': return 'INFO'
    if cat == 'U2R':    return 'CRITICAL'
    if conf >= 90: return 'CRITICAL'
    if conf >= 70: return 'HIGH'
    if conf >= 50: return 'MEDIUM'
    return 'LOW'

def _alert_counts(alerts):
    c = {'critical': 0, 'high': 0, 'medium': 0, 'total': len(alerts)}
    for a in alerts:
        s = a.get('severity','').upper()
        if s == 'CRITICAL': c['critical'] += 1
        elif s == 'HIGH':   c['high'] += 1
        elif s == 'MEDIUM': c['medium'] += 1
    return c

# ════════════════════════════════════════════════════════════
#  AUTH DECORATOR
# ════════════════════════════════════════════════════════════
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════
def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#0D1B2A')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img

def safe_float(data, key, default=0.0, lo=None, hi=None):
    try:
        v = float(data.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid value for '{key}'. Expected a number.")
    if lo is not None and v < lo:
        raise ValueError(f"'{key}' must be >= {lo}.")
    if hi is not None and v > hi:
        raise ValueError(f"'{key}' must be <= {hi}.")
    return v

def _extract_shap(shap_values, index=0, class_idx=1):
    if isinstance(shap_values, list):
        ci = min(class_idx, len(shap_values)-1)
        sv = np.array(shap_values[ci])
        return sv[index] if sv.ndim == 2 else sv
    if hasattr(shap_values, 'values'):
        sv = np.array(shap_values.values)
        if sv.ndim == 3:
            return sv[index, :, min(class_idx, sv.shape[2]-1)]
        elif sv.ndim == 2:
            return sv[index]
        return sv
    sv = np.array(shap_values)
    if sv.ndim == 3:
        return sv[index, :, class_idx]
    elif sv.ndim == 2:
        return sv[index]
    return sv

# ════════════════════════════════════════════════════════════
#  TRAINING (on startup)
# ════════════════════════════════════════════════════════════
def load_and_train():
    global rf_model, dt_model, scaler, X_train_scaled, X_test_scaled
    global y_test, rf_metrics, dt_metrics, feature_names, is_trained

    train_path = os.path.join(DATA_DIR, 'KDDTrain+.txt')
    test_path  = os.path.join(DATA_DIR, 'KDDTest+.txt')

    # Fallback: download if not present
    if not os.path.exists(train_path):
        import urllib.request
        print("[*] Downloading NSL-KDD dataset ...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
            train_path)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
            test_path)

    # Prepare data (multi-class: 5 categories)
    data = prepare_datasets(train_path, test_path, multiclass=True)
    X_train_scaled = data['X_train']
    X_test_scaled  = data['X_test']
    y_test         = data['y_test']
    scaler         = data['scaler']
    feature_names  = data['feature_names']

    # Save scaler
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Train
    rf_model, dt_model, rf_metrics, dt_metrics = train_models(
        X_train_scaled, data['y_train'],
        X_test_scaled,  y_test
    )

    is_trained = True
    print("[*] System ready!\n")


# ════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ════════════════════════════════════════════════════════════
@app.route('/login', methods=['GET','POST'])
def login():
    if 'user' in session: return redirect(url_for('index'))
    error = username = ''
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','')
        if not username or not password:
            error = 'Please enter both username and password.'
        else:
            users = _load_users()
            if username not in users or users[username] != _hash(password):
                error = 'Invalid username or password.'
            else:
                session['user'] = username
                return redirect(url_for('index'))
    return render_template('login.html', error=error, username=username)

@app.route('/register', methods=['GET','POST'])
def register():
    if 'user' in session: return redirect(url_for('index'))
    error = success = None
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','')
        confirm  = request.form.get('confirm','')
        if not username or not password or not confirm:
            error = 'All fields are required.'
        elif len(username) < 3:
            error = 'Username must be at least 3 characters.'
        elif len(password) < 6:
            error = 'Password must be at least 6 characters.'
        elif password != confirm:
            error = 'Passwords do not match.'
        else:
            users = _load_users()
            if username in users:
                error = 'Username already taken.'
            else:
                _save_user(username, password)
                success = 'Account created! You can now log in.'
    return render_template('register.html', error=error, success=success)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


# ════════════════════════════════════════════════════════════
#  MAIN ROUTES
# ════════════════════════════════════════════════════════════
@app.route('/')
@login_required
def index():
    return render_template('index.html',
                           rf_metrics=rf_metrics,
                           dt_metrics=dt_metrics,
                           is_trained=is_trained,
                           class_names=CLASS_LIST)

@app.route('/predict')
@login_required
def predict_page():
    return render_template('predict.html')

@app.route('/simulate')
@login_required
def simulate_page():
    return render_template('simulate.html')

@app.route('/visualize')
@login_required
def visualize_page():
    return render_template('visualize.html')


# ════════════════════════════════════════════════════════════
#  API: MANUAL PREDICTION
# ════════════════════════════════════════════════════════════
@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data received.'}), 400

        model_choice = data.get('model', 'rf')
        if not is_trained:
            return jsonify({'error': 'Models not ready yet. Please wait.'}), 503

        # Build feature vector from form inputs
        features = {
            'duration':                    safe_float(data, 'duration', 0, lo=0),
            'protocol_type':               PROTOCOL_MAP.get(data.get('protocol_type','tcp'), 0),
            'service':                     SERVICE_MAP.get(data.get('service','http'), 0),
            'flag':                        FLAG_MAP.get(data.get('flag','SF'), 0),
            'src_bytes':                   safe_float(data, 'src_bytes', 0, lo=0),
            'dst_bytes':                   safe_float(data, 'dst_bytes', 0, lo=0),
            'land':                        safe_float(data, 'land', 0, lo=0, hi=1),
            'wrong_fragment':              safe_float(data, 'wrong_fragment', 0, lo=0),
            'urgent':                      safe_float(data, 'urgent', 0, lo=0),
            'hot':                         safe_float(data, 'hot', 0, lo=0),
            'num_failed_logins':           safe_float(data, 'num_failed_logins', 0, lo=0),
            'logged_in':                   safe_float(data, 'logged_in', 1, lo=0, hi=1),
            'num_compromised':             safe_float(data, 'num_compromised', 0, lo=0),
            'root_shell':                  safe_float(data, 'root_shell', 0, lo=0, hi=1),
            'su_attempted':                safe_float(data, 'su_attempted', 0, lo=0, hi=1),
            'num_root':                    safe_float(data, 'num_root', 0, lo=0),
            'num_file_creations':          safe_float(data, 'num_file_creations', 0, lo=0),
            'num_shells':                  safe_float(data, 'num_shells', 0, lo=0),
            'num_access_files':            safe_float(data, 'num_access_files', 0, lo=0),
            'num_outbound_cmds':           safe_float(data, 'num_outbound_cmds', 0, lo=0),
            'is_host_login':               safe_float(data, 'is_host_login', 0, lo=0, hi=1),
            'is_guest_login':              safe_float(data, 'is_guest_login', 0, lo=0, hi=1),
            'count':                       safe_float(data, 'count', 1, lo=0),
            'srv_count':                   safe_float(data, 'srv_count', 1, lo=0),
            'serror_rate':                 safe_float(data, 'serror_rate', 0, lo=0, hi=1),
            'srv_serror_rate':             safe_float(data, 'srv_serror_rate', 0, lo=0, hi=1),
            'rerror_rate':                 safe_float(data, 'rerror_rate', 0, lo=0, hi=1),
            'srv_rerror_rate':             safe_float(data, 'srv_rerror_rate', 0, lo=0, hi=1),
            'same_srv_rate':               safe_float(data, 'same_srv_rate', 1.0, lo=0, hi=1),
            'diff_srv_rate':               safe_float(data, 'diff_srv_rate', 0, lo=0, hi=1),
            'srv_diff_host_rate':          safe_float(data, 'srv_diff_host_rate', 0, lo=0, hi=1),
            'dst_host_count':              safe_float(data, 'dst_host_count', 255, lo=0),
            'dst_host_srv_count':          safe_float(data, 'dst_host_srv_count', 255, lo=0),
            'dst_host_same_srv_rate':      safe_float(data, 'dst_host_same_srv_rate', 1.0, lo=0, hi=1),
            'dst_host_diff_srv_rate':      safe_float(data, 'dst_host_diff_srv_rate', 0, lo=0, hi=1),
            'dst_host_same_src_port_rate': safe_float(data, 'dst_host_same_src_port_rate', 0, lo=0, hi=1),
            'dst_host_srv_diff_host_rate': safe_float(data, 'dst_host_srv_diff_host_rate', 0, lo=0, hi=1),
            'dst_host_serror_rate':        safe_float(data, 'dst_host_serror_rate', 0, lo=0, hi=1),
            'dst_host_srv_serror_rate':    safe_float(data, 'dst_host_srv_serror_rate', 0, lo=0, hi=1),
            'dst_host_rerror_rate':        safe_float(data, 'dst_host_rerror_rate', 0, lo=0, hi=1),
            'dst_host_srv_rerror_rate':    safe_float(data, 'dst_host_srv_rerror_rate', 0, lo=0, hi=1),
        }

        input_df     = pd.DataFrame([features])
        input_df     = input_df.reindex(columns=feature_names, fill_value=0)
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)

        model       = rf_model if model_choice == 'rf' else dt_model
        prediction  = int(model.predict(input_scaled)[0])
        proba       = model.predict_proba(input_scaled)[0]
        classes     = model.classes_.tolist()
        pred_idx    = classes.index(prediction)
        confidence  = round(float(proba[pred_idx]) * 100, 2)
        attack_cat  = CLASS_NAMES.get(prediction, 'Unknown')

        # SHAP bar chart
        shap_img, sv, top_idx = shap_bar_for_sample(model, input_scaled, pred_idx)
        sv = np.array(sv, dtype=float).flatten()
        feat_names = [input_scaled.columns[i] for i in top_idx[:10]]
        feat_vals  = [float(sv[i]) for i in top_idx[:10]]

        # LIME
        lime_img, lime_list = lime_explanation(model, input_scaled, X_train_scaled)

        # Save alert
        severity = _get_severity(attack_cat, confidence)
        top_feat = feat_names[0] if feat_names else 'N/A'
        if attack_cat != 'Normal':
            _save_alert({
                'timestamp':   datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user':        session.get('user','unknown'),
                'src_ip':      data.get('src_ip', 'manual'),
                'dst_ip':      data.get('dst_ip', 'manual'),
                'label':       attack_cat,
                'severity':    severity,
                'confidence':  confidence,
                'model':       'Random Forest' if model_choice == 'rf' else 'Decision Tree',
                'top_feature': top_feat,
            })

        # Per-class probabilities
        class_probs = {}
        for ci, cls in enumerate(classes):
            class_probs[CLASS_NAMES.get(cls, str(cls))] = round(float(proba[ci]) * 100, 2)

        return jsonify({
            'prediction':    prediction,
            'label':         attack_cat,
            'confidence':    confidence,
            'severity':      severity,
            'class_probs':   class_probs,
            'shap_chart':    shap_img,
            'lime_chart':    lime_img,
            'top_features':  [{'name': n, 'value': round(v, 4)}
                              for n, v in zip(feat_names, feat_vals)],
        })

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════
#  API: SIMULATION — single packet
# ════════════════════════════════════════════════════════════
@app.route('/api/simulate', methods=['POST'])
@login_required
def api_simulate():
    """Generate one simulated packet, run detection, return result."""
    try:
        body         = request.json or {}
        attack_type  = body.get('attack_type', 'auto')
        model_choice = body.get('model', 'rf')

        if not is_trained:
            return jsonify({'error': 'Models not ready'}), 503

        pkt    = generate_packet(attack_type)
        model  = rf_model if model_choice == 'rf' else dt_model
        result = detect(pkt, model, scaler, feature_names)

        # Save to alert log if attack
        if result['label'] != 'Normal':
            _save_alert({
                'timestamp':   result['timestamp'],
                'user':        session.get('user','system'),
                'src_ip':      result['src_ip'],
                'dst_ip':      result['dst_ip'],
                'label':       result['label'],
                'severity':    result['severity'],
                'confidence':  result['confidence'],
                'model':       'Random Forest' if model_choice=='rf' else 'Decision Tree',
                'top_feature': result['top_features'][0]['name'] if result['top_features'] else 'N/A',
            })

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════
#  API: SIMULATION STREAM (Server-Sent Events)
# ════════════════════════════════════════════════════════════
@app.route('/api/simulate/stream')
@login_required
def simulate_stream():
    """
    Server-Sent Events endpoint.
    Streams one detection result every ~1.5 seconds.
    Stops after max_packets (default=20).
    """
    import time
    model_choice = request.args.get('model', 'rf')
    attack_type  = request.args.get('attack_type', 'auto')
    max_packets  = int(request.args.get('max', 20))

    def event_gen():
        model = rf_model if model_choice == 'rf' else dt_model
        for i in range(max_packets):
            try:
                pkt    = generate_packet(attack_type)
                result = detect(pkt, model, scaler, feature_names)

                # Save to alert store
                if result['label'] != 'Normal':
                    _save_alert({
                        'timestamp':   result['timestamp'],
                        'user':        'system',
                        'src_ip':      result['src_ip'],
                        'dst_ip':      result['dst_ip'],
                        'label':       result['label'],
                        'severity':    result['severity'],
                        'confidence':  result['confidence'],
                        'model':       'Random Forest' if model_choice=='rf' else 'Decision Tree',
                        'top_feature': result['top_features'][0]['name'] if result['top_features'] else 'N/A',
                    })

                yield f"data: {json.dumps(result)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

            time.sleep(1.2)

        yield "data: {\"done\": true}\n\n"

    return Response(stream_with_context(event_gen()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


# ════════════════════════════════════════════════════════════
#  API: VISUALIZATIONS (global)
# ════════════════════════════════════════════════════════════
@app.route('/api/visualize/cm')
@login_required
def api_confusion_matrix():
    """Return base64 confusion matrix for RF and DT."""
    if not is_trained:
        return jsonify({'error': 'Not trained'}), 503
    try:
        result = {}
        for name, model, metrics in [('rf', rf_model, rf_metrics),
                                      ('dt', dt_model, dt_metrics)]:
            labels     = sorted(y_test.unique())
            y_pred     = model.predict(X_test_scaled)
            cm         = confusion_matrix(y_test, y_pred, labels=labels)
            label_names= [CLASS_NAMES.get(l, str(l)) for l in labels]

            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor('#0D1B2A')
            ax.set_facecolor('#0D1B2A')
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=label_names, yticklabels=label_names,
                        linewidths=0.5, linecolor='#1A2A3A',
                        annot_kws={'size': 11, 'color': 'white'}, ax=ax)
            ax.set_xlabel('Predicted', color='#B0BEC5', fontsize=11)
            ax.set_ylabel('Actual',    color='#B0BEC5', fontsize=11)
            ax.set_title(f'{"Random Forest" if name=="rf" else "Decision Tree"} — Confusion Matrix',
                         color='#00C8FF', fontsize=13, fontweight='bold')
            ax.tick_params(colors='#B0BEC5', labelsize=9)
            plt.tight_layout()
            result[name] = fig_to_base64(fig)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize/shap_summary')
@login_required
def api_shap_summary():
    """Global SHAP summary for RF and DT."""
    if not is_trained:
        return jsonify({'error': 'Not trained'}), 503
    try:
        rf_img = shap_summary_plot(rf_model, X_test_scaled, max_samples=150)
        dt_img = shap_summary_plot(dt_model, X_test_scaled, max_samples=150)
        return jsonify({'rf': rf_img, 'dt': dt_img})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize/feature_importance')
@login_required
def api_feature_importance():
    """Feature importance bar chart."""
    if not is_trained:
        return jsonify({'error': 'Not trained'}), 503
    try:
        rf_img = feature_importance_plot(rf_model, feature_names)
        dt_img = feature_importance_plot(dt_model, feature_names)
        return jsonify({'rf': rf_img, 'dt': dt_img})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════
#  ALERT ROUTES
# ════════════════════════════════════════════════════════════
@app.route('/alerts')
@login_required
def alerts_page():
    alerts = list(reversed(_load_alerts()))
    counts = _alert_counts(alerts)
    return render_template('alerts.html', alerts=alerts, counts=counts)

@app.route('/alerts/count')
@login_required
def alerts_count():
    alerts = _load_alerts()
    return jsonify({'count': len([a for a in alerts if a.get('label') != 'Normal'])})

@app.route('/alerts/clear', methods=['POST'])
@login_required
def alerts_clear():
    with open(ALERTS_FILE, 'w') as f:
        json.dump([], f)
    return jsonify({'success': True})

@app.route('/api/log/recent')
@login_required
def api_log_recent():
    """Return recent lines from alerts.log."""
    lines = get_recent_log_lines(n=30)
    return jsonify({'lines': lines})


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("  XAI-IDS — Intrusion Detection System (Upgraded)")
    print("  Loading & training models on NSL-KDD, please wait ...")
    print("=" * 60)
    load_and_train()
    print("\n✓ Open: http://127.0.0.1:5000   (admin / ids2025)\n")
    app.run(debug=False, port=5000, threaded=True)
