# XAI-Driven Intrusion Detection System (Upgraded)

**Final Year Project | NSL-KDD · Random Forest · Decision Tree · SHAP · LIME · Simulation**

---

## Project Structure

```
ids_final/
│
├── data/
│   ├── KDDTrain+.txt          ← NSL-KDD training set
│   └── KDDTest+.txt           ← NSL-KDD test set
│
├── models/
│   ├── random_forest.pkl      ← Saved RF model (auto-generated on first run)
│   ├── decision_tree.pkl      ← Saved DT model (auto-generated on first run)
│   └── scaler.pkl             ← Saved StandardScaler
│
├── logs/
│   └── alerts.log             ← All attack alerts with timestamp, IPs, confidence
│
├── templates/
│   ├── base.html              ← Navbar + layout
│   ├── index.html             ← Home / dashboard
│   ├── predict.html           ← Manual detection form
│   ├── simulate.html          ← Real-time simulation terminal
│   ├── visualize.html         ← Confusion matrix, SHAP, feature importance
│   ├── alerts.html            ← Alert log viewer
│   ├── login.html
│   └── register.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── data_preprocessing.py      ← Load & preprocess NSL-KDD dataset
├── model.py                   ← Train RF and DT classifiers
├── simulation.py              ← Simulate normal and attack network packets
├── detection.py               ← Real-time detection engine
├── explainability.py          ← SHAP and LIME explanation generators
├── app.py                     ← Main Flask web application
├── requirements.txt
└── README.md
```

---

## Step-by-Step Setup and Execution

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

> Python 3.8+ recommended. Use a virtual environment (optional but clean):
> ```bash
> python -m venv venv
> source venv/bin/activate   # Windows: venv\Scripts\activate
> pip install -r requirements.txt
> ```

### Step 2 — Run the Application

```bash
python app.py
```

The app will:
1. Load the NSL-KDD dataset from `data/` (downloads from GitHub if missing)
2. Preprocess and encode features
3. Train Random Forest and Decision Tree models
4. Launch the Flask web server

This takes **1–3 minutes** on first run (training time). Subsequent runs reuse saved models.

### Step 3 — Open in Browser

```
http://127.0.0.1:5000
```

**Default login:** `admin` / `ids2025`

---

## Running Individual Modules (Optional)

You can test each module independently:

```bash
# Test data preprocessing only
python data_preprocessing.py

# Train models and save to disk
python model.py

# Simulate 10 network packets (prints to console)
python simulation.py

# Run real-time detection on 5 simulated packets
python detection.py

# No self-test for explainability.py — it's imported by app.py
```

---

## How the System Works

### Dataset — NSL-KDD

The **NSL-KDD** dataset is a cleaned version of the KDD Cup 1999 dataset.

| Split    | Records   |
|----------|-----------|
| Train    | 125,973   |
| Test     | 22,544    |
| Features | 41        |

**5 Classes (multi-class):**

| Class  | Description                                       |
|--------|---------------------------------------------------|
| Normal | Legitimate network traffic                        |
| DoS    | Denial of Service (neptune, smurf, teardrop, …)  |
| Probe  | Port/network scanning (ipsweep, nmap, satan, …)  |
| R2L    | Remote to Local (guess_passwd, ftp_write, …)     |
| U2R    | User to Root (buffer_overflow, rootkit, …)       |

### Preprocessing (`data_preprocessing.py`)

1. Drop the `difficulty` column
2. Encode categorical features with fixed maps:
   - `protocol_type`: tcp→0, udp→1, icmp→2
   - `flag`: SF→0, S0→1, REJ→2, …
   - `service`: http→0, ftp→1, ssh→2, … (70+ services)
3. Map attack label strings → 0–4 numeric class
4. Normalize all features with `StandardScaler`

### Models (`model.py`)

| Model         | Config                              |
|---------------|-------------------------------------|
| Random Forest | 100 trees, max_depth=20, balanced   |
| Decision Tree | max_depth=15, balanced class weights|

Both trained on 125,973 samples with balanced class weights to handle class imbalance (U2R is very rare).

---

## How Simulation Works (`simulation.py`)

The simulation module generates **fake but realistic network packets** without needing a real network interface.

Each packet mimics an NSL-KDD row:
- **Metadata:** `src_ip`, `dst_ip`, `timestamp`, `protocol`, `service`
- **Features:** All 41 NSL-KDD numeric features tuned per attack type

**Traffic Generators:**

| Generator        | Key Characteristics                                    |
|------------------|--------------------------------------------------------|
| `generate_normal()` | SF flag, TCP, HTTP/HTTPS, low error rates          |
| `generate_dos()`    | S0 flag, high SYN error rate, 200–511 connections  |
| `generate_probe()`  | Many different services, low src_bytes             |
| `generate_r2l()`    | SSH/FTP, multiple failed logins, REJ error rate    |
| `generate_u2r()`    | root_shell=1, su_attempted=1, file creations       |

**Default traffic mix (auto mode):**
- 50% Normal, 25% DoS, 12% Probe, 8% R2L, 5% U2R

---

## Real-Time Detection (`detection.py`)

The detection engine:
1. Takes a simulated packet dict
2. Converts it to a DataFrame with the 41 feature columns
3. Scales with the fitted `StandardScaler`
4. Runs `model.predict()` and `model.predict_proba()`
5. Returns: predicted class, confidence %, severity level
6. Writes to `logs/alerts.log` for non-Normal traffic

**Severity mapping:**

| Severity  | Condition                          |
|-----------|------------------------------------|
| CRITICAL  | U2R attack OR confidence ≥ 90%     |
| HIGH      | confidence ≥ 70%                   |
| MEDIUM    | confidence ≥ 50%                   |
| LOW       | confidence < 50%                   |
| INFO      | Normal traffic                     |

---

## How SHAP is Used (`explainability.py`)

**SHAP (SHapley Additive exPlanations)** uses cooperative game theory to assign each feature a contribution score for a specific prediction.

- `shap_bar_for_sample()` — bar chart for one prediction (red = pushes toward attack, green = pushes toward normal)
- `shap_summary_plot()` — global mean |SHAP| over 150 test samples
- `feature_importance_plot()` — model-level importance (mean decrease in impurity)

TreeSHAP is used (fast algorithm for tree-based models — no sampling needed).

---

## How LIME is Used (`explainability.py`)

**LIME (Local Interpretable Model-agnostic Explanations)** explains one prediction at a time by:
1. Creating 300 perturbed versions of the input
2. Asking the model to predict each one
3. Fitting a simple linear model on those results
4. The linear model coefficients = feature weights

`lime_explanation()` returns a bar chart showing the top 10 features and their weights.

---

## How to View the Alert Log

### In the Browser
Navigate to **Alerts** page (`/alerts`) — shows all alerts with severity badges, confidence, and timestamp.

### In the File System
```bash
cat logs/alerts.log
```

Format:
```
2026-03-26 14:23:01 [CRITICAL] DoS | src=10.10.99.5 dst=192.168.1.12 | confidence=97.3% | top_feature=serror_rate
2026-03-26 14:23:03 [HIGH]     Probe | src=10.10.99.8 dst=192.168.1.20 | confidence=81.2% | top_feature=diff_srv_rate
```

### Via API
```
GET /api/log/recent
```
Returns last 30 lines as JSON.

---

## Web Pages Summary

| URL         | Page                  | Description                                    |
|-------------|------------------------|------------------------------------------------|
| `/`         | Home / Dashboard       | Model metrics, attack types, system pipeline   |
| `/predict`  | Manual Detection       | Enter feature values, get SHAP + LIME charts   |
| `/simulate` | Simulation Terminal    | Real-time streaming detection, live log view   |
| `/visualize`| Visualizations         | Confusion matrix, SHAP summary, feature import |
| `/alerts`   | Alert Log              | All saved alerts with severity badges          |

---

## Troubleshooting

**Q: App takes too long to start**
A: First run trains both models. Grab a coffee ☕ — it's a one-time cost (~2–3 min).

**Q: SHAP summary takes 20+ seconds**
A: Normal — TreeSHAP runs over 150 test samples. Results are cached after the first load.

**Q: Dataset not found**
A: The app auto-downloads from GitHub if `data/KDDTrain+.txt` is missing. Ensure internet access on first run.

**Q: Port 5000 already in use**
A: Change the port in `app.py`: `app.run(port=5001)`

---

## Login Credentials

| Username | Password |
|----------|----------|
| admin    | ids2025  |

You can register new accounts via the `/register` page.
