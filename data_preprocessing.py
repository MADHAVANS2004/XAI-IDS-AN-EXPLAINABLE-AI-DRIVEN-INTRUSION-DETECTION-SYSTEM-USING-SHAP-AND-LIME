# ============================================================
#  data_preprocessing.py
#  Loads and preprocesses the NSL-KDD dataset.
#  Handles: label encoding, normalization, categorical features.
#  Supports MULTI-CLASS labels: Normal, DoS, Probe, R2L, U2R
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Column names for the NSL-KDD dataset ─────────────────────
COL_NAMES = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

# ── Attack category mapping (multi-class) ─────────────────────
# Maps each specific attack name → one of 5 categories
ATTACK_MAP = {
    'normal': 'Normal',
    # DoS attacks
    'back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS','smurf':'DoS',
    'teardrop':'DoS','apache2':'DoS','udpstorm':'DoS','processtable':'DoS',
    'mailbomb':'DoS',
    # Probe attacks
    'satan':'Probe','ipsweep':'Probe','nmap':'Probe','portsweep':'Probe',
    'mscan':'Probe','saint':'Probe',
    # R2L (Remote to Local) attacks
    'guess_passwd':'R2L','ftp_write':'R2L','imap':'R2L','phf':'R2L',
    'multihop':'R2L','warezmaster':'R2L','warezclient':'R2L','spy':'R2L',
    'xlock':'R2L','xsnoop':'R2L','snmpguess':'R2L','snmpgetattack':'R2L',
    'httptunnel':'R2L','sendmail':'R2L','named':'R2L',
    # U2R (User to Root) attacks
    'buffer_overflow':'U2R','loadmodule':'U2R','rootkit':'U2R','perl':'U2R',
    'sqlattack':'U2R','xterm':'U2R','ps':'U2R',
}

# Numerical label mapping for the 5 categories
LABEL_MAP = {'Normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4}

# ── Categorical column fixed maps ─────────────────────────────
# Kept here so simulation.py and app.py can import them
PROTOCOL_MAP = {'tcp': 0, 'udp': 1, 'icmp': 2}
FLAG_MAP     = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTO': 3, 'SH': 4,
                'S1': 5, 'S2': 6, 'RSTOS0': 7, 'S3': 8, 'OTH': 9}
SERVICE_LIST = [
    'http','ftp','smtp','ssh','dns','ftp_data','irc','pop_3',
    'telnet','other','private','domain_u','auth','finger','eco_i',
    'ntp_u','ecr_i','nntp','time','urp_i','red_i','netbios_ssn',
    'rje','X11','netbios_ns','vmnet','bgp','link','supdup',
    'name','mtp','sql_net','uucp','pm_dump','IRC','daytime',
    'iso_tsap','printer','ldap','klogin','pop_2','nnsp','netstat',
    'gopher','remote_job','login','shell','kshell','efs','exec',
    'courier','ctf','csnet_ns','uucp_path','harvest','aol','http_443',
    'http_8001','discard','systat','echo','tim_i','whois','domain',
    'sunrpc','tftp_u','urh_i','Z39_50'
]
SERVICE_MAP = {s: i for i, s in enumerate(SERVICE_LIST)}

# ─────────────────────────────────────────────────────────────

def _map_label_multiclass(label: str) -> int:
    """Map a raw attack name to a numeric 5-class label."""
    category = ATTACK_MAP.get(label.strip().lower(), 'DoS')  # unknown → DoS
    return LABEL_MAP[category]


def _map_label_binary(label: str) -> int:
    """Map a raw label to binary 0 (normal) / 1 (attack)."""
    return 0 if label.strip().lower() == 'normal' else 1


def load_data(train_path: str, test_path: str):
    """
    Load NSL-KDD train and test files.
    Returns raw DataFrames (train_df, test_df).
    """
    train_df = pd.read_csv(train_path, header=None, names=COL_NAMES)
    test_df  = pd.read_csv(test_path,  header=None, names=COL_NAMES)
    print(f"[Data] Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    return train_df, test_df


def preprocess(df: pd.DataFrame, multiclass: bool = True) -> pd.DataFrame:
    """
    Preprocess a raw NSL-KDD DataFrame:
      1. Drop the 'difficulty' column
      2. Encode categorical features (protocol_type, service, flag)
      3. Map attack labels → numeric class
    Returns the processed DataFrame.
    """
    df = df.copy()

    # Step 1: Remove difficulty column (not useful for ML)
    df.drop(columns=['difficulty'], inplace=True, errors='ignore')

    # Step 2: Encode categorical columns using fixed maps
    # protocol_type: tcp=0, udp=1, icmp=2
    df['protocol_type'] = df['protocol_type'].map(PROTOCOL_MAP).fillna(0).astype(int)

    # service: mapped to integer index
    df['service'] = df['service'].map(SERVICE_MAP).fillna(len(SERVICE_MAP)).astype(int)

    # flag: SF=0, S0=1, REJ=2, …
    df['flag'] = df['flag'].map(FLAG_MAP).fillna(0).astype(int)

    # Step 3: Map label strings → numeric class
    if multiclass:
        df['label'] = df['label'].apply(_map_label_multiclass)
    else:
        df['label'] = df['label'].apply(_map_label_binary)

    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit StandardScaler on X_train, transform both X_train and X_test.
    Returns (X_train_scaled, X_test_scaled, scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(
        scaler.transform(X_test),      columns=X_test.columns)
    return X_train_scaled, X_test_scaled, scaler


def prepare_datasets(train_path: str, test_path: str, multiclass: bool = True):
    """
    Full pipeline: load → preprocess → scale.
    Returns dict with all splits and the fitted scaler.
    """
    train_df, test_df = load_data(train_path, test_path)

    train_p = preprocess(train_df, multiclass=multiclass)
    test_p  = preprocess(test_df,  multiclass=multiclass)

    X_train = train_p.drop(columns=['label'])
    y_train = train_p['label']
    X_test  = test_p.drop(columns=['label'])
    y_test  = test_p['label']

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print(f"[Data] Features: {X_train.shape[1]} | Classes: {sorted(y_train.unique())}")
    print(f"[Data] Train class distribution:\n{y_train.value_counts().sort_index().to_string()}")

    return {
        'X_train': X_train_scaled,
        'X_test':  X_test_scaled,
        'y_train': y_train,
        'y_test':  y_test,
        'scaler':  scaler,
        'feature_names': X_train.columns.tolist(),
    }


# ── Self-test ─────────────────────────────────────────────────
if __name__ == '__main__':
    BASE = os.path.dirname(os.path.abspath(__file__))
    result = prepare_datasets(
        os.path.join(BASE, 'data', 'KDDTrain+.txt'),
        os.path.join(BASE, 'data', 'KDDTest+.txt'),
        multiclass=True
    )
    print("\npreprocess OK")
    print("X_train shape:", result['X_train'].shape)
    print("y_test  shape:", result['y_test'].shape)
