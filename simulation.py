# ============================================================
#  simulation.py
#  Simulates network packets every 1–2 seconds.
#  Generates BOTH normal traffic and attack traffic.
#  Each packet includes: src_ip, dst_ip, protocol, service,
#  flag, bytes, duration, and all NSL-KDD features needed
#  for the classifier.
# ============================================================

import random
import time
import datetime
import numpy as np
import pandas as pd

from data_preprocessing import (
    PROTOCOL_MAP, FLAG_MAP, SERVICE_MAP, SERVICE_LIST, LABEL_MAP
)

# ─── Attack category → label number ──────────────────────────
ATTACK_LABEL = {
    'Normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4
}

# ─── IP pools ─────────────────────────────────────────────────
INTERNAL_IPS = [f"192.168.1.{i}"  for i in range(10, 30)]
EXTERNAL_IPS = [f"203.0.113.{i}"  for i in range(1, 50)]
ATTACKER_IPS = [f"10.10.99.{i}"   for i in range(1, 20)]

# ─── Helpers ─────────────────────────────────────────────────
def _ri(lo, hi): return random.randint(lo, hi)
def _rf(lo, hi): return round(random.uniform(lo, hi), 4)
def _pick(lst):  return random.choice(lst)

def _base_packet(src_ip: str, dst_ip: str) -> dict:
    """Build a packet dict with all 41 NSL-KDD numeric features + metadata."""
    proto   = _pick(list(PROTOCOL_MAP.keys()))
    service = _pick(SERVICE_LIST[:20])   # common services
    flag    = _pick(list(FLAG_MAP.keys()))
    return {
        # ── Metadata (not used by classifier directly) ──────
        'src_ip':   src_ip,
        'dst_ip':   dst_ip,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        # ── NSL-KDD features ────────────────────────────────
        'duration':           0,
        'protocol_type':      PROTOCOL_MAP[proto],
        'service':            SERVICE_MAP.get(service, 0),
        'flag':               FLAG_MAP[flag],
        'src_bytes':          0,
        'dst_bytes':          0,
        'land':               0,
        'wrong_fragment':     0,
        'urgent':             0,
        'hot':                0,
        'num_failed_logins':  0,
        'logged_in':          1,
        'num_compromised':    0,
        'root_shell':         0,
        'su_attempted':       0,
        'num_root':           0,
        'num_file_creations': 0,
        'num_shells':         0,
        'num_access_files':   0,
        'num_outbound_cmds':  0,
        'is_host_login':      0,
        'is_guest_login':     0,
        'count':              _ri(1, 50),
        'srv_count':          _ri(1, 30),
        'serror_rate':        0.0,
        'srv_serror_rate':    0.0,
        'rerror_rate':        0.0,
        'srv_rerror_rate':    0.0,
        'same_srv_rate':      _rf(0.7, 1.0),
        'diff_srv_rate':      _rf(0.0, 0.2),
        'srv_diff_host_rate': _rf(0.0, 0.1),
        'dst_host_count':     _ri(100, 255),
        'dst_host_srv_count': _ri(100, 255),
        'dst_host_same_srv_rate':      _rf(0.7, 1.0),
        'dst_host_diff_srv_rate':      _rf(0.0, 0.2),
        'dst_host_same_src_port_rate': _rf(0.0, 0.3),
        'dst_host_srv_diff_host_rate': _rf(0.0, 0.1),
        'dst_host_serror_rate':        0.0,
        'dst_host_srv_serror_rate':    0.0,
        'dst_host_rerror_rate':        0.0,
        'dst_host_srv_rerror_rate':    0.0,
        # ── Display fields ───────────────────────────────────
        'proto_name':   proto,
        'service_name': service,
        'flag_name':    flag,
    }


# ════════════════════════════════════════════════════════════
#  TRAFFIC GENERATORS
# ════════════════════════════════════════════════════════════

def generate_normal():
    """Simulate a normal HTTP/HTTPS/DNS connection."""
    pkt = _base_packet(_pick(INTERNAL_IPS), _pick(INTERNAL_IPS + EXTERNAL_IPS))
    pkt.update({
        'duration':    _ri(0, 5),
        'proto_name':  _pick(['tcp', 'udp']),
        'service_name':'http',
        'service':     SERVICE_MAP.get('http', 0),
        'protocol_type': PROTOCOL_MAP['tcp'],
        'flag_name':   'SF',
        'flag':        FLAG_MAP['SF'],
        'src_bytes':   _ri(200, 5000),
        'dst_bytes':   _ri(200, 20000),
        'logged_in':   1,
        'count':       _ri(1, 10),
        'srv_count':   _ri(1, 10),
        'same_srv_rate': _rf(0.85, 1.0),
        'attack_category': 'Normal',
        'true_label':  ATTACK_LABEL['Normal'],
    })
    return pkt


def generate_dos():
    """
    Simulate a DoS (Denial of Service) attack.
    Characterised by: high SYN error rate, many connections, low bytes.
    Similar to 'neptune' or 'smurf' in NSL-KDD.
    """
    pkt = _base_packet(_pick(ATTACKER_IPS), _pick(INTERNAL_IPS))
    pkt.update({
        'duration':    0,
        'proto_name':  'tcp',
        'protocol_type': PROTOCOL_MAP['tcp'],
        'flag_name':   'S0',
        'flag':        FLAG_MAP['S0'],
        'src_bytes':   0,
        'dst_bytes':   0,
        'logged_in':   0,
        'count':       _ri(200, 511),
        'srv_count':   _ri(200, 511),
        'serror_rate':        _rf(0.8, 1.0),
        'srv_serror_rate':    _rf(0.8, 1.0),
        'same_srv_rate':      _rf(0.8, 1.0),
        'diff_srv_rate':      _rf(0.0, 0.1),
        'dst_host_count':     _ri(200, 255),
        'dst_host_srv_count': _ri(200, 255),
        'dst_host_serror_rate':     _rf(0.7, 1.0),
        'dst_host_srv_serror_rate': _rf(0.7, 1.0),
        'attack_category': 'DoS',
        'true_label':  ATTACK_LABEL['DoS'],
    })
    return pkt


def generate_probe():
    """
    Simulate a Probe attack (port/network scan).
    Characterised by: many different services, low bytes, short duration.
    Similar to 'ipsweep' or 'portsweep' in NSL-KDD.
    """
    pkt = _base_packet(_pick(ATTACKER_IPS), _pick(INTERNAL_IPS))
    pkt.update({
        'duration':    0,
        'proto_name':  _pick(['tcp', 'icmp']),
        'protocol_type': PROTOCOL_MAP[_pick(['tcp', 'icmp'])],
        'flag_name':   'SF',
        'flag':        FLAG_MAP['SF'],
        'src_bytes':   _ri(0, 100),
        'dst_bytes':   _ri(0, 100),
        'logged_in':   0,
        'count':       _ri(1, 10),
        'srv_count':   _ri(1, 10),
        'diff_srv_rate':      _rf(0.4, 1.0),
        'same_srv_rate':      _rf(0.0, 0.3),
        'dst_host_count':     _ri(200, 255),
        'dst_host_srv_count': _ri(1, 30),
        'dst_host_same_srv_rate': _rf(0.0, 0.2),
        'dst_host_diff_srv_rate': _rf(0.5, 1.0),
        'attack_category': 'Probe',
        'true_label':  ATTACK_LABEL['Probe'],
    })
    return pkt


def generate_r2l():
    """
    Simulate an R2L (Remote to Local) attack.
    Characterised by: failed logins, sensitive services (ssh/ftp).
    Similar to 'guess_passwd' or 'ftp_write' in NSL-KDD.
    """
    pkt = _base_packet(_pick(EXTERNAL_IPS), _pick(INTERNAL_IPS))
    pkt.update({
        'duration':           _ri(1, 30),
        'proto_name':         'tcp',
        'protocol_type':      PROTOCOL_MAP['tcp'],
        'service_name':       _pick(['ssh', 'ftp', 'telnet']),
        'flag_name':          'SF',
        'flag':               FLAG_MAP['SF'],
        'src_bytes':          _ri(200, 3000),
        'dst_bytes':          _ri(100, 2000),
        'logged_in':          0,
        'num_failed_logins':  _ri(3, 10),
        'count':              _ri(1, 20),
        'srv_count':          _ri(1, 20),
        'rerror_rate':        _rf(0.5, 1.0),
        'srv_rerror_rate':    _rf(0.5, 1.0),
        'dst_host_rerror_rate':     _rf(0.4, 1.0),
        'dst_host_srv_rerror_rate': _rf(0.4, 1.0),
        'attack_category': 'R2L',
        'true_label':  ATTACK_LABEL['R2L'],
    })
    pkt['service'] = SERVICE_MAP.get(pkt['service_name'], 0)
    return pkt


def generate_u2r():
    """
    Simulate a U2R (User to Root) privilege escalation.
    Characterised by: root shell, su attempts, file creations.
    Similar to 'buffer_overflow' or 'rootkit' in NSL-KDD.
    """
    pkt = _base_packet(_pick(INTERNAL_IPS), _pick(INTERNAL_IPS))
    pkt.update({
        'duration':           _ri(10, 120),
        'proto_name':         'tcp',
        'protocol_type':      PROTOCOL_MAP['tcp'],
        'service_name':       _pick(['ssh', 'telnet']),
        'flag_name':          'SF',
        'flag':               FLAG_MAP['SF'],
        'src_bytes':          _ri(1000, 10000),
        'dst_bytes':          _ri(500,  5000),
        'logged_in':          1,
        'root_shell':         1,
        'su_attempted':       1,
        'num_root':           _ri(1, 10),
        'num_file_creations': _ri(1, 5),
        'num_shells':         _ri(1, 3),
        'hot':                _ri(2, 20),
        'count':              _ri(1, 5),
        'attack_category': 'U2R',
        'true_label':  ATTACK_LABEL['U2R'],
    })
    pkt['service'] = SERVICE_MAP.get(pkt['service_name'], 0)
    return pkt


# ─── Packet → DataFrame row (features only) ──────────────────
FEATURE_COLS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate',
]

def packet_to_df(pkt: dict) -> pd.DataFrame:
    """Extract only the ML feature columns from a packet dict."""
    row = {col: pkt.get(col, 0) for col in FEATURE_COLS}
    return pd.DataFrame([row])


# ─── Weighted random generator ─────────────────────────────
GENERATORS = [
    (generate_normal,  0.50),   # 50% normal
    (generate_dos,     0.25),   # 25% DoS
    (generate_probe,   0.12),   # 12% Probe
    (generate_r2l,     0.08),   #  8% R2L
    (generate_u2r,     0.05),   #  5% U2R
]

def generate_packet(attack_type: str = 'auto') -> dict:
    """
    Generate one simulated packet.
    attack_type: 'auto'   → weighted random (realistic mix)
                 'normal' → always normal
                 'dos'    → always DoS
                 'probe'  → always Probe
                 'r2l'    → always R2L
                 'u2r'    → always U2R
                 'attack' → random attack (no normal)
    """
    if attack_type == 'normal':
        return generate_normal()
    if attack_type == 'dos':
        return generate_dos()
    if attack_type == 'probe':
        return generate_probe()
    if attack_type == 'r2l':
        return generate_r2l()
    if attack_type == 'u2r':
        return generate_u2r()
    if attack_type == 'attack':
        return _pick([generate_dos, generate_probe, generate_r2l, generate_u2r])()
    # auto / weighted
    funcs, weights = zip(*GENERATORS)
    return random.choices(funcs, weights=weights, k=1)[0]()


# ── Standalone demo ───────────────────────────────────────────
if __name__ == '__main__':
    print("Simulating 10 packets (mix):\n")
    for i in range(10):
        pkt = generate_packet('auto')
        cat = pkt['attack_category']
        src = pkt['src_ip']
        dst = pkt['dst_ip']
        proto = pkt['proto_name']
        svc   = pkt['service_name']
        sb    = pkt['src_bytes']
        db    = pkt['dst_bytes']
        print(f"  [{pkt['timestamp']}] {src} → {dst} | {proto}/{svc} | "
              f"src={sb}B dst={db}B | Category={cat}")
        time.sleep(0.1)
