import os
import math
import numpy as np
from pathlib import Path
import config as C
from typing import Optional, List, Tuple, Dict

def _jclean(v):
    if isinstance(v, dict):
        return {str(k): _jclean(x) for (k, x) in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jclean(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v

def _iou(b1, b2) -> float:
    (x1, y1) = (max(b1[0], b2[0]), max(b1[1], b2[1]))
    (x2, y2) = (min(b1[2], b2[2]), min(b1[3], b2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return float(inter / max(a1 + a2 - inter, 1e-06))

def _gap(b1, b2) -> float:
    dx = max(b1[0] - b2[2], b2[0] - b1[2], 0.0)
    dy = max(b1[1] - b2[3], b2[1] - b1[3], 0.0)
    return math.sqrt(dx * dx + dy * dy)

def _dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def _diag(v: dict) -> float:
    (x1, y1, x2, y2) = v['bbox']
    return math.sqrt(float(x2 - x1) ** 2 + float(y2 - y1) ** 2)

def _adiff(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)

class Config:
    BASE_DIR = getattr(C, 'BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
    MODEL = getattr(C, 'MODEL', os.path.join(BASE_DIR, 'best.pt'))
    CONF = getattr(C, 'CONF', 0.51)
    IOU = getattr(C, 'IOU_NMS', 0.50)
    INPUT_SIZE = getattr(C, 'INPUT_SIZE', 640)
    CLASSES = getattr(C, 'CLASSES', [0, 1, 2, 3])
    CLASS_REMAP = getattr(C, 'CLASS_REMAP', {0: 'accident', 1: 'fire', 2: 'motorcycle', 3: 'vehicle'})
    NAMES = getattr(C, 'NAMES', {0: 'accident', 1: 'fire', 2: 'motorcycle', 3: 'vehicle'})
    COLORS = getattr(C, 'COLORS', {'vehicle': (0, 220, 0), 'motorcycle': (0, 220, 220), 'fire': (0, 60, 255), 'accident': (0, 0, 255)})
    LOCK_AFTER_SECONDS = getattr(C, 'LOCK_SECS', 5.0)
    MIN_W_RATIO = getattr(C, 'MIN_W_RATIO', 0.020)
    MIN_H_RATIO = getattr(C, 'MIN_H_RATIO', 0.015)
    MAX_AREA_R = getattr(C, 'MAX_AREA_R', 0.45)
    TRACK_BUFFER = getattr(C, 'TRACK_BUFFER', 120)
    MATCH_THRESH = getattr(C, 'MATCH_THRESH', 0.70)
    HIST_SECS = getattr(C, 'HIST_SECS', 3.0)
    USE_GMC = getattr(C, 'USE_GMC', True)
    COLL_REL_SPD = getattr(C, 'COLL_REL_SPD', 3.0)
    SUDDEN_DEC = getattr(C, 'SUDDEN_DEC', -8.5)
    MIN_DEC_SPD = getattr(C, 'MIN_DEC_SPD', 7.0)
    SHARP_TURN = getattr(C, 'SHARP_TURN', 70.0)
    MIN_TURN_SPD = getattr(C, 'MIN_TURN_SPD', 5.0)
    TTC_HIGH = getattr(C, 'TTC_HIGH', 0.6)
    TTC_MED = getattr(C, 'TTC_MED', 1.8)
    CONVOY_DEG = getattr(C, 'CONVOY_DEG', 35.0)
    CONVOY_SPD = getattr(C, 'CONVOY_SPD', 2.5)
    CONFIRM_FR = getattr(C, 'CONFIRM_FR', 6)
    POST_BURNED = getattr(C, 'POST_BURNED', 4.0)
    MAHAL_THRESH = getattr(C, 'MAHAL_THRESH', 5.0)
    BAYES_THRESH = getattr(C, 'BAYES_THRESH', 0.72)
    LONG_STOP_S = getattr(C, 'LONG_STOP_S', 7.0)
    CLUSTER_MIN = getattr(C, 'CLUSTER_MIN', 4)
    CLUSTER_EPS = getattr(C, 'CLUSTER_EPS', 130)
    CLUSTER_CD = getattr(C, 'CLUSTER_CD', 90)
    EVENT_CD = getattr(C, 'EVENT_CD', 90)
    FIRE_ENABLED = getattr(C, 'FIRE_ENABLED', True)
    FIRE_MODEL = getattr(C, 'FIRE_MODEL', 'keremberke/yolov8n-fire-detection')
    FIRE_CONF = getattr(C, 'FIRE_CONF', 0.45)
    FIRE_CONFIRM = getattr(C, 'FIRE_CONFIRM', 3)
    SMOKE_CONFIRM = getattr(C, 'SMOKE_CONFIRM', 4)
    FIRE_PAD_Y = getattr(C, 'FIRE_PAD_Y', 0.60)
    FIRE_PAD_X = getattr(C, 'FIRE_PAD_X', 0.20)
    FIRE_FULL_FRM = getattr(C, 'FIRE_FULL_FRM', True)
    FIRE_FULL_CD = getattr(C, 'FIRE_FULL_CD', 15)
    OUT_VIDEO = getattr(C, 'OUT_VIDEO', 'tracked_output.mp4')
    OUT_EVENTS = getattr(C, 'OUT_EVENTS', 'events.json')
    OUT_CSV = getattr(C, 'OUT_CSV', 'events.csv')
    OUT_REPORT = getattr(C, 'OUT_REPORT', 'collision_report.txt')
    OUT_SUMMARY = getattr(C, 'OUT_SUMMARY', 'summary.json')
    BOTSORT_YAML = getattr(C, 'BOTSORT_YAML', 'botsort.yaml')
    HCI_ENABLED = getattr(C, 'HCI_ENABLED', False)
    HCI_TRACKER_MODEL = getattr(C, 'HCI_TRACKER_MODEL', os.path.join(BASE_DIR, 'yolov10x.pt'))
    HCI_EVENT_MODEL = getattr(C, 'HCI_EVENT_MODEL', os.path.join(BASE_DIR, 'best.pt'))
    HCI_EVENT_PERIOD = getattr(C, 'HCI_EVENT_PERIOD', 5)
    HCI_TRIGGER_BUFFER = getattr(C, 'HCI_TRIGGER_BUFFER', 15)
    MAX_FRAMES = getattr(C, 'MAX_FRAMES', -1)
    AUTO_DL = getattr(C, 'AUTO_DL', False)
    MAX_TRACK_LOG = getattr(C, 'MAX_TRACK_LOG', 3000)
    USE_ONNX = getattr(C, 'USE_ONNX', False)
    ONNX_SIMPLIFY = getattr(C, 'ONNX_SIMPLIFY', True)
    ONNX_INT8 = getattr(C, 'ONNX_INT8', False)

def make_botsort(cfg: Config) -> str:
    p = Path(cfg.BOTSORT_YAML)
    yaml_content = (
        f"tracker_type: botsort\n"
        f"track_high_thresh: {cfg.CONF}\n"
        f"track_low_thresh: 0.10\n"
        f"new_track_thresh: 0.30\n"
        f"track_buffer: {cfg.TRACK_BUFFER}\n"
        f"match_thresh: {cfg.MATCH_THRESH}\n"
        f"fuse_score: true\n"
        f"with_reid: false\n"
        f"model: mobile\n"
        f"proximity_thresh: 0.5\n"
        f"appearance_thresh: 0.25\n"
        f"gmc_method: none\n"
    )
    p.write_text(yaml_content, encoding="utf-8")
    return str(p)