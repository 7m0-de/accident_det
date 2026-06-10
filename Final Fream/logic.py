from __future__ import annotations
import subprocess, sys

def _ensure_deps():
    pkgs = []
    try:
        import cv2
    except ImportError:
        pkgs.append('opencv-python-headless')
    try:
        import ultralytics
    except ImportError:
        pkgs.append('ultralytics')
    try:
        import sklearn
    except ImportError:
        pkgs.append('scikit-learn')
    try:
        import lap
    except ImportError:
        pkgs.append('lap')
    if pkgs:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-U', *pkgs])
_ensure_deps()
import csv
import json
import math
import os
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import config as C
try:
    from google.colab import files as _gfiles
    IN_COLAB = True
except ImportError:
    _gfiles = None
    IN_COLAB = False

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
    MODEL = getattr(C, 'MODEL', 'best.pt')
    FALLBACKS = getattr(C, 'FALLBACKS', ['yolov10x.pt', 'yolov8n.pt'])
    CONF = getattr(C, 'CONF', 0.45)
    IOU = getattr(C, 'IOU_NMS', 0.5)
    INPUT_SIZE = getattr(C, 'INPUT_SIZE', 1280)
    CLASSES = getattr(C, 'CLASSES', [0, 2, 3, 5, 7])
    CLASS_REMAP = getattr(C, 'CLASS_REMAP', {0: 'person', 2: 'vehicle', 3: 'motorcycle', 5: 'vehicle', 7: 'vehicle'})
    NAMES = getattr(C, 'NAMES', {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'})
    COLORS = getattr(C, 'COLORS', {'vehicle': (0, 220, 0), 'motorcycle': (0, 220, 220), 'person': (255, 180, 0)})
    LOCK_AFTER_SECONDS = getattr(C, 'LOCK_SECS', 5.0)
    MIN_W_RATIO = getattr(C, 'MIN_W_RATIO', 0.02)
    MIN_H_RATIO = getattr(C, 'MIN_H_RATIO', 0.015)
    MAX_AREA_R = getattr(C, 'MAX_AREA_R', 0.45)
    TRACK_BUFFER = getattr(C, 'TRACK_BUFFER', 120)
    MATCH_THRESH = getattr(C, 'MATCH_THRESH', 0.85)
    HIST_SECS = getattr(C, 'HIST_SECS', 3.0)
    USE_GMC = getattr(C, 'USE_GMC', True)
    COLL_REL_SPD = getattr(C, 'COLL_REL_SPD', 3.0)
    SUDDEN_DEC = getattr(C, 'SUDDEN_DEC', -6.0)
    MIN_DEC_SPD = getattr(C, 'MIN_DEC_SPD', 7.0)
    SHARP_TURN = getattr(C, 'SHARP_TURN', 70.0)
    MIN_TURN_SPD = getattr(C, 'MIN_TURN_SPD', 5.0)
    TTC_HIGH = getattr(C, 'TTC_HIGH', 0.6)
    TTC_MED = getattr(C, 'TTC_MED', 1.8)
    CONVOY_DEG = getattr(C, 'CONVOY_DEG', 35.0)
    CONVOY_SPD = getattr(C, 'CONVOY_SPD', 2.5)
    CONFIRM_FR = getattr(C, 'CONFIRM_FR', 4)
    POST_BURNED = getattr(C, 'POST_BURNED', 4.0)
    MAHAL_THRESH = getattr(C, 'MAHAL_THRESH', 4.0)
    BAYES_THRESH = getattr(C, 'BAYES_THRESH', 0.74)
    LONG_STOP_S = getattr(C, 'LONG_STOP_S', 7.0)
    CLUSTER_MIN = getattr(C, 'CLUSTER_MIN', 4)
    EVENT_CD = getattr(C, 'EVENT_CD', 90)
    FIRE_ENABLED = getattr(C, 'FIRE_ENABLED', True)
    FIRE_MODEL = getattr(C, 'FIRE_MODEL', 'keremberke/yolov8n-fire-detection')
    FIRE_CONF = getattr(C, 'FIRE_CONF', 0.45)
    FIRE_CONFIRM = getattr(C, 'FIRE_CONFIRM', 3)
    SMOKE_CONFIRM = getattr(C, 'SMOKE_CONFIRM', 4)
    FIRE_PAD_Y = getattr(C, 'FIRE_PAD_Y', 0.6)
    FIRE_PAD_X = getattr(C, 'FIRE_PAD_X', 0.2)
    FIRE_FULL_FRM = getattr(C, 'FIRE_FULL_FRM', True)
    FIRE_FULL_CD = getattr(C, 'FIRE_FULL_CD', 15)
    OUT_VIDEO = getattr(C, 'OUT_VIDEO', 'tracked_output.mp4')
    OUT_EVENTS = getattr(C, 'OUT_EVENTS', 'events.json')
    OUT_CSV = getattr(C, 'OUT_CSV', 'events.csv')
    OUT_REPORT = getattr(C, 'OUT_REPORT', 'collision_report.txt')
    OUT_SUMMARY = getattr(C, 'OUT_SUMMARY', 'summary.json')
    BOTSORT_YAML = getattr(C, 'BOTSORT_YAML', 'botsort.yaml')
    HCI_ENABLED = getattr(C, 'HCI_ENABLED', True)
    HCI_TRACKER_MODEL = getattr(C, 'HCI_TRACKER_MODEL', 'yolov10x.pt')
    HCI_EVENT_MODEL = getattr(C, 'HCI_EVENT_MODEL', 'best.pt')
    HCI_EVENT_PERIOD = getattr(C, 'HCI_EVENT_PERIOD', 5)
    HCI_TRIGGER_BUFFER = getattr(C, 'HCI_TRIGGER_BUFFER', 15)
    MAX_FRAMES = getattr(C, 'MAX_FRAMES', -1)
    AUTO_DL = getattr(C, 'AUTO_DL', False)
    MAX_TRACK_LOG = getattr(C, 'MAX_TRACK_LOG', 3000)

def make_botsort(cfg: Config) -> str:
    p = Path(cfg.BOTSORT_YAML)
    yaml_content = f"tracker_type: botsort\ntrack_high_thresh: {cfg.CONF}\ntrack_low_thresh: 0.10\nnew_track_thresh: {cfg.CONF + 0.05:.2f}\ntrack_buffer: {cfg.TRACK_BUFFER}\nmatch_thresh: {cfg.MATCH_THRESH}\nfuse_score: true\nwith_reid: false\nproximity_thresh: 0.5\nappearance_thresh: 0.25\ngmc_method: none"
    p.write_text(yaml_content, encoding="utf-8")
    return str(p)

class TimeLocker:

    def __init__(self, cfg: Config, fps: float):
        self.lock_frames = max(1, int(fps * cfg.LOCK_AFTER_SECONDS))
        self._first: Dict[int, int] = {}
        self._last: Dict[int, int] = {}
        self._counts: Dict[int, Dict[int, int]] = {}
        self._locked: Dict[int, int] = {}

    def get(self, tid: int, cls: int, frame_idx: int) -> Tuple[int, bool]:
        self._last[tid] = frame_idx
        if tid in self._locked:
            return (self._locked[tid], True)
        if tid not in self._first:
            self._first[tid] = frame_idx
            self._counts[tid] = defaultdict(int)
        self._counts[tid][cls] += 1
        if frame_idx - self._first[tid] >= self.lock_frames:
            best = max(self._counts[tid], key=self._counts[tid].get)
            self._locked[tid] = best
            self._counts.pop(tid, None)
            return (best, True)
        best = max(self._counts[tid], key=self._counts[tid].get)
        return (best, False)

    def cleanup(self, frame_idx: int, max_age: int=300):
        old = [t for (t, f) in self._last.items() if frame_idx - f > max_age]
        for t in old:
            self._first.pop(t, None)
            self._last.pop(t, None)
            self._counts.pop(t, None)
            self._locked.pop(t, None)

class KinematicTrigger:

    def __init__(self, fps: float, cfg: Config):
        self.fps = max(1.0, fps)
        self.cfg = cfg
        self._history = defaultdict(lambda : deque(maxlen=5))
        self.sudden_dec = cfg.SUDDEN_DEC
        self.sharp_turn = cfg.SHARP_TURN

    def update_and_check(self, frame_idx: int, boxes) -> bool:
        if boxes is None or boxes.id is None:
            return False
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        trigger = False
        for (i, tid) in enumerate(ids):
            (x1, y1, x2, y2) = xyxy[i]
            (cx, cy) = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            hist = self._history[tid]
            if len(hist) >= 2:
                (prev_f, prev_cx, prev_cy) = hist[-1]
                df = max(1, frame_idx - prev_f)
                vx = (cx - prev_cx) / df
                vy = (cy - prev_cy) / df
                spd = math.sqrt(vx * vx + vy * vy)
                if len(hist) >= 3:
                    (p_f, p_cx, p_cy) = hist[-2]
                    pdf = max(1, prev_f - p_f)
                    p_vx = (prev_cx - p_cx) / pdf
                    p_vy = (prev_cy - p_cy) / pdf
                    p_spd = math.sqrt(p_vx * p_vx + p_vy * p_vy)
                    accel = (spd - p_spd) / df
                    if accel <= self.sudden_dec:
                        trigger = True
                        print(f'[HCI TRIGGER] Vehicle #{tid}decelerated sharply:{accel:.2f}px/f^2')
                    if spd > 2.0 and p_spd > 2.0:
                        dot = vx * p_vx + vy * p_vy
                        cos_val = dot / (spd * p_spd)
                        cos_val = max(-1.0, min(1.0, cos_val))
                        angle = math.degrees(math.acos(cos_val))
                        if angle >= self.sharp_turn:
                            trigger = True
                            print(f'[HCI TRIGGER] Vehicle #{tid}turned sharply:{angle:.1f}°')
            hist.append((frame_idx, cx, cy))
        return trigger

    def cleanup(self, active_ids):
        for t in list(self._history):
            if t not in active_ids:
                self._history.pop(t, None)

class DLDetectorPersistence:

    def __init__(self, max_age: int=15):
        self.active_dets = []
        self.max_age = max_age

    def update(self, new_dets: Optional[List[dict]], vehicles: List[dict]) -> List[dict]:
        vehicles_dict = {v['id']: v for v in vehicles}
        if new_dets is not None:
            updated_active = []
            for nd in new_dets:
                best_iou = 0.0
                best_v = None
                for v in vehicles:
                    iou_val = _iou(v['bbox'], nd['bbox'])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_v = v
                assoc_id = best_v['id'] if best_v is not None and best_iou >= 0.3 else None
                updated_active.append({'cls_id': nd['cls_id'], 'cls_name': nd['cls_name'], 'bbox': nd['bbox'], 'center': nd['center'], 'conf': nd['conf'], 'level': nd['level'], 'assoc_id': assoc_id, 'age': 0})
            self.active_dets = updated_active
        else:
            for ad in self.active_dets:
                ad['age'] += 1
                if ad['assoc_id'] is not None and ad['assoc_id'] in vehicles_dict:
                    veh = vehicles_dict[ad['assoc_id']]
                    ad['bbox'] = veh['bbox']
                    ad['center'] = veh['center']
            self.active_dets = [ad for ad in self.active_dets if ad['age'] < self.max_age or ad['assoc_id'] in vehicles_dict]
        return self.active_dets

class GlobalMotionComp:

    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None

    def update(self, gray: np.ndarray) -> np.ndarray:
        identity = np.eye(2, 3, dtype=np.float32)
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_pts = self._good_pts(gray)
            return identity
        curr_pts = self._good_pts(gray)
        if len(curr_pts) < 8 or (self.prev_pts is not None and len(self.prev_pts) < 8):
            self._update_store(gray, curr_pts)
            return identity
        (new_pts, status, _) = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        ok = (status == 1).reshape(-1) if status is not None else np.zeros(len(self.prev_pts), bool)
        if ok.sum() < 8:
            self._update_store(gray, curr_pts)
            return identity
        try:
            (M, _) = cv2.estimateAffinePartial2D(self.prev_pts[ok], new_pts[ok], method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if M is None:
                M = identity
        except Exception:
            M = identity
        self._update_store(gray, curr_pts)
        return M.astype(np.float32)

    def _good_pts(self, gray: np.ndarray, max_corners: int=300) -> np.ndarray:
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=7)
        if corners is None:
            return np.zeros((0, 2), dtype=np.float32)
        return corners.reshape(-1, 2).astype(np.float32)

    def _update_store(self, gray: np.ndarray, pts: np.ndarray):
        self.prev_gray = gray.copy()
        self.prev_pts = pts

class KalmanTracker:

    def __init__(self):
        dt = 1.0
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 1.0
        self.P = np.eye(4) * 10.0
        self.x = np.zeros((4, 1), dtype=float)
        self.ok = False
        self._innovations: Deque[np.ndarray] = deque(maxlen=30)

    def update(self, cx: float, cy: float) -> Tuple[float, float]:
        if not self.ok:
            self.x[0, 0] = cx
            self.x[1, 0] = cy
            self.ok = True
            return (cx, cy)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        z = np.array([[cx], [cy]], dtype=float)
        innov = z - self.H @ self.x
        self._innovations.append(innov.flatten())
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ innov
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return (float(self.x[0, 0]), float(self.x[1, 0]))

    def vel(self) -> Tuple[float, float]:
        return (float(self.x[2, 0]), float(self.x[3, 0]))

    def pred(self, t: int) -> Tuple[float, float]:
        t_val = float(max(0, t))
        cx = float(self.x[0, 0]) + float(self.x[2, 0]) * t_val
        cy = float(self.x[1, 0]) + float(self.x[3, 0]) * t_val
        return (cx, cy)

    def mahalanobis_anomaly(self, cx: float, cy: float) -> float:
        if len(self._innovations) < 5:
            return 0.0
        arr = np.array(list(self._innovations))
        mu = arr.mean(axis=0)
        cov = np.cov(arr.T) + np.eye(2) * 1e-06
        diff = np.array([cx - float(self.x[0, 0]), cy - float(self.x[1, 0])]) - mu
        try:
            d2 = float(diff @ np.linalg.inv(cov) @ diff)
            return math.sqrt(max(0.0, d2))
        except Exception:
            return 0.0
_OF_GRAY_SCALE = 0.5

class OpticalFlow:

    def __init__(self):
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_boxes: Dict[int, Tuple] = {}

    def estimate(self, gray: np.ndarray, vehicles: List[dict], gmc_M: Optional[np.ndarray]=None) -> Dict[int, Optional[Tuple[float, float]]]:
        result: Dict[int, Optional[Tuple[float, float]]] = {}
        (H, W) = gray.shape[:2]
        if self._prev_gray is None:
            self._store(gray, vehicles)
            return {v['id']: None for v in vehicles}
        sh = int(H * _OF_GRAY_SCALE)
        sw = int(W * _OF_GRAY_SCALE)
        curr_small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        (cam_dx, cam_dy) = (0.0, 0.0)
        if gmc_M is not None and Config.USE_GMC:
            try:
                inv = cv2.invertAffineTransform(gmc_M)
                cam_dx = float(inv[0, 2]) * _OF_GRAY_SCALE
                cam_dy = float(inv[1, 2]) * _OF_GRAY_SCALE
            except Exception:
                pass
        for v in vehicles:
            tid = v['id']
            pb = self._prev_boxes.get(tid)
            if pb is None:
                result[tid] = None
                continue
            x1 = max(0, int(pb[0] * _OF_GRAY_SCALE))
            y1 = max(0, int(pb[1] * _OF_GRAY_SCALE))
            x2 = min(sw - 1, int(pb[2] * _OF_GRAY_SCALE))
            y2 = min(sh - 1, int(pb[3] * _OF_GRAY_SCALE))
            if x2 - x1 < 5 or y2 - y1 < 5:
                result[tid] = None
                continue
            pad = 15
            (cx1, cy1) = (max(0, x1 - pad), max(0, y1 - pad))
            (cx2, cy2) = (min(sw, x2 + pad), min(sh, y2 + pad))
            prev_crop = self._prev_gray[cy1:cy2, cx1:cx2]
            curr_crop = curr_small[cy1:cy2, cx1:cx2]
            mask = np.zeros_like(prev_crop, dtype=np.uint8)
            mask[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1] = 255
            pts = cv2.goodFeaturesToTrack(prev_crop, maxCorners=50, qualityLevel=0.2, minDistance=5, mask=mask)
            if pts is None or len(pts) < 3:
                result[tid] = None
                continue
            (new_pts, status, _) = cv2.calcOpticalFlowPyrLK(prev_crop, curr_crop, pts, None)
            if new_pts is None or status is None:
                result[tid] = None
                continue
            ok = status.reshape(-1) == 1
            if ok.sum() < 3:
                result[tid] = None
                continue
            delta = new_pts[ok].reshape(-1, 2) - pts[ok].reshape(-1, 2)
            dx = (float(np.median(delta[:, 0])) - cam_dx) / _OF_GRAY_SCALE
            dy = (float(np.median(delta[:, 1])) - cam_dy) / _OF_GRAY_SCALE
            result[tid] = (dx, dy)
        self._store(gray, vehicles)
        return result

    def _store(self, gray: np.ndarray, vehicles: List[dict]):
        (H, W) = gray.shape[:2]
        sh = int(H * _OF_GRAY_SCALE)
        sw = int(W * _OF_GRAY_SCALE)
        self._prev_gray = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        self._prev_boxes = {v['id']: tuple(v['bbox']) for v in vehicles}

class MotionAnalyzer:

    def __init__(self, fps: float):
        self.fps = max(fps, 1.0)
        self.maxlen = max(5, int(Config.HIST_SECS * self.fps))
        self.hist: Dict[int, Deque[dict]] = defaultdict(lambda : deque(maxlen=self.maxlen))
        self.kal: Dict[int, KalmanTracker] = {}
        self.of = OpticalFlow()

    def update(self, tid: int, cx: float, cy: float, fidx: int, flow: Optional[Tuple[float, float]]=None) -> dict:
        k = self.kal.setdefault(tid, KalmanTracker())
        h = self.hist[tid]
        (mx, my) = (cx, cy)
        if flow and h:
            (fx, fy) = flow
            mx = 0.6 * cx + 0.4 * (float(h[-1]['cx']) + fx)
            my = 0.6 * cy + 0.4 * (float(h[-1]['cy']) + fy)
        mahal = k.mahalanobis_anomaly(mx, my)
        (sx, sy) = k.update(mx, my)
        (vx, vy) = k.vel()
        spd = math.sqrt(vx * vx + vy * vy)
        accel = 0.0
        heading = None
        hdelta = None
        if h:
            prev = h[-1]
            df = max(1, fidx - int(prev['f']))
            accel = (spd - float(prev['spd'])) / df
            if abs(vx) > 1e-06 or abs(vy) > 1e-06:
                heading = math.degrees(math.atan2(vy, vx))
            hdelta = _adiff(prev.get('head'), heading)
        state = {'f': fidx, 'cx': sx, 'cy': sy, 'vx': vx, 'vy': vy, 'spd': spd, 'acc': accel, 'head': heading, 'hdelta': hdelta, 'mahal': mahal}
        h.append(state)
        return state

    def process_frame(self, vehicles: List[dict], gray: np.ndarray, fidx: int, gmc_M: Optional[np.ndarray]=None):
        flows = self.of.estimate(gray, vehicles, gmc_M)
        for v in vehicles:
            st = self.update(v['id'], float(v['center'][0]), float(v['center'][1]), fidx, flows.get(v['id']))
            v.update({'vx': st['vx'], 'vy': st['vy'], 'spd': st['spd'], 'acc': st['acc'], 'heading': st['head'], 'hdelta': st['hdelta'], 'mahal': st['mahal']})

    def max_spd(self, tid: int, secs: float=2.0) -> float:
        h = self.hist.get(tid)
        if not h:
            return 0.0
        cut = h[-1]['f'] - secs * self.fps
        return max((s['spd'] for s in h if s['f'] >= cut), default=0.0)

    def min_acc(self, tid: int, secs: float=1.5) -> float:
        h = self.hist.get(tid)
        if not h:
            return 0.0
        cut = h[-1]['f'] - secs * self.fps
        return min((s['acc'] for s in h if s['f'] >= cut), default=0.0)

    def cleanup(self, active_ids):
        for t in list(self.hist):
            if t not in active_ids:
                self.hist.pop(t)
                self.kal.pop(t, None)

class St(Enum):
    NORMAL = auto()
    APPROACHING = auto()
    CRITICAL = auto()
    COLLIDED = auto()
    POST = auto()
    BURNED = auto()
_LOCKED_STATES = {St.COLLIDED, St.POST, St.BURNED}
_STATE_ORDER = [St.COLLIDED, St.POST, St.BURNED]

class VehState:
    __slots__ = ('tid', 'st', 'partners', 'cf', 'ts', 'post_fr')

    def __init__(self, tid: int):
        self.tid = tid
        self.st = St.NORMAL
        self.partners: List[int] = []
        self.cf = None
        self.ts = None
        self.post_fr = 0

    def go(self, new: St, **kw) -> bool:
        if self.st in _LOCKED_STATES:
            if new not in _STATE_ORDER or self.st not in _STATE_ORDER:
                return False
            if _STATE_ORDER.index(new) <= _STATE_ORDER.index(self.st):
                return False
        self.st = new
        for (k, v) in kw.items():
            setattr(self, k, v)
        return True

@dataclass
class CollRec:
    ids: Tuple[int, int]
    frame: int
    ts: float
    iou: float
    rel_spd: float
    bayes: float
    mahal: float
    ev: dict
    msg: str = field(default='', init=False)

    def __post_init__(self):
        (a, b) = sorted(self.ids)
        self.ids = (a, b)
        self.msg = f'Collision: #{a}& #{b}at t={self.ts:.2f}s'

class Registry:

    def __init__(self):
        self._r: Dict[frozenset, CollRec] = {}
        self.dl_accidents: List[dict] = []

    def add(self, r: CollRec) -> bool:
        k = frozenset(r.ids)
        if k in self._r:
            return False
        self._r[k] = r
        print(f'[COLLISION] frame={r.frame}t={r.ts:.2f}s bayes={r.bayes:.0%}mahal={r.mahal:.1f}{r.msg}')
        return True

    def add_dl_accident(self, frame: int, ts: float, conf: float, bbox: Tuple[int, int, int, int]):
        (cx2, cy2) = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        for old in self.dl_accidents:
            (cx1, cy1) = ((old['bbox'][0] + old['bbox'][2]) / 2.0, (old['bbox'][1] + old['bbox'][3]) / 2.0)
            dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            if (_iou(old['bbox'], bbox) > 0.1 or dist < 250.0) and frame - old['frame'] < 10800:
                old['frame'] = frame
                old['ts'] = ts
                old['conf'] = max(old['conf'], conf)
                return
        self.dl_accidents.append({'frame': frame, 'ts': ts, 'conf': conf, 'bbox': bbox})
        print(f'[DL ACCIDENT] frame={frame}t={ts:.2f}s conf={conf:.0%}registered.')

    def has(self, a: int, b: int) -> bool:
        return frozenset((a, b)) in self._r

    def all(self) -> List[CollRec]:
        return list(self._r.values())

    def persistent(self, fidx: int, ts: float, states: Dict[int, VehState]) -> List[dict]:
        out = []
        for r in self._r.values():
            pst = {str(t): states[t].st.name for t in r.ids if t in states}
            out.append({'level': 'HIGH', 'type': 'COLLISION', 'ids': list(r.ids), 'score': int(r.bayes * 100), 'message': r.msg, 'color': (0, 0, 255), 'frame': fidx, 'ts': round(ts, 3), 'persistent': True, 'ev': {**r.ev, 'pst': pst}})
        return out

    def save_report(self, path: str, states: Dict[int, VehState], gen_time: str):
        lines = ['COLLISION & ACCIDENT REPORT', f'Generated:{gen_time}', '', f'Total confirmed collisions (Physics-based):{len(self._r)}', f'Total confirmed accidents (Deep Learning):{len(self.dl_accidents)}', f'Grand Total of Confirmed Incidents:{len(self._r) + len(self.dl_accidents)}', '', '=========================================', '--- PHYSICAL COLLISIONS (KINETIC) ---', '=========================================']
        for (i, r) in enumerate(self.all(), 1):
            (a, b) = r.ids
            pst = {t: states[t].st.name for t in r.ids if t in states}
            lines += [f'#{i}CONFIRMED COLLISION', f'Vehicles   : #{a}<-> #{b}', f'Time       :{r.ts:.2f}s  (frame{r.frame})', f'IoU        :{r.iou:.4f}', f'Rel speed  :{r.rel_spd:.2f}px/f', f'Bayes prob :{r.bayes:.0%}', f'Mahalanobis:{r.mahal:.2f}', f'Post-state :{pst}', '']
        lines += ['=========================================', '--- DEEP LEARNING DETECTED ACCIDENTS ---', '=========================================']
        for (i, r) in enumerate(self.dl_accidents, 1):
            lines += [f'#{i}DL ACCIDENT ALERT', f"Time       :{r['ts']:.2f}s  (frame{r['frame']})", f"Confidence :{r['conf']:.0%}", f"BBox (xyxy):{r['bbox']}", '']
        Path(path).write_text(''.join(lines), encoding='utf-8')

class BayesianFusion:
    PRIOR_COLL = 0.015
    LIKELIHOODS = {'iou_high': (0.88, 0.04), 'iou_med': (0.68, 0.12), 'contact': (0.82, 0.18), 'ttc_critical': (0.87, 0.07), 'ttc_warning': (0.62, 0.22), 'rel_spd_high': (0.77, 0.12), 'sudden_decel': (0.72, 0.1), 'post_sig': (0.9, 0.03), 'impact_sig': (0.86, 0.05), 'mahal_high': (0.7, 0.14), 'convoy': (0.06, 0.68)}

    @classmethod
    def compute(cls, flags: Dict[str, bool]) -> float:
        log_odds = math.log(cls.PRIOR_COLL / (1.0 - cls.PRIOR_COLL))
        for (key, active) in flags.items():
            if key not in cls.LIKELIHOODS:
                continue
            (pc, pn) = cls.LIKELIHOODS[key]
            lr = pc / max(pn, 1e-09) if active else (1.0 - pc) / max(1.0 - pn, 1e-09)
            log_odds += math.log(max(lr, 1e-09))
        return float(1.0 / (1.0 + math.exp(-log_odds)))

class KinematicModel:

    @staticmethod
    def ttc_kinematic(v1: dict, v2: dict, fps: float, mot: MotionAnalyzer) -> Optional[dict]:
        (x1, y1) = v1['center']
        (x2, y2) = v2['center']
        (vx1, vy1) = mot.kal[v1['id']].vel() if v1['id'] in mot.kal else (0.0, 0.0)
        (vx2, vy2) = mot.kal[v2['id']].vel() if v2['id'] in mot.kal else (0.0, 0.0)
        (rx, ry) = (x2 - x1, y2 - y1)
        (rvx, rvy) = (vx2 - vx1, vy2 - vy1)
        rv2 = rvx * rvx + rvy * rvy
        if rv2 < 1e-06:
            return None
        dist = math.sqrt(rx * rx + ry * ry)
        closing = -(rx * rvx + ry * rvy) / max(dist, 1e-06)
        if closing < 2.0:
            return None
        t_f = -(rx * rvx + ry * rvy) / rv2
        if t_f <= 0:
            return None
        fdx = rx + rvx * t_f
        fdy = ry + rvy * t_f
        fdist = math.sqrt(fdx * fdx + fdy * fdy)
        thresh = max(80.0, 0.28 * (_diag(v1) + _diag(v2)))
        if fdist > thresh * 1.3:
            return None
        return {'secs': t_f / fps, 'fdist': fdist, 'thresh': thresh, 'closing': closing}

    @staticmethod
    def ttc_predictive(v1: dict, v2: dict, fps: float, mot: MotionAnalyzer) -> Optional[float]:
        k1 = mot.kal.get(v1['id'])
        k2 = mot.kal.get(v2['id'])
        if not k1 or not k2:
            return None
        thresh = 0.28 * (_diag(v1) + _diag(v2))
        look = int(fps * 2.5)
        for t in range(1, look + 1):
            p1 = np.array(k1.pred(t))
            p2 = np.array(k2.pred(t))
            if np.linalg.norm(p1 - p2) <= thresh:
                return t / fps
        return None

    @staticmethod
    def approach_angle(v1: dict, v2: dict) -> Optional[float]:
        (h1, h2) = (v1.get('heading'), v2.get('heading'))
        if h1 is None or h2 is None:
            return None
        d1x = math.cos(math.radians(h1))
        d1y = math.sin(math.radians(h1))
        d2x = math.cos(math.radians(h2))
        d2y = math.sin(math.radians(h2))
        rx = v2['center'][0] - v1['center'][0]
        ry = v2['center'][1] - v1['center'][1]
        r = math.sqrt(rx * rx + ry * ry)
        if r < 1e-06:
            return None
        rx /= r
        ry /= r
        return float(d1x * rx + d1y * ry - (d2x * rx + d2y * ry))

class RiskEngine:

    def __init__(self, fps: float, mot: MotionAnalyzer, W: int, H: int):
        self.fps = max(fps, 1.0)
        self.mot = mot
        self.reg = Registry()
        self.states: Dict[int, VehState] = {}
        self._cool: Dict[str, int] = {}
        self._stop: Dict[int, int] = {}
        self._pair: Dict[str, dict] = {}
        self._cluster_last = -9999
        self.physics_trigger = False
        scale = (W / 1280.0 + H / 720.0) / 2.0
        ff = fps / 30.0
        self.prox = max(20, int(80 * scale))
        self.lnrby = max(50, int(180 * scale))
        self.ceps = max(40, int(130 * scale))
        self.safe_spd = 1.2 * ff
        self.long_spd = 2.5 * ff
        self.crel_spd = 3.0 * ff
        self.sud_dec = -6.0 * ff
        self.min_dec_spd = 7.0 * ff
        self.long_fr = max(1, int(fps * Config.LONG_STOP_S))
        self.burned_fr = max(1, int(fps * Config.POST_BURNED))
        self.conf_fr = Config.CONFIRM_FR
        self.evt_cd = max(1, int(fps * 4))
        self.clus_cd = max(1, int(fps * 3))
        self.static_obs = self._build_static_obs(W, H)
        self.dl_dets: List[dict] = []

    def _build_static_obs(self, W: int, H: int) -> List[dict]:
        m = max(20, int(0.05 * W))
        return [{'id': -1, 'name': 'left_wall', 'bbox': (0, 0, m, H)}, {'id': -2, 'name': 'right_wall', 'bbox': (W - m, 0, W, H)}, {'id': -3, 'name': 'top_wall', 'bbox': (0, 0, W, m)}, {'id': -4, 'name': 'bottom_wall', 'bbox': (0, H - m, W, H)}]

    def _st(self, tid: int) -> VehState:
        return self.states.setdefault(tid, VehState(tid))

    def _cooled(self, k: str, f: int) -> bool:
        if f - self._cool.get(k, -9999) >= self.evt_cd:
            self._cool[k] = f
            return True
        return False

    def _analyze_pair(self, v1: dict, v2: dict, fidx: int) -> Optional[dict]:
        pk = f"{min(v1['id'], v2['id'])}_{max(v1['id'], v2['id'])}"
        iou = _iou(v1['bbox'], v2['bbox'])
        dist = _dist(v1['center'], v2['center'])
        gap = _gap(v1['bbox'], v2['bbox'])
        glim = min(45.0, max(12.0, 0.05 * (_diag(v1) + _diag(v2))))
        contact = iou >= 0.3 or gap <= glim
        (vx1, vy1) = (v1.get('vx', 0.0), v1.get('vy', 0.0))
        (vx2, vy2) = (v2.get('vx', 0.0), v2.get('vy', 0.0))
        rel_spd = math.sqrt((vx1 - vx2) ** 2 + (vy1 - vy2) ** 2)
        hdiff = _adiff(v1.get('heading'), v2.get('heading'))
        convoy = hdiff is not None and hdiff <= Config.CONVOY_DEG and (rel_spd <= Config.CONVOY_SPD * self.fps / 30.0)
        ttc_k = KinematicModel.ttc_kinematic(v1, v2, self.fps, self.mot)
        ttc_p = KinematicModel.ttc_predictive(v1, v2, self.fps, self.mot)
        ttc_s: Optional[float] = None
        if ttc_k:
            ttc_s = ttc_k['secs']
        if ttc_p:
            ttc_s = min(ttc_s, ttc_p) if ttc_s else ttc_p
        approach = KinematicModel.approach_angle(v1, v2)
        mahal_max = max(v1.get('mahal', 0.0), v2.get('mahal', 0.0))
        _PAIR_RESET_GAP = 30
        existing = self._pair.get(pk)
        if existing is None or fidx - existing['lf'] > _PAIR_RESET_GAP:
            self._pair[pk] = {'cf': 0, 'scf': 0, 'lf': fidx, 'maxspd': 0.0, 'minacc': 0.0}
        ps = self._pair[pk]
        consec = fidx - ps['lf'] <= 2
        ps['cf'] = (ps['cf'] + 1 if consec else 1) if contact else 0
        both_slow = v1.get('spd', 0.0) <= self.safe_spd and v2.get('spd', 0.0) <= self.safe_spd
        ps['scf'] = (ps['scf'] + 1 if consec else 1) if contact and both_slow else 0
        ps['lf'] = fidx
        ps['maxspd'] = max(self.mot.max_spd(v1['id']), self.mot.max_spd(v2['id']))
        ps['minacc'] = min(self.mot.min_acc(v1['id']), self.mot.min_acc(v2['id']))
        is_id_switch = iou >= 0.7
        real_contact = iou > 0.05 or gap < 8.0
        decel_or_anomaly = v1.get('acc', 0.0) <= self.sud_dec or v2.get('acc', 0.0) <= self.sud_dec or ps['minacc'] <= self.sud_dec or (mahal_max >= Config.MAHAL_THRESH + 1.0 and rel_spd >= self.crel_spd)
        impact_sig = real_contact and (not convoy) and (not is_id_switch) and decel_or_anomaly
        post_sig = ps['scf'] >= self.conf_fr and ps['maxspd'] >= 5.0 and contact and (not is_id_switch) and (ps['minacc'] <= self.sud_dec or mahal_max >= Config.MAHAL_THRESH or ps['maxspd'] >= 10.0)
        flags = {'iou_high': iou >= 0.45, 'iou_med': iou >= 0.3, 'contact': contact, 'ttc_critical': ttc_s is not None and ttc_s <= Config.TTC_HIGH, 'ttc_warning': ttc_s is not None and ttc_s <= Config.TTC_MED, 'rel_spd_high': rel_spd >= self.crel_spd, 'sudden_decel': v1.get('acc', 0.0) <= self.sud_dec or v2.get('acc', 0.0) <= self.sud_dec, 'post_sig': post_sig, 'impact_sig': impact_sig, 'mahal_high': mahal_max >= Config.MAHAL_THRESH, 'convoy': convoy}
        bayes_physics = BayesianFusion.compute(flags)
        if bayes_physics >= 0.35 and (not convoy):
            self.physics_trigger = True
        model_conf = 0.0
        if hasattr(self, 'dl_dets') and self.dl_dets:
            for d in self.dl_dets:
                if d['cls_name'] == 'accident':
                    iou1 = _iou(v1['bbox'], d['bbox'])
                    iou2 = _iou(v2['bbox'], d['bbox'])
                    gap1 = _gap(v1['bbox'], d['bbox'])
                    gap2 = _gap(v2['bbox'], d['bbox'])
                    dist1 = _dist(v1['center'], d['center'])
                    dist2 = _dist(v2['center'], d['center'])
                    if iou1 > 0.0 or iou2 > 0.0 or gap1 < 50.0 or (gap2 < 50.0) or (dist1 < 150.0) or (dist2 < 150.0):
                        model_conf = max(model_conf, d['conf'])
        bayes = 0.7 * model_conf + 0.3 * bayes_physics
        ev = {'iou': round(iou, 4), 'dist': round(dist, 2), 'gap': round(gap, 2), 'contact': bool(contact), 'rel_spd': round(rel_spd, 3), 'hdiff': hdiff, 'convoy': bool(convoy), 'both_slow': bool(both_slow), 'approach': approach, 'ttc_secs': ttc_s, 'ttc_high': bool(ttc_s is not None and ttc_s <= Config.TTC_HIGH), 'ttc_med': bool(ttc_s is not None and ttc_s <= Config.TTC_MED), 'cf': ps['cf'], 'scf': ps['scf'], 'maxspd': round(ps['maxspd'], 3), 'minacc': round(ps['minacc'], 3), 'impact_sig': bool(impact_sig), 'post_sig': bool(post_sig), 'mahal_max': round(mahal_max, 3), 'bayes': round(bayes, 4)}
        s1 = self._st(v1['id'])
        s2 = self._st(v2['id'])
        confirmed = (post_sig or impact_sig) and (not is_id_switch)
        if confirmed and bayes >= Config.BAYES_THRESH:
            s1.go(St.CRITICAL)
            s2.go(St.CRITICAL)
            r = CollRec(ids=(v1['id'], v2['id']), frame=fidx, ts=round(fidx / self.fps, 3), iou=iou, rel_spd=rel_spd, bayes=bayes, mahal=mahal_max, ev=_jclean(ev))
            if self.reg.add(r):
                for (s, oid) in ((s1, v2['id']), (s2, v1['id'])):
                    s.go(St.COLLIDED, partners=[oid], cf=fidx, ts=fidx / self.fps)
                    s.go(St.POST)
                return {'level': 'HIGH', 'type': 'COLLISION', 'ids': [v1['id'], v2['id']], 'score': int(bayes * 100), 'message': r.msg, 'color': (0, 0, 255), 'ev': ev}
            elif self.reg.has(v1['id'], v2['id']):
                return None
        if bayes >= 0.35 and (not convoy):
            ns = St.CRITICAL if ev['ttc_high'] else St.APPROACHING
            s1.go(ns)
            s2.go(ns)
            k = f'wp_{pk}'
            if not self._cooled(k, fidx):
                return None
            wt = 'TTC_WARNING' if ev['ttc_high'] else 'CONTACT_WARNING' if contact else 'PROXIMITY_WARNING'
            return {'level': 'MEDIUM', 'type': wt, 'ids': [v1['id'], v2['id']], 'score': int(bayes * 100), 'message': f"WARNING({wt}): #{v1['id']}+#{v2['id']}bayes={bayes:.0%}", 'color': (0, 165, 255), 'ev': ev}
        return None

    def _analyze_single(self, v: dict, vehicles: List[dict], fidx: int) -> Optional[dict]:
        tid = v['id']
        if v.get('acc', 0.0) <= self.sud_dec and v.get('spd', 0.0) >= self.min_dec_spd:
            if self._cooled(f'dec_{tid}', fidx):
                return {'level': 'MEDIUM', 'type': 'SUDDEN_DECEL', 'ids': [tid], 'score': 55, 'message': f'Sudden decel: #{tid}', 'color': (0, 165, 255)}
        if v.get('hdelta') is not None and v['hdelta'] >= Config.SHARP_TURN and (v.get('spd', 0.0) >= Config.MIN_TURN_SPD):
            if self._cooled(f'trn_{tid}', fidx):
                return {'level': 'LOW', 'type': 'SHARP_TURN', 'ids': [tid], 'score': 30, 'message': f'Sharp turn: #{tid}', 'color': (255, 180, 0)}
        if v.get('spd', 0.0) < self.safe_spd:
            self._stop.setdefault(tid, fidx)
        else:
            self._stop.pop(tid, None)
        stopped_fr = fidx - self._stop.get(tid, fidx)
        if stopped_fr >= self.long_fr:
            nearby = any((o['id'] != tid and o.get('spd', 0.0) >= self.long_spd and (_dist(v['center'], o['center']) <= self.lnrby) for o in vehicles))
            if nearby and self._cooled(f'ls_{tid}', fidx):
                return {'level': 'HIGH', 'type': 'LONG_STOP', 'ids': [tid], 'score': 75, 'message': f'Long stop in traffic: #{tid}', 'color': (0, 0, 200)}
        return None

    def _post_updates(self, vehicles: List[dict], fidx: int) -> List[dict]:
        vmap = {v['id']: v for v in vehicles}
        evs = []
        for r in self.reg.all():
            for tid in r.ids:
                s = self._st(tid)
                veh = vmap.get(tid)
                if s.st == St.COLLIDED:
                    s.go(St.POST)
                if s.st == St.POST and veh:
                    if veh.get('spd', 0.0) <= 0.5:
                        s.post_fr += 1
                    else:
                        s.post_fr = 0
                    if s.post_fr >= self.burned_fr and s.go(St.BURNED):
                        evs.append({'level': 'HIGH', 'type': 'BURNED', 'ids': [tid], 'score': 90, 'message': f'#{tid}stationary post-collision', 'color': (0, 0, 180)})
        return evs

    def _cluster(self, vehicles: List[dict], fidx: int) -> List[dict]:
        return []
        if fidx - self._cluster_last < self.clus_cd:
            return []
        if len(vehicles) < Config.CLUSTER_MIN:
            return []
        coords = np.array([v['center'] for v in vehicles])
        labels = DBSCAN(eps=self.ceps, min_samples=Config.CLUSTER_MIN).fit(coords).labels_
        (best_label, best_size) = (-1, 0)
        for l in set(labels):
            if l == -1:
                continue
            s = int(np.sum(labels == l))
            if s > best_size:
                (best_size, best_label) = (s, l)
        if best_label == -1 or best_size < Config.CLUSTER_MIN:
            return []
        bv = [vehicles[i] for i in np.where(labels == best_label)[0]]
        cx = sum((v['center'][0] for v in bv)) / len(bv)
        cy = sum((v['center'][1] for v in bv)) / len(bv)
        self._cluster_last = fidx
        return [{'level': 'LOW', 'type': 'TRAFFIC_CLUSTER', 'ids': [v['id'] for v in bv], 'score': 25, 'message': f'Traffic cluster:{len(bv)}vehicles', 'color': (255, 180, 0), 'center': (cx, cy)}]

    def analyze(self, vehicles: List[dict], fidx: int) -> List[dict]:
        ts = fidx / self.fps
        evs = self.reg.persistent(fidx, ts, self.states)
        evs += self._post_updates(vehicles, fidx)
        for v in vehicles:
            self._st(v['id'])
            e = self._analyze_single(v, vehicles, fidx)
            if e:
                evs.append(e)
        active_pk: set = set()
        n = len(vehicles)
        if n > 1:
            centers = np.array([v['center'] for v in vehicles])
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            (i_idx, j_idx) = np.where((dists <= 300) & np.triu(np.ones((n, n)), k=1).astype(bool))
            for (i, j) in zip(i_idx, j_idx):
                pk = f"{min(vehicles[i]['id'], vehicles[j]['id'])}_{max(vehicles[i]['id'], vehicles[j]['id'])}"
                active_pk.add(pk)
                e = self._analyze_pair(vehicles[i], vehicles[j], fidx)
                if e:
                    evs.append(e)
        for k in list(self._pair):
            if k not in active_pk and fidx - self._pair[k]['lf'] > 120:
                self._pair.pop(k)
        evs += self._cluster(vehicles, fidx)
        pri = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        evs.sort(key=lambda e: (pri.get(e['level'], 9), -e.get('score', 0)))
        return evs

def process_detections(result_tracker, frame_idx: int, locker: TimeLocker, cfg: Config, fw: int, fh: int, fps: float, result_event=None) -> Tuple[List[dict], List[dict], List[dict]]:
    vehicles: List[dict] = []
    dl_detections: List[dict] = []
    dl_events: List[dict] = []
    if result_tracker is not None and result_tracker.boxes is not None:
        xyxy = result_tracker.boxes.xyxy
        confs = result_tracker.boxes.conf
        clss = result_tracker.boxes.cls
        ids = result_tracker.boxes.id
        xyxy_np = xyxy.cpu().numpy()
        confs_np = confs.cpu().numpy()
        clss_np = clss.cpu().numpy().astype(int)
        ids_np = ids.cpu().numpy().astype(int) if ids is not None else None
        min_w = fw * cfg.MIN_W_RATIO
        min_h = fh * cfg.MIN_H_RATIO
        max_area = fw * fh * cfg.MAX_AREA_R
        vehicle_classes = [2, 3, 5, 7] if cfg.HCI_ENABLED else [2, 3]
        COCO_TO_INTERNAL = {2: 3, 3: 2, 5: 3, 7: 3}
        for i in range(len(clss_np)):
            box = xyxy_np[i]
            cls = clss_np[i]
            conf = confs_np[i]
            if cls not in vehicle_classes:
                if not cfg.HCI_ENABLED and cls in (0, 1):
                    (x1, y1, x2, y2) = map(int, box)
                    name = 'accident' if cls == 0 else 'fire'
                    is_confirmed = cls == 0 and conf >= 0.75 or (cls == 1 and conf >= 0.6)
                    is_potential = cls == 0 and conf >= 0.35 or (cls == 1 and conf >= 0.3)
                    if not is_potential:
                        continue
                    level = 'HIGH' if is_confirmed else 'MEDIUM'
                    color = (0, 0, 255) if cls == 0 else (0, 60, 255)
                    if not is_confirmed:
                        color = (0, 255, 255)
                    dl_detections.append({'cls_id': cls, 'cls_name': name, 'bbox': (x1, y1, x2, y2), 'center': ((x1 + x2) / 2.0, (y1 + y2) / 2.0), 'conf': float(conf), 'level': level})
                    dl_events.append({'level': level, 'type': 'ACCIDENT_DL' if cls == 0 else 'FIRE_DL', 'message': f"Detected{('Accident' if cls == 0 else 'Fire')}(Deep Learning) with confidence{conf:.0%}({('CONFIRMED' if is_confirmed else 'POTENTIAL')})", 'color': color, 'frame': frame_idx, 'ts': round(frame_idx / max(fps, 1.0), 3), 'bbox': (x1, y1, x2, y2)})
                continue
            (x1, y1, x2, y2) = map(int, box)
            w_box = x2 - x1
            h_box = y2 - y1
            area_box = w_box * h_box
            if w_box < min_w or h_box < min_h or area_box > max_area:
                continue
            if ids_np is None or ids_np[i] < 0:
                continue
            tid = ids_np[i]
            cls_mapped = COCO_TO_INTERNAL.get(cls, cls) if cfg.HCI_ENABLED else cls
            (stable_cls, locked) = locker.get(tid, cls_mapped, frame_idx)
            name = cfg.CLASS_REMAP.get(stable_cls, cfg.NAMES.get(stable_cls, 'vehicle'))
            vehicles.append({'id': tid, 'cls_id': stable_cls, 'cls_name': name, 'locked': bool(locked), 'bbox': (x1, y1, x2, y2), 'center': ((x1 + x2) / 2.0, (y1 + y2) / 2.0), 'conf': float(conf), 'w': w_box, 'h': h_box, 'vx': 0.0, 'vy': 0.0, 'spd': 0.0, 'acc': 0.0, 'heading': None, 'hdelta': None, 'mahal': 0.0})
    if cfg.HCI_ENABLED and result_event is not None and (result_event.boxes is not None):
        xyxy = result_event.boxes.xyxy
        confs = result_event.boxes.conf
        clss = result_event.boxes.cls
        xyxy_np = xyxy.cpu().numpy()
        confs_np = confs.cpu().numpy()
        clss_np = clss.cpu().numpy().astype(int)
        for i in range(len(clss_np)):
            cls = clss_np[i]
            conf = confs_np[i]
            if cls not in (0, 1):
                continue
            (x1, y1, x2, y2) = map(int, xyxy_np[i])
            name = 'accident' if cls == 0 else 'fire'
            is_confirmed = cls == 0 and conf >= 0.75 or (cls == 1 and conf >= 0.6)
            is_potential = cls == 0 and conf >= 0.35 or (cls == 1 and conf >= 0.3)
            if not is_potential:
                continue
            level = 'HIGH' if is_confirmed else 'MEDIUM'
            color = (0, 0, 255) if cls == 0 else (0, 60, 255)
            if not is_confirmed:
                color = (0, 255, 255)
            dl_detections.append({'cls_id': cls, 'cls_name': name, 'bbox': (x1, y1, x2, y2), 'center': ((x1 + x2) / 2.0, (y1 + y2) / 2.0), 'conf': float(conf), 'level': level})
            dl_events.append({'level': level, 'type': 'ACCIDENT_DL' if cls == 0 else 'FIRE_DL', 'message': f"Detected{('Accident' if cls == 0 else 'Fire')}(Deep Learning) with confidence{conf:.0%}({('CONFIRMED' if is_confirmed else 'POTENTIAL')})", 'color': color, 'frame': frame_idx, 'ts': round(frame_idx / max(fps, 1.0), 3), 'bbox': (x1, y1, x2, y2)})
    return (vehicles, dl_detections, dl_events)

def draw(frame: np.ndarray, vehicles: List[dict], events: List[dict], states: Dict[int, VehState], reg: Registry, fidx: int, fps: float, dl_detections: List[dict]=None, mot: Optional['MotionAnalyzer']=None) -> np.ndarray:
    (H, W) = frame.shape[:2]
    risk_col: Dict[int, Tuple] = {}
    for r in reg.all():
        for vid in r.ids:
            risk_col[vid] = (0, 0, 255)
    for e in events:
        col = {'HIGH': (0, 0, 255), 'MEDIUM': (0, 255, 255), 'LOW': (255, 180, 0)}.get(e['level'])
        if col:
            for vid in e.get('ids', []):
                if vid not in risk_col or e['level'] == 'HIGH':
                    risk_col[vid] = col
    if mot:
        for v in vehicles:
            tid = v['id']
            h = mot.hist.get(tid)
            if h and len(h) > 1:
                col = risk_col.get(tid, Config.COLORS.get(v['cls_name'], (200, 200, 200)))
                points = [(int(p['cx']), int(p['cy'])) for p in h]
                for idx in range(len(points) - 1):
                    thickness = max(1, int(4 * (idx + 1) / len(points)))
                    cv2.line(frame, points[idx], points[idx + 1], col, thickness, cv2.LINE_AA)
    if dl_detections:
        for d in dl_detections:
            (x1, y1, x2, y2) = d['bbox']
            col = (0, 255, 255) if d.get('level') == 'MEDIUM' else Config.COLORS.get(d['cls_name'], (0, 0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
            label = f"{d['cls_name'].upper()}DL{d['conf']:.0%}({('CONFIRMED' if d.get('level') == 'HIGH' else 'POTENTIAL')})"
            (fs, sc) = (cv2.FONT_HERSHEY_SIMPLEX, 0.46)
            ((tw, th), _) = cv2.getTextSize(label, fs, sc, 1)
            ty = y1 - 4 if y1 - th - 8 >= 0 else y1 + th + 6
            cv2.rectangle(frame, (x1, ty - th - 3), (x1 + tw + 4, ty + 2), col, -1)
            txt_col = (0, 0, 0) if d.get('level') == 'MEDIUM' else (255, 255, 255)
            cv2.putText(frame, label, (x1 + 2, ty), fs, sc, txt_col, 1, cv2.LINE_AA)
    for v in vehicles:
        (x1, y1, x2, y2) = v['bbox']
        tid = v['id']
        col = risk_col.get(tid, Config.COLORS.get(v['cls_name'], (200, 200, 200)))
        thick = 3 if tid in risk_col else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)
        suffix = ''
        if tid in states:
            s = states[tid].st
            if s == St.BURNED:
                suffix = 'BURNED'
            elif s in (St.COLLIDED, St.POST):
                suffix = 'COLLIDED'
            elif s == St.CRITICAL:
                suffix = 'CRITICAL'
        mahal_str = f"M{v.get('mahal', 0.0):.1f}" if v.get('mahal', 0.0) > 2 else ''
        label = f"ID:{tid}{v['cls_name']}{v['conf']:.0%}{v['spd']:.1f}px/f{mahal_str}{suffix}"
        (fs, sc) = (cv2.FONT_HERSHEY_SIMPLEX, 0.46)
        ((tw, th), _) = cv2.getTextSize(label, fs, sc, 1)
        bw = max(1, x2 - x1)
        if tw > bw - 4:
            sc = max(0.28, sc * (bw - 4) / max(tw, 1))
            ((tw, th), _) = cv2.getTextSize(label, fs, sc, 1)
        ty = y1 - 4 if y1 - th - 8 >= 0 else y1 + th + 6
        cv2.rectangle(frame, (x1, ty - th - 3), (x1 + tw + 4, ty + 2), col, -1)
        cv2.putText(frame, label, (x1 + 2, ty), fs, sc, (0, 0, 0), 1, cv2.LINE_AA)
        (cx, cy) = map(int, v['center'])
        avx = int(v.get('vx', 0) * 8)
        avy = int(v.get('vy', 0) * 8)
        if abs(avx) + abs(avy) > 2:
            cv2.arrowedLine(frame, (cx, cy), (cx + avx, cy + avy), col, 2, tipLength=0.35)
    high_evs = [e for e in events if e['level'] == 'HIGH']
    med_evs = [e for e in events if e['level'] == 'MEDIUM']
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, min(H, 28 + 22 * max(len(high_evs) + len(med_evs), 1))), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    ts_str = f'{fidx / fps:.2f}s' if fps > 0 else ''
    cv2.putText(frame, f'Frame{fidx}|{ts_str}| Vehicles:{len(vehicles)}', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    y_hud = 36
    for e in (high_evs + med_evs)[:6]:
        col = {'HIGH': (60, 60, 255), 'MEDIUM': (0, 255, 255)}.get(e['level'], (200, 200, 0))
        tag = f"[{e['level']}]{e['message'][:80]}"
        cv2.putText(frame, tag, (6, y_hud), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
        y_hud += 20
    total_coll = len(reg.all()) + len(reg.dl_accidents)
    if total_coll:
        cv2.putText(frame, f'COLLISIONS DETECTED:{total_coll}', (6, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2, cv2.LINE_AA)
    return frame

class FireDetector:
    _CLS_NAMES = {0: 'fire', 1: 'smoke'}
    _CLS_COLORS = {'fire': (0, 60, 255), 'smoke': (80, 80, 80)}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Optional[object] = None
        self._counters: Dict[str, int] = {}
        self._cool: Dict[str, int] = {}
        self._loaded = False
        self._load_fail = False
        self._evt_cd = 60

    def _load(self):
        if self._loaded or self._load_fail:
            return
        try:
            from ultralytics import YOLO as _YOLO
            self.model = _YOLO(self.cfg.FIRE_MODEL)
            self._loaded = True
            print(f'[FireDetector] Loaded:{self.cfg.FIRE_MODEL}')
        except Exception as ex:
            print(f'[FireDetector] Could not load fire model:{ex}')
            self._load_fail = True

    def _roi(self, bbox: Tuple, W: int, H: int) -> Tuple[int, int, int, int]:
        (x1, y1, x2, y2) = bbox
        (bw, bh) = (x2 - x1, y2 - y1)
        px = int(bw * self.cfg.FIRE_PAD_X)
        py_u = int(bh * self.cfg.FIRE_PAD_Y)
        py_d = int(bh * 0.1)
        return (max(0, x1 - px), max(0, y1 - py_u), min(W, x2 + px), min(H, y2 + py_d))

    def _infer_region(self, frame: np.ndarray, roi: Tuple) -> List[dict]:
        (rx1, ry1, rx2, ry2) = roi
        if rx2 - rx1 < 20 or ry2 - ry1 < 20:
            return []
        crop = frame[ry1:ry2, rx1:rx2]
        try:
            results = self.model.predict(crop, conf=self.cfg.FIRE_CONF, verbose=False, imgsz=320)
        except Exception:
            return []
        out = []
        for r in results:
            if r.boxes is None:
                continue
            for (box, conf, cls) in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                name = self._CLS_NAMES.get(cls, f'cls{cls}')
                abs_box = (int(box[0]) + rx1, int(box[1]) + ry1, int(box[2]) + rx1, int(box[3]) + ry1)
                out.append({'cls_name': name, 'conf': float(conf), 'bbox': abs_box})
        return out

    def analyze(self, frame: np.ndarray, vehicles: List[dict], states: Dict[int, 'VehState'], fidx: int) -> List[dict]:
        if not self.cfg.FIRE_ENABLED:
            return []
        self._load()
        if self._load_fail or self.model is None:
            return []
        (H, W) = frame.shape[:2]
        events: List[dict] = []
        fired_keys: set = set()
        for v in vehicles:
            tid = v['id']
            roi = self._roi(v['bbox'], W, H)
            dets = self._infer_region(frame, roi)
            for det in dets:
                cls = det['cls_name']
                key = f'{cls}_veh_{tid}'
                fired_keys.add(key)
                need = self.cfg.FIRE_CONFIRM if cls == 'fire' else self.cfg.SMOKE_CONFIRM
                self._counters[key] = self._counters.get(key, 0) + 1
                if self._counters[key] >= need:
                    if fidx - self._cool.get(key, -9999) >= self._evt_cd:
                        self._cool[key] = fidx
                        col = self._CLS_COLORS.get(cls, (0, 0, 200))
                        lvl = 'HIGH' if cls == 'fire' else 'MEDIUM'
                        msg = f'FIRE detected on vehicle #{tid}!' if cls == 'fire' else f'SMOKE detected near vehicle #{tid}'
                        events.append({'level': lvl, 'type': cls.upper(), 'ids': [tid], 'score': int(det['conf'] * 100), 'message': msg, 'color': col, 'bbox': det['bbox'], 'source': 'vehicle_roi'})
        if self.cfg.FIRE_FULL_FRM and fidx % self.cfg.FIRE_FULL_CD == 0:
            dets = self._infer_region(frame, (0, 0, W, H))
            for det in dets:
                cls = det['cls_name']
                key = f'{cls}_global'
                need = self.cfg.FIRE_CONFIRM if cls == 'fire' else self.cfg.SMOKE_CONFIRM
                self._counters[key] = self._counters.get(key, 0) + 1
                if self._counters[key] >= need:
                    if fidx - self._cool.get(key, -9999) >= self._evt_cd * 2:
                        self._cool[key] = fidx
                        col = self._CLS_COLORS.get(cls, (0, 0, 200))
                        lvl = 'HIGH' if cls == 'fire' else 'MEDIUM'
                        msg = 'FIRE detected in scene!' if cls == 'fire' else 'SMOKE detected in scene'
                        events.append({'level': lvl, 'type': cls.upper(), 'ids': [], 'score': int(det['conf'] * 100), 'message': msg, 'color': col, 'bbox': det['bbox'], 'source': 'full_frame'})
        for key in list(self._counters):
            if key not in fired_keys and (not key.startswith('smoke_global')) and (not key.startswith('fire_global')):
                self._counters[key] = max(0, self._counters[key] - 1)
        return events

    @staticmethod
    def draw_fire_events(frame: np.ndarray, fire_events: List[dict]) -> np.ndarray:
        for ev in fire_events:
            col = ev.get('color', (0, 0, 255))
            bbox = ev.get('bbox')
            if bbox:
                (x1, y1, x2, y2) = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
                cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)
                label = ev['message'][:40]
                (fs, sc) = (cv2.FONT_HERSHEY_SIMPLEX, 0.5)
                ((tw, th), _) = cv2.getTextSize(label, fs, sc, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 3), fs, sc, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

def reader_worker(cap: cv2.VideoCapture, fq: queue.Queue, stop: threading.Event):
    while not stop.is_set():
        (ret, frame) = cap.read()
        if not ret:
            fq.put(None)
            break
        fq.put(frame)

def writer_worker(out: cv2.VideoWriter, rq: queue.Queue, locker: TimeLocker, cfg: Config, W: int, H: int, fps: float, gmc: GlobalMotionComp, mot: MotionAnalyzer, engine: RiskEngine, fire_det: 'FireDetector', track_log: dict, event_log: list):
    frame_idx = 0
    dl_persistence = DLDetectorPersistence(max_age=cfg.HCI_TRIGGER_BUFFER)
    while True:
        item = rq.get()
        if item is None:
            break
        if len(item) == 3:
            (raw_frame, result_tracker, result_event) = item
        else:
            (raw_frame, result_tracker) = item
            result_event = None
        frame_idx += 1
        (vehicles, raw_dl_dets, dl_evs) = process_detections(result_tracker, frame_idx, locker, cfg, W, H, fps, result_event)
        if cfg.HCI_ENABLED:
            dl_dets = dl_persistence.update(raw_dl_dets if result_event is not None else None, vehicles)
        else:
            dl_dets = raw_dl_dets
        for d in dl_dets:
            if d['cls_name'] == 'accident' and d.get('level') == 'HIGH' and (d.get('age', 0) == 0):
                engine.reg.add_dl_accident(frame_idx, frame_idx / max(fps, 1.0), d['conf'], d['bbox'])
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        gmc_M = gmc.update(gray)
        mot.process_frame(vehicles, gray, frame_idx, gmc_M)
        engine.dl_dets = dl_dets
        events = engine.analyze(vehicles, frame_idx)
        fire_events = fire_det.analyze(raw_frame, vehicles, engine.states, frame_idx)
        events = dl_evs + fire_events + events
        out_frame = draw(raw_frame, vehicles, events, engine.states, engine.reg, frame_idx, fps, dl_detections=dl_dets, mot=mot)
        FireDetector.draw_fire_events(out_frame, fire_events)
        out.write(out_frame)
        track_log[frame_idx] = vehicles
        for e in events:
            if not e.get('persistent', False):
                etype = e.get('type')
                if e.get('level') == 'HIGH' or etype in ('COLLISION', 'ACCIDENT_DL', 'FIRE', 'FIRE_DL', 'BURNED'):
                    is_dup = False
                    if etype in ('ACCIDENT_DL', 'FIRE_DL'):
                        bbox = e.get('bbox')
                        if bbox:
                            (cx2, cy2) = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                            for old_e in reversed(event_log[-50:]):
                                if old_e.get('type') == etype:
                                    old_bbox = old_e.get('bbox')
                                    if old_bbox:
                                        (cx1, cy1) = ((old_bbox[0] + old_bbox[2]) / 2.0, (old_bbox[1] + old_bbox[3]) / 2.0)
                                        dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                                        if (_iou(old_bbox, bbox) > 0.1 or dist < 250.0) and frame_idx - old_e['frame'] < 10800:
                                            is_dup = True
                                            break
                    if not is_dup:
                        event_log.append({'frame': frame_idx, 'ts': round(frame_idx / fps, 3), **_jclean(e)})
        if len(track_log) > cfg.MAX_TRACK_LOG:
            oldest_key = min(track_log.keys())
            del track_log[oldest_key]
        if frame_idx % 300 == 0:
            locker.cleanup(frame_idx)
            mot.cleanup({v['id'] for v in vehicles})
        rq.task_done()

def compute_model_metrics(track_log: dict, locker: 'TimeLocker', fps: float, elapsed: float, cfg: 'Config') -> dict:
    if not track_log:
        return {}
    all_confs: List[float] = []
    class_confs: Dict[str, List[float]] = defaultdict(list)
    class_counts: Dict[str, int] = defaultdict(int)
    track_frames: Dict[int, int] = defaultdict(int)
    peak_det_frame = (0, 0)
    for (fidx, dets) in track_log.items():
        n = len(dets)
        if n > peak_det_frame[0]:
            peak_det_frame = (n, fidx)
        for d in dets:
            c = float(d['conf'])
            name = d.get('cls_name', 'unknown')
            all_confs.append(c)
            class_confs[name].append(c)
            class_counts[name] += 1
            track_frames[int(d['id'])] += 1
    total_frames = len(track_log)
    total_detections = len(all_confs)
    if total_detections == 0:
        return {'total_detections': 0}
    confs_arr = np.array(all_confs, dtype=np.float32)
    avg_conf = float(np.mean(confs_arr))
    std_conf = float(np.std(confs_arr))
    p50 = float(np.percentile(confs_arr, 50))
    p90 = float(np.percentile(confs_arr, 90))
    high_conf = float(np.mean(confs_arr >= cfg.CONF))
    min_track_len = max(5, int(fps * 0.5))
    unique_tracks = sum((1 for (tid, count) in track_frames.items() if count >= min_track_len))
    locked_tracks = len(locker._locked)
    lock_rate = locked_tracks / max(unique_tracks, 1)
    avg_trk_len = float(np.mean(list(track_frames.values()))) if track_frames else 0.0
    return {'avg_conf': round(avg_conf, 4), 'conf_std': round(std_conf, 4), 'high_conf_ratio': round(high_conf, 4), 'conf_p50': round(p50, 4), 'conf_p90': round(p90, 4), 'unique_tracks': unique_tracks, 'locked_tracks': locked_tracks, 'lock_rate': round(lock_rate, 4), 'avg_track_len': round(avg_trk_len, 2), 'total_detections': total_detections, 'avg_det_per_frame': round(total_detections / max(total_frames, 1), 2), 'peak_det_count': peak_det_frame[0], 'peak_det_frame': peak_det_frame[1], 'det_per_class': dict(class_counts), 'conf_per_class': {cls: round(float(np.mean(v)), 4) for (cls, v) in class_confs.items()}, 'actual_fps': round(total_frames / max(elapsed, 1e-06), 2), 'processing_secs': round(elapsed, 2)}

def save_outputs(cfg: Config, event_log: list, engine: RiskEngine, total_fr: int, fps: float, metrics: Optional[dict]=None) -> dict:
    gen = time.strftime('%Y-%m-%d %H:%M:%S')
    Path(cfg.OUT_EVENTS).write_text(json.dumps(_jclean(event_log), ensure_ascii=False, indent=2), encoding='utf-8')
    keys = ['frame', 'ts', 'level', 'type', 'ids', 'score', 'message']
    with open(cfg.OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for e in event_log:
            w.writerow({k: str(e.get(k, '')) for k in keys})
    engine.reg.save_report(cfg.OUT_REPORT, engine.states, gen)
    coll_recs = engine.reg.all()
    dl_accs = engine.reg.dl_accidents
    summary = {'generated': gen, 'total_frames': total_fr, 'duration_secs': round(total_fr / max(fps, 1), 2), 'total_events': len(event_log), 'confirmed_collisions': len(coll_recs), 'dl_accidents': len(dl_accs), 'total_incidents': len(coll_recs) + len(dl_accs), 'total_collisions': len(coll_recs) + len(dl_accs), 'proc_fps': metrics.get('actual_fps', 0.0) if metrics else 0.0, 'unique_ids': metrics.get('unique_tracks', 0) if metrics else 0, 'collisions': [_jclean({'ids': list(r.ids), 'ts': r.ts, 'frame': r.frame, 'bayes': r.bayes, 'iou': r.iou}) for r in coll_recs], 'dl_accident_details': [_jclean(r) for r in dl_accs], 'model_metrics': _jclean(metrics) if metrics else {}, 'transcoding_failed': False}
    Path(cfg.OUT_SUMMARY).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary

def run(video_path: Optional[str]=None, run_dir: Optional[str]=None, progress_callback: Optional[callable]=None) -> dict:
    cfg = Config()
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        cfg.OUT_VIDEO = os.path.join(run_dir, C.OUT_VIDEO)
        cfg.OUT_EVENTS = os.path.join(run_dir, C.OUT_EVENTS)
        cfg.OUT_CSV = os.path.join(run_dir, C.OUT_CSV)
        cfg.OUT_REPORT = os.path.join(run_dir, C.OUT_REPORT)
        cfg.OUT_SUMMARY = os.path.join(run_dir, C.OUT_SUMMARY)
    if video_path is None:
        if IN_COLAB:
            print('Uploading video file...')
            uploaded = _gfiles.upload()
            if not uploaded:
                return {'error': 'No file uploaded.', 'transcoding_failed': False}
            video_path = next(iter(uploaded.keys()))
        else:
            raise ValueError('Must provide video_path when running outside Colab.')
    tracker_model = None
    event_model = None
    model = None
    if cfg.HCI_ENABLED:
        print('[HCI] Loading Stage 1 Tracker...')
        base_dir = getattr(cfg, 'BASE_DIR', getattr(C, 'BASE_DIR', '.'))
        for mname in [cfg.HCI_TRACKER_MODEL, 'yolov10x.pt', 'yolov10n.pt', 'yolov8n.pt']:
            paths_to_try = [mname, os.path.join(base_dir, mname), os.path.join(base_dir, '..', mname) if os.path.basename(mname) == mname else mname]
            for p in paths_to_try:
                try:
                    if os.path.exists(p) or p == mname:
                        tracker_model = YOLO(p)
                        print(f'Stage 1 Loaded:{p}')
                        break
                except Exception as e:
                    pass
            if tracker_model is not None:
                break
        if tracker_model is None:
            raise RuntimeError('Failed to load Stage 1 Tracker.')
        print('[HCI] Loading Stage 2 Event Detector...')
        base_dir = getattr(cfg, 'BASE_DIR', getattr(C, 'BASE_DIR', '.'))
        for mname in [cfg.HCI_EVENT_MODEL, 'best.pt', 'best (3).pt']:
            paths_to_try = [mname, os.path.join(base_dir, mname), os.path.join(base_dir, '..', mname) if os.path.basename(mname) == mname else mname]
            for p in paths_to_try:
                try:
                    if os.path.exists(p) or p == mname:
                        event_model = YOLO(p)
                        print(f'Stage 2 Loaded:{p}')
                        break
                except Exception as e:
                    pass
            if event_model is not None:
                break
        if event_model is None:
            raise RuntimeError('Failed to load Stage 2 Event Detector.')
    else:
        print('Loading YOLO model...')
        for mname in [cfg.MODEL] + cfg.FALLBACKS:
            try:
                model = YOLO(mname)
                print(f'Loaded:{mname}')
                break
            except Exception as e:
                print(f'Failed:{mname}:{e}')
        if model is None:
            raise RuntimeError('Failed to load any YOLO model.')
    tracker_cfg = make_botsort(cfg)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video:{video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    print(f'{video_path}|{W}×{H}|{fps:.1f}FPS |{total_raw}frames')
    locker = TimeLocker(cfg, fps)
    gmc = GlobalMotionComp()
    mot = MotionAnalyzer(fps)
    engine = RiskEngine(fps, mot, W, H)
    fire_det = FireDetector(cfg)
    raw_video_path = cfg.OUT_VIDEO + '_raw.mp4'
    out = cv2.VideoWriter(raw_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    print(f'Processing started | Bayes >={Config.BAYES_THRESH:.0%}')
    track_log: dict = {}
    event_log: list = []
    frame_queue = queue.Queue(maxsize=4)
    result_queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    r_thread = threading.Thread(target=reader_worker, args=(cap, frame_queue, stop_event), daemon=True)
    w_thread = threading.Thread(target=writer_worker, args=(out, result_queue, locker, cfg, W, H, fps, gmc, mot, engine, fire_det, track_log, event_log), daemon=True)
    r_thread.start()
    w_thread.start()
    frame_idx = 0
    t0 = time.time()
    max_fr = cfg.MAX_FRAMES
    kinematic_trigger = KinematicTrigger(fps, cfg)
    trigger_countdown = 0
    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            frame_idx += 1
            if max_fr > 0 and frame_idx > max_fr:
                break
            if cfg.HCI_ENABLED:
                result_tracker = tracker_model.track(frame, persist=True, tracker=tracker_cfg, conf=cfg.CONF, iou=cfg.IOU, imgsz=cfg.INPUT_SIZE, classes=[2, 3, 5, 7], verbose=False)[0]
                run_stage2 = False
                if frame_idx % cfg.HCI_EVENT_PERIOD == 0:
                    run_stage2 = True
                if kinematic_trigger.update_and_check(frame_idx, result_tracker.boxes):
                    run_stage2 = True
                    trigger_countdown = cfg.HCI_TRIGGER_BUFFER
                if getattr(engine, 'physics_trigger', False):
                    run_stage2 = True
                    trigger_countdown = cfg.HCI_TRIGGER_BUFFER
                    engine.physics_trigger = False
                if trigger_countdown > 0:
                    run_stage2 = True
                    trigger_countdown -= 1
                if run_stage2:
                    result_event = event_model.predict(frame, conf=cfg.CONF, iou=cfg.IOU, imgsz=cfg.INPUT_SIZE, classes=[0, 1], verbose=False)[0]
                else:
                    result_event = None
                result_queue.put((frame, result_tracker, result_event))
            else:
                result = model.track(frame, persist=True, tracker=tracker_cfg, conf=cfg.CONF, iou=cfg.IOU, imgsz=cfg.INPUT_SIZE, classes=cfg.CLASSES, verbose=False)[0]
                result_queue.put((frame, result))
            if frame_idx % 30 == 0 and progress_callback:
                elapsed_now = time.time() - t0
                cur_fps = frame_idx / max(elapsed_now, 1e-06)
                pct = int(frame_idx * 100 / max(total_raw, 1)) if total_raw > 0 else 0
                try:
                    progress_callback(frame_idx, total_raw, cur_fps, min(pct, 99))
                except Exception:
                    pass
            if frame_idx % 100 == 0:
                elapsed = time.time() - t0
                print(f'frames{frame_idx:5d}|{frame_idx / elapsed:5.1f}FPS | collisions:{len(engine.reg.all())}')
            if frame_idx % 300 == 0 and cfg.HCI_ENABLED:
                active_ids = set()
                if 'result_tracker' in locals() and result_tracker is not None and (result_tracker.boxes is not None) and (result_tracker.boxes.id is not None):
                    active_ids = set(result_tracker.boxes.id.cpu().numpy().astype(int))
                kinematic_trigger.cleanup(active_ids)
    finally:
        stop_event.set()
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
        r_thread.join(timeout=5)
        result_queue.put(None)
        w_thread.join(timeout=30)
        cap.release()
        out.release()
    elapsed = time.time() - t0
    metrics = compute_model_metrics(track_log, locker, fps, elapsed, cfg)
    if metrics:
        print(f"{'=' * 56}")
        print(f"Processing Done!{metrics.get('actual_fps', 0):.1f}FPS avg")
        print(f"Total Detections  :{metrics.get('total_detections', 0)}")
        print(f"Avg Confidence    :{metrics.get('avg_conf', 0):.2%}")
        print(f"Unique Tracks     :{metrics.get('unique_tracks', 0)}")
        print(f'Confirmed Collisions:{len(engine.reg.all())}')
        print(f"{'=' * 56}")
    summary = save_outputs(cfg, event_log, engine, frame_idx, fps, metrics)
    transcoding_failed = False
    try:
        from utils import transcode_for_web
        ok = transcode_for_web(raw_video_path, cfg.OUT_VIDEO)
        if ok:
            os.remove(raw_video_path)
        else:
            import shutil
            shutil.move(raw_video_path, cfg.OUT_VIDEO)
            transcoding_failed = True
    except Exception as e:
        print(f'[TRANSCODE]{e}')
        import shutil
        if os.path.exists(raw_video_path):
            shutil.move(raw_video_path, cfg.OUT_VIDEO)
        transcoding_failed = True
    summary['transcoding_failed'] = transcoding_failed
    try:
        Path(cfg.OUT_SUMMARY).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass
    if IN_COLAB and cfg.AUTO_DL and _gfiles:
        for fpath in [cfg.OUT_VIDEO, cfg.OUT_EVENTS, cfg.OUT_CSV, cfg.OUT_REPORT, cfg.OUT_SUMMARY]:
            if Path(fpath).exists():
                _gfiles.download(fpath)
    return summary
if __name__ == '__main__':
    if IN_COLAB:
        summary = run()
    else:
        import argparse
        parser = argparse.ArgumentParser(description='AI Accident Detection — Final v1.0')
        parser.add_argument('video', nargs='?', default=None, help='video path')
        parser.add_argument('--model', default=None, help='YOLO model')
        parser.add_argument('--max-frames', type=int, default=-1, help='max frames')
        parser.add_argument('--no-gmc', action='store_true', help='disable GMC')
        (args, _) = parser.parse_known_args()
        if args.model:
            Config.MODEL = args.model
        if args.max_frames > 0:
            Config.MAX_FRAMES = args.max_frames
        if args.no_gmc:
            Config.USE_GMC = False
        summary = run(args.video)
        print(f"Done! Collisions={summary.get('confirmed_collisions', 0)}| Events={summary.get('total_events', 0)}")