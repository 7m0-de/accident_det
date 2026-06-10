import cv2
import math
import numpy as np
from collections import defaultdict, deque
from typing import Optional, List, Tuple, Dict, Deque
from helpers import Config, _iou, _adiff

_OF_GRAY_SCALE = 0.5

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
                        print(f'[HCI TRIGGER] Vehicle #{tid} decelerated sharply: {accel:.2f}px/f^2')
                    if spd > 2.0 and p_spd > 2.0:
                        dot = vx * p_vx + vy * p_vy
                        cos_val = dot / (spd * p_spd)
                        cos_val = max(-1.0, min(1.0, cos_val))
                        angle = math.degrees(math.acos(cos_val))
                        if angle >= self.sharp_turn:
                            trigger = True
                            print(f'[HCI TRIGGER] Vehicle #{tid} turned sharply: {angle:.1f}°')
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
