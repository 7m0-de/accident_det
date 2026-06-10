import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
from helpers import Config

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
            print(f'[FireDetector] Loaded: {self.cfg.FIRE_MODEL}')
        except Exception as ex:
            print(f'[FireDetector] Could not load fire model: {ex}')
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

    def analyze(self, frame: np.ndarray, vehicles: List[dict], states: Dict[int, any], fidx: int) -> List[dict]:
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
