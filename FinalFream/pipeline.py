import csv
import json
import math
import os
import queue
import threading
import time
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Callable
import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

import config as C
from helpers import Config, make_botsort, _jclean, _iou, _gap, _dist, _diag, _adiff
from trackers import TimeLocker, KinematicTrigger, VisualDetectorPersistence, GlobalMotionComp, KalmanTracker, OpticalFlow, MotionAnalyzer
from risk import St, VehState, CollRec, Registry, BayesianFusion, KinematicModel, RiskEngine
from fire import FireDetector

try:
    from google.colab import files as _gfiles
    IN_COLAB = True
except ImportError:
    _gfiles = None
    IN_COLAB = False

live_frame_queue = queue.Queue(maxsize=2)

class VisualEventStabilizer:
    def __init__(self, max_age: int = 30):
        self.max_age = max_age
        self.events = []
        self.new_events_logged = []

    def update(self, new_dets: List[dict], frame_idx: int, fps: float) -> Tuple[List[dict], List[dict]]:
        self.new_events_logged = []
        updated_events = []
        
        for nd in new_dets:
            best_ev = None
            best_score = -1.0
            
            for ev in self.events:
                if ev['cls_name'] != nd['cls_name']:
                    continue
                iou_val = _iou(ev['bbox'], nd['bbox'])
                dist_val = _dist(ev['center'], nd['center'])
                
                if iou_val >= 0.3 or dist_val < 150.0:
                    score = iou_val if iou_val >= 0.3 else (150.0 - dist_val) / 150.0
                    if score > best_score:
                        best_score = score
                        best_ev = ev
            
            if best_ev is not None:
                best_ev['bbox'] = nd['bbox']
                best_ev['center'] = nd['center']
                best_ev['age'] = 0
                best_ev['frames_detected'] = best_ev.get('frames_detected', 0) + 1
                
                cls = nd['cls_name']
                old_dom_cls = max(best_ev['max_conf_by_cls'], key=best_ev['max_conf_by_cls'].get)
                old_dom_conf = best_ev['max_conf_by_cls'][old_dom_cls]
                old_level = best_ev.get('level', 'MEDIUM')
                last_logged = best_ev.get('last_logged_conf', old_dom_conf)
                
                best_ev['max_conf_by_cls'][cls] = max(best_ev['max_conf_by_cls'].get(cls, 0.0), nd['conf'])
                
                dom_cls = max(best_ev['max_conf_by_cls'], key=best_ev['max_conf_by_cls'].get)
                dom_conf = best_ev['max_conf_by_cls'][dom_cls]
                
                persists = best_ev.get('frames_detected', 0) >= 10
                level = 'HIGH' if (dom_cls == 'accident' and (dom_conf >= 0.75 or persists)) or (dom_cls == 'fire' and dom_conf >= 0.6) else 'MEDIUM'
                
                if dom_cls != old_dom_cls or level != old_level or (dom_conf >= last_logged + 0.15):
                    color = (0, 0, 255) if dom_cls == 'accident' else (0, 60, 255)
                    if level == 'MEDIUM':
                        color = (0, 255, 255)
                    self.new_events_logged.append({
                        'level': level,
                        'type': 'ACCIDENT_VISUAL' if dom_cls == 'accident' else 'FIRE_VISUAL',
                        'message': f"Detected {dom_cls.capitalize()} with confidence {dom_conf:.0%} ({('CONFIRMED' if level == 'HIGH' else 'POTENTIAL')})",
                        'color': color,
                        'frame': frame_idx,
                        'ts': round(frame_idx / max(fps, 1.0), 3),
                        'bbox': nd['bbox'],
                        'score': int(dom_conf * 100)
                    })
                    best_ev['last_logged_conf'] = dom_conf
                
                best_ev['cls_name'] = dom_cls
                best_ev['conf'] = dom_conf
                best_ev['level'] = level
                
                if best_ev not in updated_events:
                    updated_events.append(best_ev)
            else:
                level = 'HIGH' if (nd['cls_name'] == 'accident' and nd['conf'] >= 0.75) or (nd['cls_name'] == 'fire' and nd['conf'] >= 0.6) else 'MEDIUM'
                new_ev = {
                    'cls_id': nd['cls_id'],
                    'cls_name': nd['cls_name'],
                    'bbox': nd['bbox'],
                    'center': nd['center'],
                    'conf': nd['conf'],
                    'level': level,
                    'frames_detected': 1,
                    'max_conf_by_cls': {nd['cls_name']: nd['conf']},
                    'last_logged_conf': nd['conf'],
                    'age': 0
                }
                updated_events.append(new_ev)
                
                color = (0, 0, 255) if nd['cls_name'] == 'accident' else (0, 60, 255)
                if level == 'MEDIUM':
                    color = (0, 255, 255)
                self.new_events_logged.append({
                    'level': level,
                    'type': 'ACCIDENT_VISUAL' if nd['cls_name'] == 'accident' else 'FIRE_VISUAL',
                    'message': f"Detected {nd['cls_name'].capitalize()} with confidence {nd['conf']:.0%} ({('CONFIRMED' if level == 'HIGH' else 'POTENTIAL')})",
                    'color': color,
                    'frame': frame_idx,
                    'ts': round(frame_idx / max(fps, 1.0), 3),
                    'bbox': nd['bbox'],
                    'score': int(nd['conf'] * 100)
                })
                
        for ev in self.events:
            if ev not in updated_events:
                ev['age'] += 1
                if ev['age'] <= self.max_age:
                    updated_events.append(ev)
                    
        self.events = updated_events
        
        draw_dets = [
            {
                'cls_id': ev['cls_id'],
                'cls_name': ev['cls_name'],
                'bbox': ev['bbox'],
                'center': ev['center'],
                'conf': ev['conf'],
                'level': ev['level']
            }
            for ev in self.events
        ]
        return draw_dets, self.new_events_logged

def get_inference_device(use_onnx: bool) -> str:
    if use_onnx:
        try:
            import onnxruntime
            providers = onnxruntime.get_available_providers()
            if 'CUDAExecutionProvider' in providers and torch.cuda.is_available():
                return 'cuda:0'
        except ImportError:
            pass
        return 'cpu'
    else:
        if torch.cuda.is_available():
            return 'cuda:0'
        return 'cpu'

def get_accelerated_model_path(pt_path: str, imgsz: int = 640, simplify: bool = True, int8: bool = False) -> str:
    if not pt_path or not pt_path.endswith('.pt'):
        return pt_path
    
    onnx_path = pt_path.replace('.pt', '.onnx')
    
    if os.path.exists(onnx_path):
        logger.info(f"[ONNX] Accelerated model already exists at: {onnx_path}")
        return onnx_path
        
    logger.info(f"[ONNX] Exporting {pt_path} to ONNX format (imgsz={imgsz}, simplify={simplify}, int8={int8})...")
    try:
        temp_model = YOLO(pt_path)
        try:
            temp_model.export(
                format='onnx',
                imgsz=imgsz,
                simplify=simplify,
                int8=int8,
                dynamic=True,
                opset=17
            )
        except Exception as e:
            if int8:
                logger.warning(f"[ONNX] INT8 export failed: {e}. Falling back to FP32 ONNX export.")
                temp_model.export(
                    format='onnx',
                    imgsz=imgsz,
                    simplify=simplify,
                    int8=False,
                    dynamic=True,
                    opset=17
                )
            else:
                raise e
        
        if os.path.exists(onnx_path):
            logger.info(f"[ONNX] Export successful! Accelerated model saved at: {onnx_path}")
            return onnx_path
        else:
            logger.warning(f"[ONNX] Export finished but {onnx_path} was not found. Falling back to PyTorch.")
            return pt_path
    except Exception as e:
        logger.error(f"[ONNX] Failed to export model to ONNX: {e}. Falling back to PyTorch .pt model.")
        return pt_path

def verify_and_load_model(model_path: str, default_pt_path: str, imgsz: int, device: str) -> YOLO:
    logger.info(f"[ONNX] Attempting to load model from: {model_path} on device: {device}")
    try:
        loaded_model = YOLO(model_path)
        dummy_frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        loaded_model.predict(dummy_frame, imgsz=imgsz, device=device, verbose=False)
        logger.info(f"[ONNX] Model loaded and warmed up successfully: {model_path}")
        return loaded_model
    except Exception as e:
        logger.error(f"[ONNX] Failed to load/warmup model {model_path}: {e}")
        if model_path != default_pt_path:
            logger.warning(f"[ONNX] Falling back to PyTorch model: {default_pt_path}")
            try:
                loaded_model = YOLO(default_pt_path)
                dummy_frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
                loaded_model.predict(dummy_frame, imgsz=imgsz, device=device, verbose=False)
                return loaded_model
            except Exception as fallback_err:
                raise RuntimeError(f"Failed to load fallback PyTorch model {default_pt_path}: {fallback_err}")
        else:
            raise e

def process_detections(result_tracker, frame_idx: int, locker: TimeLocker, cfg: Config, fw: int, fh: int, fps: float, result_event=None,
                       vehicle_classes=None, accident_classes=None, fire_classes=None, class_remap=None) -> Tuple[List[dict], List[dict], List[dict]]:
    vehicles: List[dict] = []
    visual_detections: List[dict] = []
    visual_events: List[dict] = []
    
    if vehicle_classes is None:
        vehicle_classes = [2, 3, 5, 7] if cfg.HCI_ENABLED else [2, 3]
    if accident_classes is None:
        accident_classes = [0]
    if fire_classes is None:
        fire_classes = [1]
    if class_remap is None:
        class_remap = {0: 'accident', 1: 'fire', 2: 'motorcycle', 3: 'vehicle'}
        
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
        
        for i in range(len(clss_np)):
            box = xyxy_np[i]
            cls = clss_np[i]
            conf = confs_np[i]
            (x1, y1, x2, y2) = map(int, box)
            w_box = x2 - x1
            h_box = y2 - y1
            area_box = w_box * h_box
            if w_box < min_w or h_box < min_h or area_box > max_area:
                continue
                
            if cls in vehicle_classes and ids_np is not None and i < len(ids_np) and ids_np[i] >= 0:
                tid = ids_np[i]
                (stable_cls, locked) = locker.get(tid, cls, frame_idx)
                name = class_remap.get(stable_cls, 'vehicle')
                vehicles.append({
                    'id': tid, 'cls_id': stable_cls, 'cls_name': name, 'locked': bool(locked),
                    'bbox': (x1, y1, x2, y2), 'center': ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    'conf': float(conf), 'w': w_box, 'h': h_box, 'vx': 0.0, 'vy': 0.0, 'spd': 0.0,
                    'acc': 0.0, 'heading': None, 'hdelta': None, 'mahal': 0.0
                })
                
            if cls in accident_classes or cls in fire_classes:
                name = 'accident' if cls in accident_classes else 'fire'
                is_confirmed = (cls in accident_classes and conf >= 0.75) or (cls in fire_classes and conf >= 0.6)
                is_potential = (cls in accident_classes and conf >= 0.35) or (cls in fire_classes and conf >= 0.3)
                if not is_potential:
                    continue
                level = 'HIGH' if is_confirmed else 'MEDIUM'
                color = (0, 0, 255) if name == 'accident' else (0, 60, 255)
                if not is_confirmed:
                    color = (0, 255, 255)
                visual_detections.append({
                    'cls_id': cls, 'cls_name': name, 'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) / 2.0, (y1 + y2) / 2.0), 'conf': float(conf), 'level': level
                })
                visual_events.append({
                    'level': level, 'type': 'ACCIDENT_VISUAL' if name == 'accident' else 'FIRE_VISUAL',
                    'message': f"Detected {name.capitalize()} with confidence {conf:.0%} ({('CONFIRMED' if is_confirmed else 'POTENTIAL')})",
                    'color': color, 'frame': frame_idx, 'ts': round(frame_idx / max(fps, 1.0), 3),
                    'bbox': (x1, y1, x2, y2), 'score': int(conf * 100)
                })
                
    if cfg.HCI_ENABLED and result_event is not None and result_event.boxes is not None:
        xyxy = result_event.boxes.xyxy
        confs = result_event.boxes.conf
        clss = result_event.boxes.cls
        xyxy_np = xyxy.cpu().numpy()
        confs_np = confs.cpu().numpy()
        clss_np = clss.cpu().numpy().astype(int)
        for i in range(len(clss_np)):
            cls = clss_np[i]
            conf = confs_np[i]
            if cls not in accident_classes and cls not in fire_classes:
                continue
            (x1, y1, x2, y2) = map(int, xyxy_np[i])
            name = 'accident' if cls in accident_classes else 'fire'
            is_confirmed = (cls in accident_classes and conf >= 0.75) or (cls in fire_classes and conf >= 0.6)
            is_potential = (cls in accident_classes and conf >= 0.35) or (cls in fire_classes and conf >= 0.3)
            if not is_potential:
                continue
            level = 'HIGH' if is_confirmed else 'MEDIUM'
            color = (0, 0, 255) if name == 'accident' else (0, 60, 255)
            if not is_confirmed:
                color = (0, 255, 255)
            visual_detections.append({
                'cls_id': cls, 'cls_name': name, 'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) / 2.0, (y1 + y2) / 2.0), 'conf': float(conf), 'level': level
            })
            visual_events.append({
                'level': level, 'type': 'ACCIDENT_VISUAL' if name == 'accident' else 'FIRE_VISUAL',
                'message': f"Detected {name.capitalize()} with confidence {conf:.0%} ({('CONFIRMED' if is_confirmed else 'POTENTIAL')})",
                'color': color, 'frame': frame_idx, 'ts': round(frame_idx / max(fps, 1.0), 3),
                'bbox': (x1, y1, x2, y2), 'score': int(conf * 100)
            })
    return (vehicles, visual_detections, visual_events)

def draw(frame: np.ndarray, vehicles: List[dict], events: List[dict], states: Dict[int, VehState], reg: Registry, fidx: int, fps: float, visual_detections: List[dict]=None, mot: Optional[MotionAnalyzer]=None) -> np.ndarray:
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
    if visual_detections:
        for d in visual_detections:
            (x1, y1, x2, y2) = d['bbox']
            col = (0, 255, 255) if d.get('level') == 'MEDIUM' else Config.COLORS.get(d['cls_name'], (0, 0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
            label = f"{d['cls_name'].upper()} {d['conf']:.0%} ({('CONFIRMED' if d.get('level') == 'HIGH' else 'POTENTIAL')})"
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
                suffix = ' BURNED'
            elif s in (St.COLLIDED, St.POST):
                suffix = ' COLLIDED'
            elif s == St.CRITICAL:
                suffix = ' CRITICAL'
        mahal_str = f" M{v.get('mahal', 0.0):.1f}" if v.get('mahal', 0.0) > 2 else ''
        label = f"ID:{tid} {v['cls_name']} {v['conf']:.0%} {v['spd']:.1f}px/f{mahal_str}{suffix}"
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
    cv2.putText(frame, f'Frame {fidx} | {ts_str} | Vehicles: {len(vehicles)}', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    y_hud = 36
    for e in (high_evs + med_evs)[:6]:
        col = {'HIGH': (60, 60, 255), 'MEDIUM': (0, 255, 255)}.get(e['level'], (200, 200, 0))
        tag = f"[{e['level']}] {e['message'][:80]}"
        cv2.putText(frame, tag, (6, y_hud), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
        y_hud += 20
    total_coll = len(reg.all()) + len(reg.visual_accidents)
    if total_coll:
        cv2.putText(frame, f'COLLISIONS DETECTED: {total_coll}', (6, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2, cv2.LINE_AA)
    return frame

def reader_worker(cap: cv2.VideoCapture, fq: queue.Queue, stop: threading.Event):
    while not stop.is_set():
        (ret, frame) = cap.read()
        if not ret:
            fq.put(None)
            break
        fq.put(frame)

def writer_worker(out: cv2.VideoWriter, rq: queue.Queue, locker: TimeLocker, cfg: Config, W: int, H: int, fps: float, gmc: GlobalMotionComp, mot: MotionAnalyzer, engine: RiskEngine, fire_det: FireDetector, track_log: dict, event_log: list,
                  vehicle_classes: list, accident_classes: list, fire_classes: list, class_remap: dict):
    frame_idx = 0
    visual_persistence = VisualDetectorPersistence(max_age=cfg.HCI_TRIGGER_BUFFER)
    display_cache = {}
    visual_stabilizer = VisualEventStabilizer(max_age=30)
    
    while not live_frame_queue.empty():
        try:
            live_frame_queue.get_nowait()
        except queue.Empty:
            break
            
    while True:
        item = rq.get()
        if item is None:
            try:
                if live_frame_queue.full():
                    try:
                        live_frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                live_frame_queue.put_nowait(None)
            except Exception:
                pass
            break

        if len(item) == 3:
            (raw_frame, result_tracker, result_event) = item
        else:
            (raw_frame, result_tracker) = item
            result_event = None
        frame_idx += 1
        (vehicles, raw_dl_dets, raw_dl_evs) = process_detections(
            result_tracker, frame_idx, locker, cfg, W, H, fps, result_event,
            vehicle_classes, accident_classes, fire_classes, class_remap
        )
        if cfg.HCI_ENABLED:
            visual_dets = visual_persistence.update(raw_dl_dets if result_event is not None else None, vehicles)
            visual_evs = raw_dl_evs
        else:
            visual_dets, visual_evs = visual_stabilizer.update(raw_dl_dets, frame_idx, fps)
        for d in visual_dets:
            if d['cls_name'] == 'accident' and d.get('level') == 'HIGH' and (d.get('age', 0) == 0):
                engine.reg.add_dl_accident(frame_idx, frame_idx / max(fps, 1.0), d['conf'], d['bbox'])
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        gmc_M = gmc.update(gray)
        mot.process_frame(vehicles, gray, frame_idx, gmc_M, raw_frame=raw_frame)
        engine.visual_dets = visual_dets
        events = engine.analyze(vehicles, frame_idx)
        fire_events = fire_det.analyze(raw_frame, vehicles, engine.states, frame_idx)
        events = visual_evs + fire_events + events

        vehicles_for_draw = [dict(v) for v in vehicles]
        
        for v in vehicles_for_draw:
            tid = v['id']
            if tid not in display_cache or frame_idx >= display_cache[tid]['next_update']:
                display_cache[tid] = {
                    'conf': v['conf'],
                    'spd': v['spd'],
                    'mahal': v.get('mahal', 0.0),
                    'next_update': frame_idx + int(5.0 * fps)
                }
            v['conf'] = display_cache[tid]['conf']
            v['spd'] = display_cache[tid]['spd']
            if 'mahal' in v:
                v['mahal'] = display_cache[tid]['mahal']

        out_frame = draw(raw_frame, vehicles_for_draw, events, engine.states, engine.reg, frame_idx, fps, visual_detections=visual_dets, mot=mot)
        FireDetector.draw_fire_events(out_frame, fire_events)
        out.write(out_frame)
        try:
            _, jpeg_buf = cv2.imencode('.jpg', out_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_bytes = jpeg_buf.tobytes()
            if live_frame_queue.full():
                try:
                    live_frame_queue.get_nowait()
                except queue.Empty:
                    pass
            live_frame_queue.put_nowait(jpeg_bytes)
        except Exception as e:
            logger.error(f"[STREAM] Failed to queue live frame: {e}")
        
        track_log[frame_idx] = vehicles
        for e in events:
            if not e.get('persistent', False):
                etype = e.get('type')
                if e.get('level') == 'HIGH' or etype in ('COLLISION', 'ACCIDENT_VISUAL', 'FIRE', 'FIRE_VISUAL', 'BURNED'):
                    is_dup = False
                    if etype in ('ACCIDENT_VISUAL', 'FIRE_VISUAL'):
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
                                            if e.get('level') == 'HIGH' and old_e.get('level') != 'HIGH':
                                                old_e['level'] = 'HIGH'
                                                if 'message' in e: old_e['message'] = e['message']
                                                if 'score' in e: old_e['score'] = e['score']
                                            elif e.get('score', 0) > old_e.get('score', 0):
                                                if 'message' in e: old_e['message'] = e['message']
                                                old_e['score'] = e['score']
                                            break
                    if not is_dup:
                        event_log.append({'frame': frame_idx, 'ts': round(frame_idx / fps, 3), **_jclean(e)})
        if len(track_log) > cfg.MAX_TRACK_LOG:
            oldest_key = min(track_log.keys())
            del track_log[oldest_key]
        if frame_idx % 300 == 0:
            locker.cleanup(frame_idx)
            mot.cleanup({v['id'] for v in vehicles})
                                                             
            active_ids = {v['id'] for v in vehicles}
            for tid in list(display_cache.keys()):
                if tid not in active_ids:
                    del display_cache[tid]
        rq.task_done()

def compute_model_metrics(track_log: dict, locker: TimeLocker, fps: float, elapsed: float, cfg: Config) -> dict:
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
    dl_accs = engine.reg.visual_accidents
    summary = {'generated': gen, 'total_frames': total_fr, 'duration_secs': round(total_fr / max(fps, 1), 2), 'total_events': len(event_log), 'confirmed_collisions': len(coll_recs), 'visual_accidents': len(dl_accs), 'total_incidents': len(coll_recs) + len(dl_accs), 'total_collisions': len(coll_recs) + len(dl_accs), 'proc_fps': metrics.get('actual_fps', 0.0) if metrics else 0.0, 'unique_ids': metrics.get('unique_tracks', 0) if metrics else 0, 'collisions': [_jclean({'ids': list(r.ids), 'ts': r.ts, 'frame': r.frame, 'bayes': r.bayes, 'iou': r.iou}) for r in coll_recs], 'visual_accident_details': [_jclean(r) for r in dl_accs], 'model_metrics': _jclean(metrics) if metrics else {}, 'transcoding_failed': False}
    Path(cfg.OUT_SUMMARY).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary

def run(video_path: Optional[str]=None, run_dir: Optional[str]=None, progress_callback: Optional[Callable]=None) -> dict:
    cfg = Config()
    result_tracker = None
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        cfg.OUT_VIDEO = os.path.join(run_dir, C.OUT_VIDEO)
        cfg.OUT_EVENTS = os.path.join(run_dir, C.OUT_EVENTS)
        cfg.OUT_CSV = os.path.join(run_dir, C.OUT_CSV)
        cfg.OUT_REPORT = os.path.join(run_dir, C.OUT_REPORT)
        cfg.OUT_SUMMARY = os.path.join(run_dir, C.OUT_SUMMARY)
    if video_path is None:
        if IN_COLAB:
            logger.info('Uploading video file...')
            uploaded = _gfiles.upload()
            if not uploaded:
                return {'error': 'No file uploaded.', 'transcoding_failed': False}
            video_path = next(iter(uploaded.keys()))
        else:
            raise ValueError('Must provide video_path when running outside Colab.')
    tracker_model = None
    event_model = None
    model = None

    tracker_device = get_inference_device(cfg.USE_ONNX)
    event_device = get_inference_device(cfg.USE_ONNX)
    model_device = get_inference_device(cfg.USE_ONNX)

    if cfg.HCI_ENABLED:
        logger.info('[HCI] Loading Stage 1 Tracker...')
        tracker_path = cfg.HCI_TRACKER_MODEL
        if cfg.USE_ONNX:
            tracker_path = get_accelerated_model_path(cfg.HCI_TRACKER_MODEL, imgsz=cfg.INPUT_SIZE, simplify=cfg.ONNX_SIMPLIFY, int8=cfg.ONNX_INT8)
        tracker_model = verify_and_load_model(tracker_path, cfg.HCI_TRACKER_MODEL, cfg.INPUT_SIZE, tracker_device)
        
        logger.info('[HCI] Loading Stage 2 Event Detector...')
        event_path = cfg.HCI_EVENT_MODEL
        if cfg.USE_ONNX:
            event_path = get_accelerated_model_path(cfg.HCI_EVENT_MODEL, imgsz=cfg.INPUT_SIZE, simplify=cfg.ONNX_SIMPLIFY, int8=cfg.ONNX_INT8)
        event_model = verify_and_load_model(event_path, cfg.HCI_EVENT_MODEL, cfg.INPUT_SIZE, event_device)
    else:
        logger.info('Loading YOLO model...')
        model_path = cfg.MODEL
        if cfg.USE_ONNX:
            model_path = get_accelerated_model_path(cfg.MODEL, imgsz=cfg.INPUT_SIZE, simplify=cfg.ONNX_SIMPLIFY, int8=cfg.ONNX_INT8)
        model = verify_and_load_model(model_path, cfg.MODEL, cfg.INPUT_SIZE, model_device)

    vehicle_classes = []
    accident_classes = []
    fire_classes = []
    class_remap = {}

    if cfg.HCI_ENABLED:
        tracker_names = tracker_model.names if hasattr(tracker_model, 'names') else {}
        for cid, name in tracker_names.items():
            name_lower = name.lower()
            if any(term in name_lower for term in ['car', 'truck', 'bus', 'vehicle', 'motorcycle', 'cycle', 'van', 'suv', 'sedan']):
                vehicle_classes.append(cid)
                class_remap[cid] = 'motorcycle' if any(t in name_lower for t in ['motorcycle', 'cycle']) else 'vehicle'
        event_names = event_model.names if hasattr(event_model, 'names') else {}
        for cid, name in event_names.items():
            name_lower = name.lower()
            if any(term in name_lower for term in ['accident', 'crash', 'collision', 'acc']):
                accident_classes.append(cid)
            elif any(term in name_lower for term in ['fire', 'smoke', 'burn']):
                fire_classes.append(cid)
        if not vehicle_classes:
            vehicle_classes = [2, 3, 5, 7]
        if not accident_classes:
            accident_classes = [0]
        if not fire_classes:
            fire_classes = [1]
    else:
        model_names = model.names if hasattr(model, 'names') else {}
        for cid, name in model_names.items():
            name_lower = name.lower()
            is_veh = any(term in name_lower for term in ['car', 'truck', 'bus', 'vehicle', 'motorcycle', 'cycle', 'van', 'suv', 'sedan'])
            is_acc = any(term in name_lower for term in ['accident', 'crash', 'collision', 'acc'])
            is_fire = any(term in name_lower for term in ['fire', 'smoke', 'burn'])
            if is_veh:
                vehicle_classes.append(cid)
            if is_acc:
                accident_classes.append(cid)
                class_remap[cid] = 'accident'
            elif is_fire:
                fire_classes.append(cid)
                class_remap[cid] = 'fire'
            else:
                class_remap[cid] = name_lower
        if not vehicle_classes:
            vehicle_classes = [2, 3, 5, 7]
        cfg.CLASSES = list(model_names.keys())
        cfg.NAMES = {k: v for k, v in model_names.items()}
        cfg.CLASS_REMAP = class_remap

    tracker_cfg = make_botsort(cfg)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    logger.info(f'{video_path} | {W}x{H} | {fps:.1f}FPS | {total_raw} frames')
    locker = TimeLocker(cfg, fps)
    gmc = GlobalMotionComp()
    mot = MotionAnalyzer(fps)
    engine = RiskEngine(fps, mot, W, H)
    fire_det = FireDetector(cfg)
    raw_video_path = cfg.OUT_VIDEO + '_raw.mp4'
    out = cv2.VideoWriter(raw_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    logger.info(f'Processing started | Bayes >= {Config.BAYES_THRESH:.0%}')
    track_log: dict = {}
    event_log: list = []
    frame_queue = queue.Queue(maxsize=4)
    result_queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    r_thread = threading.Thread(target=reader_worker, args=(cap, frame_queue, stop_event), daemon=True)
    w_thread = threading.Thread(target=writer_worker, args=(out, result_queue, locker, cfg, W, H, fps, gmc, mot, engine, fire_det, track_log, event_log, vehicle_classes, accident_classes, fire_classes, class_remap), daemon=True)
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
                result_tracker = tracker_model.track(frame, persist=True, tracker=tracker_cfg, conf=cfg.CONF, iou=cfg.IOU, imgsz=cfg.INPUT_SIZE, classes=vehicle_classes, device=tracker_device, verbose=False)[0]
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
                    result_event = event_model.predict(frame, conf=cfg.CONF, iou=cfg.IOU, imgsz=cfg.INPUT_SIZE, classes=accident_classes + fire_classes, device=event_device, verbose=False)[0]
                else:
                    result_event = None
                result_queue.put((frame, result_tracker, result_event))
            else:
                result = model.track(frame, persist=True, tracker=tracker_cfg, conf=cfg.CONF, iou=cfg.IOU, imgsz=cfg.INPUT_SIZE, classes=cfg.CLASSES, device=model_device, verbose=False)[0]
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
                logger.info(f'frames {frame_idx:5d} | {frame_idx / elapsed:5.1f}FPS | collisions: {len(engine.reg.all())}')
            if frame_idx % 300 == 0 and cfg.HCI_ENABLED:
                active_ids = set()
                if result_tracker is not None and (result_tracker.boxes is not None) and (result_tracker.boxes.id is not None):
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
        logger.info('=' * 56)
        logger.info(f"Processing Done! {metrics.get('actual_fps', 0):.1f}FPS avg")
        logger.info(f"Total Detections  : {metrics.get('total_detections', 0)}")
        logger.info(f"Avg Confidence    : {metrics.get('avg_conf', 0):.2%}")
        logger.info(f"Unique Tracks     : {metrics.get('unique_tracks', 0)}")
        logger.info(f'Confirmed Collisions: {len(engine.reg.all())}')
        logger.info('=' * 56)
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
        logger.error(f'[TRANSCODE] {e}')
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