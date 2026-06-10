import math
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from sklearn.cluster import DBSCAN
from helpers import Config, _iou, _dist, _gap, _diag, _adiff, _jclean
from trackers import MotionAnalyzer

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
        self.msg = f'Collision: #{a} & #{b} at t={self.ts:.2f}s'

class Registry:
    def __init__(self):
        self._r: Dict[frozenset, CollRec] = {}
        self.dl_accidents: List[dict] = []

    def add(self, r: CollRec) -> bool:
        k = frozenset(r.ids)
        if k in self._r:
            return False
        self._r[k] = r
        print(f'[COLLISION] frame={r.frame} t={r.ts:.2f}s bayes={r.bayes:.0%} mahal={r.mahal:.1f} {r.msg}')
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
        print(f'[DL ACCIDENT] frame={frame} t={ts:.2f}s conf={conf:.0%} registered.')

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
        lines = ['COLLISION & ACCIDENT REPORT\n', f'Generated: {gen_time}\n', '\n', f'Total confirmed collisions (Physics-based): {len(self._r)}\n', f'Total confirmed accidents (Deep Learning): {len(self.dl_accidents)}\n', f'Grand Total of Confirmed Incidents: {len(self._r) + len(self.dl_accidents)}\n', '\n', '=========================================\n', '--- PHYSICAL COLLISIONS (KINETIC) ---\n', '=========================================\n']
        for (i, r) in enumerate(self.all(), 1):
            (a, b) = r.ids
            pst = {t: states[t].st.name for t in r.ids if t in states}
            lines += [f'#{i} CONFIRMED COLLISION\n', f'Vehicles   : #{a} <-> #{b}\n', f'Time       : {r.ts:.2f}s  (frame {r.frame})\n', f'IoU        : {r.iou:.4f}\n', f'Rel speed  : {r.rel_spd:.2f}px/f\n', f'Bayes prob : {r.bayes:.0%}\n', f'Mahalanobis: {r.mahal:.2f}\n', f'Post-state : {pst}\n', '\n']
        lines += ['=========================================\n', '--- DEEP LEARNING DETECTED ACCIDENTS ---\n', '=========================================\n']
        for (i, r) in enumerate(self.dl_accidents, 1):
            lines += [f'#{i} DL ACCIDENT ALERT\n', f"Time       : {r['ts']:.2f}s  (frame {r['frame']})\n", f"Confidence : {r['conf']:.0%}\n", f"BBox (xyxy): {r['bbox']}\n", '\n']
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
            return {'level': 'MEDIUM', 'type': wt, 'ids': [v1['id'], v2['id']], 'score': int(bayes * 100), 'message': f"WARNING({wt}): #{v1['id']}+#{v2['id']} bayes={bayes:.0%}", 'color': (0, 165, 255), 'ev': ev}
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
                        evs.append({'level': 'HIGH', 'type': 'BURNED', 'ids': [tid], 'score': 90, 'message': f'#{tid} stationary post-collision', 'color': (0, 0, 180)})
        return evs

    def _cluster(self, vehicles: List[dict], fidx: int) -> List[dict]:
        return []

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
