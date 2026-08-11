import os
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Clear any existing handlers
for h in list(logger.handlers):
    logger.removeHandler(h)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] %(message)s')
# Console handler
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
# File handler
fh = logging.FileHandler(os.path.join(BASE_DIR, 'app.log'), encoding='utf-8')
fh.setFormatter(formatter)
logger.addHandler(fh)

logging.getLogger('config').info("Root logger configured successfully in config.py")

HOST = os.environ.get('ACCIDENT_HOST', '0.0.0.0')
PORT = int(os.environ.get('ACCIDENT_PORT', '8000'))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_DIR = os.path.join(STATIC_DIR, 'uploads')
RUNS_DIR = os.path.join(STATIC_DIR, 'runs')
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
TIMEOUT_SECONDS = int(os.environ.get('PROCESSING_TIMEOUT_SECS', '1800'))
MODEL = os.environ.get('DETECTION_MODEL', os.path.join(BASE_DIR, 'best.pt'))
CONF = float(os.environ.get('CONF_THRESH', '0.51'))
IOU_NMS = float(os.environ.get('IOU_THRESH', '0.50'))
INPUT_SIZE = int(os.environ.get('INPUT_SIZE', '640'))
CLASSES = [0, 1, 2, 3]
NAMES = {0: 'accident', 1: 'fire', 2: 'motorcycle', 3: 'vehicle'}
CLASS_REMAP = {0: 'accident', 1: 'fire', 2: 'motorcycle', 3: 'vehicle'}
COLORS = {'vehicle': (0, 220, 0), 'motorcycle': (0, 220, 220), 'fire': (0, 60, 255), 'accident': (0, 0, 255)}
TRACK_BUFFER = int(os.environ.get('TRACK_BUFFER', '120'))
MATCH_THRESH = float(os.environ.get('MATCH_THRESH', '0.70'))
LOCK_SECS = float(os.environ.get('LOCK_SECS', '5.0'))
MIN_W_RATIO = float(os.environ.get('MIN_W_RATIO', '0.020'))
MIN_H_RATIO = float(os.environ.get('MIN_H_RATIO', '0.015'))
MAX_AREA_R = float(os.environ.get('MAX_AREA_R', '0.45'))
HIST_SECS = float(os.environ.get('HIST_SECS', '3.0'))
COLL_IOU = float(os.environ.get('COLL_IOU', '0.30'))
CONTACT_GAP = float(os.environ.get('CONTACT_GAP', '12'))
CONTACT_GAP_R = float(os.environ.get('CONTACT_GAP_R', '0.05'))
COLL_REL_SPD = float(os.environ.get('COLL_REL_SPD', '3.0'))
PRE_IMPACT = float(os.environ.get('PRE_IMPACT', '6.0'))
POST_SLOW = float(os.environ.get('POST_SLOW', '1.2'))
SUDDEN_DEC = float(os.environ.get('SUDDEN_DEC', '-8.5'))
MIN_DEC_SPD = float(os.environ.get('MIN_DEC_SPD', '7.0'))
SHARP_TURN = float(os.environ.get('SHARP_TURN', '70.0'))
MIN_TURN_SPD = float(os.environ.get('MIN_TURN_SPD', '5.0'))
TTC_HIGH = float(os.environ.get('TTC_HIGH', '0.6'))
TTC_MED = float(os.environ.get('TTC_MED', '1.8'))
MIN_CLOSE = float(os.environ.get('MIN_CLOSE', '2.5'))
CONVOY_DEG = float(os.environ.get('CONVOY_DEG', '35.0'))
CONVOY_SPD = float(os.environ.get('CONVOY_SPD', '2.5'))
CONFIRM_FR = int(os.environ.get('CONFIRM_FR', '6'))
POST_BURNED = float(os.environ.get('POST_BURNED', '4.0'))
MAHAL_THRESH = float(os.environ.get('MAHAL_THRESH', '5.0'))
BAYES_THRESH = float(os.environ.get('BAYES_THRESH', '0.72'))
MAX_PAIR_PX = int(os.environ.get('MAX_PAIR_PX', '300'))
PROX_PX = int(os.environ.get('PROX_PX', '80'))
LONG_STOP_S = float(os.environ.get('LONG_STOP_S', '7.0'))
LONG_NRBY = int(os.environ.get('LONG_NRBY', '180'))
LONG_SPD = float(os.environ.get('LONG_SPD', '2.5'))
CLUSTER_MIN = int(os.environ.get('CLUSTER_MIN', '4'))
CLUSTER_EPS = int(os.environ.get('CLUSTER_EPS', '130'))
CLUSTER_CD = int(os.environ.get('CLUSTER_CD', '90'))
EVENT_CD = int(os.environ.get('EVENT_CD', '90'))
FIRE_ENABLED = os.environ.get('FIRE_ENABLED', 'True').lower() == 'true'
FIRE_MODEL = os.environ.get('FIRE_MODEL', 'keremberke/yolov8n-fire-detection')
FIRE_CONF = float(os.environ.get('FIRE_CONF', '0.45'))
FIRE_CONFIRM = int(os.environ.get('FIRE_CONFIRM', '3'))
SMOKE_CONFIRM = int(os.environ.get('SMOKE_CONFIRM', '4'))
FIRE_PAD_Y = float(os.environ.get('FIRE_PAD_Y', '0.60'))
FIRE_PAD_X = float(os.environ.get('FIRE_PAD_X', '0.20'))
FIRE_FULL_FRM = os.environ.get('FIRE_FULL_FRM', 'True').lower() == 'true'
FIRE_FULL_CD = int(os.environ.get('FIRE_FULL_CD', '15'))
USE_GMC = os.environ.get('USE_GMC', 'True').lower() == 'true'
USE_OPTICAL_FLOW = os.environ.get('USE_OPTICAL_FLOW', 'True').lower() == 'true'
GMC_METHOD = os.environ.get('GMC_METHOD', 'sparse')
MAX_FRAMES = int(os.environ.get('MAX_FRAMES', '-1'))
AUTO_DL = False
MAX_TRACK_LOG = int(os.environ.get('MAX_TRACK_LOG', '3000'))
OUT_VIDEO = 'tracked_output.mp4'
OUT_EVENTS = 'events.json'
OUT_CSV = 'events.csv'
OUT_REPORT = 'collision_report.txt'
OUT_SUMMARY = 'summary.json'
BOTSORT_YAML = os.path.join(BASE_DIR, 'botsort.yaml')
HCI_ENABLED = os.environ.get('HCI_ENABLED', 'False').lower() == 'true'
HCI_TRACKER_MODEL = os.environ.get('HCI_TRACKER_MODEL', os.path.join(BASE_DIR, 'yolov10x.pt'))
HCI_EVENT_MODEL = os.environ.get('HCI_EVENT_MODEL', os.path.join(BASE_DIR, 'best.pt'))
HCI_EVENT_PERIOD = int(os.environ.get('HCI_EVENT_PERIOD', '5'))
HCI_TRIGGER_BUFFER = int(os.environ.get('HCI_TRIGGER_BUFFER', '15'))

USE_ONNX = os.environ.get('USE_ONNX', 'False').lower() == 'true'
ONNX_SIMPLIFY = os.environ.get('ONNX_SIMPLIFY', 'True').lower() == 'true'
ONNX_INT8 = os.environ.get('ONNX_INT8', 'False').lower() == 'true'

# Load overrides from config_override.json
override_path = os.path.join(BASE_DIR, 'config_override.json')
if os.path.exists(override_path):
    try:
        import json
        with open(override_path, 'r', encoding='utf-8') as f:
            overrides = json.load(f)
        if isinstance(overrides, dict):
            globals().update(overrides)
            logging.getLogger('config').info(f"Loaded config overrides from config_override.json: {list(overrides.keys())}")
    except Exception as e:
        logging.getLogger('config').error(f"Failed to load config_override.json: {e}")