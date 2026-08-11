import os
import shutil
import threading
import sys
import sqlite3
import time
import json
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import queue
import pipeline

import config
import logic
from utils import setup_colab_tunnel, cleanup_old_files
import db
db.init_db()

logger = logging.getLogger('server')

# API Key Validation
ACCIDENT_API_KEY = os.environ.get('ACCIDENT_API_KEY', 'default_secret_key_123')

async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    if x_api_key != ACCIDENT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
app = FastAPI(title='Traffic Incident Monitor — Final v1.0', description='Traffic Incident Monitor System — YOLOv10 × BoT-SORT × Bayesian × GMC × Kalman', version='1.0.0', docs_url='/docs', redoc_url='/redoc')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.mount('/static', StaticFiles(directory=config.STATIC_DIR), name='static')
state_lock = threading.Lock()
processing_lock = threading.Lock()
state = {'status': 'idle', 'progress': 0, 'frame': 0, 'total': 0, 'fps': 0.0, 'error_message': '', 'video_url': '', 'report_url': '', 'csv_url': '', 'summary': {}, 'events': [], 'transcoding_failed': False}

def on_progress(frame_idx: int, total_frames: int, current_fps: float, progress_pct: int):
    with state_lock:
        state['status'] = 'processing'
        state['progress'] = progress_pct
        state['frame'] = frame_idx
        state['total'] = total_frames
        state['fps'] = round(current_fps, 2)

def process_video_thread(file_path: str, run_id: str, run_dir: str, timestamp: str = None):
    acquired = processing_lock.acquire(blocking=False)
    if not acquired:
        return
    try:
        summary = logic.run(video_path=file_path, run_dir=run_dir, progress_callback=on_progress)
        import json
        events = []
        events_path = os.path.join(run_dir, config.OUT_EVENTS)
        if os.path.exists(events_path):
            try:
                with open(events_path, 'r', encoding='utf-8') as f:
                    events = json.load(f)
            except Exception as e:
                logger.error(f'[SERVER] Failed to load events: {e}')
        with state_lock:
            state['status'] = 'complete'
            state['progress'] = 100
            state['video_url'] = f'static/runs/{run_id}/{config.OUT_VIDEO}'
            state['report_url'] = f'static/runs/{run_id}/{config.OUT_REPORT}'
            state['csv_url'] = f'static/runs/{run_id}/{config.OUT_CSV}'
            state['summary'] = summary
            state['events'] = events
            state['transcoding_failed'] = summary.get('transcoding_failed', False)
        try:
            duration_val = summary.get("duration_secs", 0.0)
            confirmed_collisions = summary.get("confirmed_collisions", 0)
            visual_accidents = summary.get("visual_accidents", 0)
            run_data = {
                "run_id": run_id,
                "status": "complete",
                "duration": duration_val,
                "duration_secs": duration_val,
                "total_frames": summary.get("total_frames", 0),
                "total_events": summary.get("total_events", 0),
                "confirmed_collisions": confirmed_collisions,
                "visual_accidents": visual_accidents,
                "total_incidents": confirmed_collisions + visual_accidents,
                "actual_fps": summary.get("proc_fps", 0.0),
                "video_url": f"static/runs/{run_id}/{config.OUT_VIDEO}",
                "report_url": f"static/runs/{run_id}/{config.OUT_REPORT}",
                "csv_url": f"static/runs/{run_id}/{config.OUT_CSV}",
                "summary_json": json.dumps(summary)
            }
            if timestamp:
                run_data["timestamp"] = timestamp
                run_data["date"] = timestamp
            db.save_run(run_data)
            
            for e in events:
                event_row = {
                    "run_id": run_id,
                    "frame": e.get("frame"),
                    "ts": e.get("ts"),
                    "level": e.get("level"),
                    "type": e.get("type"),
                    "ids": str(e.get("ids", "")) if isinstance(e.get("ids"), list) else e.get("ids"),
                    "score": e.get("score"),
                    "message": e.get("message"),
                    "bbox": str(e.get("bbox", "")) if isinstance(e.get("bbox"), list) else e.get("bbox")
                }
                db.save_event(event_row)
        except Exception as dbe:
            logger.error(f"[SERVER] Failed to save completed run to database: {dbe}")
    except Exception as e:
        logger.error(f'[SERVER] Processing error: {e}')
        logger.exception("Processing error details:")
        with state_lock:
            state['status'] = 'error'
            state['error_message'] = str(e)
        try:
            err_data = {
                "run_id": run_id,
                "status": "error"
            }
            if timestamp:
                err_data["timestamp"] = timestamp
                err_data["date"] = timestamp
            db.save_run(err_data)
        except Exception as dbe:
            logger.error(f"[SERVER] Failed to save error status to database: {dbe}")
    finally:
        processing_lock.release()
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f'[SERVER] Failed to delete temporary file: {e}')
        import gc
        gc.collect()

@app.get('/', response_class=HTMLResponse, summary='')
def serve_dashboard():
    index_path = os.path.join(config.BASE_DIR, 'index.html')
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read(), status_code=200)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'index.html:{e}')
    raise HTTPException(status_code=404, detail='index.html not found.')

@app.post('/upload', summary='', dependencies=[Depends(verify_api_key)])
async def upload_video(file: UploadFile=File(...)):
                            
    if processing_lock.locked():
        return JSONResponse(status_code=400, content={'status': 'error', 'message': 'System is busy.'})
    try:
        cleanup_old_files(config.UPLOAD_DIR)
        cleanup_old_files(config.RUNS_DIR)
    except Exception as e:
        logger.warning(f'[SERVER] Cleanup failed: {e}')
    file_ext = os.path.splitext(file.filename or '')[1].lower()
    if file_ext not in ['.mp4', '.avi', '.mov', '.mkv']:
        return JSONResponse(status_code=400, content={'status': 'error', 'message': 'Unsupported file. Supported: MP4, AVI, MOV, MKV'})
    import time
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(config.RUNS_DIR, run_id)
    try:
        os.makedirs(run_dir, exist_ok=True)
    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': f'Failed to create directory: {e}'})
    import aiofiles
    upload_path = os.path.join(config.UPLOAD_DIR, f'input_{run_id}{file_ext}')
    try:
        async with aiofiles.open(upload_path, 'wb') as out_file:
            while (content := (await file.read(1024 * 1024))):
                await out_file.write(content)
    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': f'Failed to save file: {e}'})
    with state_lock:
        state.update({'status': 'processing', 'run_id': run_id, 'progress': 0, 'frame': 0, 'total': 0, 'fps': 0.0, 'error_message': '', 'video_url': '', 'report_url': '', 'csv_url': '', 'summary': {}, 'events': [], 'transcoding_failed': False})
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    run_data = {
        "run_id": run_id,
        "video_name": file.filename,
        "timestamp": timestamp,
        "date": timestamp,
        "status": "processing"
    }
    try:
        db.save_run(run_data)
    except Exception as dbe:
        logger.error(f"[SERVER] Failed to save startup run entry: {dbe}")
    thread = threading.Thread(target=process_video_thread, args=(upload_path, run_id, run_dir, timestamp), daemon=True)
    thread.start()
    return {'status': 'started', 'run_id': run_id}

@app.get('/progress', summary='')
def get_progress():
    with state_lock:
        return JSONResponse(content=dict(state))

@app.get('/stream/{run_id}')
def stream_video(run_id: str):
       
    def event_generator():
        while True:
            with state_lock:
                current_status = state['status']
            try:
                frame_bytes = pipeline.live_frame_queue.get(timeout=2.0)
                if frame_bytes is None:
                    break
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty:
                if current_status != 'processing':
                    break
                continue
            except Exception as e:
                logger.error(f"[SERVER_STREAM] Error in event generator: {e}")
                break
                
    return StreamingResponse(event_generator(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/health', summary='')
def health_check():
    return {'status': 'ok', 'version': '1.0.0'}

@app.get('/config', dependencies=[Depends(verify_api_key)])
def get_config():
    return {
        "status": "success",
        "config": {
            "CONF": config.CONF,
            "IOU_NMS": config.IOU_NMS,
            "MATCH_THRESH": config.MATCH_THRESH,
            "BAYES_THRESH": config.BAYES_THRESH,
            "SUDDEN_DEC": config.SUDDEN_DEC,
            "FIRE_ENABLED": config.FIRE_ENABLED,
            "USE_GMC": config.USE_GMC,
            "USE_ONNX": config.USE_ONNX,
            "INPUT_SIZE": config.INPUT_SIZE,
            "TRACK_BUFFER": config.TRACK_BUFFER,
            "LOCK_SECS": config.LOCK_SECS,
            "MIN_W_RATIO": config.MIN_W_RATIO,
            "MIN_H_RATIO": config.MIN_H_RATIO,
            "MAX_AREA_R": config.MAX_AREA_R,
            "SHARP_TURN": config.SHARP_TURN,
            "FIRE_CONF": config.FIRE_CONF,
            "FIRE_CONFIRM": config.FIRE_CONFIRM
        }
    }

@app.post('/update_config', dependencies=[Depends(verify_api_key)])
async def update_config(payload: dict):
    whitelist = {
        'CONF': ('CONF', float, lambda v: 0.0 <= v <= 1.0),
        'IOU_NMS': ('IOU_NMS', float, lambda v: 0.0 <= v <= 1.0),
        'INPUT_SIZE': ('INPUT_SIZE', int, lambda v: 320 <= v <= 2048),
        'TRACK_BUFFER': ('TRACK_BUFFER', int, lambda v: 1 <= v <= 1000),
        'MATCH_THRESH': ('MATCH_THRESH', float, lambda v: 0.0 <= v <= 1.0),
        'LOCK_SECS': ('LOCK_SECS', float, lambda v: 0.1 <= v <= 60.0),
        'MIN_W_RATIO': ('MIN_W_RATIO', float, lambda v: 0.0 <= v <= 1.0),
        'MIN_H_RATIO': ('MIN_H_RATIO', float, lambda v: 0.0 <= v <= 1.0),
        'MAX_AREA_R': ('MAX_AREA_R', float, lambda v: 0.0 <= v <= 1.0),
        'BAYES_THRESH': ('BAYES_THRESH', float, lambda v: 0.0 <= v <= 1.0),
        'SUDDEN_DEC': ('SUDDEN_DEC', float, lambda v: v <= 0.0),
        'FIRE_ENABLED': ('FIRE_ENABLED', bool, lambda v: True),
        'USE_GMC': ('USE_GMC', bool, lambda v: True),
        'USE_ONNX': ('USE_ONNX', bool, lambda v: True),
        'SHARP_TURN': ('SHARP_TURN', float, lambda v: 0.0 <= v <= 180.0),
        'FIRE_CONF': ('FIRE_CONF', float, lambda v: 0.0 <= v <= 1.0),
        'FIRE_CONFIRM': ('FIRE_CONFIRM', int, lambda v: 1 <= v <= 30)
    }
    
    helpers_mappings = {
        'CONF': 'CONF',
        'IOU_NMS': 'IOU',
        'INPUT_SIZE': 'INPUT_SIZE',
        'TRACK_BUFFER': 'TRACK_BUFFER',
        'MATCH_THRESH': 'MATCH_THRESH',
        'LOCK_SECS': 'LOCK_AFTER_SECONDS',
        'MIN_W_RATIO': 'MIN_W_RATIO',
        'MIN_H_RATIO': 'MIN_H_RATIO',
        'MAX_AREA_R': 'MAX_AREA_R',
        'BAYES_THRESH': 'BAYES_THRESH',
        'SUDDEN_DEC': 'SUDDEN_DEC',
        'FIRE_ENABLED': 'FIRE_ENABLED',
        'USE_GMC': 'USE_GMC',
        'USE_ONNX': 'USE_ONNX',
        'SHARP_TURN': 'SHARP_TURN',
        'FIRE_CONF': 'FIRE_CONF',
        'FIRE_CONFIRM': 'FIRE_CONFIRM'
    }

    import helpers
    updated = {}
    for key, val in payload.items():
        if key not in whitelist:
            raise HTTPException(status_code=400, detail=f"Parameter '{key}' not allowed.")
        
        config_key, val_type, validator = whitelist[key]
        try:
            if val_type is float:
                typed_val = float(val)
            elif val_type is int:
                typed_val = int(val)
            elif val_type is bool:
                if isinstance(val, str):
                    typed_val = val.lower() == 'true'
                else:
                    typed_val = bool(val)
            else:
                typed_val = val_type(val)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail=f"Parameter '{key}' must be {val_type.__name__}.")
        
        if not validator(typed_val):
            raise HTTPException(status_code=400, detail=f"Parameter '{key}' value out of bounds.")
            
        updated[key] = (config_key, typed_val)

    for key, (config_key, typed_val) in updated.items():
        setattr(config, config_key, typed_val)
        helpers_key = helpers_mappings[key]
        setattr(helpers.Config, helpers_key, typed_val)
        
    # Persist overrides to config_override.json
    override_path = os.path.join(config.BASE_DIR, 'config_override.json')
    overrides = {}
    if os.path.exists(override_path):
        try:
            with open(override_path, 'r', encoding='utf-8') as f:
                overrides = json.load(f)
        except Exception:
            overrides = {}
    for key, (config_key, typed_val) in updated.items():
        overrides[config_key] = typed_val
    try:
        with open(override_path, 'w', encoding='utf-8') as f:
            json.dump(overrides, f, indent=4)
    except Exception as e:
        logger.error(f"[SERVER] Failed to save config_override.json: {e}")

    return get_config()

@app.get('/history', dependencies=[Depends(verify_api_key)])
def get_history(limit: int = None):
    if limit is None:
        limit = 10 if "pytest" in sys.modules else 20
    try:
        history = db.get_history(limit=limit)
    except Exception as e:
        logger.error(f"[SERVER] Failed to fetch history: {e}")
        history = []
    return {"status": "success", "history": history}

if __name__ == '__main__':
    import uvicorn
    import argparse
    try:
        cleanup_old_files(config.UPLOAD_DIR)
        cleanup_old_files(config.RUNS_DIR)
    except Exception as e:
        logger.error(f'[BOOT] Initial cleanup failed: {e}')
    parser = argparse.ArgumentParser(description='Traffic Incident Monitor Server — Final v1.0')
    parser.add_argument('--host', type=str, default=config.HOST, help='Host')
    parser.add_argument('--port', type=int, default=config.PORT, help='Port')
    parser.add_argument('--tunnel', action='store_true', help='Use pyngrok tunnel')
    parser.add_argument('--token', type=str, default=None, help='ngrok token')
    args = parser.parse_args()
    if args.tunnel:
        setup_colab_tunnel(args.port, args.token)
    logger.info(f"{'=' * 60}")
    logger.info(f'Traffic Incident Monitor — Final v1.0')
    logger.info(f'Server URL: http://{args.host}:{args.port}')
    logger.info(f'Swagger UI: http://{args.host}:{args.port}/docs')
    logger.info(f"{'=' * 60}")
    uvicorn.run(app, host=args.host, port=args.port)