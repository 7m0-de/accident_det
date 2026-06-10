import os
import shutil
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import config
import logic
from utils import setup_colab_tunnel, cleanup_old_files
app = FastAPI(title='AI Accident Detection System — Final v1.0', description='AI Traffic Accident Detection System — YOLOv10 × BoT-SORT × Bayesian × GMC × Kalman', version='1.0.0', docs_url='/docs', redoc_url='/redoc')
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

def process_video_thread(file_path: str, run_id: str, run_dir: str):
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
                print(f'[SERVER] Failed to load events:{e}')
        with state_lock:
            state['status'] = 'complete'
            state['progress'] = 100
            state['video_url'] = f'static/runs/{run_id}/{config.OUT_VIDEO}'
            state['report_url'] = f'static/runs/{run_id}/{config.OUT_REPORT}'
            state['csv_url'] = f'static/runs/{run_id}/{config.OUT_CSV}'
            state['summary'] = summary
            state['events'] = events
            state['transcoding_failed'] = summary.get('transcoding_failed', False)
    except Exception as e:
        import traceback
        print(f'[SERVER] Processing error:{e}')
        traceback.print_exc()
        with state_lock:
            state['status'] = 'error'
            state['error_message'] = str(e)
    finally:
        processing_lock.release()
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f'[SERVER] Failed to delete temporary file:{e}')
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
    raise HTTPException(status_code=404, detail='index.html  .')

@app.post('/upload', summary='')
async def upload_video(file: UploadFile=File(...)):
    """.
     : MP4, AVI, MOV, MKV"""
    if processing_lock.locked():
        return JSONResponse(status_code=400, content={'status': 'error', 'message': '.    .'})
    try:
        cleanup_old_files(config.UPLOAD_DIR)
        cleanup_old_files(config.RUNS_DIR)
    except Exception as e:
        print(f'[SERVER] Cleanup failed:{e}')
    file_ext = os.path.splitext(file.filename or '')[1].lower()
    if file_ext not in ['.mp4', '.avi', '.mov', '.mkv']:
        return JSONResponse(status_code=400, content={'status': 'error', 'message': '. : MP4, AVI, MOV, MKV'})
    import time
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(config.RUNS_DIR, run_id)
    try:
        os.makedirs(run_dir, exist_ok=True)
    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': f':{e}'})
    import aiofiles
    upload_path = os.path.join(config.UPLOAD_DIR, f'input_{run_id}{file_ext}')
    try:
        async with aiofiles.open(upload_path, 'wb') as out_file:
            while (content := (await file.read(1024 * 1024))):
                await out_file.write(content)
    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': f':{e}'})
    with state_lock:
        state.update({'status': 'processing', 'progress': 0, 'frame': 0, 'total': 0, 'fps': 0.0, 'error_message': '', 'video_url': '', 'report_url': '', 'csv_url': '', 'summary': {}, 'events': [], 'transcoding_failed': False})
    thread = threading.Thread(target=process_video_thread, args=(upload_path, run_id, run_dir), daemon=True)
    thread.start()
    return {'status': 'started', 'run_id': run_id}

@app.get('/progress', summary='')
def get_progress():
    with state_lock:
        return JSONResponse(content=dict(state))

@app.get('/health', summary='')
def health_check():
    return {'status': 'ok', 'version': '1.0.0'}
if __name__ == '__main__':
    import uvicorn
    import argparse
    try:
        cleanup_old_files(config.UPLOAD_DIR)
        cleanup_old_files(config.RUNS_DIR)
    except Exception as e:
        print(f'[BOOT] Initial cleanup failed:{e}')
    parser = argparse.ArgumentParser(description='AI Accident Detection Server — Final v1.0')
    parser.add_argument('--host', type=str, default=config.HOST, help='')
    parser.add_argument('--port', type=int, default=config.PORT, help='')
    parser.add_argument('--tunnel', action='store_true', help='pyngrok')
    parser.add_argument('--token', type=str, default=None, help='ngrok')
    args = parser.parse_args()
    if args.tunnel:
        setup_colab_tunnel(args.port, args.token)
    print(f"{'=' * 60}")
    print(f'AI Accident Detection — Final v1.0')
    print(f'Server URL: http://{args.host}:{args.port}')
    print(f'Swagger UI: http://{args.host}:{args.port}/docs')
    print(f"{'=' * 60}")
    uvicorn.run(app, host=args.host, port=args.port)