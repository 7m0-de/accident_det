import os
import shutil
import subprocess
import sys
import time
import logging

logger = logging.getLogger(__name__)

def check_ffmpeg() -> bool:
    if shutil.which('ffmpeg') is not None:
        return True
    try:
        import imageio_ffmpeg
        return os.path.exists(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        return False

def get_ffmpeg_exe() -> str:
    path = shutil.which('ffmpeg')
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'

def transcode_for_web(input_path: str, output_path: str) -> bool:
    if not os.path.exists(input_path):
        return False
    ffmpeg_exe = get_ffmpeg_exe()
    cmd = [ffmpeg_exe, '-y', '-i', input_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-movflags', '+faststart', '-an', output_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info(f'[TRANSCODE] Transcoding completed: {os.path.basename(output_path)}')
            return True
        else:
            logger.error(f'[TRANSCODE] FFmpeg error: {result.stderr[-300:]}')
            return False
    except subprocess.TimeoutExpired:
        logger.error('[TRANSCODE] Timeout expired during FFmpeg transcoding (300s).')
        return False
    except Exception as e:
        logger.error(f'[TRANSCODE] Error during transcoding: {e}')
        return False

def setup_colab_tunnel(port: int, authtoken: str=None) -> str:
    try:
        try:
            from pyngrok import ngrok
        except ImportError:
            logger.warning('[TUNNEL] pyngrok is not installed. Please install it with: pip install pyngrok>=7.2.0')
            return None
        token = authtoken or os.environ.get('NGROK_AUTHTOKEN')
        if token:
            ngrok.set_auth_token(token)
        tunnel = ngrok.connect(port)
        url = tunnel.public_url
        logger.info('=' * 60)
        logger.info(f'Public tunnel active: {url}')
        logger.info('Open this link in your browser to access the interface.')
        logger.info('=' * 60)
        return url
    except Exception as e:
        logger.error(f'[TUNNEL] Error setting up tunnel: {e}')
        return None

def cleanup_old_files(directory: str, max_age_seconds: int=3600):
    if not os.path.exists(directory):
        return
    now = time.time()
    deleted = 0
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        try:
            age = now - os.path.getmtime(item_path)
            if age > max_age_seconds:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
                deleted += 1
        except Exception as e:
            logger.error(f'[CLEANUP] Failed to delete {item}: {e}')
    if deleted > 0:
        logger.info(f'[CLEANUP] Cleaned up {deleted} items from {os.path.basename(directory)}.')