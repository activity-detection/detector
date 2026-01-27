import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODE = os.getenv('APP_MODE', 'VIDEO').upper()
    
    SOURCE_PATH = os.getenv('SOURCE_PATH')
    
    MODEL_PATH = os.getenv('MODEL_PATH')
    POSE_MODEL_PATH = os.getenv('POSE_MODEL_PATH')
    PLATES_MODEL_PATH = os.getenv('PLATES_MODEL_PATH')
    CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', '0.5'))
    
    SHOW_VIDEO = os.getenv('SHOW_VIDEO', 'True').lower() == 'true'
    VIDEO_WIDTH = int(os.getenv('VIDEO_WIDTH')) if os.getenv('VIDEO_WIDTH') else None
    VIDEO_HEIGHT = int(os.getenv('VIDEO_HEIGHT')) if os.getenv('VIDEO_HEIGHT') else None
    
    RECORD = os.getenv('RECORD_RESULTS', 'False').lower() == 'true'
    RECORD_NAME = os.getenv('RECORD_NAME')

    PIXELATION_SCALE = float(os.getenv('PIXELATION_SCALE'))

    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '4'))
    CLIP_FOLDER = os.getenv('CLIP_FOLDER')

    CAM_USER = os.getenv('CAMERA_USER')
    CAM_PASS = os.getenv('CAMERA_PASSWORD')
    CAM_IP = os.getenv('CAMERA_IP')
    CAM_PORT = os.getenv('CAMERA_PORT')
    CAM_PATH = os.getenv('RTSP_PATH')

    @staticmethod
    def get_rtsp_url():
        return f"rtsp://{Config.CAM_USER}:{Config.CAM_PASS}@{Config.CAM_IP}:{Config.CAM_PORT}/{Config.CAM_PATH}"