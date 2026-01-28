import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    MODE = os.getenv('APP_MODE', 'VIDEO').upper()
    
    BASE_DIR = os.getcwd()

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
    
    TRAIN_PROJECT_DIR = os.path.join(BASE_DIR, os.getenv('TRAIN_PROJECT_DIR'))

    TRAIN_MODEL_PATH = os.path.join(BASE_DIR, os.getenv('TRAIN_MODEL_PATH'))
    TRAIN_RUN_NAME = os.getenv('TRAIN_RUN_NAME')
    TRAIN_DATA_YAML = os.getenv('TRAIN_DATA_YAML')
    TRAIN_EPOCHS = int(os.getenv('TRAIN_EPOCHS'))
    TRAIN_IMG_SIZE = int(os.getenv('TRAIN_IMG_SIZE'))
    TRAIN_DEVICE = os.getenv('TRAIN_DEVICE')
    TRAIN_BATCH = int(os.getenv('TRAIN_BATCH'))

    LSTM_MODEL_PATH = os.path.join(BASE_DIR, os.getenv('LSTM_MODEL_PATH'))