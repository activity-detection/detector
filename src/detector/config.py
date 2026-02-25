import os
from dotenv import load_dotenv

load_dotenv()

BASE_YOLO_MAPPING = {
    0: 'bicycle',
    1: 'car',
    2: 'bird',
    3: 'cat',
    4: 'dog',
    5: 'handbag',
    6: 'suitcase',
    7: 'knife'
}

LICENSE_PLATE_ID = 8

LSTM_MAPPING = {
    0 : 'normal',
    1 : 'jumping_jacks',
    2 : 'squat'
}

class Config:
    MODE = os.getenv('APP_MODE', 'VIDEO').upper()

    BASE_DIR = os.getcwd()

    SOURCE_PATH = os.getenv('SOURCE_PATH')

    ACTION_VECTORS_PATH = os.getenv('ACTION_VECTORS_PATH')
    
    BASE_MODEL_PATH = os.getenv('BASE_MODEL_PATH')
    POSE_MODEL_PATH = os.getenv('POSE_MODEL_PATH')
    LSTM_MODEL_PATH = os.path.join(BASE_DIR, os.getenv('LSTM_MODEL_PATH'))

    FRAME_RATE = 25 # default

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