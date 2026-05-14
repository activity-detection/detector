from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field
import os

BASE_YOLO_MAPPING: dict[int, str] = {
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

WINDOW_SIZE = 30
TARGET_FPS = 25.0

LSTM_MAPPING: dict[int, str] = {
    -1: 'error',
    0: 'normal',
    1: 'jumping_jacks',
    2: 'squats',
    3: 'falling',
    4: 'running'
}

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_MODE: str = Field(default="VIDEO")
    BASE_DIR: str = Field(default_factory=os.getcwd)
    FRAME_RATE: float = TARGET_FPS
    BATCH_SIZE: int = 8
    BACKEND_UPLOAD_URL: str = ""
    SEQUENCE_FRAMES_GAP: int = 15 # TODO może to gdzieś przeniść

    SOURCE_PATH: str
    ACTION_VECTORS_PATH: str
    BASE_MODEL_PATH: str
    POSE_MODEL_PATH: str
    CLIP_FOLDER: str
    
    # TODO coś z tym może zrobić
    RAW_LSTM_MODEL_PATH: str = Field(default="", alias="LSTM_MODEL_PATH")

    CAMERA_USER: str
    CAMERA_PASSWORD: str
    CAMERA_IP: str
    CAMERA_PORT: str
    RTSP_PATH: str

    UPLOAD_CLIPS: bool
    SAVE_CLIPS: bool

    @computed_field
    @property
    def LSTM_MODEL_PATH(self) -> str:
        return os.path.join(self.BASE_DIR, self.RAW_LSTM_MODEL_PATH)

    def get_rtsp_url(self) -> str:
        return f"rtsp://{self.CAMERA_USER}:{self.CAMERA_PASSWORD}@{self.CAMERA_IP}:{self.CAMERA_PORT}/{self.RTSP_PATH}"

Config = Settings()  # pyright: ignore[reportCallIssue]