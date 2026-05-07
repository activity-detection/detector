from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field
from dotenv import load_dotenv

import os

load_dotenv()

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

LSTM_MAPPING: dict[int, str] = {
    -1: 'error',
    0: 'normal',
    1: 'jumping_jacks',
    2: 'squat',
    3: 'fall',
    4: 'running',
}

SCENE_FLAGS: tuple[str, ...] = ('fight', 'crowd', 'lying_still')


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_MODE: str = Field(default="VIDEO")
    BASE_DIR: str = Field(default_factory=os.getcwd)
    FRAME_RATE: float = 25.
    BATCH_SIZE: int = 8
    DB_URL: str = ""
    SEQUENCE_FRAMES_GAP: int = 15

    # FPS / okna LSTM
    TARGET_FPS: float = 25.0
    LSTM_WINDOW: int = 30
    SCENE_WINDOW: int = 30
    LSTM_CONF_THRESHOLD: float = 0.6

    # Heurystyki
    LYING_ASPECT_THRESHOLD: float = 1.5
    LYING_SUSTAIN_FRAMES: int = 30
    CROWD_PERSON_THRESHOLD: int = 20
    CROWD_SUSTAIN_FRAMES: int = 15

    SOURCE_PATH: str
    ACTION_VECTORS_PATH: str
    BASE_MODEL_PATH: str
    POSE_MODEL_PATH: str
    CLIP_FOLDER: str

    RAW_LSTM_MODEL_PATH: str = Field(default="", alias="LSTM_MODEL_PATH")
    RAW_FIGHT_LSTM_MODEL_PATH: str = Field(default="", alias="FIGHT_LSTM_MODEL_PATH")
    RAW_FIGHT_SCALER_PATH: str = Field(default="", alias="FIGHT_SCALER_PATH")
    RAW_TRAIN_PROJECT_DIR: str = Field(default="", alias="TRAIN_PROJECT_DIR")

    CAMERA_USER: str
    CAMERA_PASSWORD: str
    CAMERA_IP: str
    CAMERA_PORT: str
    RTSP_PATH: str

    ROBOFLOW_API_KEY: str
    ROBOFLOW_WORKSPACE: str
    ROBOFLOW_PROJECT: str
    ROBOFLOW_DATASET_VERSION: int

    @computed_field
    @property
    def LSTM_MODEL_PATH(self) -> str:
        return os.path.join(self.BASE_DIR, self.RAW_LSTM_MODEL_PATH)

    @computed_field
    @property
    def FIGHT_LSTM_MODEL_PATH(self) -> str:
        return os.path.join(self.BASE_DIR, self.RAW_FIGHT_LSTM_MODEL_PATH)

    @computed_field
    @property
    def FIGHT_SCALER_PATH(self) -> str:
        return os.path.join(self.BASE_DIR, self.RAW_FIGHT_SCALER_PATH)

    @computed_field
    @property
    def TRAIN_PROJECT_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, self.RAW_TRAIN_PROJECT_DIR)

    def get_rtsp_url(self) -> str:
        return f"rtsp://{self.CAMERA_USER}:{self.CAMERA_PASSWORD}@{self.CAMERA_IP}:{self.CAMERA_PORT}/{self.RTSP_PATH}"


Config = Settings()  # pyright: ignore[reportCallIssue]
