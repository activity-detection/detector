from ultralytics import YOLO
import os
from detector.config import Config

PROJECT_DIR = Config.TRAIN_PROJECT_DIR

MODEL_PATH = Config.TRAIN_MODEL_PATH
RUN_NAME = Config.TRAIN_RUN_NAME
DATA_YAML = Config.TRAIN_DATA_YAML
EPOCHS = Config.TRAIN_EPOCHS
IMG_SIZE = Config.TRAIN_IMG_SIZE
DEVICE = Config.TRAIN_DEVICE
BATCH = Config.TRAIN_BATCH

def train():
    if not os.path.exists(MODEL_PATH):
        print(f"BŁĄD: Nie znaleziono modelu: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        device=DEVICE,
    )
    
    print(f"\nTrening zakończony. Wyniki zapisano w:\n{os.path.join(PROJECT_DIR, RUN_NAME)}")

if __name__ == "__main__":
    train()