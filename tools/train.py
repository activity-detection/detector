from ultralytics import YOLO
import os

BASE_DIR = os.getcwd()

MODEL_DIR = os.path.join(BASE_DIR, "models", "saved-models")
PROJECT_DIR = os.path.join(BASE_DIR, "models", "runs")

MODEL_PATH = os.path.join(MODEL_DIR, "yolo26s.pt")
RUN_NAME = "license_plate_v11_1"
DATA_YAML = "data/License Plate Recognition.v11i.yolo26/data.yaml"
EPOCHS = 5
IMG_SIZE = 640
DEVICE = "0"
BATCH = 20

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