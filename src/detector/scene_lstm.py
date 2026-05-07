"""
LSTM scene-level dla detekcji bójek.

Wejście: 10D feature vector per klatka (z scene_features.SceneFeatureExtractor).
Bufor 30 klatek (SCENE_WINDOW @ TARGET_FPS=25 → 1.2s) → forward → softmax → label.

Wymaga StandardScaler z treningu (joblib pickle), żeby normalizować cechy
identycznie jak podczas treningu.
"""

import joblib
import torch
import torch.nn as nn

from src.detector.config import Config
from src.detector import logger


class FightLSTM(nn.Module):
    INPUT_DIM = 10
    HIDDEN_DIM = 64
    OUTPUT_DIM = 2
    NUM_LAYERS = 2
    DROPOUT = 0.3

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            self.INPUT_DIM,
            self.HIDDEN_DIM,
            self.NUM_LAYERS,
            batch_first=True,
            dropout=self.DROPOUT,
        )
        self.fc = nn.Linear(self.HIDDEN_DIM, self.OUTPUT_DIM)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.scaler = None
        self.is_loaded = False
        self.conf_threshold = Config.LSTM_CONF_THRESHOLD

    def load_model(self) -> None:
        model_path = Config.FIGHT_LSTM_MODEL_PATH
        scaler_path = Config.FIGHT_SCALER_PATH

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.load_state_dict(checkpoint)
        self.eval()
        self.scaler = joblib.load(scaler_path)
        self.is_loaded = True
        logger.info(f"FightLSTM loaded from {model_path} (scaler: {scaler_path})")

    def forward(self, x):  # type: ignore[override]
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def predict(self, sequence_2d) -> int:
        """
        sequence_2d: np.ndarray (window, 10) — surowe cechy. Skalowanie wewnątrz.
        Zwraca: 1 jeśli bójka (>= conf_threshold), 0 inaczej, -1 gdy model niezaładowany.
        """
        if not self.is_loaded:
            return -1

        scaled = self.scaler.transform(sequence_2d).astype("float32")
        tensor = torch.tensor(scaled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(1)
            cls = idx.item()
            if cls == 1 and conf.item() < self.conf_threshold:
                cls = 0
        return int(cls)
