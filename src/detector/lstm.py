# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportUnknownParameterType=false

import torch.nn as nn
import torch

from src.detector.config import Config, LSTM_MAPPING
from src.detector import logger

INPUT_DIM = 34
HIDDEN_DIM = 64
OUTPUT_DIM = len(LSTM_MAPPING) - 1  # bez 'error' (-1)


class MultiClassLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_layers=3):
        super(MultiClassLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.isLoaded = False
        self.conf_threshold = Config.LSTM_CONF_THRESHOLD

    def load_model(self):
        lstm_path = Config.LSTM_MODEL_PATH

        checkpoint = torch.load(lstm_path, map_location=self.device, weights_only=True)
        self.load_state_dict(checkpoint)
        self.eval()
        self.isLoaded = True
        logger.info(f"LSTM Model correctly loaded from {lstm_path}")

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def predict(self, sequence_tensor):
        if not self.isLoaded:
            return -1

        sequence_tensor = sequence_tensor.to(self.device)

        with torch.no_grad():
            outputs = self(sequence_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, idx = probs.max(1)
            class_id = int(idx.item())
            if class_id != 0 and conf.item() < self.conf_threshold:
                class_id = 0
        return class_id
