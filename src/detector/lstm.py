import torch
import torch.nn as nn

from .config import Config, BASE_YOLO_MAPPING, LSTM_MAPPING

INPUT_DIM = 34
HIDDEN_DIM = 64
OUTPUT_DIM = 3

class MultiClassLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_layers=3):
        super(MultiClassLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.isLoaded = False
        self._load_model()

    def _load_model(self): # loads a model from a set location
        lstm_path = Config.LSTM_MODEL_PATH

        try:
            checkpoint = torch.load(lstm_path, map_location=self.device)
            self.load_state_dict(checkpoint)
            self.eval()
            self.isLoaded = True
            print(f"[INFO] LSTM Model correctly loaded from {lstm_path}")
        except FileNotFoundError:
            print(f"[ERROR] File not found: {lstm_path}")
        except Exception as e:
            print(f"[ERROR] Error during loading LSTM model: {e}")

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out
    
    def predict(self, sequence_tensor): # predicts action using lstm
        if not self.isLoaded:
            return -1
        else:
            sequence_tensor = sequence_tensor.to(self.device)

            with torch.no_grad():
                outputs = self(sequence_tensor)
                
                _, predicted_idx = torch.max(outputs, 1)
                class_id = predicted_idx.item()
                
            return class_id