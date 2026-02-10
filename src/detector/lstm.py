import torch.nn as nn

INPUT_DIM = 34
HIDDEN_DIM = 64
OUTPUT_DIM = 3

CLASS_NAMES = {
            0: "normal", 
            1: "jumping_jacks",
            2: "squat"
        }

class MultiClassLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_layers=3):
        super(MultiClassLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out