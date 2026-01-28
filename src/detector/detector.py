import torch
import numpy as np
from collections import defaultdict, deque
import cv2

from .config import Config
from .action_recorder import ActionRecorder

class Detector:
    def __init__(self, models, anonymizer, buffer_seconds=5, post_event_seconds=3, fps=30):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.track_history = defaultdict(lambda: deque(maxglen=30))
        self.recorders = {}
        self.dangers = {1, 2}
        
        self.CLASS_NAMES = {
            0: "inne", 
            1: "przysiad",
            2: "pajacyk"
        }

        self.detection_model = models["detection"]["model"]
        self.pose_model = models["pose"]["model"]
        self.plates_model = models["plates"]["model"]


        self.anonymizer = anonymizer
        self.buffer_seconds = buffer_seconds
        self.post_event_seconds = post_event_seconds
        self.fps = fps
        
        lstm_path = Config.LSTM_MODEL_PATH
        
        try:
            checkpoint = torch.load(lstm_path, map_location=self.device)
            self.detection_model.load_state_dict(checkpoint)
            self.detection_model.to(self.device)
            self.detection_model.eval()
            print(f"[INFO] Model LSTM załadowany pomyślnie z {lstm_path}")
        except FileNotFoundError:
            print(f"[ERROR] Nie znaleziono pliku modelu: {lstm_path}")
            self.detection_model = None
        except Exception as e:
            print(f"[ERROR] Błąd ładowania modelu LSTM: {e}")
            self.detection_model = None

    def predict_action(self, sequence_tensor):
        """
        Wykonuje predykcję akcji na podstawie sekwencji 30 klatek.
        
        Args:
            sequence_tensor (torch.Tensor): Tensor o kształcie (1, 30, 34)
                                            Batch=1, Seq=30, Features=34
        Returns:
            str: Nazwa wykrytej akcji (np. "Boksowanie")
        """

        if self.detection_model is None:
            return "LSTM się nie załadował!"

        # Przeniesienie danych na to samo urządzenie co model
        sequence_tensor = sequence_tensor.to(self.device)

        # Sekcja bez liczenia gradientów (szybciej i lżej dla pamięci)
        with torch.no_grad():
            # 1. Inferencja (Forward pass)
            outputs = self.detection_model(sequence_tensor) # Wynik np. [-2.5, 4.1, 0.2]
            
            # 2. Wybór klasy o najwyższym wyniku
            _, predicted_idx = torch.max(outputs, 1)
            class_id = predicted_idx.item() # Zamiana tensora na zwykłą liczbę (int)
            
            # Opcjonalnie: Progowanie pewności (Confidence Threshold)
            # probs = torch.softmax(outputs, dim=1)
            # confidence = probs[0][class_id].item()
            # if confidence < 0.6: return "Unknown"

        # 3. Mapowanie ID na nazwę (np. 0 -> "Boksowanie")
        return class_id, self.CLASS_NAMES.get(class_id, f"Unknown ({class_id})")
    
    def process_batch_multiperson(self, batch_frames):

        results = self.pose_model.track(batch_frames, persist=True, verbose=False)

        processed_frames = []

        # Dla każdej klatki i wyników dla niej w batchu
        for _, (frame, result) in enumerate(zip(batch_frames, results)):
            
            # ID osób obecnych w klatce
            present_ids = set()

            if result.boxes is not None and result.boxes.id is not None:
                # Pobieramy ID i Keypoints z wyników
                # id to tensor, zamieniamy na inty
                track_ids = result.boxes.id.int().cpu().tolist()

                keypoints = result.keypoints.xy.cpu().numpy() # (N, 17, 2)

                # Iterujemy po KAŻDEJ wykrytej osobie w tej klatce
                for person_idx, track_id in enumerate(track_ids):            

                    # Dodajemy ID do listy obecnych
                    present_ids.add(track_id)

                    kps = keypoints[person_idx] # (17, 2)

                    # Musisz pobrać wymiary obrazu, np. z pierwszej klatki batcha
                    h, w = batch_frames[0].shape[:2]

                    # Kopiujemy, żeby nie psuć oryginału (jeśli potrzebny do rysowania)
                    kps_norm = kps.copy()
                    kps_norm[:, 0] /= w  # X dzielimy przez szerokość
                    kps_norm[:, 1] /= h  # Y dzielimy przez wysokość
                    flat_kps = kps_norm.flatten() # (34, )             

                    self.track_history[track_id].append(flat_kps)

                    # Inicjalizacja rekordera jeśli nowy
                    if track_id not in self.recorders:
                        self.recorders[track_id] = ActionRecorder(
                            self.buffer_seconds, self.post_event_seconds, self.fps
                        )
                    recorder = self.recorders[track_id]

                    action_id = 0
                    action_label = "inne"

                    # Czy osoba dla id ma historię 30 klatek
                    if len(self.track_history[track_id]) == 30:
                        # --- TUTAJ URUCHAMIASZ LSTM DLA TEJ OSOBY ---     

                        # Przygotowanie danych (1, 30, 34)
                        sequence = np.array(self.track_history[track_id])
                        input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device) 

                        action_id, action_label = self.predict_action(input_tensor)

                    if self.recorders:
                        plates_results = self.plates_model(frame, verbose=False)[0]

                        self.anonymizer.anonymize(frame, result, "pose")
                        self.anonymizer.anonymize(frame, plates_results, "box")

                    # Logika nagrywania DLA WIDOCZNYCH
                    if action_id in self.dangers:
                        recorder.process(frame, interest=True)
                    else:
                        saved_file = recorder.process(frame, interest=False)
                        if saved_file:
                            print(f"[INFO] Zapisano klip dla ID {track_id}: {saved_file}")


                    # Wizualizacja (np. wypisanie akcji nad głową tej osoby)
                    box = result.boxes.xyxy[person_idx].cpu().numpy()
                    x1, y1, _, _ = map(int, box)
                    cv2.putText(frame, f"ID {track_id}: {action_label}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
            # Kopiujemy klucze (list(self.recorders.keys())), bo możemy usuwać elementy w pętli
            for r_id in list(self.recorders.keys()):
                if r_id not in present_ids:
                    recorder = self.recorders[r_id]
                    
                    # Wywołujemy process z interest=False.
                    # To spowoduje dekrementację frames_left_to_record w ActionRecorderze.
                    saved_file = recorder.process(frame, interest=False)
                    
                    if saved_file:
                        print(f"[INFO] Zapisano klip (po zniknięciu) dla ID {r_id}: {saved_file}")
                        # Skoro plik zapisany, to resetujemy stan rekordera. 
                        # Można go usunąć, jeśli nie chcemy trzymać historii bufora dla nieobecnych.
                        del self.recorders[r_id]
                        # Warto też wyczyścić historię LSTM
                        if r_id in self.track_history:
                            del self.track_history[r_id]

            processed_frames.append(frame)

        return processed_frames
