import cv2
from ultralytics import YOLO
from action_detector import ActionDetector
import os

# --- KONFIGURACJA ---
VIDEO_PATH = "examples/video4.mp4"
MODEL_PATH = "models/yolo11m-pose.pt"
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
# --------------------

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"BŁĄD: Nie znaleziono pliku {VIDEO_PATH}")
        return

    print("Ładowanie modelu...")
    model = YOLO(MODEL_PATH)
    
    print("Inicjalizacja detektora akcji...")
    detector = ActionDetector(buffer_seconds=2, post_event_seconds=3)
    
    cap = cv2.VideoCapture(VIDEO_PATH)

    print("Rozpoczynanie analizy. Naciśnij 'q' aby wyjść.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Koniec pliku wideo.")
            break

        # --- ZMIANA: Skalowanie do Full HD ---
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

        # 3. Inferencja YOLO (teraz działa na obrazie FullHD)
        results = model(frame, verbose=False)
        result = results[0]

        # 4. Przetwarzanie w ActionDetector
        saved_file = detector.process(frame, result)

        # 5. --- WIZUALIZACJA ---
        annotated_frame = result.plot(boxes=False) 

        # B. Status nagrywania
        if detector.is_recording_event:
            # Pozycjonowanie względem nowej rozdzielczości
            cv2.circle(annotated_frame, (TARGET_WIDTH - 50, 50), 20, (0, 0, 255), -1)
            cv2.putText(annotated_frame, "REC", (TARGET_WIDTH - 90, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            frames_left = detector.frames_left_to_record
            cv2.putText(annotated_frame, f"Left: {frames_left}", (TARGET_WIDTH - 150, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        if saved_file:
            print(f"-> Zapisano na dysku: {saved_file}")

        # Wyświetlanie (możesz dodać cv2.WINDOW_NORMAL jeśli ekran jest mniejszy niż 1080p)
        cv2.namedWindow("Action Detection Demo", cv2.WINDOW_NORMAL)
        cv2.imshow("Action Detection Demo", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()