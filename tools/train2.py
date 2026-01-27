import argparse
from ultralytics import YOLO

def train(args):
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        project=args.project,
        exist_ok=args.exist_ok,
        device=args.device
    )

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Trenowanie modelu YOLO")
    arg_parser.add_argument("--model", type=str, required=True, help="Ścieżka do pliku modelu YOLO")
    arg_parser.add_argument("--data", type=str, required=True, help="Ścieżka do pliku danych YAML")
    arg_parser.add_argument("--epochs", type=int, default=100, help="Liczba epok treningu")
    arg_parser.add_argument("--batch-size", type=int, default=-1, help="Rozmiar partii wsadowej")
    arg_parser.add_argument("--img-size", type=int, default=640, help="Rozmiar obrazu wejściowego")
    arg_parser.add_argument("--project", type=str, default="runs/train", help="Katalog projektu do zapisu wyników")
    arg_parser.add_argument("--exist-ok", type=bool, default=False, help="Nadpisz istniejący katalog projektu")
    arg_parser.add_argument("--device", type=str, default="0", help="Urządzenie do treningu (np. '0' dla GPU 0, 'cpu' dla CPU)")
    
    args = arg_parser.parse_args()
    train(args)