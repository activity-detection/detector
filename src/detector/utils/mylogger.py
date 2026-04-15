import logging.config
import pathlib
import json

def setup_logging():
    # 1. Określamy główny folder projektu
    current_file = pathlib.Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent
    
    # 2. TWORZYMY FOLDER LOGS (Jeśli nie istnieje)
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Wczytujemy plik konfiguracyjny (pamiętaj o braku "s" w nazwie folderu!)
    config_file = project_root / "logging_config" / "stdout.json"
    
    with open(config_file) as f_in:
        config = json.load(f_in)
        
    # Opcjonalnie: Jeśli chcesz, aby ścieżka do logów była zawsze absolutna
    # i nie zależała od tego, skąd odpalisz projekt, możesz nadpisać ją w kodzie:
    config["handlers"]["file"]["filename"] = str(log_dir / "detector.log")

    # 4. Odpalamy konfigurację
    logging.config.dictConfig(config)
    
    # (Usunąłem kod queue_handler, skoro nie używasz go w tym JSON-ie)