import json
import os
import random
import shutil
import sys
from pathlib import Path

# Stała globalna dla rozszerzeń obrazów
IMG_EXTS = ('.jpg', '.jpeg', '.png')

# --- FUNKCJE POMOCNICZE (PRIVATE/UTILS) ---

def _get_files(path, extensions):
    """
    Pobiera listę plików o zadanych rozszerzeniach z podanej ścieżki.
    Obsługuje zarówno ścieżkę do folderu, jak i pojedynczego pliku.
    """
    p = Path(path)
    if not p.exists():
        return []
        
    if p.is_file():
        if p.suffix.lower() in extensions:
            return [p]
        return []
    
    # Jeśli folder
    files = [f for f in p.glob("*") if f.suffix.lower() in extensions]
    return sorted(files)

def _prepare_dirs(output_root, split):
    """
    Tworzy strukturę folderów images/labels dla danego splitu (train/val/test).
    Wymaga podania output_root.
    """
    out_path = Path(output_root)
    img_dir = out_path / "images" / split
    lbl_dir = out_path / "labels" / split
    
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    return img_dir, lbl_dir

def _rollback(log, was_moved):
    """Cofa operacje na plikach w przypadku błędu (dla funkcji add_data)."""
    print("Cofanie zmian (Rollback)...")
    for s, d in reversed(log):
        if Path(d).exists():
            if was_moved:
                shutil.move(str(d), str(s))
            else:
                os.remove(str(d))

# --- NARZĘDZIA DO KONWERSJI I EDYCJI (TOOLS) ---

def coco_to_yolo_pose(json_path, output_dir, target_class_id=0):
    """
    Konwertuje plik COCO JSON (annotations) bezpośrednio na format YOLO Pose (.txt).
    """
    json_path = Path(json_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Rozpoczynam konwersję COCO: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    count = 0

    for ann in data['annotations']:
        if 'keypoints' not in ann or not ann['keypoints']:
            continue
            
        img = images.get(ann['image_id'])
        if img is None:
            continue

        w, h = img['width'], img['height']
        
        # Normalizacja BBox (center_x, center_y, width, height)
        xc = (ann['bbox'][0] + ann['bbox'][2] / 2) / w
        yc = (ann['bbox'][1] + ann['bbox'][3] / 2) / h
        wn = ann['bbox'][2] / w
        hn = ann['bbox'][3] / h
        
        # Normalizacja Punktów Kluczowych (x, y, visibility)
        kp = ann['keypoints']
        yolo_kp = []
        for i in range(0, len(kp), 3):
            kx = kp[i] / w
            ky = kp[i+1] / h
            kv = int(kp[i+2])
            yolo_kp.append(f"{kx:.6f} {ky:.6f} {kv}")
        
        line = f"{target_class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f} " + " ".join(yolo_kp)
        
        file_stem = Path(img['file_name']).stem
        with open(out_dir / f"{file_stem}.txt", 'a') as f_out:
            f_out.write(line + "\n")
        count += 1
        
    print(f"Konwersja zakończona: {count} adnotacji zapisano w {output_dir}")

def update_class_id(labels_path, new_class_id):
    """
    Zmienia numer klasy (pierwsza kolumna) we wszystkich plikach .txt w danej ścieżce.
    """
    files = _get_files(labels_path, extensions=('.txt',))
    
    for txt_f in files:
        updated_lines = []
        with open(txt_f, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                parts[0] = str(new_class_id)
                updated_lines.append(" ".join(parts) + "\n")
        
        with open(txt_f, 'w') as f:
            f.writelines(updated_lines)
            
    print(f"Zmieniono ID klasy na {new_class_id} w {len(files)} plikach.")

def update_visibility(labels_path, visibility):
    """
    Zmienia parametr widoczności (v) dla wszystkich punktów kluczowych w plikach .txt.
    """
    files = _get_files(labels_path, extensions=('.txt',))

    for txt_f in files:
        updated_lines = []
        with open(txt_f, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Format YOLO Pose: class x y w h px1 py1 pv1 ...
                if len(parts) <= 5: continue 

                # pv (visibility) jest co trzecim elementem od indeksu 7
                for i in range(7, len(parts), 3):
                    parts[i] = str(int(visibility))
                
                updated_lines.append(" ".join(parts) + "\n")
        
        with open(txt_f, 'w') as f:
            f.writelines(updated_lines)
            
    print(f"Zaktualizowano widoczność (v={visibility}) w {len(files)} plikach.")

def add_pose_to_bbox(labels_path, class_id=None, visibility=0, num_keypoints=17):
    """
    Konwertuje format Detection (tylko bbox) na format Pose, dodając zerowe punkty kluczowe.
    Parametr num_keypoints musi być podany (domyślnie 17).
    """
    files = _get_files(labels_path, extensions=('.txt',))
    
    # Przygotowanie "ogona" z pustymi punktami
    pose_tail = [f"0.0 0.0 {int(visibility)}"] * num_keypoints
    pose_str = " ".join(pose_tail)

    for txt_f in files:
        updated_lines = []
        with open(txt_f, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                cid = str(class_id) if class_id is not None else parts[0]
                # Składanie nowej linii: ID + BBOX + POSE
                new_line = f"{cid} " + " ".join(parts[1:5]) + f" {pose_str}\n"
                updated_lines.append(new_line)
        
        with open(txt_f, 'w') as f:
            f.writelines(updated_lines)
            
    print(f"Dodano puste punkty (v={visibility}, kpts={num_keypoints}) do {len(files)} plików.")

# --- BUDOWANIE DATASETU (BUILDER) ---

def add_data(src_img_dir, src_lbl_dir, output_root, split, count=None, move=False, seed=42):
    """
    Kopiuje lub przenosi pary obraz-etykieta do folderu wyjściowego.
    Wymaga podania output_root.
    """
    if not output_root:
        print("BŁĄD: Musisz podać output_root!")
        return

    src_img = Path(src_img_dir)
    src_lbl = Path(src_lbl_dir)
    dst_img, dst_lbl = _prepare_dirs(output_root, split)
    
    all_imgs = _get_files(src_img, extensions=IMG_EXTS)
    if not all_imgs:
        print(f"Ostrzeżenie: Brak zdjęć w {src_img}")
        return

    # Wybór próbek
    if count is not None and count < len(all_imgs):
        random.seed(seed)
        selected = random.sample(all_imgs, count)
    else:
        selected = all_imgs

    log = []
    try:
        for img_p in selected:
            # Zakładamy, że plik txt ma taką samą nazwę jak obraz
            lbl_p = src_lbl / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                raise FileNotFoundError(f"Brak etykiety dla obrazu: {img_p.name}")

            # Definicja par źródło -> cel
            pairs = [
                (img_p, dst_img / img_p.name),
                (lbl_p, dst_lbl / lbl_p.name)
            ]
            
            for s, d in pairs:
                op = shutil.move if move else shutil.copy2
                op(str(s), str(d))
                log.append((s, d))

    except Exception as e:
        print(f"\n[BŁĄD KRYTYCZNY]: {e}. Uruchamiam rollback...")
        _rollback(log, was_moved=move)
        sys.exit(1)

    print(f"[{split.upper()}] Pomyślnie {'przeniesiono' if move else 'skopiowano'} {len(selected)} par do {output_root}.")

def add_complete_dataset(source_root, output_root, move=False):
    """
    Automatycznie przetwarza strukturę source/train, source/test, source/val.
    """
    root = Path(source_root)
    for split in ['train', 'test', 'val']:
        split_p = root / split
        if split_p.exists():
            add_data(
                src_img_dir=split_p / "images", 
                src_lbl_dir=split_p / "labels", 
                output_root=output_root,
                split=split, 
                move=move
            )

def create_yaml(output_root, classes_dict, num_keypoints=17, relative_path_name="."):
    """
    Generuje plik data.yaml potrzebny do treningu.
    """
    out_root = Path(output_root)
    if not out_root.exists():
        print(f"BŁĄD: Folder {output_root} nie istnieje.")
        return

    # Standardowe indeksy flip dla COCO (dla 17 punktów)
    # Jeśli masz inną liczbę punktów, ta lista może być niepoprawna (można ją sparametryzować)
    coco_flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    
    yaml_lines = [
        f"path: {relative_path_name}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(classes_dict)}",
        "names:",
    ]
    
    for idx, name in classes_dict.items():
        yaml_lines.append(f"  {idx}: {name}")
    
    yaml_lines.extend([
        "",
        f"kpt_shape: [{num_keypoints}, 3]",
        f"flip_idx: {coco_flip_idx}"
    ])

    with open(out_root / "data.yaml", "w") as f:
        f.write("\n".join(yaml_lines))
    print(f"Sukces: Wygenerowano YAML w {out_root}/data.yaml")

# --- PRZYKŁAD UŻYCIA (CLI) ---
if __name__ == "__main__":
    
    # add_complete_dataset("data\License-Plate-Recognition-11", "data\License-Plate-Recognition-11-2")
    
    # create_yaml(output_root="data\License-Plate-Recognition-11-2", classes_dict={1: 'registration plate'})

    for split in ['train', 'test', 'val']:
        update_class_id("data\License-Plate-Recognition-11-2\labels\\"+split, 0)