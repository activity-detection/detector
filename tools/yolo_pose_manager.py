import json
import os
import random
import shutil
import sys
from pathlib import Path

class YoloPoseManager:
    def __init__(self, output_root=None, num_keypoints=17):
        """
        Inicjalizuje menedżera zbioru danych YOLO Pose.
        output_root: Główny folder, w którym powstanie gotowy dataset.
        num_keypoints: Liczba punktów kluczowych (domyślnie 17 dla COCO).
        """
        self.output_root = Path(output_root) if output_root else None
        self.num_keypoints = num_keypoints
        self.img_exts = ('.jpg', '.jpeg', '.png')

    # --- PRYWATNE METODY POMOCNICZE ---

    def _get_images(self, path):
        """Pobiera posortowaną listę obrazów z danej ścieżki."""
        p = Path(path)
        if not p.exists():
            return []
        return sorted([f for f in p.iterdir() if f.suffix.lower() in self.img_exts])

    def _prepare_dirs(self, split):
        """Tworzy strukturę images/labels dla danego splitu wewnątrz output_root."""
        if not self.output_root:
            print("BŁĄD: Nie zdefiniowano output_root w konstruktorze klasy!")
            sys.exit(1)
            
        img_dir = self.output_root / "images" / split
        lbl_dir = self.output_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        return img_dir, lbl_dir

    # --- NARZĘDZIA DO EDYCJI ETYKIET (TOOLS) ---

    def update_class_id(self, labels_dir, new_class_id):
        """Zmienia numer klasy (pierwsza kolumna) we wszystkich plikach .txt w folderze."""
        labels_path = Path(labels_dir)
        txt_files = list(labels_path.glob("*.txt"))

        for txt_f in txt_files:
            updated_lines = []
            with open(txt_f, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    parts[0] = str(new_class_id)
                    updated_lines.append(" ".join(parts) + "\n")
            
            with open(txt_f, 'w') as f:
                f.writelines(updated_lines)
        print(f"Zmieniono ID klasy na {new_class_id} w {len(txt_files)} plikach.")

    def update_visibility(self, labels_dir, visibility):
        """Zmienia tylko parametr widoczności (v) dla wszystkich istniejących punktów kluczowych."""
        labels_path = Path(labels_dir)
        txt_files = list(labels_path.glob("*.txt"))

        for txt_f in txt_files:
            updated_lines = []
            with open(txt_f, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    # YOLO Pose ma format: class x y w h px1 py1 pv1 ...
                    # Punkty kluczowe (v) znajdują się na indeksach 7, 10, 13...
                    if len(parts) <= 5: continue 

                    for i in range(7, len(parts), 3):
                        parts[i] = str(int(visibility))
                    
                    updated_lines.append(" ".join(parts) + "\n")
            
            with open(txt_f, 'w') as f:
                f.writelines(updated_lines)
        print(f"Zaktualizowano widoczność (v={visibility}) w {len(txt_files)} plikach.")

    def add_pose_to_bbox(self, labels_dir, class_id=None, visibility=0):
        """Konwertuje format Detection (tylko bbox) na format Pose, dodając zerowe punkty kluczowe."""
        labels_path = Path(labels_dir)
        txt_files = list(labels_path.glob("*.txt"))
        
        # Przygotowanie "ogona" z pustymi punktami
        pose_tail = [f"0.0 0.0 {int(visibility)}"] * self.num_keypoints
        pose_str = " ".join(pose_tail)

        for txt_f in txt_files:
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
        print(f"Dodano puste punkty (v={visibility}) do {len(txt_files)} plików w {labels_dir}")

    # --- NARZĘDZIA DO BUDOWANIA DATASETU (BUILDER) ---

    def add_data(self, src_img_dir, src_lbl_dir, split, count=None, move=False, seed=42):
        """Kopiuje lub przenosi pary obraz-etykieta do folderu wyjściowego."""
        src_img, src_lbl = Path(src_img_dir), Path(src_lbl_dir)
        dst_img, dst_lbl = self._prepare_dirs(split)
        
        all_imgs = self._get_images(src_img)
        if not all_imgs:
            print(f"Ostrzeżenie: Brak zdjęć w {src_img}")
            return

        # Wybór próbek (losowy z seedem lub wszystkie alfabetycznie)
        if count is not None and count < len(all_imgs):
            random.seed(seed)
            selected = random.sample(all_imgs, count)
        else:
            selected = all_imgs

        moved_log = []
        try:
            for img_p in selected:
                lbl_p = src_lbl / f"{img_p.stem}.txt"
                if not lbl_p.exists():
                    raise FileNotFoundError(f"Brak etykiety dla obrazu: {img_p.name}")

                d_img_p, d_lbl_p = dst_img / img_p.name, dst_lbl / lbl_p.name
                
                op = shutil.move if move else shutil.copy2
                op(str(img_p), str(d_img_p))
                moved_log.append((img_p, d_img_p))
                op(str(lbl_p), str(d_lbl_p))
                moved_log.append((lbl_p, d_lbl_p))

        except Exception as e:
            print(f"\n[BŁĄD KRYTYCZNY]: {e}. Uruchamiam rollback...")
            for src, dst in reversed(moved_log):
                if dst.exists():
                    if move: shutil.move(str(dst), str(src))
                    else: os.remove(str(dst))
            sys.exit(1)

        print(f"[{split.upper()}] Pomyślnie {'przeniesiono' if move else 'skopiowano'} {len(selected)} par.")

    def add_complete_dataset(self, source_root, move=False):
        """Automatycznie przetwarza strukturę source/train, source/test, source/val."""
        root = Path(source_root)
        for split in ['train', 'test', 'val']:
            split_p = root / split
            if not split_p.exists(): continue
            self.add_data(split_p/"images", split_p/"labels", split, move=move)

    def create_yaml(self, classes_dict):
        """Generuje plik data.yaml niezbędny do treningu YOLO Pose."""
        if not self.output_root: return
        
        coco_flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        
        yaml_content = f"path: {self.output_root.absolute()}\n"
        yaml_content += "train: images/train\n"
        yaml_content += "val: images/val\n"
        yaml_content += "test: images/test\n\n"
        yaml_content += f"nc: {len(classes_dict)}\n"
        yaml_content += "names:\n"
        for idx, name in classes_dict.items():
            yaml_content += f"  {idx}: {name}\n"
        
        yaml_content += f"\nkpt_shape: [{self.num_keypoints}, 3]\n"
        yaml_content += f"flip_idx: {coco_flip_idx}\n"

        with open(self.output_root / "data.yaml", "w") as f:
            f.write(yaml_content)
        print(f"Wygenerowano plik konfiguracyjny: {self.output_root}/data.yaml")