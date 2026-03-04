import argparse
from ultralytics import YOLO
import cv2
import fiftyone as fo
import fiftyone.zoo as foz
from roboflow import Roboflow
from src.detector.config import Config
from pathlib import Path
import shutil
import yaml
from itertools import batched
from tqdm import tqdm
import math

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEFAULT_DATASET = 'data/con_dataset/dataset.yaml'
LICENSE_ID = 8
DEFAULT_COCO_CLASSES = [
    "bicycle", 'car', "bird", "cat", "dog",
    "handbag", "suitcase", "knife"
]

NEW_NAMES_MAP = {
    0: 'bicycle', 1: 'car', 2: 'bird', 3: 'cat', 
    4: 'dog', 5: 'handbag', 6: 'suitcase', 7: 'knife', 
    8: 'license_plate'
}

YOLO_TO_OUR_IDS_MAP = {1: 0, 2: 1, 14: 2, 15: 3, 16: 4, 26: 5, 28: 6, 43: 7} # mapping from yolo ids to our ids

def get_model_path(model: str) -> str:
    return f'runs/detect/{model}/weights/last.pt'

def get_dataset(path: str | Path) -> Path:
    file = Path(path)
    files = [file, file / 'data.yaml', file / 'dataset.yaml']
    for file in files:
        if file.exists() and file.suffix == '.yaml':
            return file
    raise FileNotFoundError('Dataset not found')

def train_script(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    dataset = get_dataset(args.dataset)
    model.train(
        name=args.name,
        data=dataset,
        batch=args.batch,
        epochs=args.epochs
    )

def resume_script(args: argparse.Namespace) -> None:
    path = get_model_path(args.name)
    model = YOLO(path)
    model.train(resume=True)

def validate_script(args: argparse.Namespace) -> None:
    path = get_model_path(args.name)
    name = f'{args.name}_val'
    model = YOLO(path)
    dataset = get_dataset(args.dataset)
    model.val(data=dataset, name=name)

def show_script(args: argparse.Namespace) -> None:
    path = get_model_path(args.name)
    model = YOLO(path)
    
    cap = cv2.VideoCapture(args.example)
    if not cap.isOpened():
        print('File not found')
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.track(frame, verbose=False)
            result = results[0]
            plot = result.plot()
            plot = cv2.resize(plot, tuple(args.size))
            cv2.imshow('Model', plot)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()

def download_coco_script(args: argparse.Namespace) -> None:

    my_classes = args.classes if args.classes else [
        "bicycle", 'car', "bird", "cat", "dog",
        "handbag", "suitcase", "knife"
    ]

    splits = args.splits
    
    for split_name in splits:
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split=split_name,
            label_types=["detections"],
            classes=my_classes,
            seed=42,
            shuffle=True
        )

        dataset = dataset.filter_labels("ground_truth", fo.ViewField("label").is_in(my_classes))

        dataset.export(
            export_dir=args.path, 
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            split=split_name,
            classes=my_classes
        )

def download_license_script(args: argparse.Namespace) -> None:
    rf = Roboflow(Config.ROBOFLOW_API_KEY)
    project = rf.workspace(Config.ROBOFLOW_WORKSPACE).project(Config.ROBOFLOW_PROJECT)
    version = project.version(Config.ROBOFLOW_DATASET_VERSION)
    version.download('yolo26', location=args.path)

def update_dataset_yaml(dataset_path: str) -> None:
    yaml_path = get_dataset(dataset_path)
    
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        data['names'] = NEW_NAMES_MAP
        data['nc'] = len(NEW_NAMES_MAP)
        data['path'] = str(yaml_path.parent.absolute())
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)

def copy_dataset(args: argparse.Namespace) -> Path:
    dataset_path = Path(args.dataset)
    if args.output is not None:
        output_path = Path(args.output)
        if output_path.exists():
            raise FileExistsError(f'{output_path} already exists')
        shutil.copytree(dataset_path, output_path)
        dataset_path = output_path
    return dataset_path

def process_label_file(label_file: Path, result, license_id: int, mapped_classes: dict = None) -> None:
    lines = []
    
    if label_file.exists():
        with open(label_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

    if mapped_classes is not None:
        new_lines = []
        for line in lines:
            parts = line.split()
            if parts:
                parts[0] = str(license_id)
                new_lines.append(' '.join(parts))
        lines = new_lines

    boxes = result.boxes.xywhn
    classes = result.boxes.cls
    for box, cls in zip(boxes, classes):
        class_id = mapped_classes[int(cls)] if mapped_classes else license_id
        line = f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
        lines.append(line)

    if lines:
        with open(label_file, 'w') as f:
            f.write('\n'.join(lines) + '\n')

def _label_dataset_logic(args: argparse.Namespace, splits: list, mapped_classes: dict = None) -> None:
    dataset_path = copy_dataset(args)
    model = YOLO(args.model)
    
    for split in splits:
        img_folder = dataset_path / split / 'images'
        if not img_folder.exists():
            img_folder = dataset_path / 'images' / split
            label_folder = dataset_path / 'labels' / split
        else:
            label_folder = dataset_path / split / 'labels'
            
        if not img_folder.exists():
            continue
            
        images = [img for img in img_folder.iterdir() if img.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            continue
        
        max_iter = math.ceil(len(images) / args.batch)
        for batch in tqdm(batched(iterable=images, n=args.batch), total=max_iter):
            results = model.predict(batch, verbose=False,
                                    classes=list(mapped_classes.keys()) if mapped_classes else None)
            for img, result in zip(batch, results):
                label_file = label_folder / f'{img.stem}.txt'
                process_label_file(label_file, result, LICENSE_ID, mapped_classes=mapped_classes)
            
    update_dataset_yaml(dataset_path)

def label_license_dataset_script(args: argparse.Namespace) -> None:
    mapped = YOLO_TO_OUR_IDS_MAP
    _label_dataset_logic(args, splits=['train', 'valid'], mapped_classes=mapped)

def label_coco_dataset_script(args: argparse.Namespace) -> None:
    _label_dataset_logic(args, splits=['train', 'validation'], mapped_classes=None)

def combine_datasets_script(args: argparse.Namespace) -> None:
    license_dataset_path = Path(args.license_dataset)
    coco_dataset_path = Path(args.coco_dataset)
    output_dataset_path = Path(args.output)

    if not output_dataset_path.exists():
        shutil.copytree(coco_dataset_path, output_dataset_path)
    else:
        raise FileExistsError(f'{output_dataset_path} exists')

    coco_splits = ['train', 'validation']
    license_splits = ['train', 'valid']
    
    for coco_split, license_split in zip(coco_splits, license_splits):
        for part in ['images', 'labels']:
            input_path = license_dataset_path / license_split / part
            output_path = output_dataset_path / part / coco_split
            
            if not input_path.exists():
                continue
                
            output_path.mkdir(parents=True, exist_ok=True)
            
            for entry in input_path.iterdir():
                if entry.is_file():
                    dest_file = output_path / f"license_{entry.name}"
                    shutil.copy(entry, dest_file)
                    
    update_dataset_yaml(output_dataset_path)

def setup_parser():
    parser = argparse.ArgumentParser(prog='tools')

    subparsers = parser.add_subparsers(title='subcommands', help='YOLO and dataset commands')

    train_parser = subparsers.add_parser('train', help='trains YOLO model')
    train_parser.add_argument('name', help='name of the run')
    train_parser.add_argument('model', help='path to model')
    train_parser.add_argument('-d', '--dataset', default=DEFAULT_DATASET, help='path to dataset')
    train_parser.add_argument('-b', '--batch', action='store', default=32, type=int, help='batch size used in training (DEFAULT=32)')
    train_parser.add_argument('-e', '--epochs', action='store', default=150, type=int, help='sets epochs amount (DEFAULT=150)')
    train_parser.set_defaults(func=train_script)

    resume_parser = subparsers.add_parser('resume', help='resume YOLO training')
    resume_parser.add_argument('name', help='name of the run')
    resume_parser.set_defaults(func=resume_script)

    validate_parser = subparsers.add_parser('validate', help='validates YOLO model')
    validate_parser.add_argument('name', help='name of the run')
    validate_parser.add_argument('-d', '--dataset', default=DEFAULT_DATASET, help='path to dataset')
    validate_parser.set_defaults(func=validate_script)

    show_parser = subparsers.add_parser('show', help='shows results from YOLO on example')
    show_parser.add_argument('name', help='name of the run')
    show_parser.add_argument('example', help='path to example video')
    show_parser.add_argument('--size', '-s', nargs=2, default=(1200, 900), type=int, help='size of the window')
    show_parser.set_defaults(func=show_script)

    download_coco_parser = subparsers.add_parser('download_coco', help='downloads COCO dataset')
    download_coco_parser.add_argument(
        'path',
        help='determines path to folder in which dataset will be downloaded'
    )
    download_coco_parser.add_argument(
        '-c', '--classes',
        action='extend', nargs="+",
        help='used to select classes in dataset')
    download_coco_parser.add_argument(
        '-s', '--splits',
        action='extend', nargs="+",
        help='selects splits of dataset',
        default=('train', 'validation'),
        choices=['train', 'validation'])
    download_coco_parser.set_defaults(func=download_coco_script)

    download_license_parser = subparsers.add_parser('download_license', help='downloads license dataset')
    download_license_parser.add_argument(
        'path',
        help='determines path to folder in which dataset will be downloaded'
    )
    download_license_parser.set_defaults(func=download_license_script)

    label_license_parser = subparsers.add_parser('label_license', help='labels license dataset using base YOLO model')
    label_license_parser.add_argument('dataset', help='path to dataset')
    label_license_parser.add_argument('model', help='path to model')
    label_license_parser.add_argument('-o', '--output', help='path to output folder. If not specified label in place')
    label_license_parser.add_argument('-b', '--batch', help='length of batch used in predicting labels', default=32, type=int)
    label_license_parser.set_defaults(func=label_license_dataset_script)

    label_coco_parser = subparsers.add_parser('label_coco', help='labels COCO dataset using base YOLO model')
    label_coco_parser.add_argument('dataset', help='path to dataset')
    label_coco_parser.add_argument('model', help='path to model')
    label_coco_parser.add_argument('-o', '--output', help='path to output folder. If not specified label in place')
    label_coco_parser.add_argument('-b', '--batch', help='length of batch used in predicting labels', default=32, type=int)
    label_coco_parser.set_defaults(func=label_coco_dataset_script)

    combine_datasets_parser = subparsers.add_parser('combine_datasets', help='combines license dataset and COCO dataset')
    combine_datasets_parser.add_argument('license_dataset', help='path to license dataset')
    combine_datasets_parser.add_argument('coco_dataset', help='path to COCO dataset')
    combine_datasets_parser.add_argument('output', help='path to output folder')
    combine_datasets_parser.set_defaults(func=combine_datasets_script)
    return parser

def main():
    parser = setup_parser()

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()