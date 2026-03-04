## Using script.py

To use script.py write command below into terminal
``` uv run script.py ```

It will print out help containing short description of every subcommand.

Example pipeline to create dataset containing both license plates and selected yolo classes. It will also label each dataset using model to fill gaps (mostly cars in license dataset and license plates in COCO dataset):

```
# Downloading datasets
uv run -m src.tools.script download_coco data/coco_dataset
uv run -m src.tools.script download_license data/license_dataset

# Training YOLO model using license dataset
uv run -m src.tools.script train yolo26_license yolo26m.pt -d data/license_dataset -b 26

# Labeling datasets in place
uv run -m src.tools.script label_license data/license_dataset yolo26x.pt
uv run -m src.tools.script label_coco data/coco_dataset/ runs/detect/yolo26_license/weights/best.pt

# Combining datasets
uv run -m src.tools.script combine_datasets data/license_dataset data/coco_dataset data/dataset

# Training final model
uv run -m src.tools.script train yolo26_all yolo26m.pt -d data/dataset -b 26
```