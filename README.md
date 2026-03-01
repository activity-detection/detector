## Using script.py

To use script.py write command below into terminal
``` uv run script.py ```

It will print out help containing short description of every subcommand.

Example pipeline to create dataset containing both license plates and selected yolo classes. It will also label each dataset using model to fill gaps (mostly cars in license dataset and license plates in COCO dataset):

```
# Downloading datasets
uv run src/tools/script.py download_coco data/coco_dataset
uv run src/tools/script.py download_license data/license_dataset

# Training YOLO model using license dataset
uv run src/tools/script.py train yolo26_license yolo26m.pt -d data/license_dataset -b 26

# Labeling datasets in place
uv run src/tools/script.py label_license data/license_dataset yolo26x.pt
uv run src/tools/script.py label_coco data/coco_dataset/ runs/detect/yolo26_license/weights/best.pt

# Combining datasets
uv run src/tools/script.py combine_datasets data/license_dataset data/coco_dataset data/dataset

# Training final model
uv run src/tools/script.py train yolo26_all yolo26m.pt -d data/dataset -b 26
```