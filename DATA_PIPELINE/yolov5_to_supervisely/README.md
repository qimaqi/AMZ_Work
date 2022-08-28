
# This code can be used to convert YOLOv5 to Supervisely format
### from supervisely [app](https://github.com/supervisely-ecosystem/convert-yolov5-to-supervisely-format) and you can get help from [tutorial](https://github.com/supervisely/supervisely/blob/master/help/jupyterlab_scripts/src/tutorials/01_project_structure/project.ipynb)



<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Preparation">Preparation</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a>
</p>


## Overview
App transforms folder or `tar` archive with images and labels in [YOLOv5 format](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) to [Supervisely format](https://docs.supervise.ly/data-organization/00_ann_format_navi) and uploads data to Supervisely Platform.


## Preparation

Upload images and labels in YOLO v5 format to team files. It is possible to upload folders ([download example](https://drive.google.com/drive/folders/1CqG0bmDRoGF33r5gLWnmEHgkp9u196DZ?usp=sharing)) or tar archives ([download example](https://drive.google.com/drive/folders/1YmbEBqgOVrL9IiBVRpKJ-_7ZnV31Wc7r?usp=sharing)).

![](https://i.imgur.com/BRA0Bjt.png)

Example of `data_config.yaml`:

```yaml
nc: 2                           # number of classes       
train: ../data_example/images/  # path to train imgs
val: ../data_example/images/    # path to val imgs
colors:                         # classes box color
- [0, 0, 255]
- [255, 255, 0]
- [255, 165, 0]
- [255, 0, 0]
names: ['blue cone', 'yellow cone','orange cone', 'big orange cone']  # class names
```

## How To Run 

**Step 1**: Prepare the data and install all necessary package.

**Step 2**: Run the command, replace the path which fit your case
```python
python3 convert_yolov5_to_sly.py -p ../data_example
```

## How to use

After that you can then upload your project to supervisely. If you can see both annotation and images then you are suceessful with this transformation.
<img src="https://i.imgur.com/KFiRU6K.png"/>


