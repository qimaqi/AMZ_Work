from pylabel import importer
import os


path_to_annotations = "/scratch/SF2022_merge_4_classes_reorder/train/labels/"

#Identify the path to get from the annotations to the images 
path_to_images = "/scratch/SF2022_merge_4_classes_reorder/train/images/"

#Import the dataset into the pylable schema 
#Class names are defined here https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml
yoloclasses = ['blue_cone', 'yellow_cone','orange_cone', 'large_orange_cone']
dataset = importer.ImportYoloV5(path=path_to_annotations, path_to_images=path_to_images, cat_names=yoloclasses,
    img_ext="png", name="amz_coco")

dataset.export.ExportToCoco(cat_id_index=1)