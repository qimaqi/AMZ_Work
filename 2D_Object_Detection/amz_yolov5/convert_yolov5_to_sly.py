import os
import yaml
import tarfile
from pathlib import Path

import supervisely_lib as sly

# my_app = sly.AppService()

# TEAM_ID = os.environ["context.teamId"]
# WORKSPACE_ID = os.environ["context.workspaceId"]
# INPUT_DIR = os.environ.get("modal.state.slyFolder")
# INPUT_FILE = os.environ.get("modal.state.slyFile")

DATA_CONFIG_NAME = "data_config.yaml"

coco_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"]

input_dir = '/home/qimaqi/Downloads/datasets/data_annotation_2022' #'/home/qimaqi/Downloads/datasets/sf2020'
project_name = 'sf2022_supervisely'
DATA_CONFIG_NAME = 'data_config.yaml'


def generate_colors(count):
    colors = []
    for _ in range(count):
        new_color = sly.color.generate_rgb(colors)
        colors.append(new_color)
    return colors


def get_coco_names(config_yaml):
    if "names" not in config_yaml:
        print("['names'] key is empty in {}. Class names will be taken from default coco classes names".format(DATA_CONFIG_NAME))
    return config_yaml.get("names", coco_classes)


def get_coco_classes_colors(config_yaml, default_count):
    return config_yaml.get("colors", generate_colors(default_count))


def read_config_yaml(config_yaml_path):
    result = {"names": None, "colors": None, "datasets": []}

    if not os.path.isfile(config_yaml_path):
        raise Exception("File {!r} not found".format(config_yaml_path))

    with open(config_yaml_path, "r") as config_yaml_info:
        config_yaml = yaml.safe_load(config_yaml_info)
        result["names"] = get_coco_names(config_yaml)
        result["colors"] = get_coco_classes_colors(config_yaml, len(result["names"]))

        if "nc" not in config_yaml:
            print("Number of classes is not defined in {}. Actual number of classes is {}.".format(DATA_CONFIG_NAME, len(result["names"])))
        elif config_yaml.get("nc", []) != len(result["names"]):
            print("Defined number of classes {} doesn't match with actual number of classes {}".format(config_yaml.get("nc", int), len(result["names"]), DATA_CONFIG_NAME))

        if len(config_yaml.get("colors", [])) == 0:
            print("Colors not found in {}. Colors will be generated for classes automatically.".format(DATA_CONFIG_NAME))
            result["colors"] = generate_colors(len(result["names"]))
        elif result["names"] == coco_classes or len(result["names"]) != len(config_yaml.get("colors")):
            print("len(config_yaml['colors']) !=  len(config_yaml['names']). New colors will be generated for classes automatically.")
            result["colors"] = generate_colors(len(result["names"]))

        conf_dirname = os.path.dirname(config_yaml_path)
        for t in ["train", "val"]:
            if t not in config_yaml:
                raise Exception('{!r} path is not defined in {!r}'.format(t, DATA_CONFIG_NAME))

            if t in config_yaml:
               cur_dataset_path = os.path.normpath(os.path.join(conf_dirname, config_yaml[t]))

               if len(result["datasets"]) == 1 and config_yaml["train"] == config_yaml["val"]:
                   print("'train' and 'val' paths for images are the same in {}. Images will be uploaded to 'train' dataset".format(DATA_CONFIG_NAME))
                   continue

               if os.path.isdir(cur_dataset_path):
                   result["datasets"].append((t, cur_dataset_path))

               elif len(result["datasets"]) == 0:
                   raise Exception("No datasets given, check your project Directory or Archive")

               elif len(result["datasets"]) == 1:
                   raise Exception("Directory: {!r} not found.".format(cur_dataset_path))

    return result


def upload_project_meta(config_yaml_info):
    classes = []
    for class_id, class_name in enumerate(config_yaml_info["names"]):
        yaml_class_color = config_yaml_info["colors"][class_id]
        obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Rectangle, color=yaml_class_color)
        classes.append(obj_class)

    tags_arr = [
        sly.TagMeta(name="train", value_type=sly.TagValueType.NONE),
        sly.TagMeta(name="val", value_type=sly.TagValueType.NONE)
    ]
    project_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(items=classes), tag_metas=sly.TagMetaCollection(items=tags_arr))
    # api.project.update_meta(project_id, project_meta.to_json())
    return project_meta


def convert_geometry(x_center, y_center, ann_width, ann_height, img_width, img_height):
    x_center = float(x_center)
    y_center = float(y_center)
    ann_width = float(ann_width)
    ann_height = float(ann_height)

    px_x_center = x_center * img_width
    px_y_center = y_center * img_height

    px_ann_width = ann_width * img_width
    px_ann_height = ann_height * img_height

    left = px_x_center - (px_ann_width / 2)
    right = px_x_center + (px_ann_width / 2)

    top = px_y_center - (px_ann_height / 2)
    bottom = px_y_center + (px_ann_height / 2)

    return sly.Rectangle(top, left, bottom, right)


def parse_line(line, img_width, img_height, project_meta, config_yaml_info):
    line_parts = line.split()
    if len(line_parts) != 5:
        line_parts.pop(1)
        # print('line_parts',line_parts)
        # raise Exception("Invalid annotation format")
        
    if int(1000*float(line_parts[1])) == 0 and int(1000*float(line_parts[2])) == 0 and int(1000*float(line_parts[3])) == 0 and int(1000*float(line_parts[4])) == 0:
        raise Exception("Invalid annotation format of duplication")
    # else:
        # print('line_parts',line_parts)
    class_id, x_center, y_center, ann_width, ann_height = line_parts
    class_name = config_yaml_info["names"][int(float(class_id))]
    return sly.Label(convert_geometry(x_center, y_center, ann_width, ann_height, img_width, img_height), project_meta.get_obj_class(class_name))


def process_coco_dir(input_dir,dest_dataset0, project_meta, config_yaml_info): # dest_dataset1,dest_dataset2,dest_dataset3,dest_dataset4,
    for dataset_type, dataset_path in config_yaml_info["datasets"]:
        tag_meta = project_meta.get_tag_meta(dataset_type)
        dataset_name = os.path.basename(dataset_path)

        images_list = sorted(sly.fs.list_files(dataset_path, valid_extensions=sly.image.SUPPORTED_IMG_EXTS))
        if len(images_list) == 0:
            raise Exception("Dataset: {!r} is empty. Check {!r} directory in project folder".format(dataset_name, dataset_path))

        # dataset = dest_dataset
        # dataset = api.dataset.create(project.id, dataset_name, change_name_if_conflict=True)
        from tqdm import tqdm
        pbar = tqdm(total= len(images_list), # [3000:]
                    desc='images')
        # progress = sly.Progress("Processing {} dataset".format(dataset_name), len(images_list), sly.logger)
        # for batch in sly._utils.batched(images_list):
        cur_img_names = []
        cur_img_paths = []
        cur_anns = []
        # print('images_list',images_list)
        for image_file_name in images_list: #batch:
            image_name = os.path.basename(image_file_name)
            cur_img_names.append(image_name)
            cur_img_paths.append(image_file_name)
            ann_file_name = os.path.join(input_dir, "labels", "{}.txt".format(os.path.splitext(image_name)[0]))

            curr_img = sly.image.read(image_file_name)
            height, width = curr_img.shape[:2]

            labels_arr = []
            if os.path.isfile(ann_file_name):
                with open(ann_file_name, "r") as f:
                    for idx, line in enumerate(f):
                        try:
                            label = parse_line(line, width, height, project_meta, config_yaml_info)
                            labels_arr.append(label)
                        except Exception as e:
                            print(e, {"filename": ann_file_name, "line": line, "line_num": idx})

            tags_arr = sly.TagCollection(items=[sly.Tag(tag_meta)])
            ann = sly.Annotation(img_size=(height, width), labels=labels_arr)# img_tags=tags_arr
            cur_anns.append(ann)

            # print('Loaded annotation has {} labels and {} image tags.'.format(len(ann.labels), len(ann.img_tags)))
            # print('Label class names: ' + (', '.join(label.obj_class.name for label in ann.labels)))
            # print('Image tags: ' + (', '.join(tag.get_compact_str() for tag in ann.img_tags)))

            file_num = int(image_name[:-4])
            dest_dataset0.add_item_np(
                    image_name, curr_img,
                    ann=ann)
            # if file_num%5==0:
            #     dest_dataset0.add_item_np(
            #         image_name, curr_img,
            #         ann=ann)
            # if file_num%5==1:
            #     dest_dataset1.add_item_np(
            #         image_name, curr_img,
            #         ann=ann)
            # if file_num%5==2:
            #     dest_dataset2.add_item_np(
            #         image_name, curr_img,
            #         ann=ann)
            # if file_num%5==3:
            #     dest_dataset3.add_item_np(
            #         image_name, curr_img,
            #         ann=ann)
            # if file_num%5==4:
            #     dest_dataset4.add_item_np(
            #         image_name, curr_img,
            #         ann=ann)

            pbar.update(1)
        pbar.close()
            #img_infos = api.image.upload_paths(dataset.id, cur_img_names, cur_img_paths)
            #img_ids = [x.id for x in img_infos]

            #api.annotation.upload_anns(img_ids, cur_anns)
            # print('cur_anns',cur_anns)
            # progress.iters_done_report(len(batch))


#@my_app.callback("yolov5_sly_converter")
#@sly.timeit
def yolov5_sly_converter(api: sly.Api, task_id, context, state, app_logger):
    # storage_dir = my_app.data_dir
    # Do transformation locally
    input_dir = '/home/qimaqi/Downloads/datasets/sf2020'
    project_name = 'sf2020_supervisely'
    DATA_CONFIG_NAME = 'data_config.yaml'
    # if INPUT_DIR:
    #     cur_files_path = INPUT_DIR
    #     extract_dir = os.path.join(storage_dir, str(Path(cur_files_path).parent).lstrip("/"))
    #     input_dir = os.path.join(extract_dir, Path(cur_files_path).name)
    #     archive_path = os.path.join(storage_dir, cur_files_path.strip("/") + ".tar")
    #     project_name = Path(cur_files_path).name
    # else:
    #     cur_files_path = INPUT_FILE
    #     extract_dir = os.path.join(storage_dir, sly.fs.get_file_name(cur_files_path))
    #     archive_path = os.path.join(storage_dir, sly.fs.get_file_name_with_ext(cur_files_path))
    #     input_dir = extract_dir
    #     project_name = sly.fs.get_file_name_with_ext(INPUT_FILE)

    # api.file.download(TEAM_ID, cur_files_path, archive_path)
    # if tarfile.is_tarfile(archive_path):
    #     with tarfile.open(archive_path) as archive:
    #          archive.extractall(extract_dir)
    # else:
    #     raise Exception("No such file".format(INPUT_FILE))

    config_yaml_info = read_config_yaml(os.path.join(input_dir, DATA_CONFIG_NAME), app_logger) # read yolov5 config data
    project = api.project.create(WORKSPACE_ID, project_name, change_name_if_conflict=True) # build project in supervisely website
    project_meta = upload_project_meta(api, project.id, config_yaml_info)  # build meta supervisely
    process_coco_dir(input_dir, project, project_meta, api, config_yaml_info, app_logger) #main process here
    api.task.set_output_project(task_id, project.id, project.name)
    # my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": TEAM_ID,
        "context.workspaceId": WORKSPACE_ID,
        "modal.state.slyFolder": INPUT_DIR,
        "modal.state.slyFile": INPUT_FILE,
        "CONFIG_DIR": os.environ.get("CONFIG_DIR", None)
    })

    # my_app.run(initial_events=[{"command": "yolov5_sly_converter"}])

def print_project_files(project_dir):
    files_with_sizes = {
        filename: sly.fs.get_file_size(os.path.join(project_dir, filename))
        for filename in sly.fs.list_dir_recursively(project_dir)}
    files_text = '\n'.join(
        '\t{}: {} bytes'.format(filename, files_with_sizes[filename])
        for filename in sorted(files_with_sizes.keys()))
    print('Project contents:\n{}\n'.format(str(files_text)))  

if __name__ == "__main__":
    # sly.main_wrapper("main", main)
    # test function
    # create a empty project locally
    import os

    # A helper to pretty-print project on-disk files.
    # Can be safely skipped - not essentiall for understanding the rest of the code.

    # Directory path to create the new project in
    project_0 = './sf2022'
    # project_0 = './sf2020_0'
    # project_1 = './sf2020_1'
    # project_2 = './sf2020_2'
    # project_3 = './sf2020_3'
    # project_4 = './sf2020_4'

    # Remove the target directory in case it is left over from previous runs.
    sly.io.fs.remove_dir(project_0)
    # sly.io.fs.remove_dir(project_0)
    # sly.io.fs.remove_dir(project_1)
    # sly.io.fs.remove_dir(project_2)
    # sly.io.fs.remove_dir(project_3)
    # sly.io.fs.remove_dir(project_4)

    print('Creating project...')
    dest_project0 = sly.Project(project_0, sly.OpenMode.CREATE)
    # dest_project0 = sly.Project(project_0, sly.OpenMode.CREATE)
    # dest_project1 = sly.Project(project_1, sly.OpenMode.CREATE)
    # dest_project2 = sly.Project(project_2, sly.OpenMode.CREATE)
    # dest_project3 = sly.Project(project_3, sly.OpenMode.CREATE)
    # dest_project4 = sly.Project(project_4, sly.OpenMode.CREATE)

    # Creating a project immediately writes out a serialized meta file to disk.
    config_yaml_info = read_config_yaml(os.path.join(input_dir, DATA_CONFIG_NAME))
    print(config_yaml_info)
    project_meta = upload_project_meta(config_yaml_info)  # build meta supervisely
    dest_project0.set_meta(project_meta)
    # dest_project1.set_meta(project_meta)
    # dest_project2.set_meta(project_meta)
    # dest_project3.set_meta(project_meta)
    # dest_project4.set_meta(project_meta)

    dataset_name0 = 'rainy'
    dest_dataset0 = dest_project0.create_dataset(dataset_name0) 
    # dataset_name1 = 'sf2020_1'
    # dest_dataset1 = dest_project1.create_dataset(dataset_name1) 
    # dataset_name2 = 'sf2020_2'
    # dest_dataset2 = dest_project2.create_dataset(dataset_name2) 
    # dataset_name3 = 'sf2020_3'            # if file_num%5==0:
            #     dest_dataset0.add_item_np(
            #         image_name, curr_img,
            #         ann=ann)
    # dest_dataset1,dest_dataset2,dest_dataset3,dest_dataset4



    #######################################3

    #### read project and show
#     project = sly.Project(MODIFIED_PROJECT_DIR, sly.OpenMode.READ)
#     # Print basic project metadata.
#     print("Project name: ", project.name)
#     print("Project directory: ", project.directory)
#     print("Total images: ", project.total_items)
#     print("Dataset names: ", project.datasets.keys())
#     print("\n")

# # What datasets and images are there and where are they on disk?
#     for dataset in project:
#         print("Dataset: ", dataset.name)
        
#         # A dataset item is a pair of an image and its annotation.
#         # The annotation contains all the labeling information about
#         # the image - segmentation masks, objects bounding boxes etc.
#         # We will look at annotations in detail shortly.
#         for item_name in dataset:
#             print(item_name)
#             img_path = dataset.get_img_path(item_name)
#             print("  image: ", img_path)
#         print()
#     print(project.meta)

#     # Grab the file paths for both raw image and annotation in one call.
#     item_paths = project.datasets.get('sf2020_supervisely').get_item_paths('00000001.png')

#     # Load and deserialize annotation from JSON format.
#     # Annotation data is cross-checked again project meta, and references to
#     # the right LabelMeta and TagMeta objects are set up.
#     ann = sly.Annotation.load_json_file(item_paths.ann_path, project.meta)

#     print('Loaded annotation has {} labels and {} image tags.'.format(len(ann.labels), len(ann.img_tags)))
#     print('Label class names: ' + (', '.join(label.obj_class.name for label in ann.labels)))
#     print('Image tags: ' + (', '.join(tag.get_compact_str() for tag in ann.img_tags)))
