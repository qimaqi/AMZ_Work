# this script is used to put forward, left, right head images and merge to one folder so the labels
# be more bold and take all images and labels without autocross name

import os
import pathlib
import glob
from tqdm import tqdm
import shutil
import argparse

parser = argparse.ArgumentParser(description='Process the path .')
parser.add_argument('-p', type=str, metavar='path',
                    help='starting and ending character')
args = parser.parse_args()

src_folder_path = args.p #'autocross_2020-07-05-11-58-07'


target_imgs_path = os.path.join(os.getcwd(),'images')
target_labels_path = os.path.join(os.getcwd(),'labels')

# main task rename
# show all images
all_labels_list = glob.glob(os.path.join(os.getcwd(),'labels')+'/*/*/*.txt')
all_labels_list.sort()
pbar = tqdm(total=len(all_labels_list), 
                    desc='labels')
for count, label_i in enumerate(all_labels_list):
    # first find corresponnding image name
    img_i = label_i.replace('labels','images')[:-3] + 'png'
    # rename images and txt
    target_img_i_name = str(count).zfill(8) +'.png'
    target_label_i_name = str(count).zfill(8) + '.txt'
    target_img_i_path = os.path.join(target_imgs_path,target_img_i_name)
    target_label_i_path = os.path.join(target_labels_path,target_label_i_name)
    shutil.move(img_i, target_img_i_path)
    shutil.move(label_i, target_label_i_path)
    pbar.update(1)
pbar.close()