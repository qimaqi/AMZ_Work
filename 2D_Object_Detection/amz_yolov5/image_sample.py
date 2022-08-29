import os
import glob
import shutil
from tqdm import tqdm

data_list = ["/media/qimaqi/My Passport/tmp/amz/new_lens_ann/new_lens_data/05_04"]
for data_folder in data_list:#glob.glob('./*/*/data/*/forward_camera'):
    print('data folder',data_folder)
    target_path = data_folder
    #'/home/qimaqi/Downloads/datasets/wet/2021-07-14_Duebendorf/data/autocross-lidar-cone-color-camera_2021-07-14-14-48-36' + '/forward_camera_filtered'
    dst_path = os.path.dirname(data_folder) +'_sample'
    #'/home/qimaqi/Downloads/datasets/wet/2021-07-14_Duebendorf/data/autocross-lidar-learnt_2021-07-14-14-15-05' + '/forward_camera_filtered_sample'
    os.mkdir(dst_path)

    count = 0
    file_list = os.listdir(target_path)
    file_list.sort()
    for image_file in tqdm(file_list):
        if image_file.endswith('.png'):
            if count % 3 ==0:
                src_file_path = os.path.join(target_path, image_file)
                dst_file_path = os.path.join(dst_path, image_file)
                print('dst_file_path',dst_file_path)
                shutil.copy(src_file_path,
                                    dst_file_path)
            count+=1

