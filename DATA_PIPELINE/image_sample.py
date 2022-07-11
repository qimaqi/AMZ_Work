import os
import glob
import shutil

for data_folder in glob.glob('./*/*/data/*/forward_camera'):
    print('data folder',data_folder)
    target_path = data_folder
    #'/home/qimaqi/Downloads/datasets/wet/2021-07-14_Duebendorf/data/autocross-lidar-cone-color-camera_2021-07-14-14-48-36' + '/forward_camera_filtered'
    dst_path = data_folder +'_sample'
    #'/home/qimaqi/Downloads/datasets/wet/2021-07-14_Duebendorf/data/autocross-lidar-learnt_2021-07-14-14-15-05' + '/forward_camera_filtered_sample'
    os.mkdir(dst_path)

    count = 0
    file_list = os.listdir(target_path)
    file_list.sort()
    for image_file in file_list:
        if image_file.endswith('.png'):
            if count % 5 ==0:
                src_file_path = os.path.join(target_path, image_file)
                dst_file_path = os.path.join(dst_path, image_file)
                shutil.copy(src_file_path,
                                    dst_file_path)
            count+=1

