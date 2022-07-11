

import cv2
import os
import tqdm
import argparse
from glob import glob

parser = argparse.ArgumentParser(description='Process the path .')
parser.add_argument('-p', type=str, metavar='path',required=True,
                    help='Path example')
parser.add_argument('-r','--rate', type=str, metavar='rate',default=1,
                    help='frame rate')
parser.add_argument('-o','--output', type=str, metavar='output',default='out.avi',
                    help='output name')
args = parser.parse_args()

frame_path = args.p 
frame_rate = args.rate
output=args.output

print(glob(os.path.join(frame_path ,'/%8d.png')))

# import ffmpeg
# (
#     ffmpeg
#     .pattern_type(glob)
#     .input(os.path.join(frame_path ,'*.png'), framerate = 1)
#     .output(os.path.join(frame_path,'moive.mp4'))
#     .run()
# )
# using terminal: ffmpeg -framerate 10 -pattern_type glob -i "*.png" out.mkv

images = [img for img in os.listdir(frame_path) if img.endswith(".png")]
# reformulate the imgs
img_name_list = []
for img_name in images:
    img_name_list.append(img_name)
img_name_list.sort()
frame = cv2.imread(os.path.join(frame_path, images[0]))
height, width, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
save_path = os.path.join(frame_path, output)
# fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video = cv2.VideoWriter(save_path, fourcc, frame_rate, (width, height),isColor =True)
count = 0
for img_name in tqdm.tqdm(img_name_list):
    if count>0:
        video.write(cv2.imread(os.path.join(frame_path, img_name)))
    count+=1
    

# cv2.destroyAllWindows()
# video.release()
