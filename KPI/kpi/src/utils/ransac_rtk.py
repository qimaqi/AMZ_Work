import numpy as np
import os
from sklearn.linear_model import RANSACRegressor
import pymap3d as pm
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import pickle

from scipy.spatial.transform import Rotation
RTK_COG_OFFSET = 1.166  # meter

def get_tmat_at_timestamp(transform_data: np.ndarray, timestamp_ns: int) -> np.ndarray:
    """
    Given a numpy array of tranform data for a specific parent-child pair,
    return the transformation matrix at a given timestamp.
    """

    # print('transform_data[:, 0] ',transform_data[:, 0] )
    # print('timestamp_ns',timestamp_ns)
    # 1622997016045964032
    # 1622997029332927225

    tmat = np.zeros((4, 4))
    # index = np.where(transform_data[:, 0] == timestamp_ns)
    # find the closest one
    time_diff = np.abs(transform_data[:, 0]-timestamp_ns)
    nearest_index = np.argmin(time_diff)
    index = nearest_index
    # print('time difference between cone timestamps and nearest tf',time_diff[nearest_index])
    if index is None:
        raise RuntimeError("No transform found: check that transform and boundary timestamps match")

    quaternion = transform_data[index, 4:8].reshape((1, 4))
    # print('quaternion',quaternion)
    tmat[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    tmat[:3, 3] = transform_data[index, 1:4]
    # print('tmat',tmat)
    return tmat


gnss_projection_trans = Transformer.from_crs(
        "epsg:4326",
        "+proj=utm +zone=32 +ellps=WGS84",
        always_xy=True,
    )

def load_csv(filename):
    # gnss_path = os.path.join(folder_path, 'gnss.csv')
    if not os.path.isfile(filename):
        raise RuntimeError(f"File {filename} doesn't exist.")
    data_array = np.loadtxt(filename, delimiter=',')
    return data_array

def load_dataset(foldername):
    gnss = load_csv(os.path.join(foldername, 'gnss.csv'))
    steering = load_csv(os.path.join(foldername, 'steering.csv'))
    cones_df = pd.read_csv(os.path.join(foldername, 'cones.csv'), sep=',')
    cones = np.array(cones_df)
    tf_path = os.path.join(foldername, 'tf.pickle')
    with open(tf_path, 'rb') as f:
        tf_ = pickle.load(f)
    return gnss, steering, cones, tf_

def extract_rtk_cart(gnss, origin):
    gnss_enu = pm.geodetic2enu(gnss[:,0], gnss[:,1], gnss[:,2], \
        origin[0], origin[1], origin[2], ell=pm.utils.Ellipsoid('wgs84'))
    rtk_cart_x = gnss_enu[0]
    rtk_cart_y = gnss_enu[1]
    return rtk_cart_x, rtk_cart_y

def ransac_rtk(rtk_cart_x, rtk_cart_y, segment_size):
    rtk_inlier = []
    for i in range(np.floor(np.size(rtk_cart_x)/segment_size)):
        x = rtk_cart_x[i*segment_size:(i+1)*segment_size]
        y = rtk_cart_y[i*segment_size:(i+1)*segment_size]
        estimator = RANSACRegressor(random_state=0).fit(x, y)
    return

def extract_rtk_cart_2(gnss,gnss_projection_trans,gnss_null_x, gnss_null_y, start_frame, end_frame):

    # print('gnss_null_x, gnss_null_y',gnss_null_x, gnss_null_y)
    rtk_cart_x = []
    rtk_cart_y = []
    for ii in range(start_frame, end_frame):
        gnss_long = gnss[ii,1]
        gnss_lat = gnss[ii,2]
        gnss_x, gnss_y = gnss_projection_trans.transform(gnss_long, gnss_lat)
        rtk_cart_x.append(gnss_x - gnss_null_x)
        rtk_cart_y.append(gnss_y - gnss_null_y)
    return np.array(rtk_cart_x), np.array(rtk_cart_y)



def extract_rtk_gtmd_2(gnss_gtmd,gnss_projection_trans,gnss_null_x, gnss_null_y ):
    longitude_avg_start, latitude_avg_start = np.average(gnss_gtmd[0:100, 1:3], axis=0)
    rtk_cart_x = []
    rtk_cart_y = []
    for ii in range(len(gnss_gtmd)):
        gnss_lat = gnss_gtmd[ii,1]
        gnss_long = gnss_gtmd[ii,2]
        gnss_x, gnss_y = gnss_projection_trans.transform(gnss_long, gnss_lat)
        rtk_cart_x.append(gnss_x - gnss_null_x)
        rtk_cart_y.append(gnss_y - gnss_null_y)
    return np.array(rtk_cart_x), np.array(rtk_cart_y)




go, s, co, tf_ = load_dataset('06-06/')
egomotion_to_world = np.array(tf_['egomotion_to_world']).astype(np.float128)
# print('egomotion_to_world',egomotion_to_world)
# print('total rtk frame', len(go))
c = co[:,1:].astype('float')
# g = go[:,1:]
# origin = g[0,1:4]
frames = [] # for storing the generated images
fig = plt.figure()

longitude_avg_start, latitude_avg_start = np.average(go[0:100, 1:3], axis=0)
heading_null = np.average(go[0:50, 7])
start_frame_list = np.arange(600,len(go),2)

phi = (heading_null + 90) / 180.0 * np.pi
rot_mat = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

gnss_null_x, gnss_null_y = gnss_projection_trans.transform(longitude_avg_start, latitude_avg_start)

for i, start_frame_i in enumerate(start_frame_list):
    end_frame_i = start_frame_i + 10
    end_time_stamps = go[end_frame_i,0]
    # print('end_time_stamps',end_time_stamps)
    tmat = get_tmat_at_timestamp(egomotion_to_world, end_time_stamps)
    car_pos_world = np.array([tmat[0,3] - RTK_COG_OFFSET ,tmat[1,3]])
    car_pos_rtk = np.matmul(rot_mat.T, car_pos_world.T).T
    tf_car_x = car_pos_rtk[0]
    tf_car_y = car_pos_rtk[1]
    # = gtmd_2d = np.matmul(rot_mat, gtmd_2d.T).T
    print('car position tf',tf_car_x,tf_car_y)
    
    gx, gy = extract_rtk_cart_2(go, gnss_projection_trans,gnss_null_x, gnss_null_y ,start_frame = start_frame_i, end_frame=end_frame_i)

    cx, cy = extract_rtk_gtmd_2(co, gnss_projection_trans,gnss_null_x, gnss_null_y )
# gx, gy = extract_rtk_cart2(g[:,1:], origin)
# cx, cy = extract_rtk_cart(c, c[0,1:4])

    fig, ax = plt.subplots()
    ax.plot(gx, gy, linewidth=1.0)
    ax.scatter(gx[-1], gy[-1],c='green')
    print('car position rtk',gx[-1], gy[-1])
    ax.scatter(tf_car_x, tf_car_y,c='orange') # tf coordinate
    # need to change this coordinate to rtk coordinate
    # based on the init heading
    # can we get the heading angle 
    ax.scatter(cx, cy,c='red')
    plt.savefig(os.path.join('/home/qimaqi/Downloads/06-06/video_folder', str(i).zfill(5)+'.png'))
    plt.close('all')


    # frames.append(fig)
# plt.show()

# save a video to see the car moving
# ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
#                                 repeat_delay=1000)
# # ani.save('movie.mp4')
# plt.show()