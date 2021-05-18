import os, sys, json

import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.extend(['optimize'])
from OneEuroFilter import OneEuroFilter

BODY_25_ADJ_LIST = [[1,8], [1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [8,9], [9,10], [10,11], [8,12], [12,13], [13,14], [1,0], [0,15], [15,17], [0,16], [16,18], [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]
COMBINED_ADJ_LIST = [[1, 27], [27, 26], [26, 25], [25, 8], [1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [8,9], [9,10], [10,11], [8,12], [12,13], [13,14], [1,0], [0,15], [15,17], [0,16], [16,18], [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]

def pad_image(im, new_size):
    ''' Pads the image with black so that it is the new size. new_size should be (W, H) '''
    if im.shape[1] >= new_size[0] and im.shape[0] >= new_size[1]:
        return im # already this size
    old_H, old_W = im.shape[:2]
    new_W, new_H = new_size

    delta_W = new_W - old_W
    delta_H = new_H - old_H
    top = delta_H // 2
    bottom = delta_H - top
    left = delta_W // 2
    right = delta_W - left

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return im

def resize_image(im, new_size):
    ''' 
    Resizes the image and interpolates so that it is the given new_size (W, H).
    Keeps original image aspect ratio, only one dimension will be the desired size.
    '''
    if im.shape[1] == new_size[0] or im.shape[0] == new_size[1]:
        return im # already this size
    old_H, old_W = im.shape[:2]
    des_W, des_H = new_size
    width_ratio = float(des_W) / old_W
    height_ratio = float(des_H) / old_H
    width_min = width_ratio < height_ratio
    new_W = des_W if width_min else min(des_W, int(old_W * height_ratio))
    new_H = des_H if not width_min else min(des_H, int(old_H * width_ratio))
    
    im = cv2.resize(im, (new_W, new_H), interpolation=cv2.INTER_AREA) #cv2.INTER_CUBIC)
    return im

def load_keypoint_file(file_path, num_joints=25):
    ''' 
    Loads info from a single keypoint file.
    Only returns 2d pose keypoints of the first person.
    If no people are detected this frame, returns (1, 3) of all 0,
    otherwise it's (num_keypoints, 3). It's 3 because [x, y, confidence].
    '''
    if not os.path.isfile(file_path):
        print('Could not find keypoint file')
        return None
    with open(file_path, 'r') as json_file:
        json_dict = json.load(json_file)
    if json_dict is None:
        print('error reading keypoint file')
        return None
    if len(json_dict['people']) == 0:
        # print('no people found')
        return np.zeros((num_joints, 3))

    joint2d = np.array(json_dict['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
    return joint2d

def load_keypoint_dir(dir_path):
    ''' Loads keypoints for an entire video contained in a single directory '''
    if not os.path.isdir(dir_path):
        return None
    keypoint_files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.split('.')[-1] == 'json'])
    joint2d_list = [load_keypoint_file(f) for f in keypoint_files]
    return np.stack(joint2d_list, axis=0)

def filter_poses(poses, fcmin=0.05, beta=0.4, freq=1):
    config = {
        'freq': freq,  # Hz
        'mincutoff': fcmin,  
        'beta': beta,  
        'dcutoff': 1.0 
    }

    poses_filtered = 0 * poses
    for j in range(poses.shape[1]):
        for i in range(poses.shape[2]):
            f = OneEuroFilter(**config)
            timestamp = 0.0
            for t in range(poses.shape[0]):
                filtered = f(poses[t, j, i], t)
                timestamp += 1.0 / config["freq"]
                poses_filtered[t, j, i] = filtered

    return poses_filtered

def render_2d_keypoints(joint2d, video_path=None, joint_colors=[None], adj_list=None, flipy=True, dimensions=(1920, 1080), conf_thresh=0.1):
    ''' 
    Renders 2D keypoints. If video path is specified, overlays on the original video.
    Specify the color of joints at each frame with a 2d list joint_colors = (FxJ) with each entry a color i.e. 'g' or 'r'
    '''
    if not isinstance(joint2d, list):
        joint2d = [joint2d]
        if adj_list != None:
            adj_list = [adj_list]
        if joint_colors != [None]:
            joint_colors = [joint_colors]
    if not isinstance(joint_colors, list):
        joint_colors = [joint_colors]
    if adj_list is None:
        # didn't pass one in
        adj_list = [BODY_25_ADJ_LIST]*len(joint2d) 

    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_vid_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ret, frame = cap.read()
        # frame = pad_image(frame, (1920, 1080))
        # frame = resize_image(frame, dimensions)
        height, width, _ = frame.shape
    else:
        height = dimensions[1]
        width = dimensions[0]

    ax = plt.subplot(111)
    fig = plt.gcf()
    fig_num = fig.number
    plt.ion()
    if video_path is None:
        num_frames = joint2d[0].shape[0]
    else:
        num_frames = int(min([joint2d[0].shape[0], num_vid_frames]))

    valid_thresh = conf_thresh

    color_list = ['g', 'r', 'b']
    if joint_colors == [None]:
        joint_colors = []
        for i in range(len(joint2d)):
            joint_colors.append([[color_list[i % len(color_list)]]*joint2d[i].shape[1]]*joint2d[i].shape[0])

    for i in range(num_frames):
        i = i % num_frames
        ax.clear()
        # plt.axis('off')
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        if flipy:
            plt.gca().invert_yaxis()
        # plot joints (have to do sequentially for coloring)
        for k in range(len(joint2d)):
            for j in range(joint2d[k].shape[1]):
                if joint2d[k].shape[2] < 3 or joint2d[k][i,j,2] > valid_thresh:
                    ax.plot(joint2d[k][i, j, 0], joint2d[k][i, j, 1], joint_colors[k][i][j] + 'o')
            # plot skeleton
            for pair in adj_list[k]:
                if joint2d[k].shape[2] < 3 or np.sum(joint2d[k][i,pair,2] > valid_thresh) == 2:
                    plt.plot(joint2d[k][i, pair, 0], joint2d[k][i, pair, 1], '-' + color_list[k % 3])
        # plot images
        if video_path is not None:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), extent=(0, width, height, 0))
            if i == num_frames - 1:
                cap = cv2.VideoCapture(video_path)

        plt.pause(0.001)

        if video_path is not None:
            ret, frame = cap.read()
            # frame = pad_image(frame, (1920, 1080))
            # frame = resize_image(frame, dimensions)

        if not plt.fignum_exists(fig_num):
            break

    plt.close()
    plt.ioff()
