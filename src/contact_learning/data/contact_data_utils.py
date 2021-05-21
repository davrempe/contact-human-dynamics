import os, sys, shutil, argparse, subprocess

import numpy as np

sys.path.extend(['utils'])
import openpose_utils

'''

Utility functions to load in data from the synthetic Mixamo
dataset which is generally in the form:

Character1
|___ Motion 1
|________ foot_contacts.npy
|________ viewX
|________ keyoints_viewX
|________ viewX_camera_params.npz
|___ Motion 2
|___ ...
Character2
Character3
....
CharacterN

NOTE: get_all_* functions will return data in the same order so that e.g.
each idx in the returned lists will correspond across view renders, openpose 
keypoints/data, camera params, and foot contacts.

'''

def get_all_character_paths(data_root):
    ''' 
    Given the root directory, returns a list of paths to character dirs in the dataset.
    '''
    if not os.path.exists(data_root):
        print('[contact_data_utils.get_character_paths] Could not \
                find directory of data: ' + data_root)
        return []
    character_dirs = sorted([os.path.join(data_root, f) for f \
                     in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f)) \
                     and f[0] != '.'])
    return sorted(character_dirs)

def get_character_motion_paths(character_path):
    '''
    Given the base directory for a character, returns a list of paths for the contained motions.
    '''
    if not os.path.exists(character_path):
        print('[contact_data_utils.get_motion_paths] Could not \
                find directory of character data: ' + character_path)
        return []
    motion_dirs = sorted([os.path.join(character_path, f) for f \
                in os.listdir(character_path) if os.path.isdir(os.path.join(character_path, f)) \
                and f[0] != '.'])
    return sorted(motion_dirs)

def get_all_motion_paths(character_paths):
    '''
    Given a list of character paths, returns a list of all motion directories contained
    in any of these character directories.
    '''
    motion_paths = []
    for character_dir in character_paths:
        motion_paths += get_character_motion_paths(character_dir)
    return motion_paths

def get_motion_view_paths(motion_path):
    '''
    Given a single motion path, collects and returns all directories containing
    rendered frames from viewpoints (viewN).
    '''
    if not os.path.exists(motion_path):
        print('[contact_data_utils.get_motion_view_paths] Could not \
                find directory of motion data: ' + motion_path)
        return []
    render_dirs = sorted([os.path.join(motion_path, f) for f \
                in os.listdir(motion_path) if os.path.isdir(os.path.join(motion_path, f)) \
                and f[:4] == 'view'])
    return sorted(render_dirs)

def get_all_view_paths(motion_paths):
    '''
    Given a list of motion directories, returns paths to all all directories containing
    rendered frames from viewpoints (viewN).
    '''
    view_paths = []
    for motion_dir in motion_paths:
        view_paths += get_motion_view_paths(motion_dir)
    return view_paths

def get_motion_vid_paths(motion_path):
    '''
    Given a single motion path, collects and returns the path of all rendered
    videos in the motion directory.
    '''
    motion_name = motion_path.split('/')[-1]
    motion_view_paths = get_motion_view_paths(motion_path)
    motion_vid_paths = []
    for render_img_path in motion_view_paths:
        render_name = render_img_path.split('/')[-1]
        motion_vid_paths.append(os.path.join(motion_path, motion_name + '_' + render_name + '.mp4'))
        if not os.path.exists(motion_vid_paths[-1]):
            print('WARNING: Could not find motion video for ' + motion_name + ' ' + render_name)
    return motion_vid_paths

def get_all_vid_paths(motion_paths):
    '''
    Given a list of motion directories, returns paths for the rendered view videos.
    '''
    vid_paths = []
    for motion_dir in motion_paths:
        vid_paths += get_motion_vid_paths(motion_dir)
    return vid_paths

def get_motion_cam_param_path(motion_path):
    '''
    Given a single motion path, collects and returns all cam param file paths.
    '''
    motion_view_paths = get_motion_view_paths(motion_path)
    cam_param_paths = []
    for render_img_path in motion_view_paths:
        cam_param_paths.append(render_img_path + '_camera_params.npz')
        if not os.path.exists(cam_param_paths[-1]):
            print('WARNING: Could not find camera params in ' + motion_path + '...')
    return cam_param_paths

def get_all_cam_param_paths(motion_paths):
    '''
    Given a list of motion directories, returns paths for the camera parameter data
    for each rendered viewpoint (viewX_camera_params.npz)
    '''
    cam_param_paths = []
    for motion_dir in motion_paths:
        cam_param_paths += get_motion_cam_param_path(motion_dir)
    return cam_param_paths

def get_all_cam_param_data(cam_param_paths):
    '''
    Given a list of camera parameter npz files, loads in the data and returns
    a list of dictionaries containing P, RT, and K matrices.
    '''
    cam_param_data = []
    for param_path in cam_param_paths:
        if not os.path.exists(param_path):
            cam_param_data.append(None)
        else:
            cam_param_data.append(np.load(param_path))
    return cam_param_data

def get_motion_openpose_paths(motion_path):
    '''
    Given a single motion path, collects and returns all OpenPose keypoint dirs.
    '''
    motion_view_paths = get_motion_view_paths(motion_path)
    keypoint_paths = []
    for render_img_path in motion_view_paths:
        render_name = render_img_path.split('/')[-1]
        keypoint_paths.append(os.path.join(motion_path, 'keypoints_' + render_name))
        if not os.path.exists(keypoint_paths[-1]):
            print('WARNING: Could not find OpenPose data in ' + keypoint_paths[-1])
    return keypoint_paths

def get_all_openpose_paths(motion_paths):
    '''
    Given a list of motion directories, returns paths to all contained OpenPose data
    directories (keypoints_viewN).
    '''
    openpose_paths = []
    for motion_dir in motion_paths:
        openpose_paths += get_motion_openpose_paths(motion_dir)
    return openpose_paths

def get_all_openpose_data(openpose_paths):
    '''
    Given a list of OpenPose paths, loads in the joint sequences for each one.
    Returns a list of np arrays that are each N x J x 3 where N is the number
    of frames in the sequence, J is the number of openpose joints (25), and
    the 3 is the (x, y, confidence) of that joint.
    '''
    openpose_data = []
    for keypoint_dir in openpose_paths:
        if not os.path.exists(keypoint_dir):
            joint2d_seq = None
        else:
            joint2d_seq = openpose_utils.load_keypoint_dir(keypoint_dir)
        openpose_data.append(joint2d_seq)
    return openpose_data

def get_all_foot_contact_paths(motion_paths):
    '''
    Given a list of motion directories, returns paths to the foot contact
    data for every view contained within these motion directories. NOTE: 
    this means that the returned list will directly correspond with other returned
    data even though the contacts are the same for every view of a particular motion.
    '''
    foot_contact_paths = []
    for motion_dir in motion_paths:
        num_views = len(get_motion_view_paths(motion_dir))
        cur_contact_path = os.path.join(motion_dir, 'foot_contacts.npy')
        if not os.path.exists(cur_contact_path):
            print('WARNING: Could not find contact data in ' + motion_dir + '...')
        foot_contact_paths += [cur_contact_path]*num_views
    return foot_contact_paths

def get_all_foot_contact_data(foot_contact_paths):
    '''
    Given a list of contact data paths, loads in the contact sequences for each one.
    Returns a list of np arrays that are each N x 4 where N is the number
    of frames in the sequence, 4 is the foot joints in order 
    [left_heel, left_toes, right_heel, right_toes] which each contain a binary
    flag whether the joint is is contact or not.
    '''
    foot_contact_data = []
    for foot_contact_file in foot_contact_paths:
        if not os.path.exists(foot_contact_file):
            foot_contact_data.append(None)
        else:
            foot_contact_data.append(np.load(foot_contact_file))
    return foot_contact_data

def get_frame_paths(view_path):
    '''
    Given a single view path (the directory of rendered images), returns
    the paths of the images contained in the that directory.
    '''
    if not os.path.exists(view_path):
        print('[contact_data_utils.get_frame_paths] Could not find frame path ' + view_path)
        return []
    frame_paths = sorted([os.path.join(view_path, f) for f \
                     in os.listdir(view_path) if f[0] != '.' and f.split('.')[-1] == 'png'])
    return frame_paths

if __name__ == '__main__':
    data_path = './contact_data_gen/data_tex'
    character_paths = get_all_character_paths(data_path)
    print(character_paths)
    all_motion_paths = get_all_motion_paths(character_paths)
    print(all_motion_paths)
    all_openpose_paths = get_all_openpose_paths(all_motion_paths)
    print(all_openpose_paths)
    all_openpose_data = get_all_openpose_data(all_openpose_paths)
    for keypoint_data in all_openpose_data:
        print(keypoint_data.shape)
    all_contact_paths = get_all_foot_contact_paths(all_motion_paths)
    print(all_contact_paths)
    print(len(all_openpose_paths))
    print(len(all_openpose_data))
    print(len(all_contact_paths))
    all_contact_data = get_all_foot_contact_data(all_contact_paths)
    for contact_data in all_contact_data:
        if contact_data is None:
            print('NONE')
        else:
            print(contact_data.shape)
    all_view_paths = get_all_view_paths(all_motion_paths)
    print(all_view_paths)
    print(get_frame_paths(all_view_paths[0]))
    all_cam_param_paths = get_all_cam_param_paths(all_motion_paths)
    print(all_cam_param_paths)
    all_cam_param_data = get_all_cam_param_data(all_cam_param_paths)
    print(len(all_cam_param_data))
    for cam_param_data in all_cam_param_data:
        if cam_param_data is None:
            print('NONE')
        else:
            print(cam_param_data.keys())
    