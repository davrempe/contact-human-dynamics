import os
import sys
import logging
import argparse
import glob

import numpy as np
import cv2 as cv
import copy
import optimize_trajectory
import math

sys.path.extend(['skeleton_fitting/ik'])
import BVH as BVH
import Animation as Animation

sys.path.append('utils')
import openpose_utils
import totalcap_utils
from character_info_utils import get_character_toe_inds, get_character_ankle_inds

# Total capture only works with this resolution
TOTAL_CAP_HEIGHT = 1080
TOTAL_CAP_WIDTH = 1920

# Use the same intrinsics assumed by monocular total capture
MTC_FOCAL_LENGTH = [2000.0, 2000.0] 
MTC_PP = [TOTAL_CAP_WIDTH / 2, TOTAL_CAP_HEIGHT / 2]

def optimize_2d_3d(input_path, skel_path, output_path,
                   visualize=False,
                   min_idx=0,
                   max_idx=100,
                   character='ybot',
                   use_gt_floor=False):  
  ''' 
  Full pre-processing kinematic optimization. Targets the combined skeleton and optimization to match 
  2D and 3D joints, smoothing, and foot contacts.
  '''

  if not os.path.exists(output_path):
    os.mkdir(output_path)

  #read 2d and 3d pose estimations
  input_dir = '/'.join(input_path.split('/')[:-1])
  openpose_dir = os.path.join(input_dir, 'openpose_result')
  totalcap_path = os.path.join(input_dir, 'tracked_results.json')
  foot_contacts_path = os.path.join(input_dir, 'foot_contacts.npy')
  if not os.path.isdir(openpose_dir):
    print('Could not find openpose results in ' + openpose_dir + '!')
    return
  if not os.path.isfile(totalcap_path):
    print('Could not find total capture results!')
    return
  if not os.path.isfile(foot_contacts_path):
    print('Could not find foot contact labels!')
    return

  poses2D = openpose_utils.load_keypoint_dir(openpose_dir) # N x J x 3
  if visualize:
    print('Showing input 2d keypoints...')
    openpose_utils.render_2d_keypoints(poses2D, flipy=True)
  joint_conf_2d = poses2D[:,:,2] # confidence in openpose detection
  poses2D = poses2D[:,:,:2]
  totalcap_res = totalcap_utils.load_totalcap_results(totalcap_path) # N x J x 3
  root_pos, body25_3d, smpl_3d = (totalcap_res.root_trans, totalcap_res.joint3d, totalcap_res.smpl_joint3d)
  body25_root_pos, body25_3d = totalcap_utils.normalize_root_pos(root_pos, body25_3d) # make sure mid hips are always at 0 (aka joint position are root-relative)
  smpl_root_pos, smpl_3d = totalcap_utils.normalize_root_pos(root_pos, smpl_3d, root_idx=totalcap_utils.SMPL_ROOT_IDX)
  smpl_joint_rot = totalcap_res.smpl_joint_angles.copy()

  # combine to single skeleton
  poses3D = totalcap_utils.create_combined_model(body25_3d, smpl_3d)
  root_pos = body25_root_pos.copy() # use this root b/c it was used for projection loss in totalcap optimization
  ADJ_LIST = totalcap_utils.COMBINED_ADJ_LIST

  num_frames = max_idx - min_idx
  poses2D = poses2D[min_idx:max_idx]
  joint_conf_2d = joint_conf_2d[min_idx:max_idx]
  poses3D = poses3D[min_idx:max_idx]
  root_pos = root_pos[min_idx:max_idx]
  smpl_joint_rot = smpl_joint_rot[min_idx:max_idx]

  # for visualization/debugging purposes
  body25_root_pos = body25_root_pos[min_idx:max_idx]
  body25_3d = body25_3d[min_idx:max_idx]
  smpl_root_pos = smpl_root_pos[min_idx:max_idx]
  smpl_3d = smpl_3d[min_idx:max_idx]

  # save raw total capture body 25 joints for projection evaluation
  # np.savez(os.path.join(output_path, 'totalcap_body25'), body25_root_pos=body25_root_pos, body25_joints=body25_3d)

  # need to pad poses2D to have the same number of joints as 3D data
  # also set confidence of this padding to 0 so it's not used for projection loss
  num_extra_joints = 3
  if num_extra_joints > 0:
    poses2D = np.concatenate((poses2D, np.zeros((num_frames, num_extra_joints, 2))), axis=1) # there are 3 extra joints in combined skeleton
    joint_conf_2d = np.concatenate((joint_conf_2d, np.zeros((num_frames, num_extra_joints))), axis=1) # confidence to 0

  # read in foot contact data
  # 1 for in contact (i.e. velocity should be 0), 0 otherwise
  num_frames = poses2D.shape[0]
  foot_contacts = np.load(foot_contacts_path)

  # save a clipped version
  np.save(os.path.join(output_path, 'foot_contacts'), foot_contacts[min_idx:max_idx])

  contacts_leftheel = foot_contacts[min_idx:max_idx,0]
  contacts_lefttoe = foot_contacts[min_idx:max_idx,1]
  contacts_rightheel = foot_contacts[min_idx:max_idx,2]
  contacts_righttoe = foot_contacts[min_idx:max_idx,3]
  vel_constraints = np.zeros((num_frames, poses3D.shape[1]))
  vel_constraints[:, 19] = contacts_lefttoe # LBigToe
  vel_constraints[:, 20] = contacts_lefttoe # LSmallToe
  vel_constraints[:, 21] = contacts_leftheel # LHeel
  vel_constraints[:, 22] = contacts_righttoe # RBigToe
  vel_constraints[:, 23] = contacts_righttoe # RSmallToe
  vel_constraints[:, 24] = contacts_rightheel # RHeel

  # use ground truth floor if desired
  plane_normal = None
  plane_point = None
  if use_gt_floor:
    print('Using given floor!')
    floor_file = os.path.join(input_dir, 'floor_gt.txt')
    with open(floor_file, 'r') as f:
      normal_line = f.readline()
      normal_str = normal_line.split(' ')
      plane_normal = np.array([float(x) for x in normal_str])
      point_line = f.readline().split('\n')[0]
      point_str = point_line.split(' ')
      plane_point = np.array([float(x) for x in point_str]) * 100.0 # to cm (originally in blender m)
  

  if visualize:
    print('Showing initial 3D motion and contacts on combined skeleton...')
    viz_pose_3d = poses3D.copy()
    viz_root_trans = root_pos.copy()
    viz_pose_3d[:, :, 1] *= -1.0
    viz_root_trans[:, 1] *= -1.0
    totalcap_utils.visualize_results(viz_root_trans, viz_pose_3d,
                                     contacts=vel_constraints,
                                     adj_list=ADJ_LIST,
                                     show_local=False,
                                     floor_normal=plane_normal,
                                     floor_point=plane_point)

  ppy = MTC_PP[1]
  ppx = MTC_PP[0]
  focal_length = MTC_FOCAL_LENGTH # Same as assumed in total capture

  # load in combined skeleton to optimize on
  skeleton, names, _ = BVH.load(skel_path)
  bone_dict = {}
  for bone_idx in range(len(names)):
    bone_dict[bone_idx] = names[bone_idx]

  # initialize with smpl joint angles
  init_combined_joint_rot = totalcap_utils.combined_angles_from_smpl(smpl_joint_rot)
  
  # perform refinement optimization
  kinematic_optim_res = optimize_trajectory.optimize_trajectory(
                                                                  poses2D, # GT openpose 2d keypoints
                                                                  joint_conf_2d, # confidence in 2d detection (used to weight re-projection term)
                                                                  poses3D, # initial estimate total capture 3D keypoints (relative to root)
                                                                  root_pos, # initial estimate root translation
                                                                  init_combined_joint_rot, # initial estimate of joint anlges
                                                                  skeleton, # animation containing mixamo skeleton to use
                                                                  names, # skeleton joint names
                                                                  ppx, # width of image (that was used for openpose/total capture)
                                                                  ppy, # height of image
                                                                  np.array(focal_length), # fx, fy
                                                                  vel_constraints, # contact labels
                                                                  save_dir=output_path,
                                                                  plane_normal=plane_normal,
                                                                  plane_point=plane_point
                                                                )
  anim, newPose3D, projPose2D, plane_normal, plane_point, new_vel_constraints = kinematic_optim_res

  # save information for later visualization
  # this includes original 2d poses, 3d poses, and optimized 2D and 3D poses
  # np.savez(os.path.join(output_path, 'optim_results'), poses2D=poses2D, poses3D=poses3D, root_pos=root_pos, vel_constraints=vel_constraints,
  #                       newPose2D=projPose2D, newPose3D=newPose3D, plane_normal=plane_normal, plane_point=plane_point)

  # refined contacts
  new_feet_contacts = new_vel_constraints[:,range(19, 25)]
  new_contacts_left_heel = new_feet_contacts[:, 2]
  new_contacts_left_toe = np.logical_or(new_feet_contacts[:,0], new_feet_contacts[:,1]).astype(int)
  new_contacts_right_heel = new_feet_contacts[:, 5]
  new_contacts_right_toe = np.logical_or(new_feet_contacts[:,3], new_feet_contacts[:,4]).astype(int)
  vel_constraints[:, 19] = new_contacts_left_toe # LBigToe
  vel_constraints[:, 20] = new_contacts_left_toe # LSmallToe
  vel_constraints[:, 21] = new_contacts_left_heel # LHeel
  vel_constraints[:, 22] = new_contacts_right_toe # RBigToe
  vel_constraints[:, 23] = new_contacts_right_toe # RSmallToe
  vel_constraints[:, 24] = new_contacts_right_heel # RHeel

  # F x 4
  new_contacts_left_heel = new_contacts_left_heel.reshape((-1,1))
  new_contacts_left_toe = new_contacts_left_toe.reshape((-1,1))
  new_contacts_right_heel = new_contacts_right_heel.reshape((-1,1))
  new_contacts_right_toe = new_contacts_right_toe.reshape((-1,1))
  refined_contacts = np.concatenate([new_contacts_left_heel, new_contacts_left_toe, new_contacts_right_heel, new_contacts_right_toe], axis=1).astype(int)
  # save refined contacts over clipped initial ones
  np.save(os.path.join(output_path, 'foot_contacts'), refined_contacts)

  # also output floor information
  with open(os.path.join(output_path, 'floor_out.txt'), 'w') as floor_file:
    floor_file.write(str(plane_normal[0]))
    floor_file.write(' ')
    floor_file.write(str(plane_normal[1]))
    floor_file.write(' ')
    floor_file.write(str(plane_normal[2]))
    floor_file.write('\n')
    floor_file.write(str(plane_point[0]))
    floor_file.write(' ')
    floor_file.write(str(plane_point[1]))
    floor_file.write(' ')
    floor_file.write(str(plane_point[2]))

  # visualize results
  if visualize:
    viz_results(input_path, output_path, poses2D, poses3D, root_pos, projPose2D, newPose3D, vel_constraints, plane_normal, plane_point)
  
  print('Finished kinematic optimization!')

def viz_results(input_path, output_path, poses2D, poses3D, root_pos, projPose2D, newPose3D, vel_constraints,
                plane_normal=None,
                plane_point=None):
  vidcap = cv.VideoCapture(input_path)
  vid_width = vidcap.get(cv.CAP_PROP_FRAME_WIDTH)   # float
  vid_height = vidcap.get(cv.CAP_PROP_FRAME_HEIGHT) # float

  # first 2D
  print('Visualizing final 2D results...')
  viz_2d_og = poses2D.copy()[:, :25] # show actual openpose detection
  viz_2d_new = projPose2D.copy()
  # must scale down to video since detected on scaled video
  viz_2d_og[:, :, 0] *= (vid_width / 1920.0)
  viz_2d_og[:, :, 1] *= (vid_height / 1080.0)
  viz_2d_new[:, :, 0] *= (vid_width / 1920.0)
  viz_2d_new[:, :, 1] *= (vid_height / 1080.0)
  openpose_utils.render_2d_keypoints([viz_2d_og, viz_2d_new],
                                     adj_list=[openpose_utils.BODY_25_ADJ_LIST, openpose_utils.COMBINED_ADJ_LIST],
                                     video_path=input_path,
                                     flipy=True,
                                     dimensions=(vid_width, vid_height))

  # then 3D overlaid
  viz_3d_og = poses3D.copy()
  viz_root_trans_og = root_pos.copy()
  viz_3d_og[:, :, 1] *= -1.0
  viz_root_trans_og[:, 1] *= -1.0
  viz_3d_og[:,:,:] += np.expand_dims(viz_root_trans_og, 1)
  dummy_trans = np.zeros((viz_3d_og.shape[0], 3))
  # show overlay
  print('Visualizing final 3D result overlaid on initialization...')
  save_path = os.path.join(output_path, 'result_overlay.mp4') #os.path.join('/'.join(input_path.split('/')[:-1]), 'result_overlay.mp4')
  viz_3d_new = newPose3D.copy() # new root trans is baked in
  viz_3d_new[:, :, 1] *= -1.0
  totalcap_utils.visualize_results([dummy_trans]*2, [viz_3d_og, viz_3d_new],
                                    adj_list=[totalcap_utils.COMBINED_ADJ_LIST, totalcap_utils.COMBINED_ADJ_LIST],
                                    show_local=True,
                                    save_path=save_path,
                                    fps=30)

  # then alone with contacts
  print('Visualizing final 3D results with contacts...')
  save_path = os.path.join(output_path, 'result_contacts.mp4') #os.path.join('/'.join(input_path.split('/')[:-1]), 'result_contacts.mp4')
  totalcap_utils.visualize_results([dummy_trans], [viz_3d_new],
                                    adj_list=[totalcap_utils.COMBINED_ADJ_LIST],
                                    contacts=vel_constraints,
                                    show_local=True,
                                    save_path=save_path)
  save_path = os.path.join(output_path, 'result_contacts_floor.mp4') #os.path.join('/'.join(input_path.split('/')[:-1]), 'result_contacts_floor.mp4')
  totalcap_utils.visualize_results([dummy_trans], [viz_3d_new],
                                    adj_list=[totalcap_utils.COMBINED_ADJ_LIST],
                                    contacts=vel_constraints,
                                    show_local=True,
                                    save_path=save_path,
                                    floor_normal=plane_normal,
                                    floor_point=plane_point)

def main(args):
  if args.viz_only:
    result_path = os.path.join(args.output_path, 'optim_results.npz')
    results = np.load(result_path)
    viz_results(args.input_path, args.output_path, results['poses2D'], results['poses3D'], results['root_pos'], 
                results['newPose2D'], results['newPose3D'], results['vel_constraints'], 
                results['plane_normal'], results['plane_point'])
  else:
    optimize_2d_3d(args.input_path, args.skel_path, args.output_path, args.visualize, args.start, args.end, args.character, args.use_gt_floor)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='run 2d/3d pose estimation')
    parser.add_argument(
        '--input_path',
        default='../data/example_data/dance1/dance1.mp4',
        type=str,
        help='Path to source video, total capture 3d pose results and 2d OpenPose results should be contained in the same directory.')
    parser.add_argument(
        '--output_path',
        default='../data/example_data/dance1/kinematic_results',
        type=str,
        help='Path to store result output.')
    parser.add_argument(
        '--skel_path',
        default='skeleton_fitting/combined_body_25.bvh',
        type=str)
    parser.add_argument('--start', type=int, default=0, help='start frame in the video to optimize from')
    parser.add_argument('--end', type=int, default=100, help='end frame in the video to optimize to')
    parser.add_argument('--visualize', 
                        dest='visualize', 
                        action='store_true', 
                        help='Display visualization of initial 2d/3d keypoints and final result after refinement.')
    parser.set_defaults(visualize=False)
    parser.add_argument('--viz-only', 
                        dest='viz_only', 
                        action='store_true', 
                        help='Do not run optimization, only visualize results from previous run (in optim_results.npz).')
    parser.set_defaults(viz_only=False)
    parser.add_argument('--gt-floor', 
                        dest='use_gt_floor', 
                        action='store_true', 
                        help='Will use GT floor read in from floor_gt.txt for optimization')
    parser.set_defaults(use_gt_floor=False)
    parser.add_argument(
        '--character',
        default='ybot',
        help='the character to apply motion to, this decides which mapping from SMPL to character skeleton to use if using simple optimization. If using full, if passed in this will save the totalcap initialization copied to this character (pulled from skeletonfitting/character.bvh',
        type=str)

    args = parser.parse_args()
    main(args)