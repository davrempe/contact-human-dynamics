import os, sys, json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D

sys.path.extend(['skeleton_fitting'])
from character_info_utils import get_character_to_smpl_mapping, mapping_smpl_to_combined_skel, mapping_combined_skel_to_smpl

BODY_25_ROOT_IDX = 8 # BODY_25 format from OpenPose
BODY_25_ADJ_LIST = [[1,8], [1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [8,9], [9,10], [10,11], [8,12], [12,13], [13,14], [1,0], [0,15], [15,17], [0,16], [16,18], [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]

SMPL_ROOT_IDX = 0
SMPL_ADJ_LIST = [[11, 8], [8, 5], [5, 2], [2, 0], [10, 7], [7, 4], [4, 1], [1, 0], [0, 3], [3, 6], [6, 9], [9, 12], [12, 15], [12, 13], [13, 16], [16, 18], [18, 20], [12, 14], [14, 17], [17, 19], [19, 21]]
SMPL_SPINE_JOINTS = [3, 6, 9]

COMBINED_ROOT_IDX = 8
COMBINED_ADJ_LIST = [[1, 27], [27, 26], [26, 25], [25, 8], [1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [8,9], [9,10], [10,11], [8,12], [12,13], [13,14], [1,0], [0,15], [15,17], [0,16], [16,18], [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]

class TotalCapResults():
    ''' Data structure to hold results form Total capture fitting '''
    def __init__(self):
        self.root_trans = None # F x 3
        self.joint3d = None # F x 25 x 3
        self.smpl_joint3d = None # F x 22 x 3
        self.smpl_joint_angles = None # F x 22 x 3 IN RADIANS
        self.body_coeffs = None # F x 30
        self.face_coeffs = None # F x 200

def load_totalcap_results(file_path):
    ''' 
    Loads info from a single results file.
    Returns two np array. root_trans is the global root translation (F, 3).
    joint3d is an (F, J, 3) array with F the number of frames, J the number
    of joints. 
    '''
    if not os.path.isfile(file_path):
        print('Could not find results file')
        return None
    with open(file_path, 'r') as json_file:
        json_dict = json.load(json_file)
    if json_dict is None:
        print('error reading results file')
        return None
    
    frame_dicts = json_dict['totalcapResults']
    joint3d = np.zeros((len(frame_dicts), len(frame_dicts[0]["joints"]), 3))
    smpl_joint3d = np.zeros((len(frame_dicts), len(frame_dicts[0]["SMPLJoints"]), 3))
    smpl_joint_angles = np.zeros((len(frame_dicts), len(frame_dicts[0]["SMPLJoints"]), 3))
    root_trans = np.zeros((joint3d.shape[0], 3))
    body_coeffs = np.zeros((len(frame_dicts), len(frame_dicts[0]["bodyCoeffs"])), dtype=np.float64)
    face_coeffs = np.zeros((len(frame_dicts), len(frame_dicts[0]["faceCoeffs"])), dtype=np.float64)
    for i, frame in enumerate(frame_dicts):
        root_trans[i, :] = np.array([frame['trans']['x'], frame['trans']['y'], frame['trans']['z']])
        # OpenPose Joints
        joints_dicts = frame["joints"]
        for j, joint in enumerate(joints_dicts):
            joint3d[i, j, :] = np.array([joint['pos']['x'], joint['pos']['y'], joint['pos']['z']], dtype=float)
        # SMPL Joints
        joints_dicts = frame["SMPLJoints"]
        for j, joint in enumerate(joints_dicts):
            smpl_joint3d[i, j, :] = np.array([joint['pos']['x'], joint['pos']['y'], joint['pos']['z']], dtype=float)
            smpl_joint_angles[i, j, :] = np.array([joint['rot']['x'], joint['rot']['y'], joint['rot']['z']], dtype=float)
        # Shape coefficients
        body_coeffs[i] = np.array(frame["bodyCoeffs"], dtype=np.float64)
        face_coeffs[i] = np.array(frame["faceCoeffs"], dtype=np.float64)

    res = TotalCapResults()
    res.root_trans = root_trans
    res.joint3d = joint3d
    res.smpl_joint3d = smpl_joint3d
    res.smpl_joint_angles = smpl_joint_angles
    res.body_coeffs = body_coeffs
    res.face_coeffs = face_coeffs
    
    return res

def write_array(file_out, np_array):
    ''' Writes an array on to a single line, each element seperated by one space. '''
    for i in range(np_array.shape[0] - 1):
        file_out.write(str(np_array[i]))
        file_out.write(' ')
    file_out.write(str(np_array[np_array.shape[0] - 1]))
    return

def save_totalcap_results(totalcap_results, out_path):
    '''
    Saves a TotalCapResults object as a series of txt files each containg a single frame of data.
    Each frame includes (root_trans, Adam joint rotations, body coefficients, face basis coefficients).
    This is the same format used by MonoculartotalCapture code to read in results.
    Note only smpl_joint_angles, body_coeffs, face_coeffs, and root_trans are used for saving
    from the object (with dummy hand data appended to joint angles).
    '''
    print('Saving total capture optimized results...')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    num_frames = totalcap_results.smpl_joint_angles.shape[0]
    for i in range(num_frames):
        # create a new output file
        with open(os.path.join(out_path, '%04d.txt' % (i+1)), 'w') as file_out:
            # first current root translation
            write_array(file_out, totalcap_results.root_trans[i])
            file_out.write('\n')
            # then pose parameters
            # add dummy data for hands
            hands_poses = np.zeros((20 + 20, 3))
            all_angles = np.concatenate((totalcap_results.smpl_joint_angles[i], hands_poses), axis=0)
            write_array(file_out, np.reshape(all_angles, (-1)))
            file_out.write('\n')
            # then shape coeffs
            write_array(file_out, totalcap_results.body_coeffs[i])
            file_out.write('\n')
            write_array(file_out, totalcap_results.face_coeffs[i])

def smpl_from_combined_angles(combined_joints):
    ''' 
    Takes a combined joint angles array (F x 28 x 3) and maps it to 
    a SMPL joint angles array (F x 22 x 3).
    '''
    smpl_angles = np.zeros((combined_joints.shape[0], len(mapping_smpl_to_combined_skel), 3))
    for i in range(smpl_angles.shape[0]):
        for j in range(smpl_angles.shape[1]):
            combined_idx = mapping_smpl_to_combined_skel[j]
            if combined_idx != -1:
                smpl_angles[i, j, :] = combined_joints[i, combined_idx, :]
            else:
                smpl_angles[i, j, :] = np.zeros((3))
    return smpl_angles

def combined_angles_from_smpl(smpl_angles):
    ''' 
    Takes a SMPL joint angles array (F x 22 x 3) and maps it to a
    combined joint angles array (F x 28 x 3).
    '''
    combined_angles = np.zeros((smpl_angles.shape[0], len(mapping_combined_skel_to_smpl), 3))
    for i in range(combined_angles.shape[0]):
        for j in range(combined_angles.shape[1]):
            smpl_idx = mapping_combined_skel_to_smpl[j]
            if smpl_idx != -1:
                combined_angles[i, j, :] = smpl_angles[i, smpl_idx, :]
            else:
                combined_angles[i, j, :] = np.zeros((3))
    return combined_angles

def mixamo_angles_from_smpl(smpl_angles, character='ybot'):
    ''' 
    Takes a SMPL joint angles array (F x 22 x 3) and maps it to a
    mixamo joint angles array (F x J x 3).
    '''
    mixamo_mapping = get_character_to_smpl_mapping(character)
    mixamo_angles = np.zeros((smpl_angles.shape[0], len(mixamo_mapping), 3))
    for i in range(mixamo_angles.shape[0]):
        for j in range(mixamo_angles.shape[1]):
            smpl_idx = mixamo_mapping[j]
            if smpl_idx != -1:
                mixamo_angles[i, j, :] = smpl_angles[i, smpl_idx, :]
            else:
                mixamo_angles[i, j, :] = np.zeros((3))
    return mixamo_angles

def create_combined_model(body25_joint3d, smpl_joint3d):
    '''
    Creates the combined model (body 25 with spine joints from smpl added in).
    Given joints should be normalized such that all joint positions are relative to root.
    '''
    if len(SMPL_SPINE_JOINTS) == 0:
        return body25_joint3d
    spine_joints = smpl_joint3d[:, SMPL_SPINE_JOINTS, :]
    combined_joints = np.concatenate((body25_joint3d, spine_joints), axis=1)
    return combined_joints


def normalize_root_pos(root_trans, joint3d, root_idx=BODY_25_ROOT_IDX):
    '''
    Because joint positions are regressed from body shape, the resulting root joint will not always be at the
    origin. This function removes translation of the root joint from the given 3D joint data and moves this
    translation into the global given root_trans. This way all joint positions are relative to the root and
    all global translation happens in root_trans.
    '''
    norm_joint3d = joint3d - joint3d[:, root_idx, :].reshape((joint3d.shape[0], 1, 3))
    norm_root_trans = root_trans + joint3d[:, root_idx, :]
    return norm_root_trans, norm_joint3d

def eval_plane(normal, point, xz):
    ''' eval y value at [x, z] for a given plane defined by a normal and point '''
    y = (normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2] - normal[0]*xz[0] - normal[2]*xz[1]) / normal[1]
    return y

def visualize_results(root_trans, joint3d,
                       show_local=False,
                       contacts=None,
                       adj_list=None,
                       interval=33.33,
                       save_path=None,
                       fps=30,
                       azim=-60,
                       floor_normal=None,
                       floor_point=None,
                       flip_floor=True):
    if not isinstance(joint3d, list):
        joint3d = [joint3d]
        root_trans = [root_trans]
        if adj_list != None:
            adj_list = [adj_list]
    if adj_list is None:
        # didn't pass one in
        adj_list = [BODY_25_ADJ_LIST]*len(joint3d) 

    if not show_local:
        for i in range(len(joint3d)):
            joint3d[i] = joint3d[i] + root_trans[i].reshape((joint3d[i].shape[0], 1, 3))

    draw_floor = False
    if floor_normal is not None and floor_point is not None:
        draw_floor = True
    
    anim = joint3d
    num_joints = anim[0].shape[1]
        
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    maxes = []
    mins = []
    for i in range(len(anim)):
        maxes.append(np.amax(np.amax(anim[i], axis=1), axis=0))
        mins.append(np.amin(np.amin(anim[i], axis=1), axis=0))
    max_all = np.amax(np.array(maxes), axis=0)
    min_all = np.amin(np.array(mins), axis=0)
    max_mov_rad = np.amax(np.abs(max_all - min_all) / 2.0) + 25.0 # padding
    mov_avg = (max_all + min_all) / 2.0
    ax.set_xlim3d(mov_avg[0] - max_mov_rad, mov_avg[0] + max_mov_rad)
    ax.set_zlim3d(mov_avg[1] - max_mov_rad, mov_avg[1] + max_mov_rad)
    ax.set_ylim3d(mov_avg[2] - max_mov_rad, mov_avg[2] + max_mov_rad)

    ax.azim = azim
    acolors = ['g', 'b', 'r']
    
    lines = []
    joints = []
    for i in range(len(anim)):
        lines += [plt.plot([0,0], [0,0], [0,0], color=acolors[i % len(acolors)], 
            lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for _ in range(len(adj_list[i]))]
        joints += [plt.plot([0] ,[0], [0], 'o', color=acolors[i % len(acolors)])[0] for _ in range(anim[i].shape[1])]

    if draw_floor:
        floor_center_xz = mov_avg[[0, 2]]
        floor_center_y = eval_plane(floor_normal, floor_point, floor_center_xz)
        floor_cent = np.array([floor_center_xz[0], floor_center_y, floor_center_xz[1]])

        tan1_xz = floor_center_xz + np.array([-10.0, 0.0])
        tan1_y = eval_plane(floor_normal, floor_point, tan1_xz)
        tan2_xz = floor_center_xz + np.array([0.0, 10.0])
        tan2_y = eval_plane(floor_normal, floor_point, tan2_xz)

        tan1 = np.array([tan1_xz[0], tan1_y, tan1_xz[1]]) - floor_cent
        tan1 /= np.linalg.norm(tan1)
        tan2 = np.array([tan2_xz[0], tan2_y, tan2_xz[1]]) - floor_cent
        tan2 /= np.linalg.norm(tan2)

        tile_width = 25
        tile_diam = 10
        tile_rad = tile_diam // 2
        start_pt = floor_cent + tile_rad*tile_width*tan1 + tile_rad*tile_width*tan2
        for i in range(tile_diam):
            for j in range(tile_diam):
                line1_start = start_pt - i*tile_width*tan1 - j*tile_width*tan2
                line1_end = (start_pt - tile_width*tan1) - i*tile_width*tan1 - j*tile_width*tan2
                if flip_floor:
                    premult = -1.0
                else:
                    premult=1.0
                plt.plot([line1_start[0],line1_end[0]], [line1_start[2],line1_end[2]], [premult*line1_start[1],premult*line1_end[1]], color='r', 
                            lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])

                line2_start = line1_end
                line2_end = line1_end - tile_width*tan2
                plt.plot([line2_start[0],line2_end[0]], [line2_start[2],line2_end[2]], [premult*line2_start[1],premult*line2_end[1]], color='r', 
                            lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])

                if j == tile_diam - 1:
                    line3_start = line2_end
                    line3_end = line2_end + tile_width*tan1
                    plt.plot([line3_start[0],line3_end[0]], [line3_start[2],line3_end[2]], [premult*line3_start[1],premult*line3_end[1]], color='r', 
                                lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])
                if i == 0:
                    line3_end = line2_end + tile_width*tan1
                    line4_start = line3_end
                    line4_end = line3_end + tile_width*tan2
                    plt.plot([line4_start[0],line4_end[0]], [line4_start[2],line4_end[2]], [premult*line4_start[1],premult*line4_end[1]], color='r', 
                                lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])
    
    def animate(i):
        changed = []
        joint_off = 0
        bone_off = 0
        for k in range(len(anim)):
            for j in range(len(adj_list[k])):
                idx1, idx2 = adj_list[k][j]
                # update bones
                lines[j + bone_off].set_data(
                    [ anim[k][i,idx1,0], anim[k][i,idx2,0]],
                    [anim[k][i,idx1,2],       anim[k][i,idx2,2]])
                lines[j + bone_off].set_3d_properties(
                    [ anim[k][i,idx1,1],        anim[k][i,idx2,1]])
            for j in range(anim[k].shape[1]):
                # update joints
                joints[j + joint_off].set_data([anim[k][i,j,0]], [anim[k][i,j,2]])
                joints[j + joint_off].set_3d_properties([anim[k][i,j,1]])
                if contacts is not None and contacts[i, j]:
                    joints[j + joint_off].set_color('r')
                else:
                    joints[j + joint_off].set_color(acolors[k % len(acolors)])
            bone_off += len(adj_list[k])
            joint_off += anim[k].shape[1]

        changed += lines
            
        return changed
        
    plt.tight_layout()
        
    ani = animation.FuncAnimation(fig, animate, np.arange(len(anim[0])), interval=interval)

    if save_path != None:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(save_path, writer=writer)
        
    plt.show()
    plt.close()

def compare_3d_poses(roots, poses, offset=0, show_local=False):
    if len(roots) != len(poses):
        print('Root translations and poses must be same length!')
        return
    viz_poses = []
    viz_roots = []
    for i in range(len(poses)):
        viz_pose_3d = poses[i].copy()
        viz_root_trans = roots[i].copy()
        viz_pose_3d[:, :, 1] *= -1.0 # flip for viz
        viz_root_trans[:, 1] *= -1.0
        # apply offset
        viz_pose_3d[:, :, 0] += i*offset
        viz_root_trans[:, 0] += i*offset
        viz_poses.append(viz_pose_3d)
        viz_roots.append(viz_root_trans)

    visualize_results(viz_roots, viz_poses, show_local=show_local)