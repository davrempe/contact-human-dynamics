
'''
This script has 2 functionalities. If the --viz flags is passed in, it visualizes results of the physical optimization.
Otherwise, it processes and writes out the necessary input data to perform the physical optimization.
'''

import os, sys, argparse, shutil

from copy import copy, deepcopy

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.patheffects as pe

sys.path.append('../skeleton_fitting/ik')
import BVH
import Animation
from Quaternions import Quaternions
from InverseKinematics import JacobianInverseKinematicsCK, JacobianInverseKinematics
sys.path.append('../optimize')
from OneEuroFilter import OneEuroFilter

from character_info_utils import get_character_upper_body, get_character_leg_chain, get_character_toe_inds, get_character_ankle_inds, get_character_hip_inds, get_character_mass, get_character_heel_inds # skeleton joint inds
from character_info_utils import get_character_seg_to_joint_map, get_character_seg_to_mass_perc_map # mappings
from character_info_utils import heeled_characters

class TowrResults():
    ''' Data structure to hold results form Total capture fitting '''
    def __init__(self):
        self.num_feet = None
        self.dt = None 
        self.base_pos = None # F x 3
        self.base_rot = None # F x 3 IN DEGREES
        self.base_R = None # F x 3 x 3
        self.feet_pos = None # F x nFeet x 3
        self.feet_force = None # F x nFeet x 3
        self.feet_contact = None # F x nFeet

    def copy_from(self, res):
        self.num_feet = res.num_feet 
        self.dt = res.dt 
        self.base_pos = res.base_pos.copy() if not res.base_pos is None else None # F x 3
        self.base_rot = res.base_rot.copy() if not res.base_rot is None else None # F x 3 IN DEGREES
        self.base_R = res.base_R.copy() if not res.base_R is None else None # F x 3 x 3
        self.feet_pos = res.feet_pos.copy() if not res.feet_pos is None else None # F x nFeet x 3
        self.feet_force = res.feet_force.copy() if not res.feet_force is None else None # F x nFeet x 3
        self.feet_contact = res.feet_contact.copy() if not res.feet_contact is None else None  # F x nFeet

def load_results(file_path, flip_coords=True):
    ''' Loads in results from towr optimization and transforms them back into our original coordinates '''
    if not os.path.exists(file_path):
        print('Could not find results file ' + file_path)
        return

    lines = []
    with open(file_path, 'r') as res_file:
        lines = res_file.readlines()

    lines = [line.replace('\n', '') for line in lines]

    results = TowrResults()
    idx = 1
    results.dt = float(lines[idx])
    idx += 2
    num_frames = int(lines[idx])
    idx += 2
    results.num_feet = int(lines[idx])
    idx += 2
    # COM position
    base_lin = [float(x) for x in lines[idx].split(' ')]
    results.base_pos = np.reshape(np.array(base_lin), (num_frames, 3))
    idx += 2
    # COM orientation
    base_ang = [float(x) for x in lines[idx].split(' ')]
    results.base_rot = np.reshape(np.array(base_ang), (num_frames, 3))
    idx += 2
    # end-effector position
    results.feet_pos = []
    for foot_idx in range(results.num_feet):
        foot_pos = [float(x) for x in lines[idx].split(' ')]
        results.feet_pos.append(np.reshape(np.array(foot_pos), (num_frames, 1, 3)))
        idx += 2
    results.feet_pos = np.concatenate(results.feet_pos, axis=1)
    # end-effector contact forces
    results.feet_force = []
    for foot_idx in range(results.num_feet):
        foot_force = [float(x) for x in lines[idx].split(' ')]
        results.feet_force.append(np.reshape(np.array(foot_force), (num_frames, 1, 3)))
        idx += 2
    results.feet_force = np.concatenate(results.feet_force, axis=1)
    # end-effector contacts
    results.feet_contact = []
    for foot_idx in range(results.num_feet):
        foot_contact = [int(x) for x in lines[idx].split(' ')]
        results.feet_contact.append(np.reshape(np.array(foot_contact), (num_frames, 1)))
        idx += 2
    results.feet_contact = np.concatenate(results.feet_contact, axis=1)

    # for now switch y,z
    results.base_pos = results.base_pos[:, [0,2,1]]
    if flip_coords:
        results.base_pos[:, :] *= -1.0
    for foot_idx in range(results.num_feet):
        results.feet_pos[:,foot_idx,:] = results.feet_pos[:, foot_idx, [0,2,1]]
        if flip_coords:
            results.feet_pos[:, foot_idx, :] *= -1.0
    for foot_idx in range(results.num_feet):
        results.feet_force[:,foot_idx,:] = results.feet_force[:, foot_idx, [0,2,1]]
        if flip_coords:
            results.feet_force[:, foot_idx, :] *= -1.0

    # rotation is x,y,z(up) so need to swap so y is up
    rot_angle, rot_axis = Quaternions.from_euler(np.expand_dims(np.radians(results.base_rot), axis=1), order='xyz', world=True).angle_axis()
    rot_axis = rot_axis[:, :, [0,2,1]] # swap y and z
    if flip_coords:
        rot_axis[:, :, :] *= -1.0
    results.base_rot = Quaternions.from_angle_axis(rot_angle, rot_axis).euler(order='xyz')[:,0,:]
    results.base_R = Quaternions.from_angle_axis(rot_angle, rot_axis).transforms()[:,0,:]
    
    return results

def plot_3curve(res_arr, dt, xlabel='', ylabel='', save_file=None, show=True):
    fig = plt.figure()

    t = np.arange(res_arr.shape[0]) * dt
    plt.plot(t, res_arr[:,0], '-r', label='X')
    plt.plot(t, res_arr[:,1], '-g', label='Y')
    plt.plot(t, res_arr[:,2], '-b', label='Z')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    if save_file is not None:
        plt.savefig(save_file)

    if show:
        plt.show()

    plt.close(fig)

def eval_plane(normal, point, xz):
    ''' eval y value at [x, z] for a given plane defined by a normal and point '''
    y = (normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2] - normal[0]*xz[0] - normal[2]*xz[1]) / normal[1]
    return y


def viz_results(towr_results, skeletons, start_idx, end_idx,
                floor_normal=None, floor_point=None, flip_floor=True,
                interval=33.33, save_path=None,
                draw_trace=True, draw_forces=True, draw_towr=True, draw_skel=True, 
                fps=30, draw_floor=True, flip_anim=[1], show=True, names=[]):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    offset = np.array([[2.0, 0.0, 0.0]]) # in meters

    for extra_res_idx in range(1, len(towr_results)):
        res_copy = TowrResults()
        res_copy.copy_from(towr_results[extra_res_idx])
        # then apply our offset
        res_copy.base_pos += offset * extra_res_idx
        for foot_idx in range(res_copy.num_feet):
            res_copy.feet_pos[:, foot_idx] += offset * extra_res_idx

        towr_results[extra_res_idx] = res_copy 

    contacts = [res.feet_contact for res in towr_results]

    body_pos = [Animation.positions_global(skeleton) * 0.01 for skeleton in skeletons] # to meters
    for jnt_idx in flip_anim:
        for cur_body_pos in body_pos:
            cur_body_pos[:, :, jnt_idx] *= -1.0

    # offset body positions too
    for extra_res_idx in range(1, len(skeletons)):
        body_pos[extra_res_idx] += offset * extra_res_idx

    base_pos = [res.base_pos for res in towr_results]

    maxes = np.amax(np.concatenate(base_pos, axis=0), axis=0)
    mins = np.amin(np.concatenate(base_pos, axis=0), axis=0)
    move_rad = (np.amax(np.abs(maxes - mins) / 2.0) + 1.0) / 1.5
    mov_avg = (maxes + mins) / 2.0
    ax.set_xlim3d(mov_avg[0] - move_rad, mov_avg[0] + move_rad)
    ax.set_zlim3d(mov_avg[1] - move_rad, mov_avg[1] + move_rad)
    ax.set_ylim3d(mov_avg[2] - move_rad, mov_avg[2] + move_rad)

    ax.set_xlabel('x')
    ax.set_ylabel('z')

    plt.axis('off')
    plt.grid(b=None)

    ax.azim = -127.15384
    ax.elev = 15.35064

    if draw_floor and floor_normal is not None and floor_point is not None:
        draw_floor = True
    else:
        draw_floor = False

    skel_color = ['r', 'purple', 'blue', 'green']
    base_color = 'orange'
    feet_color = ['g', 'b', 'pink', 'purple']
    forces_color = 'r'
    robot_color = 'yellow'

    if not names is None and len(names) == len(towr_results):
        text_offset = np.array([0.0, 1.3, 0.0])
        for res_idx in range(len(towr_results)):
            text_pos = np.mean(towr_results[res_idx].base_pos, axis=0) + text_offset
            ax.text(text_pos[0], text_pos[2], text_pos[1], names[res_idx], color=skel_color[res_idx])

    if draw_trace:
        for cur_towr_result in towr_results:
            for i in range(cur_towr_result.base_pos.shape[0] - 1):
                from_pt = cur_towr_result.base_pos[i, :]
                to_pt = cur_towr_result.base_pos[i+1, :]
                plt.plot([from_pt[0], to_pt[0]], [from_pt[2], to_pt[2]], [from_pt[1], to_pt[1]], lw=3, color=base_color, linestyle='dashed')

                for j in range(cur_towr_result.feet_pos.shape[1]):
                    from_pt = cur_towr_result.feet_pos[i, j, :]
                    to_pt = cur_towr_result.feet_pos[i+1, j, :]
                    plt.plot([from_pt[0], to_pt[0]], [from_pt[2], to_pt[2]], [from_pt[1], to_pt[1]], lw=3, color=feet_color[j], linestyle='dashed')

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

        tile_width = 0.5
        tile_diam = 20
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
                plt.plot([line1_start[0],line1_end[0]], [line1_start[2],line1_end[2]], [premult*line1_start[1],premult*line1_end[1]], color='grey', 
                            lw=2, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])

                line2_start = line1_end
                line2_end = line1_end - tile_width*tan2
                plt.plot([line2_start[0],line2_end[0]], [line2_start[2],line2_end[2]], [premult*line2_start[1],premult*line2_end[1]], color='grey', 
                            lw=2, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])

                if j == tile_diam - 1:
                    line3_start = line2_end
                    line3_end = line2_end + tile_width*tan1
                    plt.plot([line3_start[0],line3_end[0]], [line3_start[2],line3_end[2]], [premult*line3_start[1],premult*line3_end[1]], color='grey', 
                                lw=2, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])
                if i == 0:
                    line3_end = line2_end + tile_width*tan1
                    line4_start = line3_end
                    line4_end = line3_end + tile_width*tan2
                    plt.plot([line4_start[0],line4_end[0]], [line4_start[2],line4_end[2]], [premult*line4_start[1],premult*line4_end[1]], color='grey', 
                                lw=2, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])

    if draw_towr:
        base_joints = [plt.plot([0] ,[0], [0], 'o', color=base_color, markersize=10.0)[0] for i in range(len(towr_results))]

    all_joints = []
    bone_lines = []
    if draw_skel:
        for skel_idx in range(len(skeletons)):
            cur_all_joints = []
            cur_bone_lines = []
            for i in range(body_pos[skel_idx].shape[1]):
                cur_all_joints.append(plt.plot([0] ,[0], [0], 'o', color=skel_color[skel_idx])[0])
            all_joints.append(cur_all_joints)
            for i in range(body_pos[skel_idx].shape[1] - 1):
                cur_bone_lines.append(plt.plot([0,0], [0,0], [0,0], color=skel_color[skel_idx], lw=3, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0])
            bone_lines.append(cur_bone_lines)
            
    feet_joints = []
    hip_joints = []
    legs = []
    forces = []
    robot_lines = []
    for res_idx in range(len(towr_results)):
        cur_feet_joints = []
        cur_legs = []
        cur_forces = []
        for i in range(towr_results[res_idx].num_feet):
            if draw_towr:
                cur_feet_joints.append(plt.plot([0] ,[0], [0], 'o', color=feet_color[i], markersize=10.0)[0])
                cur_legs.append(plt.plot([0,0], [0,0], [0,0], color=feet_color[i], lw=3, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0])
            if draw_forces:
                cur_forces.append(plt.plot([0,0], [0,0], [0,0], color=forces_color, lw=1, path_effects=[pe.Stroke(linewidth=3), pe.Normal()])[0])

        feet_joints.append(cur_feet_joints)
        legs.append(cur_legs)
        forces.append(cur_forces)

        if draw_towr:
            cur_robot_lines = []
            robot_verts = np.array([[0.0, 0.0, 0.0], [0.0, -0.25, 0.0], [0.05, -0.2, 0.0], [-0.05, -0.2, 0.0], [0.0, -0.25, 0.0]])
            for i in range(robot_verts.shape[0] - 1):
                cur_robot_lines.append(plt.plot([0,0], [0,0], [0,0], color=robot_color, lw=3, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0])

            robot_lines.append(cur_robot_lines)


    animate_forces = [res.feet_force*0.001 for res in towr_results]
    
    def animate(i):
        # print('azim: ' + str(ax.azim))
        # print('elev: ' + str(ax.elev))
        # print('dist: ' + str(ax.dist))

        changed = []

        if draw_towr:
            for res_idx in range(len(towr_results)):
                base_joints[res_idx].set_data([towr_results[res_idx].base_pos[i, 0]], [towr_results[res_idx].base_pos[i, 2]])
                base_joints[res_idx].set_3d_properties([towr_results[res_idx].base_pos[i, 1]])

                cur_robot_joints = np.dot(towr_results[res_idx].base_R[i], robot_verts.T).T + np.expand_dims(towr_results[res_idx].base_pos[i], axis=0)
                for j in range(len(robot_lines[res_idx])):
                    robot_lines[res_idx][j].set_data(
                        [ cur_robot_joints[j, 0],     cur_robot_joints[j+1, 0]],
                        [cur_robot_joints[j, 2],       cur_robot_joints[j+1, 2]])
                    robot_lines[res_idx][j].set_3d_properties(
                        [ cur_robot_joints[j, 1],        cur_robot_joints[j+1, 1]])

        if draw_skel:
            for skel_idx in range(len(skeletons)):
                for j in range(body_pos[skel_idx].shape[1]):
                    cur_all_joint = body_pos[skel_idx][i, j] 
                    all_joints[skel_idx][j].set_data([cur_all_joint[0]], [cur_all_joint[2]])
                    all_joints[skel_idx][j].set_3d_properties([cur_all_joint[1]])

                for j in range(1, body_pos[skel_idx].shape[1]):
                    cur_all_joint = body_pos[skel_idx][i, j]
                    cur_par_joint = body_pos[skel_idx][i, skeletons[skel_idx].parents[j]]
                    bone_lines[skel_idx][j-1].set_data(
                        [ cur_all_joint[0],     cur_par_joint[0]],
                        [cur_all_joint[2],       cur_par_joint[2]])
                    bone_lines[skel_idx][j-1].set_3d_properties(
                        [ cur_all_joint[1],        cur_par_joint[1]])


        if draw_towr:
            for res_idx in range(len(towr_results)):
                for j in range(towr_results[res_idx].num_feet):
                    feet_joints[res_idx][j].set_data([towr_results[res_idx].feet_pos[i, j, 0]], [towr_results[res_idx].feet_pos[i, j, 2]])
                    feet_joints[res_idx][j].set_3d_properties([towr_results[res_idx].feet_pos[i, j, 1]])

                    if contacts[res_idx] is not None and contacts[res_idx][i, j]:
                        feet_joints[res_idx][j].set_color(forces_color)
                    else:
                        feet_joints[res_idx][j].set_color(feet_color[j])

                    legs[res_idx][j].set_data(
                            [ towr_results[res_idx].base_pos[i, 0], towr_results[res_idx].feet_pos[i, j, 0]],
                            [towr_results[res_idx].base_pos[i, 2],       towr_results[res_idx].feet_pos[i, j, 2]])
                    legs[res_idx][j].set_3d_properties(
                        [ towr_results[res_idx].base_pos[i, 1],        towr_results[res_idx].feet_pos[i, j, 1]])

        if draw_forces:
            for res_idx in range(len(towr_results)):
                for j in range(towr_results[res_idx].num_feet):
                    forces[res_idx][j].set_data(
                        [ towr_results[res_idx].feet_pos[i, j, 0] - animate_forces[res_idx][i, j, 0], towr_results[res_idx].feet_pos[i, j, 0]],
                        [towr_results[res_idx].feet_pos[i, j, 2] - animate_forces[res_idx][i, j, 2],       towr_results[res_idx].feet_pos[i, j, 2]])
                    forces[res_idx][j].set_3d_properties(
                        [towr_results[res_idx].feet_pos[i, j, 1] - animate_forces[res_idx][i, j, 1],       towr_results[res_idx].feet_pos[i, j, 1]])

        changed = legs

        return changed
        
    plt.tight_layout()
        
    ani = animation.FuncAnimation(fig, animate, np.arange(towr_results[0].base_pos.shape[0]), interval=interval)

    if save_path != None:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(save_path, writer=writer)

    if show:
        plt.show()

def add_heel_to_anim(anim, toe_inds, ankle_inds):
    ''' 
    Adds a dummy left and right heel joint as the last 2 joints in this animation. Toe inds should be joint index of [left, right] toe, 
        same for ankles. Places the heel at the same vertical offset as the toes.
    '''
    heel_offsets = np.zeros((2, 3))
    heel_offsets[:,1] = anim.offsets[toe_inds, 1]
    anim.offsets = np.append(anim.offsets, heel_offsets[np.newaxis, 0], axis=0)
    anim.offsets = np.append(anim.offsets, heel_offsets[np.newaxis, 1], axis=0)
    anim.parents = np.append(anim.parents, ankle_inds[0])
    anim.parents = np.append(anim.parents, ankle_inds[1])
    anim.positions = np.append(anim.positions, np.zeros((anim.positions.shape[0], 2, 3)), axis=1)
    heel_rots = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    anim.orients.qs = np.append(anim.orients.qs, heel_rots, axis=0)
    heel_rots = np.tile(heel_rots, [anim.rotations.shape[0], 1, 1])
    anim.rotations.qs = np.append(anim.rotations.qs, heel_rots, axis=1)

    num_frames = anim.positions.shape[0]
    left_heel_idx = anim.positions.shape[1] - 2
    right_heel_idx = anim.positions.shape[1] - 1
    anim.positions[:,[left_heel_idx, right_heel_idx],:] = np.expand_dims(anim.offsets[[left_heel_idx, right_heel_idx],:], axis=0).repeat(num_frames, axis=0)

    return anim, heel_offsets

def remove_heel_from_anim(anim):
    ''' assumes the last 2 joints are heels, and removes these from the animation'''
    old_len = anim.offsets.shape[0] - 2
    anim.offsets = anim.offsets[:old_len]
    anim.parents = anim.parents[:old_len]
    anim.positions = anim.positions[:,:old_len]
    anim.orients = anim.orients[:old_len]
    anim.rotations = anim.rotations[:,:old_len]
    return anim

def find_contact_durations(contacts, dt):
    ''' From sequence of binary contact flags, extract the durations of each contact '''
    prev_state = contacts[0]
    cur_duration = 0.0
    durations = []
    for i in range(0, contacts.shape[0]-1):
        cur_state = contacts[i]
        if cur_state != prev_state:
            durations.append(cur_duration)
            cur_duration = dt
        else:
            cur_duration += dt
        prev_state = cur_state
    durations.append(cur_duration)
    return durations

def prepare_input(anim_bvh, floor_file, contacts_file, out_dir, character, \
                    start_idx=None,
                    end_idx=None,
                    dt=(1.0 / 30.0),
                    combined_contacts=False):
    '''
    This prepares all data input to feed to the trajectory optimization:
       - skel_info.txt : hip offsets, leg length, heel distance, mass, and moments of inertia over time
       - motion_info.txt : COM trajectory/orientation, feet trajectory
       - terrain_info.txt : ground plane normal and one point on the plane
       - contact_info.txt : durations of contacts for toes and heels, and whether they're in contact with the ground at start
    '''
    ''' Writes out the necessary data files passed to physics optimization '''
    if not os.path.exists(anim_bvh):
        print('Could not find animated bvh file ' + anim_bvh)
        return
    if not os.path.exists(floor_file):
        print('Could not find floor file ' + floor_file)
        return
    if not os.path.exists(contacts_file):
        print('Could not find contacts file ' + contacts_file)
        return
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = anim_COM.shape[0]
    
    num_frames = end_idx - start_idx

    # load the actual motion, zero out the root orientation and position, and
    # calculate the inertia matrix for every single timestep
    anim, names, _ = BVH.load(anim_bvh)
    zero_root = np.zeros((anim.shape[0], 3))
    zero_root = Quaternions.from_euler(zero_root, order='xyz', world=True)
    anim.rotations[:,0,:] = zero_root
    anim.positions[:,0,:] = np.zeros((anim.shape[0], 3))
    normalized_positions = Animation.positions_global(anim)

    # find the leg length
    left_leg_chain = get_character_leg_chain(character, 'left')
    bone_lengths = np.linalg.norm(anim.offsets[left_leg_chain[1:]], axis=1)
    max_leg_length = np.sum(bone_lengths) * 0.01 # to meters
    # print('Max leg length (m)')
    # print(max_leg_length)

    # find COM at each frame and move the root there (since our orientations are about the root)
    hip_joints = get_character_hip_inds(character)
    map_segment_to_joints = get_character_seg_to_joint_map(character)
    map_segment_to_mass_perc = get_character_seg_to_mass_perc_map(character)
    frame_coms = []
    hip_offsets = np.zeros((anim.shape[0], 2, 3))
    for frame in range(anim.shape[0]):
        frame_COM = np.zeros(3)
        for key in map_segment_to_joints.keys():
            seg_pos = np.mean(normalized_positions[frame, map_segment_to_joints[key], :], axis=0)
            frame_COM += map_segment_to_mass_perc[key] * 0.01 * seg_pos
        frame_coms.append(frame_COM)

        for j in range(len(hip_joints)):
            hip_offsets[frame, j] = (normalized_positions[frame, hip_joints[j]] - frame_COM)

    frame_coms = np.array(frame_coms)

    hip_offsets *= 0.01 # to meters
    hip_offsets[:,:,:] *= -1.0 # flip
    hip_offsets = hip_offsets[:,:,[0,2,1]] # swap y and z

    anim.positions[:,0,:] -= frame_coms # move so COM is at origin
    normalized_positions = Animation.positions_global(anim) * 0.01
    normalized_positions[:,:,:] *= -1.0 # flip y
    normalized_positions = normalized_positions[:,:,[0,2,1]] # swap y and z
    # then calculate inertia about COM
    total_mass = get_character_mass(character)
    inertia_mats = []
    for frame in range(anim.shape[0]):
        I_frame= np.zeros((3, 3))
        for key in map_segment_to_joints.keys():
            seg_pos = np.mean(normalized_positions[frame, map_segment_to_joints[key], :], axis=0)
            seg_mass = map_segment_to_mass_perc[key] * 0.01 * total_mass            
            I_seg = (np.eye(3)*np.dot(seg_pos, seg_pos) - np.outer(seg_pos, seg_pos)) * seg_mass
            I_frame += I_seg
        inertia_mats.append(I_frame)

    toe_inds = get_character_toe_inds(character)
    left_toe_idx, right_toe_idx = toe_inds
    ankle_inds = get_character_ankle_inds(character)
    left_ankle_idx, right_ankle_idx = ankle_inds

     # animated motion
    # get COM trajectory, root orientation, left/right foot trajectories
    anim, names, _ = BVH.load(anim_bvh)
    # print('LEFT FOOT OFF:')
    # print(anim.offsets[left_toe_idx])
    # print('RIGHT FOOT OFF:')
    # print(anim.offsets[right_toe_idx])

    # append a heel to the character at the same height as foot
    if not character in heeled_characters:
        # must add a heel if doesn't already have one
        anim, heel_offsets = add_heel_to_anim(anim, toe_inds, ankle_inds)
    anim_pos = Animation.positions_global(anim)

    anim_pos[:,:,:] *= -1.0 # flip y
    anim_pos = anim_pos[:,:,[0,2,1]] # swap y and z

    anim_pos *= 0.01 # CONVERTED TO meters since towr uses them (and to make physically reasonable)

    root_pos = anim_pos[:,0,:]
    left_foot_pos = anim_pos[:,left_toe_idx,:]
    right_foot_pos = anim_pos[:, right_toe_idx,:]

    left_ankle_pos = anim_pos[:,left_ankle_idx,:]
    right_ankle_pos = anim_pos[:,right_ankle_idx,:]

    left_heel_idx = anim_pos.shape[1] - 2
    right_heel_idx = anim_pos.shape[1] - 1
    if character in heeled_characters:
        # already has a heel joint
        left_heel_idx, right_heel_idx = get_character_heel_inds(character)

    left_heel_pos = anim_pos[:, left_heel_idx, :]
    right_heel_pos = anim_pos[:, right_heel_idx, :]

    heel_dist = np.mean(np.linalg.norm(left_foot_pos - left_heel_pos, axis=1))

    left_leg_chain = get_character_leg_chain(character, 'left')
    bone_lengths = np.linalg.norm(anim.offsets[left_leg_chain[1:-1]], axis=1) # only down to ankle
    max_heel_length = (np.sum(bone_lengths) + np.linalg.norm(anim.offsets[left_heel_idx])) * 0.01 # to meters
    # print('Max heel length (m)')
    # print(max_heel_length)

    skel_out_path = os.path.join(out_dir, 'skel_info.txt')
    with open(skel_out_path, 'w') as skel_file:
        # left hip offset
        for frame in range(start_idx, end_idx):
            skel_file.write(str(hip_offsets[frame, 0, 0]) + ' ' + str(hip_offsets[frame, 0, 1]) + ' ' + str(hip_offsets[frame, 0, 2]) + '\n')
        # right hip offset
        for frame in range(start_idx, end_idx):
            skel_file.write(str(hip_offsets[frame, 1, 0]) + ' ' + str(hip_offsets[frame, 1, 1]) + ' ' + str(hip_offsets[frame, 1, 2]) + '\n')
        # leg length (hip to toe)
        skel_file.write(str(max_leg_length) + '\n')
        # heel length (hip to heel)
        skel_file.write(str(max_heel_length) + '\n')
        # heel distance (from toe)
        skel_file.write(str(heel_dist) + '\n')
        # total mass
        skel_file.write(str(total_mass) + '\n')
        # Ixx, Iyy, Izz, Ixy, Ixz, Iyz
        for frame in range(start_idx, end_idx):
            I_cur = inertia_mats[frame]
            skel_file.write(str(I_cur[0, 0]) + ' ' + str(I_cur[1, 1]) + ' ' + str(I_cur[2, 2]) + ' ' + 
                        str(I_cur[0, 1]) + ' ' + str(I_cur[0, 2]) + ' ' + str(I_cur[1, 2]) + '\n')

    rot_angle, rot_axis = anim.rotations.angle_axis()
    rot_axis[:,:,:] *= -1.0 # flip y
    rot_axis = rot_axis[:, :, [0,2,1]] # swap y and z]
    root_rot = Quaternions.from_angle_axis(rot_angle, rot_axis).euler(order='xyz')[:,0,:]

    # fix root rot to be smooth
    for dim in range(3):
        cur_val = root_rot[0, dim]
        for frame_idx in range(1, root_rot.shape[0]):
            pre_mult = 1.0 if cur_val >= 0.0 else -1.0
            next_val = root_rot[frame_idx, dim]
            while abs(next_val - cur_val) > np.pi:
                next_val += pre_mult * 2*np.pi
            root_rot[frame_idx, dim] = next_val
            cur_val = next_val 

    # plot_3curve(left_foot_pos, dt, 'time(s)', 'Left foot goal pos (m)')
    # plot_3curve(left_heel_pos, dt, 'time(s)', 'Left heel goal pos (m)')
    # plot_3curve(right_foot_pos, dt, 'time(s)', 'Right foot goal pos (m)')
    # plot_3curve(right_heel_pos, dt, 'time(s)', 'Right heel goal pos (m)')
    # plot_3curve(root_rot, dt, 'time(s)', 'target euler angle')

    # calc COM over time
    anim_COM = np.zeros((anim_pos.shape[0], 3))
    for i in range(anim_pos.shape[0]):
        cur_COM = np.zeros(3)
        for key in map_segment_to_joints.keys():
            seg_pos = np.mean(anim_pos[i, map_segment_to_joints[key], :], axis=0)
            mass_frac = map_segment_to_mass_perc[key] * 0.01
            cur_COM += mass_frac * seg_pos
        anim_COM[i] = cur_COM

    motion_out_path = os.path.join(out_dir, 'motion_info.txt')
    with open(motion_out_path, 'w') as motion_file:
        motion_file.write(str(dt) + '\n')
        # COM trajectory
        for i in range(start_idx, end_idx):
            motion_file.write(str(anim_COM[i, 0]) + ' ' + str(anim_COM[i, 1]) + ' ' + str(anim_COM[i, 2]))
            if i < end_idx - 1:
                motion_file.write(' ')
        motion_file.write('\n')
        # COM (root) orientation
        for i in range(start_idx, end_idx):
            motion_file.write(str(root_rot[i, 0]) + ' ' + str(root_rot[i, 1]) + ' ' + str(root_rot[i, 2]))
            if i < end_idx - 1:
                motion_file.write(' ')
        motion_file.write('\n')
        # left foot trajectory
        for i in range(start_idx, end_idx):
            motion_file.write(str(left_foot_pos[i, 0]) + ' ' + str(left_foot_pos[i, 1]) + ' ' + str(left_foot_pos[i, 2]))
            if i < end_idx - 1:
                motion_file.write(' ')
        motion_file.write('\n')
        # left heel trajectory
        for i in range(start_idx, end_idx):
            motion_file.write(str(left_heel_pos[i, 0]) + ' ' + str(left_heel_pos[i, 1]) + ' ' + str(left_heel_pos[i, 2]))
            if i < end_idx - 1:
                motion_file.write(' ')
        motion_file.write('\n')
        # right foot trajectory
        for i in range(start_idx, end_idx):
            motion_file.write(str(right_foot_pos[i, 0]) + ' ' + str(right_foot_pos[i, 1]) + ' ' + str(right_foot_pos[i, 2]))
            if i < end_idx - 1:
                motion_file.write(' ')
        motion_file.write('\n')
        # right heel trajectory
        for i in range(start_idx, end_idx):
            motion_file.write(str(right_heel_pos[i, 0]) + ' ' + str(right_heel_pos[i, 1]) + ' ' + str(right_heel_pos[i, 2]))
            if i < end_idx - 1:
                motion_file.write(' ')
        motion_file.write('\n')

    # floor information
    # already have it, just copy file
    floor_out_path = os.path.join(out_dir, 'terrain_info.txt')
    with open(floor_file, 'r') as f:
        normal_line = f.readline()
        normal_str = normal_line.split(' ')
        plane_normal = np.array([float(x) for x in normal_str]) # to meters
        point_line = f.readline().split('\n')[0]
        point_str = point_line.split(' ')
        plane_loc = np.array([float(x) for x in point_str]) * 0.01 # to meters

    plane_normal[:] *= -1.0 # flip y
    plane_normal = plane_normal[[0,2,1]] # swap y and z
    plane_loc[:] *= -1.0 # flip y
    plane_loc = plane_loc[[0,2,1]] # swap y and z
    with open(os.path.join(floor_out_path), 'w') as floor_file:
        floor_file.write(str(plane_normal[0]))
        floor_file.write(' ')
        floor_file.write(str(plane_normal[1]))
        floor_file.write(' ')
        floor_file.write(str(plane_normal[2]))
        floor_file.write('\n')
        floor_file.write(str(plane_loc[0]))
        floor_file.write(' ')
        floor_file.write(str(plane_loc[1]))
        floor_file.write(' ')
        floor_file.write(str(plane_loc[2]))

    foot_contacts = np.load(contacts_file)
    # heel = 0, toe = 1
    contacts_left = foot_contacts[:,[0,1]]
    contacts_left = np.amax(contacts_left, axis=1) # if either are in contact
    contacts_right = foot_contacts[:,[2,3]]
    contacts_right = np.amax(contacts_right, axis=1) # if either are in contact
    contacts_left = contacts_left[start_idx:end_idx]
    contacts_right = contacts_right[start_idx:end_idx]

    # left toe, left heel, right toe, right heel
    contacts_all = foot_contacts[start_idx:end_idx,[1, 0, 3, 2]]

    left_toe_start_in_contact = contacts_left[0] #contacts_all[0, 0]
    if combined_contacts:
        left_toe_start_in_contact = contacts_all[0, 0]
    left_heel_start_in_contact = contacts_all[0, 1]
    right_toe_start_in_contact = contacts_right[0] #contacts_all[0, 2]
    if combined_contacts:
        right_toe_start_in_contact = contacts_all[0, 2]
    right_heel_start_in_contact = contacts_all[0, 3]

    # figure out durations
    if not combined_contacts:
        left_toe_durations = find_contact_durations(contacts_all[:,0], dt)
    else:
        left_toe_durations = find_contact_durations(contacts_left, dt)
    left_heel_durations = find_contact_durations(contacts_all[:,1], dt)
    if not combined_contacts:
        right_toe_durations = find_contact_durations(contacts_all[:,2], dt)
    else:
        right_toe_durations = find_contact_durations(contacts_right, dt)
    right_heel_durations = find_contact_durations(contacts_all[:,3], dt)


    contacts_out_path = os.path.join(out_dir, 'contact_info.txt')
    with open(contacts_out_path, 'w') as contacts_file:
        # left toe start in contact
        contacts_file.write(str(left_toe_start_in_contact) + '\n')
        # left toe durations
        contacts_file.write(str(len(left_toe_durations)) + '\n')
        for i in range(len(left_toe_durations)):
            contacts_file.write(str(left_toe_durations[i]))
            if i < len(left_toe_durations) - 1:
                contacts_file.write(' ')
        contacts_file.write('\n')
        # left heel start in contact
        contacts_file.write(str(left_heel_start_in_contact) + '\n')
        # left heel durations
        contacts_file.write(str(len(left_heel_durations)) + '\n')
        for i in range(len(left_heel_durations)):
            contacts_file.write(str(left_heel_durations[i]))
            if i < len(left_heel_durations) - 1:
                contacts_file.write(' ')
        contacts_file.write('\n')
        # right toe start in contact
        contacts_file.write(str(right_toe_start_in_contact) + '\n')
        # right toe durations
        contacts_file.write(str(len(right_toe_durations)) + '\n')
        for i in range(len(right_toe_durations)):
            contacts_file.write(str(right_toe_durations[i]))
            if i < len(right_toe_durations) - 1:
                contacts_file.write(' ')
        contacts_file.write('\n')
        # right heel start in contact
        contacts_file.write(str(right_heel_start_in_contact) + '\n')
        # right heel durations
        contacts_file.write(str(len(right_heel_durations)) + '\n')
        for i in range(len(right_heel_durations)):
            contacts_file.write(str(right_heel_durations[i]))
            if i < len(right_heel_durations) - 1:
                contacts_file.write(' ')

def apply_results(towr_results, anim_bvh, start_idx, end_idx, character, run_ik=True):
    ''' Applies trajectory optim results back to our original skeleton. '''

    # read in og animation
    anim, names, _ = BVH.load(anim_bvh)
    anim.rotations = anim.rotations[start_idx:end_idx]
    anim.positions = anim.positions[start_idx:end_idx]

    toe_inds = get_character_toe_inds(character)
    ankle_inds = get_character_ankle_inds(character)

    # add heel to IK with towr results
    if (not character in heeled_characters) and (towr_results.feet_pos.shape[1] == 4):
        anim, _ = add_heel_to_anim(anim, toe_inds, ankle_inds)
   
    init_pos = Animation.positions_global(anim)
    upper_body_joints = get_character_upper_body(character)
    map_segment_to_joints = get_character_seg_to_joint_map(character)
    map_segment_to_mass_perc = get_character_seg_to_mass_perc_map(character)
    left_toe_idx, right_toe_idx = get_character_toe_inds(character)

    # calc COM over time and offsets of upper body joints (including root)
    # since we only need offsets, don't have to zero out position
    anim_COM = np.zeros((anim.shape[0], 3))
    upper_offsets = np.zeros((anim.shape[0], len(upper_body_joints), 3))
    for i in range(anim.shape[0]):
        cur_COM = np.zeros(3)
        for key in map_segment_to_joints.keys():
            seg_pos = np.mean(init_pos[i, map_segment_to_joints[key], :], axis=0)
            mass_frac = map_segment_to_mass_perc[key] * 0.01
            cur_COM += mass_frac * seg_pos
        anim_COM[i] = cur_COM

        for j in range(len(upper_body_joints)):
            upper_offsets[i, j] = init_pos[i, upper_body_joints[j]] - cur_COM

    anim_og = deepcopy(anim)
    com_og = anim_COM.copy()
    
    # desired global position of upper joints is based on optimized COM position
    seq_len = end_idx - start_idx
    desired_pos = upper_offsets + np.expand_dims(towr_results.base_pos[:seq_len], axis=1)*100.0

    # keep everything the same except replace
    # root information with optimized output
    anim.rotations[:,0,:] = Quaternions.from_euler(towr_results.base_rot, order='xyz', world=True)[:seq_len]
    anim.positions[:,0,:] = desired_pos[:,0,:]

    if run_ik:
        # add IK targets for upper body joints
        targetmap = {}
        for i in range(len(upper_body_joints)):
            targetmap[upper_body_joints[i]] = desired_pos[:,i,:]

        # setup IK with foot joints targetetd at feet position optimized outputs
        left_foot_target = towr_results.feet_pos[:seq_len,0,:] * 100.0 # out of meters
        right_foot_target = towr_results.feet_pos[:seq_len,1,:] * 100.0 # out of meters
        targetmap[left_toe_idx] = left_foot_target
        targetmap[right_toe_idx] = right_foot_target

        # and heels
        left_heel_idx = anim.positions.shape[1] - 2
        right_heel_idx = anim.positions.shape[1] - 1
        if character in heeled_characters:
            print('Using COMBINED character')
            # already has a heel joint
            left_heel_idx, right_heel_idx = get_character_heel_inds(character)
        if towr_results.feet_pos.shape[1] == 4:
            print('Found heel data, including it in IK')
            left_heel_target = towr_results.feet_pos[:seq_len,2,:] * 100.0
            right_heel_target = towr_results.feet_pos[:seq_len,3,:] * 100.0
            targetmap[left_heel_idx] = left_heel_target
            targetmap[right_heel_idx] = right_heel_target

        # run IK
        ik = JacobianInverseKinematicsCK(anim, targetmap, translate=True, iterations=30, smoothness=0.001, damping=7.0, silent=False)
        ik()

    return anim, names, anim_og, com_og

def build_towr_results_from_anim(anim, com_trajectory, contacts, dt, character):
    ''' Builds a TowrResults object from some animation. Assumes 4 feet. Contact should be in order left_toe, right_toe, left_heel, right_heel. 
        Outputs just like a TowrResults object (so in meters). '''
    res = TowrResults()
    res.num_feet = 4
    res.dt = dt 
    res.base_pos = com_trajectory / 100.0 # F x 3
    res.base_rot = anim.rotations.euler(order='xyz')[:,0,:] # F x 3 IN DEGREES
    res.base_R = anim.rotations.transforms()[:,0,:] # F x 3 x 3

    toe_inds = get_character_toe_inds(character)
    ankle_inds = get_character_ankle_inds(character)

    if character in heeled_characters:
        heel_anim = anim
    else:
        heel_anim, _ = add_heel_to_anim(anim, toe_inds, ankle_inds)
    anim_pos = Animation.positions_global(heel_anim)

    left_toe_idx, right_toe_idx = get_character_toe_inds(character)

    left_foot_pos = anim_pos[:,left_toe_idx,:].reshape((com_trajectory.shape[0], 1, -1))
    right_foot_pos = anim_pos[:, right_toe_idx,:].reshape((com_trajectory.shape[0], 1, -1))

    left_heel_idx = anim_pos.shape[1] - 2
    right_heel_idx = anim_pos.shape[1] - 1
    if character in heeled_characters:
        # already has a heel joint
        left_heel_idx, right_heel_idx = get_character_heel_inds(character)
    left_heel_pos = anim_pos[:, left_heel_idx, :].reshape((com_trajectory.shape[0], 1, -1))
    right_heel_pos = anim_pos[:, right_heel_idx, :].reshape((com_trajectory.shape[0], 1, -1))

    res.feet_pos = np.concatenate((left_foot_pos, right_foot_pos, left_heel_pos, right_heel_pos), axis=1) / 100.0 # F x nFeet x 3
    res.feet_force = np.zeros((left_foot_pos.shape[0], 4, 3)) # F x nFeet x 3

    res.feet_contact = contacts # F x nFeet

    return res


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', dest='viz', action='store_true', help='Visualize results, otherwise process input')
    parser.set_defaults(viz=False)
    flags = parser.parse_known_args()[0]

    if flags.viz:
        #
        # VISUALIZING
        #
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', required=True, nargs='+', help='List of files with towr results to visualize.')
        parser.add_argument('--name', nargs='+', help='List of files with towr results to visualize.')
        parser.add_argument('--floor', help='file with data for floor fit (normal and point)')
        parser.add_argument('--out-vid', help='out video file', default=None)
        parser.add_argument('--out-plots', help='out plots dir', default=None)
        parser.add_argument('--out-bvh', nargs='+', help='out bvh file to save the each data file given to', default=None)
        parser.add_argument('--character', help='name of the character (i.e. ybot)', default=None)
        parser.add_argument('--start', type=int, default=None, help='start idx')
        parser.add_argument('--end', type=int, default=None, help='start idx')
        parser.add_argument('--anim', required=True, help='The original motion bvh file before physical optimization')
        parser.add_argument('--contacts', default=None, help='Path to original contact labels if displaying original motion comparison')
        parser.add_argument('--fps', default=30, help='FPS to render at')
        parser.add_argument('--trace', dest='trace', action='store_true', help='Trace trajectories.')
        parser.set_defaults(trace=False)
        parser.add_argument('--forces', dest='forces', action='store_true', help='Draw contact forces.')
        parser.set_defaults(forces=False)
        parser.add_argument('--towr', dest='towr', action='store_true', help='Draw towr-optimized values (COM, foot positions)')
        parser.set_defaults(towr=False)
        parser.add_argument('--skel', dest='skel', action='store_true', help='Draw skeleton')
        parser.set_defaults(skel=False)
        parser.add_argument('--no-ik', dest='ik', action='store_false', help='Run IK when transferring back to skel')
        parser.set_defaults(ik=True)
        parser.add_argument('--no-floor', dest='draw_floor', action='store_false', help='Do not draw floor')
        parser.set_defaults(draw_floor=True)
        parser.add_argument('--hide', dest='hide', action='store_true', help='Whether to hide the plots or just save them')
        parser.set_defaults(hide=False)
        parser.add_argument('--plots', dest='plots', action='store_true', help='Whether to graph 2D plots')
        parser.set_defaults(plots=False)
        parser.add_argument('--compare-og', dest='compare_og', action='store_true', help='If enabled, will plot the towr results alongside the original skeleton animation.')
        parser.set_defaults(compare_og=False)

        flags = parser.parse_known_args()[0]

        with open(flags.floor, 'r') as f:
            normal_line = f.readline()
            normal_str = normal_line.split(' ')
            plane_normal = np.array([float(x) for x in normal_str]) * 0.01 # to meters
            point_line = f.readline().split('\n')[0]
            point_str = point_line.split(' ')
            plane_loc = np.array([float(x) for x in point_str]) * 0.01 # to meters

        res_list = []
        anim_list = []
        for data_path in flags.data:
            res = load_results(data_path, flip_coords=True)
            res_list.append(res)
            anim, names, anim_og, com_og = apply_results(res, flags.anim, flags.start, flags.end, flags.character, run_ik=flags.ik)
            anim_list.append(anim)

        # if we're comparing to original, build a towr_results object for the original motion
        res_og = None
        if flags.compare_og:
            foot_contacts = None
            if not flags.contacts is None:
                foot_contacts = np.load(flags.contacts)
                # order correctly (lt, rt, lh, rh)
                foot_contacts = foot_contacts[:, [1, 3, 0, 2]]
            res_og = build_towr_results_from_anim(anim_og, com_og, foot_contacts, res.dt, flags.character)
        
        if flags.out_bvh is not None and len(flags.out_bvh) > 0:
            print(flags.out_bvh)
            for bvh_idx, cur_out_bvh in enumerate(flags.out_bvh):
                save_anim = anim_list[bvh_idx]
                if not flags.character in heeled_characters: 
                    save_anim = remove_heel_from_anim(save_anim)
                BVH.save(cur_out_bvh, save_anim, names)

        if flags.compare_og:
            res_list = [res_og] + res_list
            anim_list = [anim_og] + anim_list

        # flip y for viz
        for cur_res in res_list:
            cur_res.base_pos[:, 1] *= -1.0
            for foot_idx in range(cur_res.num_feet):
                cur_res.feet_pos[:, foot_idx, 1] *= -1.0
            for foot_idx in range(cur_res.num_feet):
                cur_res.feet_force[:, foot_idx, 1] *= -1.0

        flip_anim = [1]

        label_names = flags.name
        if flags.compare_og:
            label_names = ['Init'] + label_names

        viz_results(res_list, anim_list, flags.start, flags.end,
                    floor_normal=plane_normal, floor_point=plane_loc, save_path=flags.out_vid,
                    flip_floor=True, draw_trace=flags.trace, draw_forces=flags.forces,
                    draw_towr=flags.towr, draw_skel=flags.skel,fps=flags.fps,
                    draw_floor=flags.draw_floor, flip_anim=flip_anim, show=(not flags.hide),
                    names=label_names)
        
        if flags.plots:
            if not flags.out_plots is None:
                if not os.path.exists(flags.out_plots):
                    os.mkdir(flags.out_plots)
                plot_3curve(res.base_pos, res.dt, 'time(s)', 'CoM pos (m)', save_file=os.path.join(flags.out_plots, 'com_pos.png'), show=(not flags.hide))
                plot_3curve(res.base_rot, res.dt, 'time(s)', 'CoM ang (degrees)', save_file=os.path.join(flags.out_plots, 'com_orient.png'), show=(not flags.hide))
                plot_3curve(res.feet_pos[:,0], res.dt, 'time(s)', 'left toe pos (m)', save_file=os.path.join(flags.out_plots, 'left_toe_pos.png'), show=(not flags.hide))
                plot_3curve(res.feet_pos[:,1], res.dt, 'time(s)', 'right toe pos (m)', save_file=os.path.join(flags.out_plots, 'right_toe_pos.png'), show=(not flags.hide))
                plot_3curve(res.feet_pos[:,2], res.dt, 'time(s)', 'left heel pos (m)', save_file=os.path.join(flags.out_plots, 'left_heel_pos.png'), show=(not flags.hide))
                plot_3curve(res.feet_pos[:,3], res.dt, 'time(s)', 'right heel pos (m)', save_file=os.path.join(flags.out_plots, 'right_heel_pos.png'), show=(not flags.hide))
                plot_3curve(res.feet_force[:,0], res.dt, 'time(s)', 'left toe contact force (N)', save_file=os.path.join(flags.out_plots, 'left_toe_force.png'), show=(not flags.hide))
                plot_3curve(res.feet_force[:,1], res.dt, 'time(s)', 'right toe contact force (N)', save_file=os.path.join(flags.out_plots, 'right_toe_force.png'), show=(not flags.hide))
                plot_3curve(res.feet_force[:,2], res.dt, 'time(s)', 'left heel contact force (N)', save_file=os.path.join(flags.out_plots, 'left_heel_force.png'), show=(not flags.hide))
                plot_3curve(res.feet_force[:,3], res.dt, 'time(s)', 'right heel contact force (N)', save_file=os.path.join(flags.out_plots, 'right_heel_force.png'), show=(not flags.hide))

    else:
        #
        # PROCESSING
        #
        parser = argparse.ArgumentParser()
        parser.add_argument('--anim', required=True, help='Animated skeleton.')
        parser.add_argument('--floor', help='file with data for floor fit (normal and point)')
        parser.add_argument('--contacts', help='file with data for foot contacts')
        parser.add_argument('--out', help='directory to write out processed towr input')
        parser.add_argument('--character', help='The character skeleton (i.e. ybot or skeletonzombie')
        parser.add_argument('--start', type=int, default=None, help='start idx')
        parser.add_argument('--end', type=int, default=None, help='start idx')
        parser.add_argument('--fps', default=30.0, help='FPS of animated skeleton')
        parser.add_argument('--no-heel', dest='heel', action='store_false', help='If included, combines toe and heel contacts into a single label.')
        parser.set_defaults(heel=True)

        flags = parser.parse_known_args()[0]

        prepare_input(flags.anim, flags.floor, flags.contacts, flags.out, flags.character,
                      start_idx=flags.start,
                      end_idx=flags.end,
                      dt=(1.0/float(flags.fps)),
                      combined_contacts=(not flags.heel))
