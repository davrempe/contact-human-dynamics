import os
import sys
import bpy
from mathutils import Matrix, Vector, Quaternion, Euler
import numpy as np
import argparse
import subprocess
import glob
from copy import deepcopy
import pickle

from os import listdir, makedirs, system
from os.path import exists
from os.path import isdir

sys.path.extend(['utils'])
from character_info_utils import get_character_mass # mappings

sys.path.extend(['skeleton_fitting/ik'])
import BVH
import Animation
from Quaternions import Quaternions

KINEMATIC_RESULT_NAME = 'kinematic_results'
TOWR_OUT_NAME = 'phys_optim_out_'

CHAR_OBJ_NAME = 'Armature'
CAM_NAME = 'Camera'
SHADOW_LIGHT_PRE = 'ShadowLight'
CAM_LIGHTS = ['CamLight_Left', 'CamLight_Right']
FLOOR_NAME = 'Floor'
NUM_LIGHTS = 4
CHARACTER_NAME_TO_ID = { 'liam'      : '44939_Liam',
                         'remy'      : '44942_Remy',
                         'malcolm'   : '44940_Malcolm',
                         'stefani'   : '44944_Stefani',
                         'douglas'   : '45049_Douglas',
                         'regina'    : '44941_Regina',
                         'shae'      : '44943_Shae', 
                         'swat'      : '254_Swat',
                         'lola'      : '893_Lola_B_Styperek', 
                         'derrick'   : '363_Derrick',
                         'pearl'     : '45051_Pearl',
                         'jasper'    : '45050_Jasper',
                         'exored'    : '132_Exo_Red', 
                         'ybot'      : '45276_Y_Bot',
                         'ty'        : '910_Ty', 
                         'skeletonzombie' : '968_Skeletonzombie_T_Avelange'}
#  scaling to normalize each character
CHARACTER_SCALING = { '44939_Liam'      : 0.005,
                      '44942_Remy'      : 0.005,
                      '44940_Malcolm'   : 0.005,
                      '44944_Stefani'   : 0.005,
                      '45049_Douglas'   : 0.01, 
                      '44941_Regina'    : 0.005,
                      '44943_Shae'      : 0.005,
                      '254_Swat'        : 0.01, 
                      '893_Lola_B_Styperek' : 0.01, 
                      '363_Derrick'     : 0.01, 
                      '45051_Pearl'     : 0.01,
                      '45050_Jasper'    : 0.01,
                      '132_Exo_Red'     : 0.01, 
                      '45276_Y_Bot'     : 0.01,
                      '910_Ty'          : 0.01,
                      '968_Skeletonzombie_T_Avelange' : 0.01}

VIEW0_CAM_ROT = (np.pi/2.0, 0.0, np.pi)

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

def setup_floor(floor_file=None, floor_tex=None, flip=True, draw_floor=True):
    if floor_file != None and os.path.exists(floor_file):
        # read in custom
        with open(floor_file, 'r') as f:
            normal_line = f.readline()
            normal_str = normal_line.split(' ')
            plane_normal = [float(x) for x in normal_str]
            point_line = f.readline().split('\n')[0]
            point_str = point_line.split(' ')
            plane_loc = tuple([float(x)*0.01 for x in point_str])

        # convert loc to blender coords
        if flip:
            plane_loc = (-plane_loc[0], -plane_loc[2], -plane_loc[1])
            plane_normal = np.array([-plane_normal[0], -plane_normal[2], -plane_normal[1]])
        else:
            plane_loc = (plane_loc[0], plane_loc[2], plane_loc[1])
            plane_normal = np.array([plane_normal[0], plane_normal[2], plane_normal[1]])
        up_axis = np.array([0.0, 0.0, 1.0])
        if np.abs(np.dot(up_axis, plane_normal / np.linalg.norm(plane_normal)) - 1.0) < 1e-6:
            # they're going in the same direction
            print('Normal is already aligned with up axis!')
            axis_to_rot = up_axis
            angle_to_rot = 0.0
        else:
            axis_to_rot = np.cross(up_axis, plane_normal)
            axis_to_rot /= np.linalg.norm(axis_to_rot)
            angle_to_rot = np.arccos(np.dot(up_axis, plane_normal))
        # convert orientation to blender transform
        plane_rotation = (angle_to_rot, axis_to_rot[0], axis_to_rot[1], axis_to_rot[2])
    else:
        if floor_file != None:
            print('Could not find given floor file, Using defaults')
        plane_loc = (0, 0.0, 0.0)
        plane_rotation = (0.0, 0.0, 0.0, 1.0)
        plane_normal = (0.0, 0.0, 1.0)
    
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects[FLOOR_NAME].select = True
    bpy.data.objects[FLOOR_NAME].location = plane_loc

    bpy.context.scene.objects.active = bpy.data.objects[FLOOR_NAME]
    bpy.data.objects[FLOOR_NAME].scale = (20.0, 20.0, 1.0)
    bpy.data.objects[FLOOR_NAME].rotation_mode = 'AXIS_ANGLE'
    bpy.data.objects[FLOOR_NAME].rotation_axis_angle = plane_rotation #(np.pi / 8.0, -1.0, 0.0, 0.0) # angle, then axis

    floor_mat = bpy.data.objects[FLOOR_NAME].active_material
    tex = floor_mat.texture_slots[0].texture
    if floor_tex is None:
        tex.image = None
        floor_mat.diffuse_color = (0.154, 0.154, 0.154)
    else:
        im = bpy.data.images.load(floor_tex, check_existing=False)
        tex.image = im
        tex.use_mipmap = False
        tex.use_interpolation = False
        tex.filter_type = 'AREA'
        tex.filter_size = 0.1
        tex.use_filter_size_min = False
        tex.repeat_x = 10
        tex.repeat_y = 10

    slot = floor_mat.texture_slots[0] #.add()
    slot.texture = tex
    slot.uv_layer = 'UVMap'

    bpy.ops.object.select_all(action="DESELECT")

    if not draw_floor:
        floor_mat.use_only_shadow = True
        floor_mat.shadow_only_type = 'SHADOW_ONLY'

    return plane_normal, plane_loc

def setup_lighting(mean_root):
    # There are 2 lights attached to the camera that only move with the camera
    for cam_light_name in CAM_LIGHTS:
        cur_light_obj = bpy.data.objects[cam_light_name]
        cur_lamp_obj = cur_light_obj.data
        # energy
        if mean_root is not None:
            cur_lamp_obj.energy = max([((-mean_root[1] - 3.5)/6.0)*0.03 + 0.005, 0.005])   #0.03 
        else:
            cur_lamp_obj.energy = 0.03

    for light_idx in range(NUM_LIGHTS):
        cur_light_name = SHADOW_LIGHT_PRE + str(light_idx)
        cur_light_obj = bpy.data.objects[cur_light_name]
        cur_lamp_obj = cur_light_obj.data
        if light_idx == 2:
            cur_lamp_obj.energy = 0.35
            cur_lamp_obj.use_only_shadow = True
            # set location
            cur_light_loc = cur_light_obj.location
            cur_light_obj.location = (-6.0, 0.5, 9.0)
        elif light_idx == 3:
            cur_lamp_obj.energy = 0.17
            cur_lamp_obj.shadow_method = 'NOSHADOW'
            # set location
            cur_light_loc = cur_light_obj.location
            cur_light_obj.location = (2, 2, 5.0)
        else:
            cur_lamp_obj.energy = 0.0

    return 

def setup_camera(view_idx, cam_params=None, flip=True, mean_root=None):
    '''
    If camera params are given, sets the camera in this position.
    Otherwise sets a side view based on the floor and mean_root position.
    '''
    cam_obj = bpy.data.objects[CAM_NAME]

    # floor info
    floor_aa = bpy.data.objects[FLOOR_NAME].rotation_axis_angle
    floor_angle = floor_aa[0]
    floor_axis = floor_aa[1:]

    if view_idx == 0:
        # this will always be true whether we have cam params or not
        # view0 (camera at 0,0,0)
        cam_obj.location = (0.0, 0.0, 0.0)
        cam_obj.rotation_mode = 'XYZ'
        cam_obj.rotation_euler = VIEW0_CAM_ROT
    elif cam_params is not None:
        #  figure out extrinsics w.r.t view0 since that's
        #   what system the animation is in
        R_bcam2cv = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        # get original camera transformation of view0
        RT_0 = cam_params[0]['RT']
        R_0 = RT_0[:,:3] # R_bcam2cv*R_world2bcam
        T_0 = RT_0[:,3] # R_bcam2cv*T_world2bcam
        R_0_og= np.dot(R_bcam2cv, R_0).T
        T_0_og = np.dot(R_0_og, -np.dot(R_bcam2cv, T_0))

        # get original camera transformation view_idx
        RT_n = cam_params[view_idx]['RT']
        R_n = RT_n[:,:3] # R_bcam2cv*R_world2bcam
        T_n = RT_n[:,3] # R_bcam2cv*T_world2bcam
        R_n_og = np.dot(R_bcam2cv, R_n).T
        T_n_og = np.dot(R_n_og, -np.dot(R_bcam2cv, T_n))

        R_view0 = Euler(VIEW0_CAM_ROT, 'XYZ').to_matrix()
        R_view0 = np.array([[R_view0[0][0], R_view0[0][1], R_view0[0][2]],
                            [R_view0[1][0], R_view0[1][1], R_view0[1][2]],
                            [R_view0[2][0], R_view0[2][1], R_view0[2][2]]], dtype=float)

        # now want cam trans of view n w.r.t. view0 (at 0,0,0)
        R_cam = np.dot(R_view0, np.dot(R_0_og.T, R_n_og))
        T_cam = np.dot(R_view0.T, np.dot(R_0_og.T, T_n_og - T_0_og))
        # to blender
        R_blend_cam = Matrix(tuple(map(tuple, R_cam)))
        T_blend_cam = tuple(T_cam)

        cam_obj.location = T_blend_cam
        cam_obj.rotation_mode = 'XYZ'
        cam_obj.rotation_euler = R_blend_cam.to_euler('XYZ')
    else:
        if view_idx == 1:             # side view -0.15
            side_offset = np.array([-5.5, 0.0, -0.15 + (floor_angle / (np.pi /5.0))])
            if flip:
                side_offset[1] *= -1.0
            camera_loc = mean_root + side_offset
            cam_obj.location = (camera_loc[0], camera_loc[1], camera_loc[2])
            cam_obj.rotation_mode = 'XYZ'
            cam_obj.rotation_euler = (np.pi*(90.0/180.0), 0.0, -np.pi / 2.0)
        if view_idx == 2:             # side view
            three_quarter_offset = np.array([-5.0, 3.0, -0.6 + (floor_angle / (np.pi /5.0))])
            camera_loc = mean_root + three_quarter_offset

            cam_obj.location = (camera_loc[0], camera_loc[1], camera_loc[2])
            cam_obj.rotation_mode = 'XYZ'
            cam_obj.rotation_euler = (np.pi*(90.0/180.0), 0.0, -np.pi / 1.5)

        if cam_params is None:
            # always align with the ground
            floor_quat = Quaternion(floor_aa[1:], floor_aa[0])
            floor_mat = floor_quat.to_matrix()
            cam_mat = cam_obj.rotation_euler.to_matrix()
            rot_mat = floor_mat * cam_mat
            rot_quat = rot_mat.to_quaternion()
            cam_obj.rotation_mode = 'QUATERNION'
            cam_obj.rotation_quaternion = rot_quat

    # must update scene to recalculate transforms
    bpy.context.scene.update()

def setup_rendering(out_dir, num_frames, cam_params):
    bpy.context.scene.render.resolution_x = cam_params[0] 
    bpy.context.scene.render.resolution_y = cam_params[1]
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.use_antialiasing = True

    file_path = os.path.join(os.path.abspath(out_dir), 'frame_')

    bpy.data.scenes["Scene"].render.filepath = (file_path)
    bpy.data.scenes["Scene"].render.image_settings.file_format = "PNG"
    bpy.data.scenes["Scene"].render.alpha_mode = 'TRANSPARENT'
    bpy.data.scenes["Scene"].render.image_settings.color_mode ='RGBA'

    bpy.data.scenes["Scene"].frame_start = 0
    bpy.data.scenes["Scene"].frame_end = num_frames

    bpy.context.scene.camera = bpy.data.objects['Camera']
    bpy.context.scene.camera.data.lens = cam_params[2]

    return file_path + '%04d.png'

def render_and_make_video(out_dir, name, num_frames, fps, cam_params):
    # setup rendering
    frame_out_dir = os.path.join(out_dir, name)
    out_file_format = setup_rendering(frame_out_dir, num_frames, cam_params)
    # render
    bpy.ops.render.render(animation=True)

    # create a video
    out_file = os.path.join(out_dir, name + '.mp4')
    subprocess.run(['ffmpeg', '-y', '-r', str(fps), '-i', out_file_format, '-vcodec', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', out_file])

def clear_parenting(char_parts):
    for part_obj in char_parts:
        bpy.ops.object.select_all(action="DESELECT")
        part_obj.select = True
        bpy.context.scene.objects.active = part_obj
        bpy.ops.object.parent_clear()

def parent_to_armature(char_parts, parent_obj):
    for part_obj in char_parts:
        print(part_obj.name)
        part_obj.location = (0,0,0)
        bpy.ops.object.select_all(action="DESELECT")
        child = part_obj
        parent_obj.select = True
        child.select = True
        bpy.context.scene.objects.active = parent_obj
        bpy.ops.object.parent_set(type="ARMATURE")

def get_mean_root(anim, new_scale, flip):
    mean_root = np.mean(anim.positions[:,0,:], axis=0) * new_scale
    # to blender
    if flip:
        mean_root = -mean_root[[0, 2, 1]]
    else:
        mean_root = mean_root[[0, 2, 1]]
        mean_root[1] *= -1.0
    return mean_root

def make_mat(name, diffuse, specular, alpha):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = diffuse
    mat.diffuse_shader = 'LAMBERT'
    mat.diffuse_intensity = 1.0
    mat.specular_color = specular
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.5
    mat.alpha = alpha
    mat.ambient = 1
    return mat

def draw_com_and_forces(draw_com, draw_forces, force_on_com,
                        com_trajectory=None,
                        feet_pos=None,
                        feet_forces=None):
    com_objs = []
    if draw_com:
        com_mat = make_mat('red',(1,0,0), (1,1,1) ,1)
        for frame_idx in range(com_trajectory.shape[0]):
            bpy.ops.object.select_all(action='DESELECT')
            sphere_loc = tuple(com_trajectory[frame_idx])
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, size=0.01, location=sphere_loc)
            bpy.context.object.data.materials.append(com_mat)
            com_objs.append(bpy.context.object)
    
    force_objs = []
    sphere_objs = []
    if draw_forces:
        force_mat = make_mat('red',(1,0,0), (1,1,1), 1)
        force_mat2 = make_mat('green',(0,1,0), (1,1,1), 1)
        # create an arrow (cylinder with cone on top)
        # fix x,y scale to be constant, keyframe z scale, position, and rotation.
        depth = 1.0
        for joint_idx in range(feet_pos.shape[1]):
            bpy.ops.object.select_all(action='DESELECT')

            if force_on_com:
                bpy.ops.mesh.primitive_uv_sphere_add(segments=32, size=0.03)
                bpy.context.object.data.materials.append(force_mat)
                sphere_objs.append(bpy.context.object)

            bpy.ops.mesh.primitive_cylinder_add(radius=0.01, depth=depth, location=(0,0,0))
            cyl_obj = bpy.context.object
            bpy.ops.mesh.primitive_cone_add(location=(0,0,depth/2.0))
            cone_obj = bpy.context.object
            cone_obj.scale = (0.025, 0.025, 0.025)
            bpy.context.scene.update()
            cyl_obj.data.materials.append(force_mat)
            cone_obj.data.materials.append(force_mat)

            bpy.ops.object.select_all(action='DESELECT')
            child = cone_obj
            cyl_obj.select = True
            child.select = True
            bpy.context.scene.objects.active = cyl_obj
            bpy.ops.object.parent_set(type="OBJECT")

            bpy.ops.object.select_all(action='DESELECT')
            cyl_obj.rotation_mode = 'QUATERNION'
            cone_obj.rotation_mode = 'QUATERNION'
            force_objs += [cyl_obj, cone_obj]
            start_frame = 1
            end_frame = com_trajectory.shape[0]
            for frame_idx in range(start_frame, end_frame+1):
                # change force direction
                cur_force_vec = feet_forces[frame_idx-1,joint_idx,:].astype(np.float64)
                cur_force_mag = np.linalg.norm(cur_force_vec)
                if cur_force_mag < 1.0:
                    cur_force_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                else:
                    cur_force_dir = cur_force_vec / cur_force_mag
                up_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)

                dotprod = np.dot(up_vec, cur_force_dir)
                if dotprod > 0.99999:
                    q_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                    q_w = 1.0
                elif dotprod < -0.99999:
                    q_xyz = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    q_w = 0.0
                else:
                    q_xyz = np.cross(up_vec, cur_force_dir)
                    q_w = 1.0 + np.dot(up_vec, cur_force_dir)

                rot_quat = np.append(q_w, q_xyz)
                rot_quat /= np.linalg.norm(rot_quat)
                rot_quat = Quaternion(rot_quat)
                cyl_obj.rotation_quaternion = rot_quat
                bpy.context.scene.update()
                bpy.context.scene.update()
                cyl_obj.keyframe_insert(data_path='rotation_quaternion', frame=frame_idx)

                # must update scene to recalculate transforms
                bpy.context.scene.update()

                if force_on_com:
                    sphere_objs[-1].location = tuple(feet_pos[frame_idx-1,joint_idx,:])
                    bpy.context.scene.update()
                    sphere_objs[-1].keyframe_insert(data_path='location', frame=frame_idx)

                # scale by magnitude
                cur_scale = cur_force_mag * 0.001
                if cur_scale < 1e-5:
                    cyl_obj.scale = (0.0, 0.0, 0.0)
                else:
                    cyl_obj.scale = (1.0, 1.0, cur_scale)
                bpy.context.scene.update()
                cyl_obj.keyframe_insert(data_path='scale', frame=frame_idx)
                # must update scene to recalculate transforms
                # offset so foot joint is at bottom of cylinder
                cyl_scale_by = (cur_scale*depth)/2.0
                if not force_on_com:
                    cyl_scale_by += 0.02
                cur_cyl_offset = cur_force_dir * cyl_scale_by
                cur_force_location = feet_pos[frame_idx-1,joint_idx,:] + cur_cyl_offset
                cyl_obj.location = tuple(cur_force_location)
                bpy.context.scene.update()
                cyl_obj.keyframe_insert(data_path='location', frame=frame_idx)
                # must update scene to recalculate transforms
                if cur_scale < 1e-5:
                    cone_obj.scale = (0.0, 0.0, 0.0)
                else:
                    cone_obj.scale = (0.025, 0.025, 0.025)
                bpy.context.scene.update()
                cone_obj.keyframe_insert(data_path='scale', frame=frame_idx)

        # must update scene to recalculate transforms
        bpy.context.scene.update()

    return com_objs, force_objs, sphere_objs

def render_multiview_eval(results_dir, character, fbx_path, scene_file, floor_tex_path, 
                        out_dir, fps, flip, kinematic_result=False, draw_com=False, 
                        draw_forces=False, force_on_com=False, combine_feet_forces=False,
                        draw_floor=True, draw_character=True, cam_params=(1280, 720, 35)):
    # load total capture initialization
    pipeline_results = os.path.join(results_dir, KINEMATIC_RESULT_NAME)
    if not os.path.exists(pipeline_results):
        print('Could not find pipeline_results for ' + results_dir)
        return None
    totalcap_bvh_path = os.path.join(pipeline_results, 'totalcap_init_%s.bvh' % (character))
    floor_file_path = os.path.join(pipeline_results, 'floor_out.txt')

    # load predicted output after physics optimization
    video_name = results_dir.split('/')[-1]
    towr_output = os.path.join(results_dir, TOWR_OUT_NAME + character)
    if not os.path.exists(towr_output):
        print('Could not find ' + towr_output)
        return None
    no_dynamics_bvh_path = os.path.join(towr_output, video_name + '_%s_no_dynamics.bvh' % (character)) # initialization
    dynamics_bvh_path = os.path.join(towr_output, video_name + '_%s_dynamics.bvh' % (character)) # after dynamics
    durations_bvh_path = os.path.join(towr_output, video_name + '_%s_durations.bvh' % (character)) # after foot contact optim

    if not os.path.exists(no_dynamics_bvh_path) or not os.path.exists(dynamics_bvh_path) or not os.path.exists(durations_bvh_path):
        print('Could not find optimized results for ' + results_dir + '. Skipping...')
        return None

    # see which optimizations actually suceeded
    success_log_path = os.path.join(towr_output, 'success_log.txt')
    dynamics_success = True
    durations_success = True
    if os.path.exists(success_log_path):
        with open(success_log_path, 'r') as f:
            dynamics_line = f.readline().rstrip().lstrip()
            dynamics_success = dynamics_line.split(' ')[-1] == '1'
            durations_line = f.readline().rstrip().lstrip()
            durations_success = durations_line.split(' ')[-1] == '1'
    else:
        # use dynamics to be safe
        durations_success = False

    if not dynamics_success and not durations_success:
        print('Optimization did not converge! Cannot evaluate for ' + results_dir.split('/')[-1])
        return None

    # viz durations if converged, else dynamics
    result_name = None
    if durations_success:
        predicted_bvh_path = durations_bvh_path
        result_name = 'durations'
    else:
        predicted_bvh_path = dynamics_bvh_path
        result_name = 'dynamics'

    # path to the character skin FBX file
    skin_path = fbx_path
    
    # have everything we need, now open the scene file and actually render
    # open the file in blender, we'll leave it open the whole time
    bpy.ops.wm.open_mainfile(filepath=scene_file)

    # setup the floor based on prediction
    plane_normal, plane_loc = setup_floor(floor_file=floor_file_path, floor_tex=floor_tex_path, flip=flip, draw_floor=draw_floor)

    feet_pos = None 
    feet_forces = None
    com_trajectory = None
    kinematic_feet_pos = None 
    kinematic_feet_forces = None
    kinematic_com_trajectory = None
    if draw_com or draw_forces:
        # load in result and get COM
        towr_results_path = os.path.join(towr_output, 'sol_out_' + result_name + '.txt')
        res = load_results(towr_results_path, flip_coords=True)
        com_trajectory = res.base_pos.copy()
        com_trajectory[:,:] *= -1.0 # flip for blender
        com_trajectory = com_trajectory[:,[0,2,1]] # flip y and z
        feet_pos = res.feet_pos.copy()
        feet_pos[:,:,:] *= -1.0 # flip for blender
        feet_pos = feet_pos[:,:,[0,2,1]] # flip y and z
        # already have forces
        feet_forces = res.feet_force.copy()
        # res.feet_force[:,:,1] *= -1.0 # y up
        feet_forces[:,:,:] *= -1.0 # flip for blender
        feet_forces = feet_forces[:,:,[0,2,1]] # flip y and z

        if kinematic_result:
            towr_results_path = os.path.join(towr_output, 'sol_out_no_dynamics.txt')
            res = load_results(towr_results_path, flip_coords=True)
            kinematic_com_trajectory = res.base_pos.copy()
            kinematic_com_trajectory[:,:] *= -1.0 # flip for blender
            kinematic_com_trajectory = kinematic_com_trajectory[:,[0,2,1]] # flip y and z
            kinematic_feet_pos = res.feet_pos.copy()
            kinematic_feet_pos[:,:,:] *= -1.0 # flip for blender
            kinematic_feet_pos = kinematic_feet_pos[:,:,[0,2,1]] # flip y and z

            # must determine force from COM trajectory
            h = 1.0 / 30.0 # assuming synthetic data at 30 fps
            anim_accel = np.zeros((kinematic_com_trajectory.shape[0]-2, 3))
            for i in range(1, kinematic_com_trajectory.shape[0]-1):
                anim_accel[i-1] = (kinematic_com_trajectory[i+1] - 2*kinematic_com_trajectory[i] + kinematic_com_trajectory[i-1]) / (h**2)

            anim_accel = np.append(anim_accel, anim_accel[-1].reshape((1,3)), axis=0)
            anim_accel = np.append(anim_accel[0].reshape((1,3)), anim_accel, axis=0)

            # now estimate ground reaction force on COM at each step F=ma-mg
            # direction of gravity is opposite of floor normal
            mass = get_character_mass(character)
            g = np.ones_like(anim_accel)
            g *= -plane_normal.reshape((1, 3))*9.81
            com_force = mass * (anim_accel - g)
            # already in blender coords
            kinematic_feet_forces = com_force.reshape((com_force.shape[0], 1, 3))
            

        # feet forces/pos are in order left toe, right toe, left heel, right heel
        if force_on_com:
            # can't viz forces on each foot separately, combine into a single mean foot trajectory
            feet_pos = np.mean(feet_pos, axis=1).reshape((feet_pos.shape[0], 1, 3))
            # need to collapse foot forces into net force
            feet_forces = np.sum(feet_forces, axis=1).reshape((feet_forces.shape[0], 1, 3))
        if kinematic_result:
            # can't viz forces on each foot separately, combine into a single mean foot trajectory
            kinematic_feet_pos = np.mean(kinematic_feet_pos, axis=1).reshape((kinematic_feet_pos.shape[0], 1, 3))

        if combine_feet_forces:
            # only want a single force for each foot placed at their mean
            foot1_mean = np.mean(feet_pos[:,[0,2],:], axis=1)    
            foot2_mean = np.mean(feet_pos[:,[1,3],:], axis=1)
            feet_pos = np.stack([foot1_mean, foot2_mean], axis=1)

            foot1_sum = np.sum(feet_forces[:,[0,2],:], axis=1)    
            foot2_sum = np.sum(feet_forces[:,[1,3],:], axis=1)
            feet_forces = np.stack([foot1_sum, foot2_sum], axis=1)

    #
    # first render predictions
    #

    # load in char fbx
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.fbx(filepath=skin_path)
    char_objects = bpy.context.selected_objects

    # make any changes to materials
    character_id = CHARACTER_NAME_TO_ID[character]
    new_scale = CHARACTER_SCALING[character_id]
    for part_obj in char_objects:
        if part_obj.name != CHAR_OBJ_NAME:
            # part_obj.active_material.specular_intensity = 0.0
            if part_obj.active_material:
                mat_keys = part_obj.material_slots.keys()
                for mat_key in mat_keys:
                    if part_obj.material_slots[mat_key].material:
                        # eyes are usually transaparent on purpose
                        if 'eyes' not in part_obj.name.lower():
                            part_obj.material_slots[mat_key].material.alpha = 1.0
                        part_obj.material_slots[mat_key].material.raytrace_mirror.use = False
                    # part_obj.active_material.alpha = 1.0
                    # part_obj.active_material.raytrace_mirror.use = False
        else:
            # rescale the entire armature
            part_obj.scale = (new_scale, new_scale, new_scale)

        if not draw_character:
            part_obj.hide_render = True 

    bpy.context.scene.update()

    # clear parenting to armature of parts and orient correctly
    char_parts = []
    for part_obj in char_objects:
        print(part_obj.name)
        if part_obj.name != CHAR_OBJ_NAME:
            bpy.ops.object.select_all(action="DESELECT")
            part_obj.select = True
            bpy.context.scene.objects.active = part_obj
            bpy.ops.object.parent_clear()
            # orient
            part_obj.scale = (new_scale, new_scale, new_scale)
            part_obj.rotation_mode = 'XYZ'
            part_obj.rotation_euler = (-np.pi/2.0, 0.0, -np.pi)

            char_parts.append(part_obj)

    # don't need OG armature anymore
    for part_obj in char_objects:
        if part_obj.name == CHAR_OBJ_NAME:
            bpy.ops.object.select_all(action="DESELECT")
            bpy.data.objects[CHAR_OBJ_NAME].select = True # motion FBX
            bpy.ops.object.delete(use_global=True)

    # load in pred BVH
    axis_up = '-Y'
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_anim.bvh(filepath=predicted_bvh_path, axis_up=axis_up, global_scale=new_scale)
    pred_motion_objects = bpy.context.selected_objects

    # move motion to character parts
    parent = bpy.data.objects[predicted_bvh_path.split('/')[-1][:-4][:63]] #max length is 63
    parent_to_armature(char_parts, parent)

    # get mean char position to place camera
    anim, names, _ = BVH.load(predicted_bvh_path)
    num_frames = anim.positions.shape[0]
    mean_root = get_mean_root(anim, new_scale, flip) 

    # set lights
    setup_lighting(mean_root)
                
    com_objs, force_objs, sphere_objs = draw_com_and_forces(draw_com, draw_forces, force_on_com,
                                                            com_trajectory=com_trajectory,
                                                            feet_pos=feet_pos,
                                                            feet_forces=feet_forces)
    # view0
    setup_camera(0, None, flip, mean_root)
    render_and_make_video(out_dir, 'pred_view0', num_frames, fps, cam_params)
    # set lights
    setup_lighting(None)
    # now view1
    setup_camera(1, None, flip, mean_root)
    render_and_make_video(out_dir, 'pred_view1', num_frames, fps, cam_params)

    # delete pred armature
    bpy.ops.object.select_all(action="DESELECT")
    parent.select = True
    bpy.ops.object.delete(use_global=True)

    # clean up COM b/c this is not the COM for kinematic result
    if draw_com:
        for com_obj in com_objs:
            bpy.ops.object.select_all(action="DESELECT")
            com_obj.select = True
            bpy.ops.object.delete(use_global=True)
    if draw_forces:
        for force_obj in force_objs:
            bpy.ops.object.select_all(action="DESELECT")
            force_obj.select = True
            bpy.ops.object.delete(use_global=True)
        for sphere_obj in sphere_objs:
            bpy.ops.object.select_all(action="DESELECT")
            sphere_obj.select = True
            bpy.ops.object.delete(use_global=True)

    if kinematic_result:
        # load in pred BVH
        axis_up = '-Y'
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.import_anim.bvh(filepath=no_dynamics_bvh_path, axis_up=axis_up, global_scale=new_scale)
        pred_motion_objects = bpy.context.selected_objects

        # move motion to character parts
        parent = bpy.data.objects[no_dynamics_bvh_path.split('/')[-1][:-4][:63]] #max length is 63
        parent_to_armature(char_parts, parent)

        # get mean char position to place camera
        anim, names, _ = BVH.load(no_dynamics_bvh_path)
        num_frames = anim.positions.shape[0]

        setup_lighting(mean_root)
                    
        com_objs, force_objs, sphere_objs = draw_com_and_forces(draw_com, draw_forces, True,
                                                                com_trajectory=kinematic_com_trajectory,
                                                                feet_pos=kinematic_feet_pos,
                                                                feet_forces=kinematic_feet_forces)
        # view0
        setup_camera(0, None, flip, mean_root)
        render_and_make_video(out_dir, 'kinematic_view0', num_frames, fps, cam_params)
        setup_lighting(None)
        # now view1
        setup_camera(1, None, flip, mean_root)
        render_and_make_video(out_dir, 'kinematic_view1', num_frames, fps, cam_params)

        # delete pred armature
        bpy.ops.object.select_all(action="DESELECT")
        parent.select = True
        bpy.ops.object.delete(use_global=True)

        # clean up COM
        if draw_com:
            for com_obj in com_objs:
                bpy.ops.object.select_all(action="DESELECT")
                com_obj.select = True
                bpy.ops.object.delete(use_global=True)
        if draw_forces:
            for force_obj in force_objs:
                bpy.ops.object.select_all(action="DESELECT")
                force_obj.select = True
                bpy.ops.object.delete(use_global=True)
            for sphere_obj in sphere_objs:
                bpy.ops.object.select_all(action="DESELECT")
                sphere_obj.select = True
                bpy.ops.object.delete(use_global=True)

    # clean up parts
    for part_obj in char_parts:
        bpy.ops.object.select_all(action="DESELECT")
        part_obj.select = True
        bpy.ops.object.delete(use_global=True)

    return

def parse_args(input):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Root directory of the sequence to visualize.', required=True)
    parser.add_argument('--fbx', help='Path to FBX file for the character to render', required=True)
    parser.add_argument('--scene', help='The scene .blend file to use for rendering (has a floor/lighting set up).', required=True)
    parser.add_argument('--texture', help='Image file containing the floor texture if desired', default=None)
    parser.add_argument('--character', help='The character to render results to [default: ybot]', default='ybot')
    parser.add_argument('--out', help='output directory to save rendered images and videos to.', required=True)
    parser.add_argument('--fps', help='fps to render at [default: 30]', default='30')
    parser.add_argument('--width', help='Image width to render', default=1280)
    parser.add_argument('--height', help='Image height to render', default=720)
    parser.add_argument('--cam-f', help='Camera focal length to use (NOTE: in mm)', default=25)
    parser.add_argument('--kinematic-results', dest='kinematic_result', action='store_true', help='Renders output of initialization along with final output.')
    parser.set_defaults(kinematic_result=False)
    parser.add_argument('--draw-com', dest='draw_com', action='store_true', help='Will render COM trajectory')
    parser.set_defaults(draw_com=False)
    parser.add_argument('--draw-forces', dest='draw_forces', action='store_true', help='Will draw the contact forces. For non physics-based results, estimates forces implied by center of mass trajectory.')
    parser.set_defaults(draw_forces=False)
    parser.add_argument('--force-on-com', dest='force_on_com', action='store_true', help='Draws the forces at the COM rather than on the feet.')
    parser.set_defaults(force_on_com=False)
    parser.add_argument('--combine-feet-forces', dest='combine_feet_forces', action='store_true', help='Draws contact forces as a single combined force between the feet.')
    parser.set_defaults(combine_feet_forces=False)
    parser.add_argument('--no-character', dest='draw_character', action='store_false', help='Will not draw the character in renderings.')
    parser.set_defaults(draw_character=True)
    parser.add_argument('--no-floor', dest='draw_floor', action='store_false', help='Will not draw the floor (but will draw shadows where the floor is supposed to be).')
    parser.set_defaults(draw_floor=True)

    return parser.parse_known_args(input)[0]

if __name__=='__main__':
    argv = sys.argv
    args = parse_args(sys.argv[argv.index('--') + 1:])

    # TODO add options for camera angle and stuff that I was messing with by hand.
    cam_params = (args.width, args.height, args.cam_f)
    render_multiview_eval(args.results, args.character, args.fbx, args.scene, args.texture, 
                        args.out, args.fps, True, kinematic_result=args.kinematic_result, draw_com=args.draw_com, 
                        draw_forces=args.draw_forces, force_on_com=args.force_on_com, combine_feet_forces=args.combine_feet_forces, 
                        draw_floor=args.draw_floor, draw_character=args.draw_character, cam_params=cam_params)