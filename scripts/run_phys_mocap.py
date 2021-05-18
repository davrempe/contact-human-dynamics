'''
Given a data folder, run kinematic optimization, character retargeting, and
physics-based optimization on each contained sequence.

Note each data directory MUST already have the following files from previous parts of the pipeline:
- tracked_results.json and openpose_result directory from Monocular Total Capture
- foot_contacts.npy from foot contact detection
'''

import os, sys, shutil, argparse, subprocess, time, glob
import cv2

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, help='Relative path to root directory containing video data. This is a directory of directories, each sub directory should contain a single mp4 along with MTC and contact detection results.')
    parser.add_argument('--character', default='ybot', choices=['combined', 'ybot', 'ty', 'skeletonzombie'], help='The character to optimize motion on. Note that `combined` will optimize on the original skeleton in the video and not an animated character.')

    # Settings for kinematic initialization
    parser.add_argument('--kinematic_viz', dest='kinematic_viz', action='store_true', help='If given, shows interactive intermediate visualizations throughout kinematic initialization.')
    parser.set_defaults(kinematic_viz=False)
    parser.add_argument('--kinematic_gt_floor', dest='kinematic_gt_floor', action='store_true', help='If given, will use the ground truth floor plane in the data director (floor_gt.txt) rather than estimating it from motion.')
    parser.set_defaults(kinematic_gt_floor=False)

    # Settings for physics-based optimization
    parser.add_argument('--towr_phys_optim_path', default='../towr_phys_optim/build', help='Path to build of physical optimization.')


    flags = parser.parse_known_args()
    flags = flags[0]
    return flags

class PhysOptimParsms():
    def __init__(self):
        # weight for COM linear position
        self.w_com_lin = 0.4
        # weight for COM angular orientation
        self.w_com_ang = 1.7 
        # weight for end-effector position
        self.w_ee = 0.3
        # weight for velocity smoothing
        self.w_smooth = 0.1
        # weight for total duration cost
        self.w_dur = 0.1

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def make_absolute(rel_path):
    ''' Makes a list of relative paths absolute '''
    return os.path.join(os.getcwd(), rel_path)


def main(args):
    data_path = args.data
    character = args.character
    if not os.path.exists(data_path):
        print('Could not find data path!')
        exit()

    video_dirs = sorted([os.path.join(data_path, f) for f \
                     in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f)) \
                     and f[0] != '.'])
    video_names = [video_dir.split('/')[-1] for video_dir in video_dirs]

    if len(video_dirs) == 0:
        print('No video directories in the data path!')
        exit()

    print(video_dirs)

    abs_data_path = make_absolute(data_path)
    abs_src_path = make_absolute('../src')
    abs_towr_path = make_absolute(args.towr_phys_optim_path)
    og_cwd = os.getcwd()
    print(og_cwd)

    # run pipeline on each video
    for video_dir, video_name in zip(video_dirs, video_names):
        print('Running pipeline for %s...' % (video_name))
        abs_video_dir = make_absolute(video_dir)
        cur_vid_file = os.path.join(abs_video_dir, video_name + '.mp4')
        if not os.path.exists(cur_vid_file):
            print('Data directory and contained video must have the same name!')
            exit()

        vidcap = cv2.VideoCapture(cur_vid_file)
        cur_fps = int(round(vidcap.get(cv2.CAP_PROP_FPS)))

        #
        # Run kinematic optimization
        #
        print('Running kinematic optimization...')
        # get number of frames in video
        cur_op_path = os.path.join(video_dir, 'openpose_result')
        cur_num_frames = len(glob.glob(os.path.join(cur_op_path, '*.json')))
        # create output path
        kinematic_optim_out = os.path.join(abs_video_dir, 'kinematic_results')
        mkdir(kinematic_optim_out)
        # run kinematic optimization procedure
        os.chdir(abs_src_path)
        kinematic_optim_cmd = ['python', 'optimize/kinematic_optimizer.py', \
                               '--input_path', os.path.join(abs_video_dir, video_name + '.mp4'), \
                               '--skel_path', 'skeleton_fitting/combined_body_25.bvh', \
                               '--output_path', kinematic_optim_out, \
                               '--end', str(cur_num_frames),
                               '--character', character
                               ]
        if args.kinematic_gt_floor:
            kinematic_optim_cmd += ['--gt-floor']
        if args.kinematic_viz:
            kinematic_optim_cmd += ['--visualize']
        print(kinematic_optim_cmd)
        subprocess.run(kinematic_optim_cmd)

        #
        # Run re-targeting to character if needed
        #
        final_kin_res = os.path.join(kinematic_optim_out, 'final_test.bvh')
        char_rtgt_res = os.path.join(kinematic_optim_out, character + '_out.bvh')
        if character != 'combined':
            print('Running retargeting...')
            retargeting_cmd = ['python', 'skeleton_fitting/combined_to_mixamo.py', \
                               '--src_bvh', final_kin_res, \
                               '--out_bvh', char_rtgt_res, \
                               '--character', character
                               ]
            print(retargeting_cmd)
            subprocess.run(retargeting_cmd)
        else:
            # copy the final output as the character out
            shutil.copyfile(final_kin_res, char_rtgt_res)            

        #
        # Generate input for physics-based optimization based on kinematic output
        #
        print('Generating input for physics-based optimization...')
        os.chdir(os.path.join(abs_src_path, 'utils'))
        phys_input_path = os.path.join(abs_video_dir, 'phys_optim_in_' + character)
        mkdir(phys_input_path)
        gen_input_cmd = ['python', 'towr_utils.py', \
                        '--anim', char_rtgt_res, \
                        '--floor', os.path.join(kinematic_optim_out, 'floor_out.txt'), \
                        '--out', phys_input_path,
                        '--contacts', os.path.join(kinematic_optim_out, 'foot_contacts.npy'),
                        '--start', str(0),
                        '--end', str(cur_num_frames),
                        '--fps', str(cur_fps),
                        '--character', character
                        ]
        print(gen_input_cmd)
        subprocess.run(gen_input_cmd)

        #
        # Run physics-based optimization
        #
        print('Running physics-based optimization...')
        os.chdir(abs_towr_path)
        phys_output_path = os.path.join(abs_video_dir, 'phys_optim_out_' + character)
        mkdir(phys_output_path)
        phys_params = PhysOptimParsms()
        phys_optim_cmd = ['./phys_optim',
                        '--in_dir', phys_input_path,
                        '--nframes', str(cur_num_frames),
                        '--out_dir', phys_output_path,
                        '--w_com_lin', str(phys_params.w_com_lin),
                        '--w_com_ang', str(phys_params.w_com_ang),
                        '--w_ee', str(phys_params.w_ee),
                        '--w_smooth', str(phys_params.w_smooth),
                        '--w_dur', str(phys_params.w_dur)
                        ]
        print(phys_optim_cmd)
        subprocess.run(phys_optim_cmd)

        #
        # Save output BVH and matplotlib visualize (simple skeleton in matplotlib).
        # For better visualization, see src/viz/viz_blender.py
        #
        print('Visualizing results...')
        os.chdir(os.path.join(abs_src_path, 'utils'))
        phys_res_names = ['sol_out_no_dynamics.txt', 'sol_out_dynamics.txt', 'sol_out_durations.txt']
        phys_res_paths = [os.path.join(phys_output_path, f) for f in phys_res_names]
        out_bvh_names = ['_no_dynamics.bvh', '_dynamics.bvh', '_durations.bvh']
        out_bvh_paths = [os.path.join(phys_output_path, video_name + '_' + character + f) for f in out_bvh_names]
        viz_res_cmd = ['python', 'towr_utils.py', \
                        '--data', phys_res_paths[0], phys_res_paths[1], phys_res_paths[2],
                        '--out-bvh', out_bvh_paths[0], out_bvh_paths[1], out_bvh_paths[2],
                        '--floor', os.path.join(kinematic_optim_out, 'floor_out.txt'), \
                        '--anim', char_rtgt_res, \
                        '--contacts', os.path.join(kinematic_optim_out, 'foot_contacts.npy'),
                        '--start', str(0),
                        '--end', str(cur_num_frames),
                        '--fps', str(cur_fps),
                        '--character', character,
                        '--out-vid', os.path.join(phys_output_path, video_name + '_compare_all_skel.mp4'),
                        '--name', 'Spline', 'Dynamics', 'Durations',
                        '--forces', '--trace', '--viz', '--hide', '--skel', '--compare-og'
                       ]
        print(viz_res_cmd)
        subprocess.run(viz_res_cmd)


        os.chdir(og_cwd) # change back to resume

if __name__=='__main__':
    flags = parse_args(sys.argv[1:])
    main(flags)