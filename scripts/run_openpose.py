import os, sys, shutil, argparse, subprocess
from multiprocessing import Pool

'''
Runs OpenPose on all videos in the given directory.
'''

video_extensions = ['avi', 'mpg', 'mp4', 'mov']
output_extension = 'mp4'
num_processes = 1

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, help='Relative path to root directory containing video data to run openpose on.')
    parser.add_argument('--out', required=True, help='Relative path to root directory to output results. Will be built with same structure as input.')
    parser.add_argument('--openpose', required=True, help='Relative path to root of openpose directory.')
    parser.add_argument('--hands', dest='hands', action='store_true', help='Detect hands')
    parser.set_defaults(hands=False)
    parser.add_argument('--face', dest='face', action='store_true', help='Detect face')
    parser.set_defaults(face=False)
    parser.add_argument('--save-video', dest='save_video', action='store_true', help='Save the OpenPose results as a video.')
    parser.set_defaults(save_video=False)

    flags = parser.parse_known_args()
    flags = flags[0]
    return flags   

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def make_absolute(rel_paths):
    ''' Makes a list of relative paths absolute '''
    return [os.path.join(os.getcwd(), rel_path) for rel_path in rel_paths]

def detect_pose(arg_paths):
    # run open pose
    input_video, json_out_path, video_out_path, image_out_path, openpose_path, \
                detect_hands, detect_face, save_video = arg_paths
    og_cwd = os.getcwd()
    os.chdir(openpose_path)

    print('Running ' + input_video.split('/')[-1] + '...')
    # first split into images
    subprocess.run(['ffmpeg', '-i', input_video, image_out_path])
    # then run openpose
    openpose_image_out = '/'.join(image_out_path.split('/')[:-1])
    run_cmds = ['./build/examples/openpose/openpose.bin', \
                    '--image_dir', openpose_image_out, '--write_json', json_out_path, \
                    '--display', '0', '--model_pose', 'BODY_25', '--number_people_max', '1']
    if detect_hands:
        run_cmds += ['--hand']
    if detect_face:
        run_cmds += ['--face']
    if save_video:
        run_cmds +=  ['--write_video', video_out_path, '--write_video_fps', '30']
    else:
        run_cmds += ['--render_pose', '0']
    subprocess.run(run_cmds)

    os.chdir(og_cwd) # change back to resume
    return

def main(data_path, out_path, openpose_path, detect_hands=False, detect_face=False, save_video=False):
    mkdir(out_path)

    # walk through and run openpose on all discovered videos
    for dirpath, dirnames, filenames in os.walk(data_path):
        if data_path == dirpath:
            cur_out_dir = out_path
        else:
            # create sub-directory in out path no matter what
            sub_path = dirpath.split(data_path)[-1][1:]
            cur_out_dir = os.path.join(out_path, sub_path)
            mkdir(cur_out_dir)

        video_files = sorted([f for f in filenames if f.split('.')[-1] in video_extensions and f[0] != '.'])
        if len(video_files) == 0:
            continue
        input_videos = [os.path.join(dirpath, f) for f in video_files]

        # create directories to hold outputs
        video_names = [f.split('.')[0] for f in video_files]
        json_out_paths = []
        video_out_paths = []
        image_out_paths = []
        for video_name in video_names:
            video_out_path = os.path.join(cur_out_dir)
            mkdir(video_out_path)
            video_json_path = os.path.join(video_out_path, 'openpose_result')
            mkdir(video_json_path)
            video_image_path = os.path.join(video_out_path, 'raw_image')
            mkdir(video_image_path)
            json_out_paths.append(video_json_path)
            video_out_paths.append(os.path.join(video_out_path, video_name + '_openpose_viz.' + output_extension))
            image_out_paths.append(video_image_path + '/' + video_name + '_%08d.png')

        # make all paths absolute
        input_videos = make_absolute(input_videos)
        json_out_paths = make_absolute(json_out_paths)
        video_out_paths = make_absolute(video_out_paths)
        image_out_paths = make_absolute(image_out_paths)

        # spawn off processes to run multiple simultaneously
        pool = Pool(processes=num_processes)
        num = len(image_out_paths)
        pool.map(detect_pose, zip(input_videos, json_out_paths, video_out_paths, image_out_paths, \
                                num*[openpose_path], num*[detect_hands], num*[detect_face], num*[save_video]))
        pool.close()
        pool.join()

if __name__=='__main__':
    flags = parse_args(sys.argv[1:])
    data_path = flags.data
    out_path = flags.out
    openpose_path = flags.openpose
    main(data_path, out_path, openpose_path, flags.hands, flags.face, flags.save_video)