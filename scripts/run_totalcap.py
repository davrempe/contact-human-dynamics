import os, sys, shutil, argparse, subprocess, time

import cv2

'''
Runs Monocular Total Capture on a given directory of directories of videos. Automatically resizes/pads videos to required 1920x1080.
'''

VID_EXTNS = ['avi', 'mpg', 'mp4', 'mov']

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, help='Relative path to root directory containing video data. This is a directory of directories, each sub directory should contain a single mp4 that will be run. Results will also be copied to this directory (openpose_results, tracked_result.json, untracked_result.json).')
    parser.add_argument('--out', required=True, help='Relative path to root directory to output visual results (tracked and untracked videos).')
    parser.add_argument('--totalcap', required=True, help='Relative path to root of monocular total capture directory.')
    parser.add_argument('--viz-data', default=None, help='(Optional) If set to a total capture data output diretory (MonocularTotalCapture/data/*), will only visualize results, not rerun the entire fitting.')
    parser.add_argument('--results-data', default=None, help='(Optional) Name of a results folder within a given viz-data folder. If provided data from this results folder will be visualized. Otherwise the usual tracked/untracked output will be used.')

    flags = parser.parse_known_args()
    flags = flags[0]
    return flags

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def make_absolute(rel_path):
    ''' Makes a list of relative paths absolute '''
    return os.path.join(os.getcwd(), rel_path)

def make_video(img_path, out_path, fps=24):
    subprocess.run(['ffmpeg', '-r', str(fps), '-i', img_path, \
                        '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', out_path])

def split_video(vid_path, out_path):
    ''' Split a video into images and save them at given out_path '''
    vid_path = make_absolute(vid_path)
    out_path = make_absolute(out_path)
    subprocess.run(['ffmpeg', '-i', vid_path, out_path])

def process_output(video_data_path, out_dir, custom_out=None):
    # collect outputs and create video results
    prefixes = []
    if custom_out is not None:
        prefixes = [custom_out + '_']
    else:
        untracked_path = os.path.join(video_data_path, 'body_3d_frontal')
        tracked_path = os.path.join(video_data_path, 'body_3d_frontal_tracking')
        make_video(untracked_path + '/%04d.png', os.path.join(out_dir, 'untracked_overlay.mp4'))
        make_video(tracked_path + '/%04d.png', os.path.join(out_dir, 'tracked_overlay.mp4'))
        prefixes = ['body_3d_frontal_tracking_', 'body_3d_frontal_']

    # make videos for all the different angles
    for prefix in prefixes:
        front_path = os.path.join(video_data_path, prefix + 'front_renders')
        make_video(front_path + '/%04d.png', os.path.join(out_dir, prefix + 'front.mp4'))
        joint_front_path = os.path.join(video_data_path, prefix + 'joint_front_renders')
        make_video(joint_front_path + '/%04d.png', os.path.join(out_dir, prefix + 'joint_front.mp4'))
        joint_side_path = os.path.join(video_data_path, prefix + 'joint_side_renders')
        make_video(joint_side_path + '/%04d.png', os.path.join(out_dir, prefix + 'joint_side.mp4'))
        side_path = os.path.join(video_data_path, prefix + 'side_renders')
        make_video(side_path + '/%04d.png', os.path.join(out_dir, prefix + 'side.mp4'))
        top_path = os.path.join(video_data_path, prefix + 'top_renders')
        make_video(top_path + '/%04d.png', os.path.join(out_dir, prefix + 'top.mp4'))

    return

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
    Resizes the image and cubically interpolates for that it is the given new_size (W, H).
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
    
    im = cv2.resize(im, (new_W, new_H), interpolation=cv2.INTER_AREA)
    # im = cv2.resize(im, (new_W, new_H), interpolation=cv2.INTER_CUBIC)
    return im

def run_totalcap(video_path, totalcap_path, totalcap_data_path, out_path, results_out_path, viz_only=False, results_name=None):
    ''' 
    Splits video into images, runs monocular total capture on it, and renders video of untracked and tracked result 
    '''
    # create new data directory for this video
    video_name = '.'.join(video_path.split('/')[-1].split('.')[:-1])
    print('Running ' + video_name + '...')
    video_data_path = os.path.join(totalcap_data_path, video_name)
    mkdir(video_data_path)
    image_out_path = os.path.join(video_data_path, 'raw_image')
    mkdir(image_out_path)

    data_name = totalcap_data_path.split('/')[-1]
    if data_name == '':
        data_name = totalcap_data_path.split('/')[-2]

    if not viz_only:
        # split video into images
        split_video(video_path, os.path.join(image_out_path, video_name + '_%08d.png'))
        # pad all images to 1920x1080
        video_frames = sorted([os.path.join(image_out_path, f) for f in os.listdir(image_out_path) if f[0] != '.'])
        for frame in video_frames:
            im = cv2.imread(frame)
            im = resize_image(im, (1920, 1080))
            im = pad_image(im, (1920, 1080))
            cv2.imwrite(frame, im)

        # run total capture
        og_cwd = os.getcwd()
        os.chdir(totalcap_path)
        totalcap_cmds = ['bash', 'run_pipeline_no_ffmpeg.sh', video_name, './data/' + data_name]
        subprocess.run(totalcap_cmds)
        os.chdir(og_cwd) # change back to resume

    # run processing to prepare data for downstream optimization
    og_cwd = os.getcwd()
    os.chdir(totalcap_path)
    # already does both tracked and untracked results
    process_cmds = ['bash', 'run_processing.sh', video_name, './data/' + data_name]
    subprocess.run(process_cmds)
    os.chdir(og_cwd) # change back to resume

    # run visualization post-processing
    og_cwd = os.getcwd()
    os.chdir(totalcap_path)
    viz_cmds = ['bash', 'run_visualization.sh', video_name, './data/' + data_name]
    if results_name is None:
        subprocess.run(viz_cmds + ['body_3d_frontal_tracking']) # tracked output
        subprocess.run(viz_cmds + ['body_3d_frontal']) # no tracking output
    else:
        custom_path = './data/' + data_name + '/' + video_name + '/' + results_name
        if not os.path.exists(custom_path):
            print('Could not find given results directory: ' + results_name)
            os.chdir(og_cwd) # change back to resume
            return
        # get number of frames
        num_frames = len([f for f in os.listdir(custom_path) if f.split('.')[-1] == 'txt' and f[0] != '.'])
        subprocess.run(viz_cmds + [results_name, str(num_frames)]) # custom output

    os.chdir(og_cwd) # change back to resume

    # collect outputs and create video results
    video_out_path = os.path.join(out_path, video_name)
    mkdir(video_out_path)
    process_output(video_data_path, video_out_path, custom_out=results_name)

    # copy over actual total capture output
    openpose_results_path = os.path.join(video_data_path, 'openpose_result')
    openpose_dest = os.path.join(results_out_path, 'openpose_result')
    tracked_results_path = os.path.join(video_data_path, 'tracked_results.json')
    tracked_dest = os.path.join(results_out_path, 'tracked_results.json')
    untracked_results_path = os.path.join(video_data_path, 'untracked_results.json')
    untracked_dest = os.path.join(results_out_path, 'untracked_results.json')
    raw_images_path = os.path.join(video_data_path, 'raw_image')
    raw_images_dest = os.path.join(results_out_path, 'raw_image')
    
    shutil.copyfile(tracked_results_path, tracked_dest)
    shutil.copyfile(untracked_results_path, untracked_dest)
    shutil.copytree(openpose_results_path, openpose_dest)
    shutil.copytree(raw_images_path, raw_images_dest)

    return

def main(data_path, totalcap_path, out_path, viz_data_path=None, results_name=None):
    mkdir(out_path)
    video_dirs = sorted([os.path.join(data_path, f) for f \
                     in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f)) \
                     and f[0] != '.'])

    video_files = []
    for video_dir in video_dirs:
        video_files += sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.split('.')[-1] in VID_EXTNS and f[0] != '.'])

    print(video_dirs)
    print(video_files)

    if len(video_dirs) != len(video_files):
        print('There may only be ONE video in each video directory!')
        return

    # create a data diretory in totalcap to hold all input/outputs
    totalcap_out_path = viz_data_path
    if totalcap_out_path is None:
        totalcap_out_path = os.path.join(totalcap_path, 'data/data_' + str(int(time.time())))
        mkdir(totalcap_out_path)

    print('TotalPose output will be in ' + totalcap_out_path + '...')

    for video, result_out in zip(video_files, video_dirs):
        run_totalcap(video, totalcap_path, totalcap_out_path, out_path, result_out, viz_only=(viz_data_path != None), results_name=results_name)


if __name__=='__main__':
    flags = parse_args(sys.argv[1:])
    data_path = flags.data
    out_path = flags.out
    totalcap_path = flags.totalcap
    viz_data_path = flags.viz_data
    main(data_path, totalcap_path, out_path, viz_data_path=viz_data_path, results_name=flags.results_data)
