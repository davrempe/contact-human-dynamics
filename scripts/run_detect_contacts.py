'''
Given a data folder, detect contacts using learned network and copies resulting contacts
back to the data folder.
'''

import os, sys, shutil, argparse, subprocess, time
from tempfile import TemporaryDirectory

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, help='Relative path to root directory containing video data. This is a directory of directories, each sub directory should contain a single mp4 that will be run. Results will also be copied to this directory (foot_contacts.npy).')
    parser.add_argument('--weights', default='../pretrained_weights/contact_detection_weights.pth', help='Relative path to the pretrained network weights.')
    parser.add_argument('--viz', dest='viz', action='store_true', help='If given, will visualize contacts detected.')
    parser.set_defaults(viz=False)

    flags = parser.parse_known_args()
    flags = flags[0]
    return flags

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def make_absolute(rel_path):
    ''' Makes a list of relative paths absolute '''
    return os.path.join(os.getcwd(), rel_path)


def main(data_path, viz, weights):
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
    with TemporaryDirectory() as abs_out_path:
        # run detect contacts on this video_dir
        og_cwd = os.getcwd()
        os.chdir(abs_src_path)
        process_cmds = ['python', 'contact_learning/test.py', \
                        '--data', abs_data_path, '--out', abs_out_path, \
                        '--weights-path', weights,
                        '--full-video', '--save-contacts', '--real-data']
        if viz:
            process_cmds += ['--viz']
        subprocess.run(process_cmds)
        os.chdir(og_cwd) # change back to resume

        # copy the results back to the original
        print('Copying results back to source data dir...')
        contact_results_path = os.path.join(abs_out_path, 'contact_results')
        full_video_results_path = os.path.join(abs_out_path, 'full_video_results')
        for (video_dir, video_name) in zip(video_dirs, video_names):
            # saved contacts
            copy_src = os.path.join(contact_results_path, video_name + '/foot_contacts.npy')
            copy_dst = os.path.join(video_dir, 'foot_contacts.npy')
            shutil.copyfile(copy_src, copy_dst)

            if viz:
                # saved videos
                copy_src = os.path.join(full_video_results_path, video_name + '.mp4')
                viz_out_dir = os.path.join(video_dir, 'contact_results_viz')
                mkdir(viz_out_dir)
                copy_dst = os.path.join(viz_out_dir, 'contact_detection_results.mp4')
                shutil.copyfile(copy_src, copy_dst)


if __name__=='__main__':
    flags = parse_args(sys.argv[1:])
    data_path = flags.data
    main(data_path, flags.viz, flags.weights)