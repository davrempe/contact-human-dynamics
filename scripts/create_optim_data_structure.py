
'''
Given a directory containing many videos to run our pipeline on, sets it up in
the structure that our pipeline expects: one directory containing each video.
'''

import os, sys, shutil, argparse

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, help='Relative path to root directory containing video data.') 
    parser.add_argument('--out', required=True, help='Relative path to the directory to output. This will be a directory of directories, each sub directory containing a single video that will be run.')

    flags = parser.parse_known_args()
    flags = flags[0]
    return flags


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def main(data_path, out_path):
    if not os.path.exists(data_path):
        print('Could not find data path!')
        exit()

    video_files = sorted([os.path.join(data_path, f) for f \
                     in os.listdir(data_path) if f[0] != '.'])
    video_names = [video_file.split('/')[-1].split('.')[0] for video_file in video_files]

    print(video_names)

    if len(video_names) == 0:
        print('No videos in the data path!')
        exit()

    mkdir(out_path)

    for video_path, video_name in zip(video_files, video_names):
        cur_video_file = video_path.split('/')[-1]
        cur_video_out_path = os.path.join(out_path, video_name)
        print(cur_video_out_path)
        mkdir(cur_video_out_path)
        shutil.copy(video_path, os.path.join(cur_video_out_path, cur_video_file))


if __name__=='__main__':
    flags = parse_args(sys.argv[1:])
    data_path = flags.data
    out_path = flags.out
    main(data_path, out_path)