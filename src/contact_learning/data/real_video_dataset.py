import os, sys
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

sys.path.extend(['contact_learning'])
from data.openpose_dataset import process_openpose_data, normalize_openpose_data, \
                                    OP_LOWER_JOINTS, OP_LOWER_JOINTS_MAP, OP_LOWER_PARENTS, OP_ROOT_JOINT, OP_JOINT_SUBSETS

sys.path.extend(['utils'])
from openpose_utils import filter_poses, render_2d_keypoints, load_keypoint_dir

# dimensions that network is trained on
TRAIN_DIM = (1280,720)
TRAIN_NORMALIZATION = 200.4160302695367 # median hip->toe dist in training
CONTACTS_FILE = 'foot_contacts.npy'
OP_DIR = 'openpose_result'
FRAME_DIR = 'raw_image'
FRAME_EXTNS = ['png', 'jpg', 'jpeg']

class RealVideoDataset(Dataset):
    ''' 
    Dataset of real videos with sequences of OpenPose joint positions.
    Should only be used for testing after training on the OpenPoseDataset.
    '''

    def __init__(self, data_root, split='test', window_size=5, contact_size=3, dimensions=(1920,1080), load_img=False, use_confidence=True, joint_set='lower'):
        '''
        Creates an RealVideoDataset object. Returned windows will always be overlapping.

        Note videos are padded to the length of the longest video for ease of batching, but original
        length can be recovered with self.seq_lens

        - data_root : path to the root directory of the dataset (contains a different directory 
                      for each video which contains the )
        - split : only 'test' is supported
        - window_size : number of frames around a target frame to use (must be odd).
        - contact_size : number of middle frames to return contact labels for (if available)
        - dimensions : dimension of the images on which OpenPose was detected
        - load_img : if True, getitem will also return the image frame corresponding to the target frame of that window
        - use_confidence : if true, joint2d data will also return confidence of openpose detection 
        '''
        self.root_dir = data_root
        if window_size % 2 == 0:
            # must be odd
            window_size += 1
        self.window_size = window_size
        self.contact_size = contact_size
        self.split = split
        self.load_img = load_img
        self.use_confidence = use_confidence
        self.joint_set = joint_set

        if self.split != 'test':
            print('Does not support split type ' + self.split)
            return
        if not self.joint_set in OP_JOINT_SUBSETS.keys():
            print('Do not support joint subset ' + self.joint_set)
            return

        print('Using joint subset ' + self.joint_set)
        print('Using windows of size ' + str(self.window_size) + '...')
        print('Using contact labels of middle ' + str(self.contact_size) + ' frames...')

        # load video directory paths
        if not os.path.exists(data_root):
            print('Could not find data path ' + data_root)
            return
        video_dirs = sorted([os.path.join(data_root, f) for f \
                     in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f)) \
                     and f[0] != '.'])
        print('Found ' + str(int(len(video_dirs))) + ' videos to load...')

        self.video_names = [f.split('/')[-1] for f in video_dirs]

        # load in OpenPose data for each
        print('Loading OpenPose data...')
        self.num_frames = -float('inf') # will set to length of longest video so we can batch
        self.op_data = [] # list of np.arrays(F, J, 3)
        self.seq_lens = []
        for i, video_dir in enumerate(video_dirs):
            cur_op_path = os.path.join(video_dir, OP_DIR)
            if not os.path.exists(cur_op_path):
                print('Could not find OpenPose data for ' + self.video_names[i])
                self.op_data.append(None)
                continue
            joint2d_seq = load_keypoint_dir(cur_op_path)
            self.seq_lens.append(joint2d_seq.shape[0])
            if joint2d_seq.shape[0] > self.num_frames:
                self.num_frames = joint2d_seq.shape[0]
            self.op_data.append(joint2d_seq)
        
        # load in frame paths, will lazily load images as needed
        print('Loading image frame data paths...')
        self.frame_paths = [] # list of lists of file paths size F
        for i, video_dir in enumerate(video_dirs):
            cur_frames_path = os.path.join(video_dir, FRAME_DIR)
            if not os.path.exists(cur_frames_path):
                print('Could not find frame data for ' + self.video_names[i])
                self.frame_paths.append([])
                continue
            cur_frames = sorted([os.path.join(cur_frames_path, f) for f \
                     in os.listdir(cur_frames_path) if os.path.isfile(os.path.join(cur_frames_path, f)) \
                     and f[0] != '.' and (f.split('.')[-1] in FRAME_EXTNS)])
            self.frame_paths.append(cur_frames)
            # print(len(self.frame_paths[-1]))
        # load in foot contact data
        print('Loading foot contact data...')
        self.contact_data = [] # list of np.arrays(F x 4)
        for i, video_dir in enumerate(video_dirs):
            cur_contact_path = os.path.join(video_dir, CONTACTS_FILE)
            if not os.path.exists(cur_contact_path):
                print('Could not find contact data for ' + self.video_names[i])
                self.contact_data.append(None)
                continue
            self.contact_data.append(np.load(cur_contact_path))
            # print(self.contact_data[-1].shape[0])
        
        if len(self.op_data) != len(self.contact_data):
            print('[RealVideoDataset] Data lengths are not compatible! Should all be the same but...')
            print('Num OpenPose dirs: ' + str(len(self.op_data)))
            print('Num contact files: ' + str(len(self.contact_data)))
        if len(self.op_data) != len(self.frame_paths):
            print('[RealVideoDataset] Data lengths are not compatible! Should all be the same but...')
            print('Num OpenPose dirs: ' + str(len(self.op_data)))
            print('Num image dirs: ' + str(len(self.frame_paths)))

        # every frame gets window except on the very edges without enough frames on each side
        print('Padding all videos to size of longest for batching: %d frames' % (self.num_frames))
        self.test_windows_per_seq = self.num_frames - 2*(self.window_size // 2)
        print('Using overlapping test frames with %d windows per video!' % (self.test_windows_per_seq))

        # data is in windows, so length of data is number of videos*windows_per_video
        self.data_len = len(self.op_data) * self.test_windows_per_seq
        print('Loaded ' + split + ' split with ' + str(self.data_len) + ' motion windows from ' + str(len(self.op_data)) + ' videos...')

        self.normalization_info = TRAIN_NORMALIZATION

        # modify our data to be the desired number of frames
        print('Truncating/padding all videos to length ' + str(self.num_frames) + '...')
        self.op_data, self.contact_data, self.frame_paths = \
                    self.fix_data_len(self.num_frames, self.op_data, self.contact_data, self.frame_paths)

        # need to scale openpose data to fit the size used in training
        print('Scaling OpenPose data to match training...')
        scale_factor_w = float(TRAIN_DIM[0]) / dimensions[0]
        scale_factor_h = float(TRAIN_DIM[1]) / dimensions[1] 
        if (scale_factor_w - scale_factor_h) > 1e-5:
            print('Videos and training data must be the same aspect ratio!')
            return
        for i, op_data_seq in enumerate(self.op_data):
            self.op_data[i][:,:,:2] *= scale_factor_w # only scale (x,y), not confidence

        # OpenPose data is noisy and has missing frames
        # need to do some preprocessing
        self.op_data = process_openpose_data(self.op_data, confidence_thresh=0.2)
        # also normalize
        self.op_data = normalize_openpose_data(self.normalization_info, self.op_data)

        return

    def fix_data_len(self, desired_len, op_data, contact_data, frame_paths):
        ''' Trims or pads the given data to the desired number of frames. '''
        for i in range(len(op_data)):
            # openpose
            cur_op = op_data[i] # F x J x 3
            if cur_op.shape[0] > desired_len:
                op_data[i] = cur_op[:desired_len]
            elif cur_op.shape[0] < desired_len:
                pad = np.repeat(cur_op[-1].reshape((1, cur_op.shape[1], cur_op.shape[2])), desired_len - cur_op.shape[0], axis=0)
                op_data[i] = np.concatenate([cur_op, pad], axis=0)
            # contacts
            cur_contacts = contact_data[i] # F x 4
            if cur_contacts is not None:
                if cur_contacts.shape[0] > desired_len:
                    contact_data[i] = cur_contacts[:desired_len]
                elif cur_contacts.shape[0] < desired_len:
                    pad = np.repeat(cur_contacts[-1].reshape((1, cur_contacts.shape[1])), desired_len - cur_contacts.shape[0], axis=0)
                    contact_data[i] = np.concatenate([cur_contacts, pad], axis=0)
            # frames
            cur_frame_paths = frame_paths[i] # F
            if len(cur_frame_paths) > 0:
                if len(cur_frame_paths) > desired_len:
                    frame_paths[i] = cur_frame_paths[:desired_len]
                elif len(cur_frame_paths) < desired_len:
                    frame_paths[i] += [cur_frame_paths[-1]]*(desired_len - len(cur_frame_paths))

        return op_data, contact_data, frame_paths

    def get_num_test_windows_per_seq(self):
        return self.test_windows_per_seq
    
    def get_joint_scaling(self):
        ''' Returns the scaling that is applied to all joint2d data '''
        return self.normalization_info

    def __len__(self):
        '''
        Gets the length of the dataset.
        '''
        return self.data_len

    def __getitem__(self, idx):
        '''
        Get a single item from the dataset. This is a dictionary containing
        'joint2d' : N x J x 3 tensor where N is the window size, J is the number
        of lower-body joints and 3 is (x, y, confidence). It also contains 
        'contacts' : [left_heel, left_toes, right_heel, right_toes] a binary tensor.
        '''
        seq_idx = -1 # the motion we will take the window from
        tgt_idx = -1 # the frame we want to classify (the center of the window)
        frame_window = [-1, -1] # the whole window to return [inclusive, exclusive)

        # for val/test, idx defines both motion sequence AND window
        # figure out which motion/view based on index
        seq_idx = idx // self.test_windows_per_seq
        window_idx = idx % self.test_windows_per_seq
        # every frame except first window_size//2 have a window
        start_frame = window_idx        
        end_frame = start_frame + self.window_size
        frame_window = [start_frame, end_frame]
        tgt_idx = start_frame + (self.window_size // 2)
        # print('Target motion: ' + str(seq_idx))
        # print('Target frame: ' + str(tgt_idx))
        # print('Window = [' + str(frame_window[0]) + ', ' + str(frame_window[1]) + ')')

        # grab data window
        cur_op_data = self.op_data[seq_idx][frame_window[0]:frame_window[1]].copy()
        cur_contact_data = None
        if self.contact_data[seq_idx] is not None:
            cur_contact_data = self.contact_data[seq_idx][frame_window[0]:frame_window[1]].copy()
        cur_frame_paths = self.frame_paths[seq_idx]
        cur_frames = None
        if len(cur_frame_paths) > 0 and self.load_img:
            # only actually load if desired
            # only load target frame
            cur_frames = io.imread(cur_frame_paths[tgt_idx])[:,:,:3] # remove alpha
            # resize to the scale of the training frames
            cur_frames = transform.resize(cur_frames, (TRAIN_DIM[1], TRAIN_DIM[0]))

        # normalize w.r.t root (MidHip)
        rel_tgt_idx = self.window_size // 2
        root_idx = OP_ROOT_JOINT
        tgt_root = cur_op_data[rel_tgt_idx, root_idx, :2].copy() # only (x,y)
        tgt_root = tgt_root.reshape((1, 1, 2))
        cur_op_data[:,:,:2] -= tgt_root

        # add back in only for target frame to give some sense of global position
        cur_op_data[rel_tgt_idx, root_idx, :2] = tgt_root

        # for OP, only want subset of joints
        cur_op_data = cur_op_data[:,OP_JOINT_SUBSETS[self.joint_set],:] # F x J x 3

        # for contacts, want middle contact_size frames
        contact_offset = (self.window_size - self.contact_size) // 2
        if cur_contact_data is not None:
            cur_contact_data = cur_contact_data[contact_offset:(self.window_size - contact_offset)]

        # take out confidence if need be
        if not self.use_confidence:
            cur_op_data = cur_op_data[:,:,:2]

        # to pytorch
        cur_op_data = torch.from_numpy(cur_op_data.astype(np.float32))
        if cur_contact_data is not None:
            cur_contact_data = torch.from_numpy(cur_contact_data.astype(np.float32))
        cur_item = {'joint2d' : cur_op_data, 'name' : self.video_names[seq_idx], 'seq_len' : self.seq_lens[seq_idx]}
        if cur_contact_data is not None:
            cur_item['contacts'] = cur_contact_data
        if cur_frames is not None and self.load_img:
            cur_item['frames'] = cur_frames
        # print(cur_item)
        return cur_item