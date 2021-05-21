import os, sys
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

sys.path.extend(['contact_learning'])
from data.contact_data_utils import get_all_character_paths, get_all_motion_paths, get_all_openpose_paths, \
                                get_all_openpose_data, get_all_foot_contact_paths, get_all_foot_contact_data, \
                                get_all_vid_paths, get_motion_view_paths, get_all_view_paths, get_frame_paths, \
                                get_character_motion_paths

sys.path.extend(['utils'])
from openpose_utils import filter_poses, render_2d_keypoints

OP_ROOT_JOINT = 8
# lower-body indicies into OpenPose 2d joints
OP_LOWER_JOINTS = [8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]
# names of each of the lower-body joints (after extracting from full OpenPose joints)
OP_LOWER_JOINTS_MAP = { "MidHip"    : 0,
                        "RHip"      : 1,
                        "RKnee"     : 2,
                        "RAnkle"    : 3,
                        "LHip"      : 4,
                        "LKnee"     : 5,
                        "LAnkle"    : 6,
                        "LBigToe"   : 7,
                        "LSmallToe" : 8,
                        "LHeel"     : 9,
                        "RBigToe"   : 10,
                        "RSmallToe" : 11,
                        "RHeel"     : 12  }
OP_LOWER_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 6, 7, 6, 3, 10, 3]

# various subsets of openpose joints
OP_JOINT_SUBSETS = { "lower" : [8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24], # full lower body (including hips)
                     "lower_knees" : [10, 11, 13, 14, 19, 20, 21, 22, 23, 24], # lower body up though the knees (so it leaves out hips)
                     "lower_ankles" : [11, 14, 19, 20, 21, 22, 23, 24], # lower body up though the ankles (so it leaves out knees/hips)
                     "lower_feet" : [11, 14, 19, 20, 21, 22, 23, 24], # lower body up though the feet (so it leaves out knees/hips/ankles)
                     "upper" : [0, 1, 2, 3, 4, 5, 6, 7], # full upper body (without hips or root)
                     "upper_hips" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12], # full upper body + hips and root
                     "upper_knees" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13], # add knees
                     "upper_ankles" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], # add ankles
                     "full" : range(25) # full body
}

def process_openpose_data(joint2d_seq_list, confidence_thresh=0.2, vid_paths=None):
    print('Preprocessing OpenPose data...')
    for i, op_seq in enumerate(joint2d_seq_list):
        cur_joint2d_seq = joint2d_seq_list[i][:,:,:2] # don't filter confidence
        cur_confidence_seq = joint2d_seq_list[i][:,:,2]

        # fix missing (very low confidence) frames caused by occlusion
        # to do this linearly interpolate from surrounding high-confidence frames
        num_frames = cur_joint2d_seq.shape[0]
        for joint_idx in range(cur_joint2d_seq.shape[1]):
            t = 0
            while t < num_frames:
                if cur_confidence_seq[t, joint_idx] < confidence_thresh:
                    # find the next valid frame
                    next_valid_frame = t+1
                    while next_valid_frame < num_frames and cur_confidence_seq[next_valid_frame, joint_idx] < confidence_thresh:
                        next_valid_frame += 1
                    # then update sequence of invalid frames accordingly
                    init_valid_frame = t - 1
                    if t == 0 and next_valid_frame == num_frames:
                        # worst case scenario, every frame is bad
                        # can't do anything
                        pass
                    elif t == 0:
                        # bad frames to start the sequence
                        # set all t < next_valid_frame to next_valid_frame
                        cur_joint2d_seq[:next_valid_frame, joint_idx, :] = cur_joint2d_seq[next_valid_frame, joint_idx, :].reshape((1,2))
                    elif next_valid_frame == num_frames:
                        # bad until the end
                        # set all t > init_valid_frame to init_valid_frame
                        cur_joint2d_seq[init_valid_frame:, joint_idx, :] = cur_joint2d_seq[init_valid_frame,joint_idx,:].reshape((1,2))
                    else:
                        # have a section of >= 1 frame that's bad
                        # linearly interpolate in this section
                        step_size = 1.0 / (next_valid_frame - init_valid_frame)
                        cur_step = step_size
                        cur_t = t
                        while cur_t < next_valid_frame:
                            cur_joint2d_seq[cur_t, joint_idx, :] = (1.0 - cur_step)*cur_joint2d_seq[init_valid_frame, joint_idx, :] + \
                                                                    cur_step*cur_joint2d_seq[next_valid_frame, joint_idx, :]
                            cur_t += 1
                            cur_step += step_size 

                    t = next_valid_frame
                else:
                    t += 1                    

        # then filter to smooth it out a bit
        # can additionally smooth if desired
        # cur_joint2d_seq = filter_poses(cur_joint2d_seq, fcmin=0.05, beta=0.005, freq=30)

        # update
        joint2d_seq_list[i][:,:,:2] = cur_joint2d_seq

        # NOTE: viz
        if vid_paths:
            render_2d_keypoints(joint2d_seq_list[i], video_path=vid_paths[i], dimensions=(1280,720), conf_thresh=-1)

        if i % 10 == 0:
            print('Finished ' + str(i+1) + ' of ' + str(len(joint2d_seq_list)) + '...')
    print('Done!')

    return joint2d_seq_list

def normalize_openpose_data(normalization_info, op_data):
    print('Normalizing OpenPose data...')
    for i in range(len(op_data)):
        # only normalize pixels, not confidence
        op_data[i][:,:,:2] /= normalization_info
        if i % 10 == 0:
            print('Finished ' + str(i) + ' of ' + str(len(op_data)))
    print('Done!')
    return op_data

class OpenPoseDataset(Dataset):
    ''' Dataset of sequences of OpenPose joint positions. '''

    def __init__(self, data_root, split='train', window_size=5, contact_size=3, train_frac=0.8, dimensions=(1280,720), load_img=False, noise_dev=0.005, overlap_test=False, use_confidence=True, joint_set='lower'):
        '''
        Creates an OpenPoseDataset object.
        - data_root : path to the root directory of the dataset
        - split : 'train', 'val', or 'test'
        - window_size : number of frames around a target frame to use (must be odd)
        - contact_size : number of middle frames to return contact labels for
        - train_frac : the fraction of motion sequences to be used at training time
        - dimensions : dimension of the images on which OpenPose was detected
        - load_img : if True, getitem will also return the image frames corresponding to that window
        - noise_dev : the standard deviation of gaussian noise to add to 2d joint position
        - overlap_test : if set to true, windows in the 'test'/'val' splits will overlap
                         so that every frame in the sequence is a target frame for some window
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
        self.noise_dev = noise_dev
        self.overlap_windows = overlap_test
        self.use_confidence = use_confidence
        self.joint_set = joint_set

        if not self.use_confidence:
            print('Not using confidence as input!')
        if not self.joint_set in OP_JOINT_SUBSETS.keys():
            print('Do not support joint subset ' + self.joint_set)
            return

        print('Using joint subset ' + self.joint_set)
        print('Using windows of size ' + str(self.window_size) + '...')
        print('Using contact labels of middle ' + str(self.contact_size) + ' frames...')

        # load in paths to get data
        character_dirs = get_all_character_paths(data_root)
        num_motions_per_character = len(get_character_motion_paths(character_dirs[0]))
        motion_dirs = get_all_motion_paths(character_dirs)

        # peek to see how many views and frames
        view_paths = get_motion_view_paths(motion_dirs[0])
        self.num_views = len(view_paths)
        self.num_frames = len(get_frame_paths(view_paths[0]))

        if self.overlap_windows:
            # every frame gets window except on the very edges without enough frames on each side
            self.test_windows_per_seq = self.num_frames - 2*(self.window_size // 2)
            print('Using overlapping test frames with %d windows per video!' % (self.test_windows_per_seq))
        else:
            self.test_windows_per_seq = self.num_frames // self.window_size

        print('Found ' + str(len(character_dirs)) + ' total characters...')
        print('Found ' + str(num_motions_per_character) + ' unique motions for each character...')
        print('Found ' + str(len(motion_dirs)) + ' total motion sequences...')
        print('Found ' + str(self.num_views) + ' views per sequences...')
        print('Found ' + str(self.num_frames) + ' frames per sequences...')
        total_frames = len(motion_dirs) * self.num_views * self.num_frames
        print('Dataset total: ' + str(total_frames) + ' unique frames...')

        # load in OpenPose and foot contact data
        op_dirs = get_all_openpose_paths(motion_dirs)
        self.op_data = get_all_openpose_data(op_dirs)

        contact_dirs = get_all_foot_contact_paths(motion_dirs)
        self.contact_data = get_all_foot_contact_data(contact_dirs)

        # load in img path for lazy loading if desired
        self.view_dirs = get_all_view_paths(motion_dirs)

        if len(self.op_data) != len(self.contact_data):
            print('[OpenPoseDataset] Data lengths are not compatible! Should all be the same but...')
            print('Num OpenPose dirs: ' + str(len(self.op_data)))
            print('Num contact files: ' + str(len(self.contact_data)))
        if load_img:
            if len(self.op_data) != len(self.view_dirs):
                print('[OpenPoseDataset] Data lengths are not compatible! Should all be the same but...')
                print('Num OpenPose dirs: ' + str(len(self.op_data)))
                print('Num contact files: ' + str(len(self.view_dirs)))

        self.data_len = len(self.op_data)

        # base normalization on entire dataset
        self.normalization_info = self.get_normalization_info(self.op_data, dimensions)

        # Here we split the data so that no motion is seen in both training and testing
        #   This DOES NOT include motion across characters, so i.e. the 277_samba motion from all viewpoints for
        #   Liam may appear in the training set, while for Stefani it is in the test set.
        np.random.seed(0) # make it the same every time
        split_inds = [[], [], []] # train, test, val
        videos_per_character = num_motions_per_character * self.num_views
        for character_idx in range(len(character_dirs)):
            # choose random set of motions for this character for splits
            motion_inds = np.arange(num_motions_per_character)
            np.random.shuffle(motion_inds) # shuffle them b/c they're sorted by name initially
            train_size = int(train_frac * num_motions_per_character)
            train_motion_inds = motion_inds[:train_size]
            test_size = (num_motions_per_character - train_size) // 2
            test_motion_inds = motion_inds[train_size:train_size+test_size]
            val_size = num_motions_per_character - train_size - test_size
            val_motion_inds = motion_inds[train_size+test_size:]
            # now map these motion indices to the corresponding global indices
            global_start_idx = character_idx * videos_per_character
            for i, split_motion_inds in enumerate([train_motion_inds, test_motion_inds, val_motion_inds]):
                for split_idx in split_motion_inds:
                    local_start_idx = split_idx*self.num_views
                    split_inds[i] += range(global_start_idx + local_start_idx, global_start_idx + local_start_idx + self.num_views)
        train_inds = split_inds[0]
        test_inds = split_inds[1]
        val_inds = split_inds[2]

        if split == 'train':
            # for training, we'll randomly sample windows from sequences
            # so data length is the same as number of sequences
            data_inds = train_inds
            self.data_len = len(data_inds)
            print('Loaded ' + split + ' split with ' + str(self.data_len) + ' FULL motion sequences to be split into windows...')
        elif split == 'val':
            # for val and test, want same windows every time, so pre-compute how
            # many windows we can get out of a sequence and set this as the data length
            # NOTE: Assume that every motion has same number of views and each view
            #       has same number of frames
            data_inds = val_inds
            self.data_len = len(data_inds) * self.test_windows_per_seq
            print('Loaded ' + split + ' split with ' + str(self.data_len) + ' motion windows from ' + str(len(data_inds)) + ' videos...')
        else:
            data_inds = test_inds
            self.data_len = len(data_inds) * self.test_windows_per_seq
            print('Loaded ' + split + ' split with ' + str(self.data_len) + ' motion windows from ' + str(len(data_inds)) + ' videos...')

        self.op_data = [self.op_data[idx] for idx in data_inds]
        self.contact_data = [self.contact_data[idx] for idx in data_inds]
        self.view_dirs = [self.view_dirs[idx] for idx in data_inds]

        # OpenPose data is noisy and has missing frames
        # need to do some preprocessing
        self.op_data = process_openpose_data(self.op_data, confidence_thresh=0.2)
        # also normalize
        self.op_data = normalize_openpose_data(self.normalization_info, self.op_data)

        return

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
        if self.split == 'train':
            seq_idx = idx
            # for training, pull random window from this sequence
            # NOTE: assuming odd window size
            min_idx = self.window_size // 2
            max_idx = self.num_frames - min_idx
            tgt_idx = np.random.randint(min_idx, max_idx)
            frame_window = [tgt_idx - min_idx, tgt_idx + min_idx + 1]
            # print('Target motion: ' + str(idx))
        else:
            # for val/test, idx defines both motion sequence AND window
            # figure out which motion/view based on index
            seq_idx = idx // self.test_windows_per_seq
            window_idx = idx % self.test_windows_per_seq

            if self.overlap_windows:
                # every frame except first window_size//2 have a window
                start_frame = window_idx
            else:
                # define window based on window index
                # we split the sequence evenly by windows
                #   ( i.e. window 0 is frame [0, window_size) )
                start_frame = window_idx * self.window_size
            
            end_frame = start_frame + self.window_size
            frame_window = [start_frame, end_frame]
            tgt_idx = start_frame + (self.window_size // 2)
            # print('Target motion: ' + str(seq_idx))

        # print('Target frame: ' + str(tgt_idx))
        # print('Window = [' + str(frame_window[0]) + ', ' + str(frame_window[1]) + ')')

        # grab data window
        # print(self.op_data[seq_idx][frame_window[0]:frame_window[1],10,:])
        cur_op_data = self.op_data[seq_idx][frame_window[0]:frame_window[1]].copy()
        cur_contact_data = self.contact_data[seq_idx][frame_window[0]:frame_window[1]].copy()
        cur_view_path = self.view_dirs[seq_idx]
        if self.load_img:
            # only actually load if desired
            cur_frame_paths = get_frame_paths(cur_view_path)
            cur_frames = np.asarray(io.imread(cur_frame_paths[tgt_idx])[:,:,:3]) # remove alpha

        # normalize w.r.t root (MidHip)
        rel_tgt_idx = self.window_size // 2
        root_idx = OP_ROOT_JOINT
        tgt_root = cur_op_data[rel_tgt_idx, root_idx, :2].copy() # only (x,y) 
        tgt_root = tgt_root.reshape((1, 1, 2))
        cur_op_data[:,:,:2] -= tgt_root

        # add back in only for target frame to give some sense of global position
        cur_op_data[rel_tgt_idx, root_idx, :2] = tgt_root

        # for OP, only want lower-body joints
        cur_op_data = cur_op_data[:,OP_JOINT_SUBSETS[self.joint_set],:] # F x J x 3

        # perturb joint positions with gaussian noise
        if self.split == 'train':
            add_noise = np.random.normal(loc=0.0, scale=self.noise_dev, size=cur_op_data[:,:,:2].shape)
            add_noise = np.concatenate([add_noise, np.zeros((cur_op_data.shape[0], cur_op_data.shape[1], 1))], axis=2)
            cur_op_data += add_noise

        # for contacts, want middle contact_size frames
        contact_offset = (self.window_size - self.contact_size) // 2
        cur_contact_data = cur_contact_data[contact_offset:(self.window_size - contact_offset)]

        # take out confidence if need be
        if not self.use_confidence:
            cur_op_data = cur_op_data[:,:,:2]

        # to pytorch
        cur_op_data = torch.from_numpy(cur_op_data.astype(np.float32))
        cur_contact_data = torch.from_numpy(cur_contact_data.astype(np.float32))
        cur_item = {'joint2d' : cur_op_data, 'contacts' : cur_contact_data, 'name' : '/'.join(cur_view_path.split('/')[-3:])}
        if self.load_img:
            cur_item['frames'] = cur_frames
        # print(cur_item)
        return cur_item

    def get_num_test_windows_per_seq(self):
        return self.test_windows_per_seq

    def get_normalization_info(self, op_data, dimensions):
        '''
        Given the OpenPose data, finds the value to normalize pixel coordinates.
        '''
        # use median distance from midhip to left big toe
        dists = []
        for op_seq in op_data:
            lower_joints = op_seq[:,OP_LOWER_JOINTS,:2]
            midhip = lower_joints[:,OP_LOWER_JOINTS_MAP['MidHip']]
            left_toe = lower_joints[:,OP_LOWER_JOINTS_MAP['LBigToe']]
            cur_dists = np.linalg.norm(midhip - left_toe, axis=1)
            dists += cur_dists.tolist()
        med_dist = np.median(np.array(dists))
        print('Median hip->toe dist: ' + str(med_dist) + ' pixels')
        return med_dist

    def get_joint_scaling(self):
        ''' Returns the scaling that is applied to all joint2d data '''
        return self.normalization_info
