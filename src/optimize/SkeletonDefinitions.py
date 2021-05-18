import numpy as np

# important projection weight is 0 on the root
BODY_25_PROJ_WEIGHTS = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.0, 1.0, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
BODY_25_SMOOTH_WEIGHTS = np.array([2.5, 2.5, 2.5, 1.5, 1.0, 2.5, 1.5, 1.0, 1.0, 2.5, 1.5, 1.0, 2.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

mapping_full_open_skel_to_body_25 = {
        0: 8,  # hips -> MidHip
        1: 1,  # Neck 
        2: 0,  # Nose 
        3: 16, # LEye
        4: 18, # LEar
        5: 15, # REye
        6: 17, # REar
        7: 5,  # LShoulder -> LShoulder 
        8: 6,  # LElbow -> LElbow
        9: 7,  # LWrist -> LWrist
        10: 2,  # RShoulder -> RShoulder
        11: 3,  # RElbow -> RElbow 
        12: 4,  # RWrist -> RWrist
        13: 12,  # LHip -> LHip
        14: 13,  # LKnee -> LKnee
        15: 14,  # LAnkle -> LAnkle
        16: 21,  # LHeel
        17: 19,  # LBigToe -> LBigToe
        18: 20,  # LSmallToe
        19: 9,  # RHip -> RHip
        20: 10,  # RKnee -> RKnee
        21: 11,  # RAnkle -> RAnkle
        22: 24,  # RHeel
        23: 22,  # RBigToe -> RBigToe
        24: 23  # RSmallToe
    }

mapping_body25_to_full_open_skel= {
        0: 2,  # Nose
        1: 1, # Neck
        2: 10, # RShoulder
        3: 11, # RElbow
        4: 12, # RWrist
        5: 7, # LShoulder
        6: 8, # LElbow
        7: 9, # LWrist
        8: 0, # MidHip
        9: 19, # RHip
        10: 20, # RKnee
        11: 21, # RAnkle
        12: 13, # LHip
        13: 14, # LKnee
        14: 15, # LAnkle
        15: 5, # REye
        16: 3, # LEye
        17: 6, # REar
        18: 4, # LEar
        19: 17, # LBigToe
        20: 18, # LSmallToe
        21: 16, # LHeel
        22: 23, # RBigToe
        23: 24, # RSmallToe
        24: 22 # RHeel
    }

BODY_25_ROOT_IDX = 8

############# Combined model with all 3 spine joints ##################

# important projection weight is 0 on the root
# also no projection on the spine
COMBINED_PROJ_WEIGHTS =   np.array([0.1, 0.1, 0.3, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0])
COMBINED_DATA_WEIGHTS =   np.array([2.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 2.5, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
COMBINED_SMOOTH_WEIGHTS = np.array([2.5, 2.5, 2.5, 1.5, 1.0, 2.5, 1.5, 1.0, 1.0, 2.5, 1.5, 1.0, 2.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5])
COMBINED_FEET_IDX = np.array([4, 5, 6, 10, 11, 12])
COMBINED_SKEL_SPINE_JOINTS = [13, 14, 15]
COMBINED_SKEL_NECK_JOINT = 16

mapping_combined_skel_to_body_25 = {
        0: 8,  # MidHip 
        1: 12,  # LHip
        2: 13,  # LKnee -> LKnee
        3: 14,  # LAnkle -> LAnkle
        4: 21,  # LHeel
        5: 19,  # LBigToe -> LBigToe
        6: 20,  # LSmallToe
        7: 9,  # RHip -> RHip
        8: 10,  # RKnee -> RKnee
        9: 11,  # RAnkle -> RAnkle
        10: 24,  # RHeel
        11: 22,  # RBigToe -> RBigToe
        12: 23,  # RSmallToe
        13: 25, # Spine
        14: 26, # Spine1
        15: 27, # Spine2
        16: 1,  # Neck 
        17: 0,  # Nose 
        18: 16, # LEye
        19: 18, # LEar
        20: 15, # REye
        21: 17, # REar
        22: 5,  # LShoulder -> LShoulder 
        23: 6,  # LElbow -> LElbow
        24: 7,  # LWrist -> LWrist
        25: 2,  # RShoulder -> RShoulder
        26: 3,  # RElbow -> RElbow 
        27: 4  # RWrist -> RWrist
    }

mapping_body_25_to_combined_skel= {
        0: 17,  # Nose
        1: 16, # Neck
        2: 25, # RShoulder
        3: 26, # RElbow
        4: 27, # RWrist
        5: 22, # LShoulder
        6: 23, # LElbow
        7: 24, # LWrist
        8: 0, # MidHip
        9: 7, # RHip
        10: 8, # RKnee
        11: 9, # RAnkle
        12: 1, # LHip
        13: 2, # LKnee
        14: 3, # LAnkle
        15: 20, # REye
        16: 18, # LEye
        17: 21, # REar
        18: 19, # LEar
        19: 5, # LBigToe
        20: 6, # LSmallToe
        21: 4, # LHeel
        22: 11, # RBigToe
        23: 12, # RSmallToe
        24: 10, # RHeel
        25: 13, # Spine
        26: 14, # Spine1
        27: 15 # Spine2
    }

COMBINED_ROOT_IDX = 8