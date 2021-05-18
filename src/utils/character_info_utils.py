
#
# To add a new character to this file:
#   -   Create a new CharacterInfo object for the character
#   -   Add it to the character_map
#   -   Initialize all info fields
#

class CharacterInfo():
    '''
    Object to hold all needed information about a Mixamo character.
    See get_* functions for description of each property. 
    '''
    def __init__(self):
        self.to_smpl_mapping = None
        self.ik_blacklist = None
        self.to_combined_mapping = None
        self.eye_indices = None
        self.upper_body_joints = None
        self.left_leg_chain = None 
        self.right_leg_chain = None
        self.heel_indices = None
        self.mass = None
        self.segment_to_joints_map = None 
        self.seg_to_mass_perc_map = None

# all the available characters
# their properties are filled in below
combined_character          = CharacterInfo()
ybot_character              = CharacterInfo()
skeletonzombie_character    = CharacterInfo()
ty_character                = CharacterInfo()

# dictionary to look up the object of each character
character_map = {
    'combined'       : combined_character,
    'ybot'           : ybot_character,
    'skeletonzombie' : skeletonzombie_character,
    'ty'             : ty_character
}

heeled_characters = ['combined']

################################################################################################################

#
# functions to retrieve character info
#

def get_character_to_smpl_mapping(character):
    ''' Returns a mapping from joint indices of the character skeleton to the SMPL skeleton '''
    char_info = character_map[character]
    return char_info.to_smpl_mapping

def get_character_ik_blacklist(character):
    ''' list of joints that shouldn't be included in IK from combined to mixamo character '''
    char_info = character_map[character]
    return char_info.ik_blacklist

def get_character_to_combined_mapping(character):
    ''' maps joints from mixamo character to combined skeleton '''
    char_info = character_map[character]
    return char_info.to_combined_mapping

def get_character_eye_inds(character):
    ''' joint indicies of the eyes. If no eyes, some joint near eye height (i.e. head) '''
    char_info = character_map[character]
    return char_info.eye_indices

def get_character_upper_body(character):
    ''' joints in the upper body of the skeleton, including root joint '''
    char_info = character_map[character]
    return char_info.upper_body_joints

def get_character_leg_chain(character, side='left'):
    ''' The leg chain starts at the hip and goes to the ToeBase. The ankle must be at -2. '''
    if side not in ['left', 'right']:
        print('Only sides left and right are supported!')
        return
    char_info = character_map[character]
    if side == 'left':
        return char_info.left_leg_chain
    else:
        return char_info.right_leg_chain

def get_character_toe_inds(character):
    ''' In order [left_toe, right_toe] '''
    left_leg_chain = get_character_leg_chain(character, 'left')
    right_leg_chain = get_character_leg_chain(character, 'right')
    toe_inds = [left_leg_chain[-1], right_leg_chain[-1]]
    return toe_inds

def get_character_ankle_inds(character):
    ''' In order [left_ankle, right_ankle] '''
    left_leg_chain = get_character_leg_chain(character, 'left')
    right_leg_chain = get_character_leg_chain(character, 'right')
    ankle_inds = [left_leg_chain[-2], right_leg_chain[-2]]
    return ankle_inds

def get_character_foot_inds(character):
    ''' Both ankle and toe in the order [left_ankle, left_toe, right_ankle, right_toe] '''
    ankle_inds = get_character_ankle_inds(character)
    toe_inds = get_character_toe_inds(character)
    foot_inds = [ankle_inds[0], toe_inds[0], ankle_inds[1], toe_inds[1]]
    return foot_inds

def get_character_heel_inds(character):
    ''' Returns indices of heels if they exist on this character [left_heel, right_heel] '''
    if character in heeled_characters:
        char_info = character_map[character]
        heel_inds = char_info.heel_indices
        return heel_inds
    else:
        return None

def get_character_hip_inds(character):
    ''' In order [left_hip, right_hip] '''
    left_leg_chain = get_character_leg_chain(character, 'left')
    right_leg_chain = get_character_leg_chain(character, 'right')
    hip_inds = [left_leg_chain[0], right_leg_chain[0]]
    return hip_inds

def get_character_mass(character):
    char_info = character_map[character]
    return char_info.mass

def get_character_seg_to_joint_map(character):
    ''' maps the body part segment (for physical properties) to the joints that define it'''
    char_info = character_map[character]
    return char_info.segment_to_joints_map

def get_character_seg_to_mass_perc_map(character):
    ''' maps the body part segment (for physical properties) to the  percentage of total mass'''
    char_info = character_map[character]
    return char_info.seg_to_mass_perc_map

################################################################################################################

#
# Human physical properties
# see http://holmeslab.ca/csb-scb/Archives/Zatsiorsky-deLeva.pdf
#
map_segment_to_mass_perc_male = {
    'head' : 6.94,
    'upper_trunk' : 15.96,
    'mid_trunk' : 16.33,
    'lower_trunk' : 11.17,
    'left_upper_arm' : 2.71,
    'left_forearm' : 1.62,
    'left_hand' : 0.61,
    'left_thigh' : 14.16,
    'left_shank' : 4.33,
    'left_foot' : 1.37,
    'right_upper_arm' : 2.71,
    'right_forearm' : 1.62,
    'right_hand' : 0.61,
    'right_thigh' : 14.16,
    'right_shank' : 4.33,
    'right_foot' : 1.37
}

map_segment_to_mass_perc_female = {
    'head' : 6.68,
    'upper_trunk' : 15.45,
    'mid_trunk' : 14.65,
    'lower_trunk' : 12.47,
    'left_upper_arm' : 2.55,
    'left_forearm' : 1.38,
    'left_hand' : 0.56,
    'left_thigh' : 14.78,
    'left_shank' : 4.81,
    'left_foot' : 1.29,
    'right_upper_arm' : 2.55,
    'right_forearm' : 1.38,
    'right_hand' : 0.56,
    'right_thigh' : 14.78,
    'right_shank' : 4.81,
    'right_foot' : 1.29
}

MALE_MASS = 73.0 #kg
FEMALE_MASS = 61.99

################################################################################################################

#
# combined skeleton (OpenPose + SMPL spine)
#

# this is not really a character so it is not accessed throught the above functions
# as "foot" is defined slightly differently (extra toe joint and no ankle)

# [left heel, big toe, small toe, right heel, big toe, small toe] NOTE: does not include the ankle
combined_foot_inds = [4,5,6,10,11,12]
combined_toe_inds = [5, 6, 11, 12]
combined_ankle_inds = [3,9]
combined_eye_inds = [18,20]

mapping_smpl_to_combined_skel = {
        0: 0,  # hips -> Hips 
        1: 1,  # leftUpLeg -> LHip
        2: 7,  # rightUpLeg -> RHip
        3: 13,  # spine -> Spine
        4: 2,  # leftLeg -> LKnee
        5: 8,  # rightLeg -> RKnee
        6: 14,  # spine1 -> Spine1
        7: 3,  # leftFoot -> LAnkle
        8: 9,  # rightFoot -> RAnkle
        9: 15,  # spine2 -> Spine2
        10: 6,  # leftToeBase -> LSmallToe # see viz, small toe is actually closer to smple toe base
        11: 12,  # rightToeBase -> RSmallToe
        12: 16,  # neck -> Neck
        13: -1, # leftShoulder
        14: -1, # rightShoulder
        15: -1, #17, # head -> Nose # these joints don't really line up positionally, but for angle it shouldn't matter
        16: 22,  # leftArm -> LShoulder
        17: 25,  # rightArm -> RShoulder
        18: 23, # leftForeArm -> LElbow
        19: 26, # rightForeArm -> RElbow
        20: 24, # lefthand -> LWrist
        21: 27 # rightHand -> RWrist
    }

mapping_combined_skel_to_smpl = {
        0: 0,  # hips -> Hips 
        1: 1,  # leftUpLeg -> LHip
        2: 4,  # leftLeg -> LKnee
        3: 7,  # leftFoot -> LAnkle
        4: -1,  # LHeel
        5: -1,  # # LBigToe
        6: 10,  # leftToeBase -> LSmallToe # see viz, small toe is actually closer to smpl toe base
        7: 2,  # rightUpLeg -> RHip
        8: 5,  # rightLeg -> RKnee
        9: 8,  # rightFoot -> RAnkle
        10: -1,  # RHeel
        11: -1,  # RBigToe
        12: 11,  # rightToeBase -> RSmallToe
        13: 3,   # spine -> Spine
        14: 6,   # spine1 -> Spine1
        15: 9,   # spine2 -> Spine2
        16: 12,  # neck -> Neck
        17: 15,  # head -> Nose # these joints don't really line up positionally, but for angle it shouldn't matter
        18: -1, # LEye
        19: -1, # LEar
        20: -1, # REye
        21: -1, # REar
        22: 16, # leftArm -> LShoulder
        23: 18, # leftForeArm -> LElbow
        24: 20, # lefthand -> LWrist
        25: 17, # rightArm -> RShoulder
        26: 19, # rightForeArm -> RElbow
        27: 21 # rightHand -> RWrist
    }

combined_character.right_leg_chain = [7, 8, 9, 11]
combined_character.left_leg_chain = [1, 2, 3, 5]
combined_character.upper_body_joints = [0] + list(range(13, 28))

combined_character.mass = MALE_MASS
combined_character.seg_to_mass_perc_map = map_segment_to_mass_perc_male

combined_character.heel_indices = [4, 10] # left, right

# maps segment names to the joints where this mass is located (estimated by avg location of all given joints)
combined_character.segment_to_joints_map = {
    'head' : [17],
    'upper_trunk' : [15, 16],
    'mid_trunk' : [14, 15],
    'lower_trunk' : [13, 14],
    'left_upper_arm' : [22, 23],
    'left_forearm' : [23, 24],
    'left_hand' : [24],
    'left_thigh' : [1, 2],
    'left_shank' : [2, 3],
    'left_foot' : [3, 4, 5, 6],
    'right_upper_arm' : [25, 26],
    'right_forearm' : [26, 27],
    'right_hand' : [27],
    'right_thigh' : [7, 8],
    'right_shank' : [8, 9],
    'right_foot' : [9, 10, 11, 12]
}

################################################################################################################

#
# ybot physics
#
ybot_character.right_leg_chain = [57, 58, 59, 60]
ybot_character.left_leg_chain = [62, 63, 64, 65] #[62, 63, 65] #
ybot_character.upper_body_joints = list(range(0, 57))# + [57, 62]

ybot_character.mass = MALE_MASS
ybot_character.seg_to_mass_perc_map = map_segment_to_mass_perc_male
# maps segment names to the joints where this mass is located (estimated by avg location of all given joints)
ybot_character.segment_to_joints_map = {
    'head' : [5],
    'upper_trunk' : [3],
    'mid_trunk' : [2],
    'lower_trunk' : [1],
    'left_upper_arm' : [10, 11],
    'left_forearm' : [11, 12],
    'left_hand' : range(12, 33),
    'left_thigh' : [62, 63],
    'left_shank' : [63, 64],
    'left_foot' : [64, 65, 66], 
    'right_upper_arm' : [34, 35],
    'right_forearm' : [35, 36],
    'right_hand' : range(36, 57),
    'right_thigh' : [57, 58],
    'right_shank' : [58, 59],
    'right_foot' : [59, 60, 61]
}

#
# ybot mappings
#
ybot_character.ik_blacklist = [10, 11, 12, 34, 35, 36] # arms + shoulder since they'll be at different height
ybot_character.eye_indices = [7,8] # left, right
ybot_character.to_combined_mapping = {
    0: 0,  # hips -> MidHip
    1: 13,  # spine -> spine
    2: 14,  # spine1 -> spine1
    3: 15,  # spine2 -> spine2
    4: 16,  # neck -> neck
    5: -1,  # head -> nose 
    6: -1, 
    7: 18,  # LeftEye -> LEye
    8: 20,  # RightEye -> REye
    9:  -1,  # leftShoulder -> Neck
    10: 22,  # leftArm -> LShoulder
    11: 23,  # leftForeArm -> LElbow
    12: 24,  # leftHand -> LWrist
    13: -1,
    14: -1,
    15: -1,
    16: -1,
    17: -1,
    18: -1,
    19: -1,
    20: -1,
    21: -1,
    22: -1,
    23: -1,
    24: -1,
    25: -1,
    26: -1,
    27: -1,
    28: -1,
    29: -1,
    30: -1,
    31: -1,
    32: -1,
    33: -1,  # rightShoulder -> neck
    34: 25,  # rightArm -> RShoulder
    35: 26,  # rightForeArm -> RElbow
    36: 27,  # rightHand -> RWrist
    37: -1,
    38: -1,
    39: -1,
    40: -1,
    41: -1,
    42: -1,
    43: -1,
    44: -1,
    45: -1,
    46: -1,
    47: -1,
    48: -1,
    49: -1,
    50: -1,
    51: -1,
    52: -1,
    53: -1,
    54: -1,
    55: -1,
    56: -1,
    57: 7,  # rightUpLeg -> RHip
    58: 8,  # rightLeg -> RKnee
    59: 9,  # rightFoot -> RAnkle
    60: 11,  # rightToeBase 
    61: -1,  # rightToeEnd -> RBigToe 
    62: 1,  # leftUpLeg -> LHip
    63: 2,  # leftLeg -> LKnee
    64: 3,  # leftFoot -> LAnkle
    65: 5,  # leftToeBase
    66: -1   # leftToeEnd -> LBigToe
}

ybot_character.to_smpl_mapping = {
    0: 0,  # hips -> MidHip
    1: 3,  # spine -> spine
    2: 6,  # spine1 -> spine1
    3: 9,  # spine2 -> spine2
    4: 12,  # neck -> neck
    5: -1,  # head 
    6: -1, 
    7: -1,  # LeftEye -> LEye
    8: -1,  # RightEye -> REye
    9:  13,  # leftShoulder -> leftShoulder
    10: 16,  # leftArm -> leftArm
    11: 18,  # leftForeArm -> leftForeArm
    12: 20,  # leftHand -> leftHand
    13: -1, # hand details (could copy over from SMPL)
    14: -1,
    15: -1,
    16: -1,
    17: -1,
    18: -1,
    19: -1,
    20: -1,
    21: -1,
    22: -1,
    23: -1,
    24: -1,
    25: -1,
    26: -1,
    27: -1,
    28: -1,
    29: -1,
    30: -1,
    31: -1,
    32: -1,
    33: 14,  # rightShoulder -> rightShoulder
    34: 17,  # rightArm -> rightArm
    35: 19,  # rightForeArm -> rightForeArm
    36: 21,  # rightHand -> rightHand
    37: -1, # hand details
    38: -1,
    39: -1,
    40: -1,
    41: -1,
    42: -1,
    43: -1,
    44: -1,
    45: -1,
    46: -1,
    47: -1,
    48: -1,
    49: -1,
    50: -1,
    51: -1,
    52: -1,
    53: -1,
    54: -1,
    55: -1,
    56: -1,
    57: 2,  # rightUpLeg -> rightUpLeg
    58: 5,  # rightLeg -> rightLeg
    59: 8,  # rightFoot -> rightFoot
    60: 11,  # rightToeBase -> rightToeBase
    61: -1,  # rightToeEnd
    62: 1,  # leftUpLeg -> leftUpLeg
    63: 4,  # leftLeg -> leftLeg
    64: 7,  # leftFoot -> leftFoot
    65: 10,  # leftToeBase
    66: -1   # leftToeEnd
}

################################################################################################################

#
# skeletonzombie physics
#
skeletonzombie_character.right_leg_chain = [60, 61, 62, 63]
skeletonzombie_character.left_leg_chain = [55, 56, 57, 58] 
skeletonzombie_character.upper_body_joints = list(range(0, 55))

skeletonzombie_character.mass = MALE_MASS * 2.0
skeletonzombie_character.seg_to_mass_perc_map = {
    'head' : 3.0,
    'upper_trunk' : 14.0,
    'mid_trunk' : 12.0,
    'lower_trunk' : 9.0,
    'left_upper_arm' : 3.0,
    'left_forearm' : 9.0,
    'left_hand' : 6.0,
    'left_thigh' : 9.0,
    'left_shank' : 3.0,
    'left_foot' : 1.0,
    'right_upper_arm' : 3.0,
    'right_forearm' : 9.0,
    'right_hand' : 6.0,
    'right_thigh' : 9.0,
    'right_shank' : 3.0,
    'right_foot' : 1.0
}
# maps segment names to the joints where this mass is located (estimated by avg location of all given joints)
skeletonzombie_character.segment_to_joints_map = {
    'head' : [29],
    'upper_trunk' : [3],
    'mid_trunk' : [2],
    'lower_trunk' : [1],
    'left_upper_arm' : [5, 6],
    'left_forearm' : [6, 7],
    'left_hand' : range(7, 28),
    'left_thigh' : [55, 56],
    'left_shank' : [56, 57],
    'left_foot' : [57, 58, 59], 
    'right_upper_arm' : [32, 33],
    'right_forearm' : [33, 34],
    'right_hand' : range(34, 55),
    'right_thigh' : [60, 61],
    'right_shank' : [61, 62],
    'right_foot' : [62, 63, 64]
}

#
# skeletonzombie mappings
#
skeletonzombie_character.ik_blacklist = [5, 6, 7, 32, 33, 34] #2, 3,#, 60, 65]
# skeletonzombie_foot_indices = [57,58,62,63] # left_ankle, left_toe, right_ankle, right_toe
skeletonzombie_character.eye_indices = [29, 29] # no eye, just use the head joint
skeletonzombie_character.to_combined_mapping = {
    0: 0,  # hips -> MidHip
    1: 13,  # spine -> spine
    2: 14,  # spine1 -> spine1
    3: 15,  # spine2 -> spine2
    4:  -1,  # leftShoulder -> Neck
    5: 22,  # leftArm -> LShoulder
    6: 23,  # leftForeArm -> LElbow
    7: 24,  # leftHand -> LWrist
    8: -1,
    9: -1,
    10: -1,
    11: -1,
    12: -1,
    13: -1,
    14: -1,
    15: -1,
    16: -1,
    17: -1,
    18: -1,
    19: -1,
    20: -1,
    21: -1,
    22: -1,
    23: -1,
    24: -1,
    25: -1,
    26: -1,
    27: -1,
    28: 16,  # neck -> neck
    29: -1,  # head -> nose 
    30: -1,  # head top end
    31: -1,  # rightShoulder -> neck
    32: 25,  # rightArm -> RShoulder
    33: 26,  # rightForeArm -> RElbow
    34: 27,  # rightHand -> RWrist
    35: -1,
    36: -1,
    37: -1,
    38: -1,
    39: -1,
    40: -1,
    41: -1,
    42: -1,
    43: -1,
    44: -1,
    45: -1,
    46: -1,
    47: -1,
    48: -1,
    49: -1,
    50: -1,
    51: -1,
    52: -1,
    53: -1,
    54: -1,
    55: 1,  # leftUpLeg -> LHip
    56: 2,  # leftLeg -> LKnee
    57: 3,  # leftFoot -> LAnkle
    58: 5,  # leftToeBase -> LBigToe
    59: -1,  # leftToeEnd
    60: 7,  # rightUpLeg -> RHip
    61: 8,  # rightLeg -> RKnee
    62: 9,  # rightFoot -> RAnkle
    63: 11,  # rightToeBase -> RBigToe 
    64: -1,  # rightToeEnd 
}

skeletonzombie_character.to_smpl_mapping = {
    0: 0,  # hips -> MidHip
    1: 3,  # spine -> spine
    2: 6,  # spine1 -> spine1
    3: 9,  # spine2 -> spine2
    4:  13,  # leftShoulder -> leftShoulder
    5: 16,  # leftArm -> leftArm
    6: 18,  # leftForeArm -> leftForeArm
    7: 20,  # leftHand -> leftHand
    8: -1,
    9: -1,
    10: -1,
    11: -1,
    12: -1,
    13: -1,
    14: -1,
    15: -1,
    16: -1,
    17: -1,
    18: -1,
    19: -1,
    20: -1,
    21: -1,
    22: -1,
    23: -1,
    24: -1,
    25: -1,
    26: -1,
    27: -1,
    28: 12,  # neck -> neck
    29: -1,  # head 
    30: -1,  # head top end
    31: 14,  # rightShoulder -> rightShoulder
    32: 17,  # rightArm -> rightArm
    33: 19,  # rightForeArm -> rightForeArm
    34: 21,  # rightHand -> rightHand
    35: -1,
    36: -1,
    37: -1,
    38: -1,
    39: -1,
    40: -1,
    41: -1,
    42: -1,
    43: -1,
    44: -1,
    45: -1,
    46: -1,
    47: -1,
    48: -1,
    49: -1,
    50: -1,
    51: -1,
    52: -1,
    53: -1,
    54: -1,
    55: 1,  # leftUpLeg -> LHip
    56: 4,  # leftLeg -> LKnee
    57: 7,  # leftFoot -> LAnkle
    58: 10,  # leftToeBase -> LBigToe
    59: -1,  # leftToeEnd
    60: 2,  # rightUpLeg -> RHip
    61: 5,  # rightLeg -> RKnee
    62: 8,  # rightFoot -> RAnkle
    63: 11,  # rightToeBase -> RBigToe 
    64: -1,  # rightToeEnd 
}

################################################################################################################

#
# ty physics
#
ty_character.right_leg_chain = [59, 60, 61, 62]
ty_character.left_leg_chain = [55, 56, 57, 58] 
ty_character.upper_body_joints = list(range(0, 55))

ty_character.mass = MALE_MASS * 0.5
ty_character.seg_to_mass_perc_map = {
    'head' : 40.0,
    'upper_trunk' : 9.0,
    'mid_trunk' : 12.0,
    'lower_trunk' : 11.0,
    'left_upper_arm' : 2.0,
    'left_forearm' : 1.0,
    'left_hand' : 1.0,
    'left_thigh' : 2.0,
    'left_shank' : 3.0,
    'left_foot' : 5.0,
    'right_upper_arm' : 2.0,
    'right_forearm' : 1.0,
    'right_hand' : 1.0,
    'right_thigh' : 2.0,
    'right_shank' : 3.0,
    'right_foot' : 5.0
}
# maps segment names to the joints where this mass is located (estimated by avg location of all given joints)
ty_character.segment_to_joints_map = {
    'head' : [53],
    'upper_trunk' : [3],
    'mid_trunk' : [2],
    'lower_trunk' : [1],
    'left_upper_arm' : [5, 6],
    'left_forearm' : [6, 7],
    'left_hand' : range(7, 28),
    'left_thigh' : [55, 56],
    'left_shank' : [56, 57],
    'left_foot' : [57, 58], 
    'right_upper_arm' : [29, 30],
    'right_forearm' : [30, 31],
    'right_hand' : range(31, 52),
    'right_thigh' : [59, 60],
    'right_shank' : [60, 61],
    'right_foot' : [61, 62]
}

#
# ty mappings
#
ty_character.ik_blacklist = [5, 6, 7, 29, 30, 31] #2, 3,#, 60, 65]
# ty_foot_indices = [57,58,61,62] # left_ankle, left_toe, right_ankle, right_toe
ty_character.eye_indices = [53, 53] # no eye, just use the head joint
ty_character.to_combined_mapping = {
    0: 0,  # hips -> MidHip
    1: 13,  # spine -> spine
    2: 14,  # spine1 -> spine1
    3: 15,  # spine2 -> spine2
    4:  -1,  # leftShoulder -> Neck
    5: 22,  # leftArm -> LShoulder
    6: 23,  # leftForeArm -> LElbow
    7: 24,  # leftHand -> LWrist
    8: -1,
    9: -1,
    10: -1,
    11: -1,
    12: -1,
    13: -1,
    14: -1,
    15: -1,
    16: -1,
    17: -1,
    18: -1,
    19: -1,
    20: -1,
    21: -1,
    22: -1,
    23: -1,
    24: -1,
    25: -1,
    26: -1,
    27: -1,
    28: -1,  # rightShoulder -> neck
    29: 25,  # rightArm -> RShoulder
    30: 26,  # rightForeArm -> RElbow
    31: 27,  # rightHand -> RWrist
    32: -1,
    33: -1,
    34: -1,
    35: -1,
    36: -1,
    37: -1,
    38: -1,
    39: -1,
    40: -1,
    41: -1,
    42: -1,
    43: -1,
    44: -1,
    45: -1,
    46: -1,
    47: -1,
    48: -1,
    49: -1,
    50: -1,
    51: -1,
    52: 16,  # neck -> neck
    53: -1,  # head -> nose 
    54: -1,  # head top end
    55: 1,  # leftUpLeg -> LHip
    56: 2,  # leftLeg -> LKnee
    57: 3,  # leftFoot -> LAnkle
    58: 5,  # leftToeBase -> LBigToe
    59: 7,  # rightUpLeg -> RHip
    60: 8,  # rightLeg -> RKnee
    61: 9,  # rightFoot -> RAnkle
    62: 11,  # rightToeBase -> RBigToe 
}

ty_character.to_smpl_mapping = {
    0: 0,  # hips -> MidHip
    1: 3,  # spine -> spine
    2: 6,  # spine1 -> spine1
    3: 9,  # spine2 -> spine2
    4:  13,  # leftShoulder -> leftShoulder
    5: 16,  # leftArm -> leftArm
    6: 18,  # leftForeArm -> leftForeArm
    7: 20,  # leftHand -> leftHand
    8: -1,
    9: -1,
    10: -1,
    11: -1,
    12: -1,
    13: -1,
    14: -1,
    15: -1,
    16: -1,
    17: -1,
    18: -1,
    19: -1,
    20: -1,
    21: -1,
    22: -1,
    23: -1,
    24: -1,
    25: -1,
    26: -1,
    27: -1,
    28: 14,  # rightShoulder -> rightShoulder
    29: 17,  # rightArm -> rightArm
    30: 19,  # rightForeArm -> rightForeArm
    31: 21,  # rightHand -> rightHand
    32: -1,
    33: -1,
    34: -1,
    35: -1,
    36: -1,
    37: -1,
    38: -1,
    39: -1,
    40: -1,
    41: -1,
    42: -1,
    43: -1,
    44: -1,
    45: -1,
    46: -1,
    47: -1,
    48: -1,
    49: -1,
    50: -1,
    51: -1,
    52: 12,  # neck -> neck
    53: -1,  # head 
    54: -1,  # head top end
    55: 1,  # leftUpLeg -> LHip
    56: 4,  # leftLeg -> LKnee
    57: 7,  # leftFoot -> LAnkle
    58: 10,  # leftToeBase -> LBigToe
    59: 2,  # rightUpLeg -> RHip
    60: 5,  # rightLeg -> RKnee
    61: 8,  # rightFoot -> RAnkle
    62: 11,  # rightToeBase -> RBigToe 
}

################################################################################################################