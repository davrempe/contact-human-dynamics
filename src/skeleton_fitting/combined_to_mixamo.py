import re
import sys
import os
import numpy as np
import scipy.io as sio
import scipy.io
import argparse
import h5py

sys.path.append('skeleton_fitting/ik')
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
from InverseKinematics import JacobianInverseKinematicsCK, JacobianInverseKinematics

sys.path.append('utils')
from character_info_utils import get_character_ik_blacklist, get_character_to_combined_mapping # mappings
from character_info_utils import get_character_foot_inds, get_character_ankle_inds, get_character_eye_inds # character joint idx
from character_info_utils import combined_eye_inds, combined_foot_inds, combined_ankle_inds # combined joint idx


def argsparser():
    parser = argparse.ArgumentParser("Retarget combined body25/smpl skeleton to mixamo 67")
    parser.add_argument('--src_bvh', help='source motion data', required=True)
    parser.add_argument('--out_bvh', help='output BVH file', required=True)
    parser.add_argument('--character', help='target skeleton', default='ybot')
    return parser.parse_args()

def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)

def retarget(src_bvh, character, out_bvh):
    skel_bvh = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.character + ".bvh")
    skel, names, _ = BVH.load(skel_bvh)
    skel.rotations = Quaternions.from_euler(0.0*skel.rotations.euler(), order='xyz', world=True)
    skel_targets = Animation.positions_global(skel)

    #
    # zeros out mixamo to have floor at 0
    #
    footIndices = get_character_foot_inds(args.character)
    fid_l, fid_r = np.array([footIndices[0],footIndices[1]]), np.array([footIndices[2],footIndices[3]])
    foot_heights = np.minimum(skel_targets[:, fid_l, 1], skel_targets[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    # print('Mixamo floor height: ' + str(floor_height))
    skel_targets[:, :, 1] -= floor_height

    # get height of character below hips
    skel_height = np.abs(np.amax(skel_targets[:,0,1]) - np.amin(skel_targets[:,footIndices,1], axis=1)).max()
    # print('Mixamo hip skeleton height: ' + str(skel_height))
    skel.positions = skel.offsets[np.newaxis]
    skel.rotations.qs = skel.orients.qs[np.newaxis]

    anim_src, names_src, ftime = BVH.load(args.src_bvh)
    anim_targets = Animation.positions_global(anim_src)

    #
    # Zero out target motion to have floor (roughly) at 0
    #
    anim_targets[:,:,1] = -anim_targets[:,:,1] # flip y to find height b/c upside down by default
    footIndicesSrc = combined_foot_inds # heel and toe joints 
    fid_l, fid_r = np.array([footIndicesSrc[0], footIndicesSrc[1], footIndicesSrc[2]]), np.array([footIndicesSrc[3], footIndicesSrc[4], footIndicesSrc[5]])
    foot_heights = np.minimum(anim_targets[:, fid_l, 1], anim_targets[:, fid_r, 1]).min(axis=1)
    src_floor_height = softmin(foot_heights, softness=0.5, axis=0)
    # print('Target floor height: ' + str(src_floor_height))
    anim_targets[:, :, 1] -= src_floor_height

    footIndicesSrc = combined_foot_inds
    # below hip height
    anim_height = np.abs(np.amax(anim_targets[:,0,1]) - np.amin(anim_targets[:,footIndicesSrc,1], axis=1)).max()
    # print('Target hip skeleton height: ' + str(anim_height))

    # scales target positions so that heights match up
    # print('Height ratio: ' + str(skel_height / anim_height))

    anim_targets[:,:,1] = -anim_targets[:,:,1] # flip back
    height_ratio = (skel_height / anim_height)

    targets = anim_targets.copy()
    targets[:,:,:] *= height_ratio # 
    targets[:,:,[0, 2]] -= np.expand_dims(targets[:, 0, [0, 2]] - anim_targets[:, 0, [0, 2]], axis=1) # don't want to scale hip translation x/z


    # set up animation to have same number of frames as target motion
    anim = skel.copy()
    anim.orients.qs = skel.orients.qs.copy()
    anim.offsets = skel.offsets.copy()
    anim.rotations.qs = skel.rotations.qs.repeat(len(targets), axis=0)
    anim.positions = skel.positions.repeat(len(targets), axis=0)
    anim.positions[:, 0] = targets[:, 0] # set root motion

    mapping = get_character_to_combined_mapping(character)
    endeffectors = range(len(mapping))
    targetmap = {}
    for i in endeffectors:
        if mapping[i] > -1 and i not in get_character_ik_blacklist(args.character): # only map joints we actually have
            targetmap[i] = targets[:,mapping[i]]
    # initialize angles for new motion with angles we have from target motion (I assume this makes IK quicker/better)
    eulerAngles = anim_src.rotations.euler()
    references = np.zeros((len(anim_src), len(mapping), 3))
    for i in range(len(mapping)):
        if mapping[i] > -1:
            references[:,i] = eulerAngles[:,mapping[i]]
            for f in range(len(anim)):
                references[f, i] = np.fmod(references[f, i]*180/3.1415, 180)
                references[f, i] = references[f, i]*3.1415/180
    anim.rotations = Quaternions.from_euler(references, order='xyz', world=True)

    # IK to actually create the retargeted motion on the mixamo character
    ik = JacobianInverseKinematicsCK(anim, targetmap, translate=True, iterations=200, smoothness=0.0, damping=7.0, silent=False)
    ik()

    # additional translation to make up for root translation during IK
    anim.positions[:, 1:, :] = skel.positions[:, 1:, :].repeat(len(targets-1), axis=0)
    ank_diff = targets[:, combined_ankle_inds, 1] - Animation.positions_global(anim)[:, get_character_ankle_inds(args.character), 1]
    ank_off = np.median(ank_diff)
    anim.positions[:, 0, 1] += ank_off
    anim.positions[:, 0, 1] -= src_floor_height # reaccount for the floor

    
    """ Save the resulting animation in BVH """
    path = os.path.dirname(out_bvh)
    if path is not '' and not os.path.exists(path):
        os.makedirs(path)
    BVH.save(out_bvh, anim, names)

    print('Finished retargeting!')

    return


if __name__ == '__main__':
    args = argsparser()
    retarget(args.src_bvh, args.character, args.out_bvh)

