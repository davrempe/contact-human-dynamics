from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np

from scipy.optimize import least_squares
from scipy import sparse
from scipy.sparse import lil_matrix
from copy import copy, deepcopy

from sklearn import linear_model

import SkeletonDefinitions

sys.path.extend(['skeleton_fitting/ik'])

import BVH as BVH
import Animation as Animation
import AnimationStructure
from Quaternions import Quaternions
from Pivots import Pivots
from InverseKinematics import JacobianInverseKinematics, JacobianInverseKinematicsCK

sys.path.append('utils')
import openpose_utils
import totalcap_utils

ROOT_IDX = SkeletonDefinitions.COMBINED_ROOT_IDX
FEET_IDX = SkeletonDefinitions.COMBINED_FEET_IDX
FORWARD_MAPPING = SkeletonDefinitions.mapping_combined_skel_to_body_25
BACKWARD_MAPPING = SkeletonDefinitions.mapping_body_25_to_combined_skel
# assume we have 3 spine joints, but this tells us which indices these are
SKEL_SPINE_IDX = SkeletonDefinitions.COMBINED_SKEL_SPINE_JOINTS
SKEL_NECK_IDX = SkeletonDefinitions.COMBINED_SKEL_NECK_JOINT
SMOOTH_WEIGHTS = SkeletonDefinitions.COMBINED_SMOOTH_WEIGHTS
PROJ_WEIGHTS = SkeletonDefinitions.COMBINED_PROJ_WEIGHTS
DATA_WEIGHTS = SkeletonDefinitions.COMBINED_DATA_WEIGHTS

SMOOTH_VEL_X = 1.0
SMOOTH_VEL_Y = 1.0
SMOOTH_VEL_Z = 2.0
SMOOTH_VEL_EULER_X = 10.0
SMOOTH_VEL_EULER_Y = 10.0
SMOOTH_VEL_EULER_Z = 10.0


def jac_root_all_for_projection(x, curPose3D, curPose2D, floorNormal, floorPoint, pointWeights, data_joint_weights, 
                                valid3DInd, valid2DInd, jointsSmoothnessWeights, velConstraint,
                                projWeight,smoothnessWeightVel,smoothnessWeightAcc,dataWeight,velWeight,floorWeight,root_idx=0):
    ''' calc jacobian '''
    noFrames = curPose3D.shape[0]
    noProjPts = len(valid3DInd)
    noPts = curPose3D.shape[1]

    # projection = noFrames * noProjPts * 2
    # smoothness in velocity = (noFrames - 1) * 3 * noPts
    # smoothness in acceleration = (noFrames - 2) * 3 * noPts
    # data for pose = noFrames * (noPts - 1) * 3
    # velocity term = (noFrames-1) * noPts * 3
    # floor term = noFrames * noPts
    noTerms = noFrames * noProjPts * 2
    if(noFrames > 1):
        noTerms += (noFrames - 1) * 3 * noPts
    if(noFrames > 2):
        noTerms += (noFrames - 2) * 3 * noPts
    noTerms += noFrames * (noPts) * 3
    if(noFrames > 1):
        noTerms += (noFrames-1)*noPts*3
    noTerms += noFrames*noPts

    jac = np.zeros((noTerms, noFrames * noPts * 3))

    count = 0

    # projection term
    for fr in range(noFrames):
        varIndex = fr * noPts * 3  # the index of the first variable that correspond to this frame
        pose3D = x[varIndex: varIndex + noPts * 3]
        currentRoot = pose3D[root_idx*3:(root_idx*3 + 3)]

        for j in range(noProjPts): # NOTE: projection weight is 0 on the root which is why we can do this (otherwise root would be added to self and doubled)
            corr2DPt = valid2DInd[j]
            corr3DPt = valid3DInd[j]
            if (pointWeights[fr, j] > 0):
                # x coordinate
                if j == root_idx:
                    f = currentRoot[0]
                    g = currentRoot[2]
                else:
                    f = pose3D[corr3DPt * 3 + 0] + currentRoot[0]
                    g = pose3D[corr3DPt * 3 + 2] + currentRoot[2]

                # derivative with respect to root or the joint position
                f_x = 1.0
                f_z = 0.0
                g_x = 0.0
                g_z = 1.0

                jac[count + j * 2 + 0, varIndex + 0] = projWeight * pointWeights[fr, j] * (f_x * g - f * g_x) / (
                        g * g)  # derivative with respect to root x
                jac[count + j * 2 + 0, varIndex + 2] = projWeight * pointWeights[fr, j] * (f_z * g - f * g_z) / (
                        g * g)  # derivative with respect to root z

                jac[count + j * 2 + 0, varIndex + corr3DPt * 3 + 0] = projWeight * pointWeights[fr, j] * (
                            f_x * g - f * g_x) / (
                                                                                    g * g)  # derivative with respect to joint position x
                jac[count + j * 2 + 0, varIndex + corr3DPt * 3 + 2] = projWeight * pointWeights[fr, j] * (
                            f_z * g - f * g_z) / (
                                                                                    g * g)  # derivative with respect to joint position z

                # y coordinate
                if j == root_idx:
                    f = currentRoot[1]
                    g = currentRoot[2]
                else:
                    f = pose3D[corr3DPt * 3 + 1] + currentRoot[1]
                    g = pose3D[corr3DPt * 3 + 2] + currentRoot[2]

                # derivative with respect to root
                f_y = 1.0
                f_z = 0.0
                g_y = 0.0
                g_z = 1.0
                jac[count + j * 2 + 1, varIndex + 1] = projWeight * pointWeights[fr, j] * (f_y * g - f * g_y) / (
                        g * g)  # derivative with respect to root y
                jac[count + j * 2 + 1, varIndex + 2] = projWeight * pointWeights[fr, j] * (f_z * g - f * g_z) / (
                        g * g)  # derivative with respect to root z
                jac[count + j * 2 + 1, varIndex + corr3DPt * 3 + 1] = projWeight * pointWeights[fr, j] * (
                            f_y * g - f * g_y) / (
                                                                                    g * g)  # derivative with respect to joint position y
                jac[count + j * 2 + 1, varIndex + corr3DPt * 3 + 2] = projWeight * pointWeights[fr, j] * (
                            f_z * g - f * g_z) / (
                                                                                    g * g)  # derivative with respect to joint position z

        count = count + noProjPts * 2

    # smoothness term for velocity
    for fr in range(noFrames - 1):
        varIndex = fr * noPts * 3
        varIndexNext = (fr+1)*noPts*3
        for j in range(noPts):
            #smoothness in x,y axis disabled
            jac[count + 0, varIndex + 0] = smoothnessWeightVel * jointsSmoothnessWeights[j] * SMOOTH_VEL_X
            jac[count + 1, varIndex + 1] = smoothnessWeightVel * jointsSmoothnessWeights[j] * SMOOTH_VEL_Y
            jac[count + 2, varIndex + 2] = smoothnessWeightVel * jointsSmoothnessWeights[j] * SMOOTH_VEL_Z
            jac[count + 0, varIndexNext + 0] = -smoothnessWeightVel * jointsSmoothnessWeights[j] * SMOOTH_VEL_X
            jac[count + 1, varIndexNext + 1] = -smoothnessWeightVel * jointsSmoothnessWeights[j] * SMOOTH_VEL_Y
            jac[count + 2, varIndexNext + 2] = -smoothnessWeightVel * jointsSmoothnessWeights[j] * SMOOTH_VEL_Z
            count = count + 3
            varIndex = varIndex + 3
            varIndexNext = varIndexNext + 3

    # smoothness for acceleration
    for fr in range(noFrames - 2):
        varIndex = fr * noPts * 3
        varIndexNext = (fr + 1) * noPts * 3
        varIndexNextNext = (fr+2)*noPts*3
        for j in range(noPts):
            jac[count + 0, varIndex + 0] = smoothnessWeightAcc
            jac[count + 1, varIndex + 1] = smoothnessWeightAcc
            jac[count + 2, varIndex + 2] = smoothnessWeightAcc
            jac[count + 0, varIndexNext + 0] = -smoothnessWeightAcc * 2
            jac[count + 1, varIndexNext + 1] = -smoothnessWeightAcc * 2
            jac[count + 2, varIndexNext + 2] = -smoothnessWeightAcc * 2
            jac[count + 0, varIndexNextNext + 0] = smoothnessWeightAcc
            jac[count + 1, varIndexNextNext + 1] = smoothnessWeightAcc
            jac[count + 2, varIndexNextNext + 2] = smoothnessWeightAcc
            count = count + 3
            varIndex = varIndex + 3
            varIndexNext = varIndexNext + 3
            varIndexNextNext = varIndexNextNext + 3

    # data term for joints
    for fr in range(noFrames):
        varIndex = fr*noPts*3
        for j in range(0, noPts):
            # if j != root_idx:
            jac[count + 0, varIndex + j * 3 + 0] = dataWeight  * data_joint_weights[fr, j]
            jac[count + 1, varIndex + j * 3 + 1] = dataWeight  * data_joint_weights[fr, j]
            jac[count + 2, varIndex + j * 3 + 2] = dataWeight  * data_joint_weights[fr, j]
            count = count + 3

    #velocity term: velocity of joint from fr to (fr+1) should be 0
    #joint position = root + joint
    for fr in range(noFrames-1):
        varIndex = fr*noPts*3
        varIndexNext = (fr + 1) * noPts * 3
        rootIdx = varIndex + root_idx*3
        rootIdxNext = varIndexNext + root_idx*3
        for j in range(noPts):
            if(velConstraint[fr, j] == 1):
                #derivative with respect to root 
                # NOTE: can do this because root will never have velocity constraint. 
                jac[count + 0, rootIdx + 0] = velWeight
                jac[count + 1, rootIdx + 1] = velWeight
                jac[count + 2, rootIdx + 2] = velWeight
                jac[count + 0, rootIdxNext + 0] = -velWeight
                jac[count + 1, rootIdxNext + 1] = -velWeight
                jac[count + 2, rootIdxNext + 2] = -velWeight

                #derivative with respect to pose
                jac[count + 0, varIndex + j * 3 + 0] = velWeight
                jac[count + 1, varIndex + j * 3 + 1] = velWeight
                jac[count + 2, varIndex + j * 3 + 2] = velWeight
                jac[count + 0, varIndexNext + j * 3 + 0] = -velWeight
                jac[count + 1, varIndexNext + j * 3 + 1] = -velWeight
                jac[count + 2, varIndexNext + j * 3 + 2] = -velWeight

            count = count + 3

    #floor term: feet joints must be at floor height
    #joint position = root + joint
    for fr in range(noFrames):
        varIndex = fr*noPts*3
        rootIdx = varIndex + root_idx*3
        for j in range(noPts):
            if(velConstraint[fr, j] == 1):
                #derivative with respect to root 
                # NOTE: can do this because root will never have velocity constraint. 
                jac[count, rootIdx + 0] = floorWeight*floorNormal[0]
                jac[count, rootIdx + 1] = floorWeight*floorNormal[1]
                jac[count, rootIdx + 2] = floorWeight*floorNormal[2]

                #derivative with respect to pose
                jac[count, varIndex + j * 3 + 0] = floorWeight*floorNormal[0]
                jac[count, varIndex + j * 3 + 1] = floorWeight*floorNormal[1]
                jac[count, varIndex + j * 3 + 2] = floorWeight*floorNormal[2]

            count = count + 1

    return jac

def jac_anim_for_projection_sparse(x, skeleton, curPose3D, rootTrans, curPose2D, floorNormal, floorPoint,
                                    pointWeights, data_joint_weights, valid3DInd, valid2DInd,
                                    jointsSmoothnessWeights, velConstraints, projWeight, smoothnessWeightVel,
                                    smoothnessWeightAcc, dataWeight, velWeight, floorWeight):
    ''' calc jacobian '''
    noFrames = curPose3D.shape[0]
    noProjPts = curPose3D.shape[1]
    valid3DInd = np.arange(noProjPts)
    valid2DInd = np.arange(noProjPts)
    noPts = curPose3D.shape[1]
    #forward kinematics
    x = np.reshape(x, [noFrames, -1])
    root = x[:,:3]
    angles = x[:,3:]
    anim = skeleton.copy()
    anim.orients.qs = skeleton.orients.qs.copy()
    anim.offsets = skeleton.offsets.copy()
    anim.positions = skeleton.positions.repeat(noFrames, axis=0)
    anim.rotations = Quaternions.from_euler(angles.reshape((noFrames, noPts, 3)), order='xyz', world=True)
    poses3D = Animation.positions_global(anim)
    poses3D[:,0] = root

    y = 0*poses3D
    for j in range(poses3D.shape[1]):
        y[:,j,:] = poses3D[:,BACKWARD_MAPPING[j],:]
    y = np.reshape(y,[-1]) #[root positions; joint positions]

    #calculate jacobian dE/dP = np.zeros((noTerms, noFrames * noPts * 3))
    jac_dp = jac_root_all_for_projection(y, curPose3D, curPose2D, floorNormal, floorPoint, pointWeights,
                                        data_joint_weights, valid3DInd, valid2DInd, jointsSmoothnessWeights, velConstraints,
                                        projWeight, smoothnessWeightVel, smoothnessWeightAcc, dataWeight, velWeight, floorWeight,
                                        root_idx=ROOT_IDX)

    #calculate jocabian dP/dR = np.zeros((noFrames * (noPts * 3) * (noPts * 3))
    targetmap = {}
    for ee in range(poses3D.shape[1]-1):
        targetmap[ee+1] = poses3D[:, ee+1]
    ik = JacobianInverseKinematics(anim, targetmap, translate=False, iterations=50, damping=2, silent=False)
    gt = Animation.transforms_global(anim)
    gp = gt[:, :, :, 3]
    gp = gp[:, :, :3] / gp[:, :, 3, np.newaxis]
    gr = Quaternions.from_transforms(gt)
    """ Calculate Descendants """
    descendants = AnimationStructure.descendants_mask(anim.parents)
    tdescendants = np.eye(anim.shape[1]) + descendants
    first_descendants = descendants[:,1:].repeat(3, axis=0).astype(int)
    first_tdescendants = tdescendants[:,1:].repeat(3, axis=0).astype(int)
    jac_dr = ik.jacobian(angles, gp, gr, ik.targets, first_descendants, first_tdescendants)

    #full jacobian dE/dR
    noVars = (noPts+1) * 3
    jac = np.zeros((jac_dp.shape[0], noFrames * noVars))
    # print("jac dimension is %d %d" % (jac.shape[0], jac.shape[1]))
    for fr in range(noFrames):
        varIndex1 = fr * noPts * 3
        varIndex2 = fr * noVars
        jt1 = jac_dp[:,varIndex1:varIndex1+noPts*3]
        jt2 = np.zeros(jt1.shape)
        for p in range(noPts):
            q = FORWARD_MAPPING[p]
            jt2[:,p*3:p*3+3] = jt1[:,q*3:q*3+3]
        jt = np.zeros((jac_dp.shape[0], noVars))
        jt[:,:3] = jt2[:,:3]
        jt[:,3:] = np.matmul(jt2[:,3:], jac_dr[fr])
        jac[:,varIndex2:varIndex2+noVars] = jt

    # smoothness term for euler angle velocity
    jac_smooth = np.zeros(((noFrames - 1) * noVars, noFrames * noVars))
    count = 0
    for fr in range(noFrames - 1):
        varIndex = fr * noVars
        varIndexNext = (fr+1) * noVars
        for j in range(noPts+1):
            jac_smooth[count + 0, varIndex + 0] = smoothnessWeightVel * SMOOTH_VEL_EULER_X
            jac_smooth[count + 1, varIndex + 1] = smoothnessWeightVel * SMOOTH_VEL_EULER_Y
            jac_smooth[count + 2, varIndex + 2] = smoothnessWeightVel * SMOOTH_VEL_EULER_Z
            jac_smooth[count + 0, varIndexNext + 0] = -smoothnessWeightVel * SMOOTH_VEL_EULER_X
            jac_smooth[count + 1, varIndexNext + 1] = -smoothnessWeightVel * SMOOTH_VEL_EULER_Y
            jac_smooth[count + 2, varIndexNext + 2] = -smoothnessWeightVel * SMOOTH_VEL_EULER_Z
            count = count + 3
            varIndex = varIndex + 3
            varIndexNext = varIndexNext + 3
    jac = np.concatenate((jac, jac_smooth), axis=0)
    jac_sparse = sparse.lil_matrix(jac)

    return jac_sparse

def fun_anim_for_projection(x, skeleton, curPose3D, rootTrans, curPose2D,
                            floorNormal, floorPoint, pointWeights, data_joint_weights,
                            valid3DInd, valid2DInd, jointsSmoothnessWeights, velConstraints,
                            projWeight,smoothnessWeightVel,smoothnessWeightAcc,dataWeight,
                            velWeight, floorWeight):
    """ calbulate objective function """

    noFrames = curPose3D.shape[0]
    noProjPts = curPose3D.shape[1] # points which will be used for projection to image space
    noPts = curPose3D.shape[1] # all joints (includes the root joint)

    f = []

    #projection = noFrames*noProjPts * 2
    #smoothness for velocity and acceleration = (noFrames-1)*3*noPts + (noFrames-2)*3*noPts
    #data term = noFrames*(noPts)*3
    # velocity term = (noFrames-1) * noPts * 3
    # floor constraints = (noFrames) * noPts
    f = np.zeros(noFrames*noProjPts * 2 + (noFrames-1)*3*noPts + (noFrames-2)*3*noPts + noFrames*(noPts)*3 + (noFrames-1)*noPts*3 + (noFrames-1)*(noPts+1)*3 + noFrames*noPts)

    #forward kinematics
    x = np.reshape(x, [noFrames, -1])
    root = x[:,:3]
    angles = x[:,3:]
    anim = skeleton.copy()
    anim.orients.qs = skeleton.orients.qs.copy()
    anim.offsets = skeleton.offsets.copy()
    anim.positions = skeleton.positions.repeat(noFrames, axis=0)
    anim.rotations = Quaternions.from_euler(angles.reshape((noFrames, noPts, 3)), order='xyz', world=True)
    initPositions = Animation.positions_global(anim)
    initPositions[:,0] = root

    y = 0*initPositions
    for j in range(initPositions.shape[1]):
        y[:,j,:] = initPositions[:,BACKWARD_MAPPING[j],:]
    y = np.reshape(y,[-1]) #[root positions; joint positions]


    count = 0

    #projection term
    for fr in range(noFrames):
        varIndex = fr*noPts*3 #what is the first variable index in x that corresponds to this frame
        pose3D = y[varIndex:varIndex + noPts*3]
        currentRoot = pose3D[ROOT_IDX*3:(ROOT_IDX*3 + 3)]
        currentPose = pose3D
        for j in range(noProjPts): # NOTE: projection weight is 0 on the root which is why we can do this (otherwise root would be added to self and doubled)
            corr2DPt = j
            corr3DPt = j
            if(pointWeights[fr, j] > 0):
                if corr3DPt == ROOT_IDX:
                    projPtX = (currentRoot[0] / currentRoot[2])
                    projPtY = (currentRoot[1] / currentRoot[2])
                    f[count+j*2+0] = projWeight*pointWeights[fr,j]*(projPtX - curPose2D[fr,corr2DPt,0])
                    f[count+j*2+1] = projWeight*pointWeights[fr,j]*(projPtY - curPose2D[fr,corr2DPt,1])
                else:
                    projPtX = (currentPose[corr3DPt*3+0] + currentRoot[0]) / (currentPose[corr3DPt*3+2] + currentRoot[2])
                    projPtY = (currentPose[corr3DPt*3+1] + currentRoot[1]) / (currentPose[corr3DPt*3+2] + currentRoot[2])
                    f[count+j*2+0] = projWeight*pointWeights[fr,j]*(projPtX - curPose2D[fr,corr2DPt,0])
                    f[count+j*2+1] = projWeight*pointWeights[fr,j]*(projPtY - curPose2D[fr,corr2DPt,1])
        count = count + noProjPts*2

    # print("projection error:%f" % np.linalg.norm(f))

    #smoothness term in velocity
    start = count
    for fr in range(noFrames-1):
        varIndex = fr * noPts * 3
        varIndexNext = (fr + 1) * noPts * 3
        for j in range(0, noPts):
            f[count + 0] = smoothnessWeightVel * jointsSmoothnessWeights[j] * SMOOTH_VEL_X * (y[varIndex+0] - y[varIndexNext+0])
            f[count + 1] = smoothnessWeightVel * jointsSmoothnessWeights[j] * SMOOTH_VEL_Y * (y[varIndex+1] - y[varIndexNext+1])
            f[count + 2] = smoothnessWeightVel * jointsSmoothnessWeights[j] * SMOOTH_VEL_Z * (y[varIndex+2] - y[varIndexNext+2])
            count = count + 3
            varIndex = varIndex + 3
            varIndexNext = varIndexNext + 3

    # print("vel smooth error:%f" % np.linalg.norm(f[start:count]))

    #smoothness term in acceleration
    start = count
    for fr in range(noFrames-2):
        varIndex = fr * noPts * 3  # what is the first variable index in x that corresponds to this frame
        varIndexNext = (fr + 1) * noPts * 3  # what is the first variable index in x that corresponds to next frame
        varIndexNextNext = (fr + 2) * noPts * 3  # what is the first variable index in x that corresponds to next next frame
        for j in range(0, noPts):
            velocityCur = y[varIndexNext:varIndexNext+3] - y[varIndex:varIndex+3]
            velocityNext = y[varIndexNextNext:varIndexNextNext+3] - y[varIndexNext:varIndexNext+3]
            f[count + 0] = smoothnessWeightAcc * (velocityNext[0]-velocityCur[0])
            f[count + 1] = smoothnessWeightAcc * (velocityNext[1]-velocityCur[1])
            f[count + 2] = smoothnessWeightAcc * (velocityNext[2]-velocityCur[2])
            count = count + 3
            varIndex = varIndex + 3
            varIndexNext = varIndexNext + 3
            varIndexNextNext = varIndexNextNext + 3

    # print("accel smooth error:%f" % np.linalg.norm(f[start:count]))

    #data term for pose
    start = count
    for fr in range(noFrames):
        varIndex = fr*noPts*3
        for j in range(0, noPts):
            if j != ROOT_IDX:
                f[count + 0] = dataWeight  * (y[varIndex] - curPose3D[fr,j,0])  * data_joint_weights[fr,j]
                f[count + 1] = dataWeight  * (y[varIndex+1] - curPose3D[fr, j, 1])  * data_joint_weights[fr,j]
                f[count + 2] = dataWeight  * (y[varIndex+2] - curPose3D[fr, j, 2])  * data_joint_weights[fr,j]
            else:
                f[count + 0] = dataWeight  * (y[varIndex] - rootTrans[fr,0])  * data_joint_weights[fr,j]
                f[count + 1] = dataWeight  * (y[varIndex+1] - rootTrans[fr, 1])  * data_joint_weights[fr,j]
                f[count + 2] = dataWeight  * (y[varIndex+2] - rootTrans[fr, 2])  * data_joint_weights[fr,j]
            count += 3
            varIndex += 3

    # print("data error:%f" % np.linalg.norm(f[start:count]))

    # velocity term: velocity of joint from fr to (fr+1) should be 0
    start = count
    # joint position = root + joint
    for fr in range(noFrames - 1):
        varIndex = fr * noPts * 3
        varIndexNext = (fr + 1) * noPts * 3
        rootIdx = varIndex + ROOT_IDX*3
        rootIdxNext = varIndexNext + ROOT_IDX*3
        for j in range(noPts): # NOTE: there will never be velocity constraints on the root which is why we can do this
            if (velConstraints[fr, j] == 1):
                jointCurFrame = y[rootIdx:rootIdx+3] + y[varIndex+j*3:varIndex+j*3+3]
                jointNextFrame = y[rootIdxNext:rootIdxNext+3] + y[varIndexNext+j*3:varIndexNext+j*3+3]
                f[count + 0] = velWeight * (jointCurFrame[0]-jointNextFrame[0])
                f[count + 1] = velWeight * (jointCurFrame[1]-jointNextFrame[1])
                f[count + 2] = velWeight * (jointCurFrame[2]-jointNextFrame[2])
            count += 3
    # print("vel constraints error:%f" % np.linalg.norm(f[start:count]))

    # feet must in contact with the floor
    start = count
    # joint position = root + joint
    for fr in range(noFrames):
        varIndex = fr * noPts * 3
        rootIdx = varIndex + ROOT_IDX*3
        for j in range(noPts): # NOTE: there will never be velocity constraints on the root which is why we can do this
            if (velConstraints[fr, j] == 1):
                jointCurFrame = y[rootIdx:rootIdx+3] + y[varIndex+j*3:varIndex+j*3+3]
                f[count] = floorWeight * (np.dot(floorNormal, jointCurFrame - floorPoint))
            count += 1 # only one residual per joint (since dot product)
    # print("floor constraints error:%f" % np.linalg.norm(f[start:count]))

    #smoothness term in euler angle velocity
    start  = count
    for fr in range(noFrames-1):
        for j in range(0, noPts+1):
            f[count + 0] = smoothnessWeightVel * SMOOTH_VEL_EULER_X * (x[fr,j*3+0] - x[fr+1,j*3+0])
            f[count + 1] = smoothnessWeightVel * SMOOTH_VEL_EULER_Y * (x[fr,j*3+1] - x[fr+1,j*3+1])
            f[count + 2] = smoothnessWeightVel * SMOOTH_VEL_EULER_Z * (x[fr,j*3+2] - x[fr+1,j*3+2])
            count = count + 3
    # print("euler angle smoothness error:%f" % np.linalg.norm(f[start:count]))

    # print("total error:%f" % np.linalg.norm(f))

    return f

def update_skeleton(skel_ref, targets, names=None):
    '''
    Update the given skeleton to have the bone lengths given in targets.
    '''
    parents = skel_ref.parents
    bones = np.zeros(len(parents))
    for j in range(len(parents) - 1):
        if (j+1) in SKEL_SPINE_IDX:
            # set length of spine and spine1 to half the dist
            # between spine1 and root (to avoid crunched spine from SMPL)
            offset = targets[:, SKEL_SPINE_IDX[2]] - targets[:, 0] # root idx of skeleton will always be zero
            bone_all = np.linalg.norm(offset, axis=1) / 3.0
            bones[j + 1] = np.median(bone_all)
        else:
            offset = targets[:, j + 1] - targets[:, parents[j + 1]]
            bone_all = np.linalg.norm(offset, axis=1)
            bones[j + 1] = np.median(bone_all)
    
    skel = skel_ref.copy()
    skel.positions = skel.offsets[np.newaxis]
    skel.rotations.qs = skel.orients.qs[np.newaxis]
    offsets = skel.offsets.copy()
    for j in range(offsets.shape[0]-1):
        offset = offsets[j + 1]
        offset = offset / np.linalg.norm(offset)
        bone = bones[j + 1]
        offset = offset * bone
        offsets[j + 1] = offset
    skel.offsets = offsets
    positions = Animation.positions_global(skel)

    skel.offsets[0, 0] = 0
    skel.offsets[0, 1] = 0
    skel.offsets[0, 2] = 0
    skel.positions = skel.offsets[np.newaxis]
    return skel

def optimize_trajectory(poses2D, joint_conf_2d, poses3D, root_pos, joint_angles,
                        skeleton, names, ppx, ppy, camFocal, velConstraints,
                        save_dir='./',
                        plane_normal=None,
                        plane_point=None):
    '''
    Runs kinematic optimization
    '''
    given_floor = True
    if plane_normal is None or plane_point is None:
        given_floor = False
        plane_normal = np.zeros((3), dtype=np.float)
        plane_point = np.zeros((3), dtype=np.float)

    num_frames = poses2D.shape[0]
    num_joints = poses2D.shape[1]
    if num_joints != poses3D.shape[1]:
        print('2D and 3D data must have the same number of joints!')
        print('2D: ' + str(num_joints))
        print('3D: ' + str(poses3D.shape[1]))
        return None

    # Fit the initial motion to the template skeleton
    # target joint positions must be in order of the skeleton to fit to
    targets = np.zeros((num_frames, len(FORWARD_MAPPING), 3))
    for fr in range(num_frames):
        for j in range(targets.shape[1]):
            targets[fr,j,:] = poses3D[fr,FORWARD_MAPPING[j],:] + root_pos[fr]
    # match bone lengths to initial 3D pose
    skeleton = update_skeleton(skeleton, targets, names)

    # calculate normalized 2D coordinates for projection loss
    # (regular projection with focal length and camera center removed)
    # also find projection weights to be used in optimization
    joints_2d_normalized = poses2D.copy()
    proj_weights = np.ones((num_frames, num_joints))     # weights for the re-projection term
    data_weights = np.ones((num_frames, num_joints))     # weights for the re-projection term
    cam_center = np.array([ppx, ppy])
    for frame_idx in range(num_frames):
        cur_2d_joints = poses2D[frame_idx, :]
        for joint_idx in range(num_joints):
            if joint_idx < 25: # only for joints that have 2D correspondence
                proj_weights[frame_idx, joint_idx] = proj_weights[frame_idx, joint_idx] * joint_conf_2d[frame_idx, joint_idx] * PROJ_WEIGHTS[joint_idx]
                # still need all joints to be used for data term, but some more than others
                data_weights[frame_idx, joint_idx] = (data_weights[frame_idx, joint_idx] + joint_conf_2d[frame_idx, joint_idx]) * DATA_WEIGHTS[joint_idx]
                # calc normalized projection
                joints_2d_normalized[frame_idx, joint_idx, 0] = (cur_2d_joints[joint_idx, 0] - cam_center[0]) / camFocal[0]
                joints_2d_normalized[frame_idx, joint_idx, 1] = (cur_2d_joints[joint_idx, 1] - cam_center[1]) / camFocal[1]
            else:
                proj_weights[frame_idx, joint_idx] = 0
                data_weights[frame_idx, joint_idx] = (data_weights[frame_idx, joint_idx] + 0.4) * DATA_WEIGHTS[joint_idx] # still want data term on spine joints (0.4 is arbitrary)

    init_root_sol = root_pos.copy()

    #
    # perform IK to get initial joint angle estimates. 
    #
    
    # Initialize the animation with fitted skeleton
    anim = skeleton.copy()
    anim.orients.qs = skeleton.orients.qs.copy()
    anim.offsets = skeleton.offsets.copy()
    anim.positions = skeleton.positions.repeat(num_frames, axis=0)
    anim.rotations.qs = anim.rotations.qs.repeat(num_frames, axis=0)
    anim.positions[:,0] = init_root_sol

    # initialize with smpl prediction
    angle_init = np.linalg.norm(joint_angles, axis=2)
    axis_init = joint_angles / np.expand_dims(angle_init + 1e-10, axis=2)
    axis_init[:,:,0] *= -1.0
    axis_init[:,:,1] *= -1.0
    axis_init[:,:,2] *= -1.0
    init_transforms = Quaternions.from_angle_axis(angle_init, axis_init)
    align_transform = Quaternions.from_angle_axis(0.0, np.array([1.0, 0.0, 0.0]))
    for i in range(num_frames):
        for j in range(1): 
            init_transforms[i, j] *= align_transform

    anim.rotations = init_transforms

    # BVH.save(os.path.join(save_dir, 'pre_ik.bvh'), anim, names)

    # Set up end effector constraints with target positions
    targetmap = {}
    for ee_idx in range(targets.shape[1]):
        # no IK on spine
        if ee_idx not in SKEL_SPINE_IDX:
            targetmap[ee_idx] = targets[:, ee_idx]
    # Solve IK for joint angles
    ik = JacobianInverseKinematicsCK(anim, targetmap,
                                     translate=False,
                                     iterations=200,
                                     smoothness=0.0,
                                     damping=7,
                                     silent=False)
    ik()
    # BVH.save(os.path.join(save_dir, 'init_test.bvh'), anim, names)
    init_root_sol = anim.positions[:,0]
    init_positions = Animation.positions_global(anim)

    # 
    # Perform the kinematic optimization
    #

    #
    # For first stage, don't care about floor
    # Weights of various optimization terms for each optim step.
    #
    projWeight = [1000.0] # projection term
    smoothWeightVel = [0.1] # velocity smoothness
    smoothWeightAcc = [0.5] # acceleration smoothness
    dataWeight = [0.3]  # keep 3d pose close to initialization
    velWeight = [10.0]  # velocity at contact frames should be 0 (e.g., foot contact)
    floorWeight = [0.0] # at contact frames, feet must be on floor

    # Compute the initial projection residual
    init_angles = np.reshape(anim.rotations.euler(), [num_frames, -1])
    print(init_angles.shape)
    init_sol = deepcopy(np.concatenate((init_root_sol, init_angles), axis=1))
    init_res = np.zeros(num_frames * num_joints * 2)
    count = 0
    for fr in range(num_frames):
        current_root = init_root_sol[fr,:]
        current_pose = init_positions[fr,:] - current_root
        for j in range(num_joints):
            joint_idx = BACKWARD_MAPPING[j]
            if (proj_weights[fr, j] > 0):
                proj_x = (current_pose[joint_idx, 0] + current_root[0]) / (current_pose[joint_idx, 2] + current_root[2])
                proj_y = (current_pose[joint_idx, 1] + current_root[1]) / (current_pose[joint_idx, 2] + current_root[2])
                init_res[count + j * 2 + 0] = projWeight[-1] * proj_weights[fr, j] * (proj_x - joints_2d_normalized[fr, j, 0])
                init_res[count + j * 2 + 1] = projWeight[-1] * proj_weights[fr, j] * (proj_y - joints_2d_normalized[fr, j, 1])
        count = count + num_joints * 2
    print('Error init:%f' % np.linalg.norm(init_res))

    init_sol = np.reshape(init_sol, [-1])

    # run stepwise optimization
    for step_idx in range(len(projWeight)):
        cur_sol = least_squares(fun_anim_for_projection, init_sol,
                                 max_nfev=50,
                                 verbose=2,
                                 jac=jac_anim_for_projection_sparse,
                                 gtol=1e-12,
                                 bounds=[-np.inf, np.inf],
                                 tr_solver = 'lsmr',
                                 args=(skeleton, poses3D, root_pos, joints_2d_normalized, plane_normal, plane_point,
                                        proj_weights, data_weights, np.arange(num_joints), np.arange(num_joints),
                                        SMOOTH_WEIGHTS, velConstraints, projWeight[step_idx], smoothWeightVel[step_idx], 
                                        smoothWeightAcc[step_idx], dataWeight[step_idx], velWeight[step_idx], floorWeight[step_idx]))
        print('solution at step ' + str(step_idx) + ':')
        print(cur_sol.cost)

        #forward kinematics
        x = np.reshape(cur_sol.x, [num_frames, -1])
        root = x[:, :3]
        angles = x[:, 3:]
        anim = skeleton.copy()
        anim.orients.qs = skeleton.orients.qs.copy()
        anim.offsets = skeleton.offsets.copy()
        anim.positions = skeleton.positions.repeat(num_frames, axis=0)
        anim.rotations = Quaternions.from_euler(angles.reshape((num_frames, num_joints, 3)), order='xyz', world=True)
        anim.positions[:, 0] = root
        # save final animation
        # BVH.save(os.path.join(save_dir, 'step' + str(step_idx) + '_test.bvh'), anim, names)

        init_sol = cur_sol.x

    #
    # fit floor
    #

    # collect foot positions at contact points
    #forward kinematics
    x = np.reshape(cur_sol.x, [num_frames, -1])
    root = x[:, :3]
    angles = x[:, 3:]
    anim = skeleton.copy()
    anim.orients.qs = skeleton.orients.qs.copy()
    anim.offsets = skeleton.offsets.copy()
    anim.positions = skeleton.positions.repeat(num_frames, axis=0)
    anim.rotations = Quaternions.from_euler(angles.reshape((num_frames, num_joints, 3)), order='xyz', world=True)
    anim.positions[:, 0] = root

    final_pos = Animation.positions_global(anim)
    feet_pos = final_pos[:, FEET_IDX, :]
    feet_contact = np.array([FORWARD_MAPPING[foot_idx] for foot_idx in FEET_IDX])
    # this will order positions by
    feet_pos = feet_pos[velConstraints[:, feet_contact] == 1]
    print(str(feet_pos.shape[0]) + ' contacts for floor fit...')

    # if we weren't given a floor to use, fit one
    if not given_floor:
        # floor normal and point won't be used now
        plane_normal = np.zeros((3), dtype=np.float)
        plane_point = np.zeros((3), dtype=np.float)

        # First fit for floor (ignore more outliers)
        huber = linear_model.HuberRegressor(epsilon=1.5)
        huber.fit(feet_pos[:,[0,2]], feet_pos[:,1])

        print('Floor fit after ' + str(huber.n_iter_) + ' LBFGS iters!')
        coeff = huber.coef_
        intercept = huber.intercept_
        print('Coeffs + Intercept = (%f, %f, %f)' % (coeff[0], coeff[1], intercept))

        plane_verts = np.array([[0.0, -1.0, 0.0], [0.0, -1.0, 100.0], [100.0, -1.0, 0.0]])
        for i in range(plane_verts.shape[0]):
            plane_verts[i, 1] = huber.predict(np.array([plane_verts[i, [0, 2]]]))
        # print(plane_verts)
        plane_normal = np.cross(plane_verts[2, :] - plane_verts[0, :], plane_verts[1, :] - plane_verts[2, :])
        plane_normal /= np.linalg.norm(plane_normal)
        plane_point = plane_verts[0]

        print('Number of outliers: %d' % (np.sum(huber.outliers_)))

    print('Normal: (%f, %f, %f)' % (plane_normal[0], plane_normal[1], plane_normal[2]))
    print('Point: (%f, %f, %f)' % (plane_point[0], plane_point[1], plane_point[2]))

    # update contacts based on floor fit only if we fit a floor
    if not given_floor:
        # second fit to find spurious contacts
        huber = linear_model.HuberRegressor(epsilon=2.2)
        huber.fit(feet_pos[:,[0,2]], feet_pos[:,1])

        print('Floor fit (for contact refinement) after ' + str(huber.n_iter_) + ' LBFGS iters!')
        coeff = huber.coef_
        intercept = huber.intercept_
        print('Coeffs + Intercept = (%f, %f, %f)' % (coeff[0], coeff[1], intercept))
        print('Number of outliers: %d' % (np.sum(huber.outliers_)))

        # print(velConstraints[:, feet_contact])
        og_vel_const = velConstraints[:, feet_contact].copy()

        # go through and figure out which contact labels were outliers
        feet_vel_constraints = velConstraints[:, feet_contact]
        fit_pts_cnt = 0
        for frame_idx in range(feet_vel_constraints.shape[0]):
            for foot_joint_idx in range(feet_vel_constraints.shape[1]):
                if feet_vel_constraints[frame_idx, foot_joint_idx] == 1:
                    # check if marked an outlier
                    if huber.outliers_[fit_pts_cnt]:
                        # set to out of contact
                        feet_vel_constraints[frame_idx, foot_joint_idx] = 0
                    fit_pts_cnt += 1

        velConstraints[:, feet_contact] = feet_vel_constraints

    #
    # Final stage optimize for foot placements
    #
    print('Now optimizing for foot placement...')
    projWeight = 1000.0 
    smoothWeightVel = 0.1 
    smoothWeightAcc = 0.5 
    dataWeight = 0.3
    velWeight = 10.0
    floorWeight = 10.0
    cur_sol = least_squares(fun_anim_for_projection, init_sol,
                            max_nfev=50,
                            verbose=2,
                            jac=jac_anim_for_projection_sparse,
                            gtol=1e-12,
                            bounds=[-np.inf, np.inf],
                            tr_solver = 'lsmr',
                            args=(skeleton, poses3D, root_pos, joints_2d_normalized, plane_normal, plane_point,
                                    proj_weights, data_weights, np.arange(num_joints), np.arange(num_joints),
                                    SMOOTH_WEIGHTS, velConstraints, projWeight, smoothWeightVel, smoothWeightAcc,
                                    dataWeight, velWeight, floorWeight))

    #output for all frames
    newPose3D = []
    projPose2D = []
    allRootPos = []

    #forward kinematics
    x = np.reshape(cur_sol.x, [num_frames, -1])
    root = x[:, :3]
    angles = x[:, 3:]
    anim = skeleton.copy()
    anim.orients.qs = skeleton.orients.qs.copy()
    anim.offsets = skeleton.offsets.copy()
    anim.positions = skeleton.positions.repeat(num_frames, axis=0)
    anim.rotations = Quaternions.from_euler(angles.reshape((num_frames, num_joints, 3)), order='xyz', world=True)
    anim.positions[:, 0] = root
    # save final animation
    BVH.save(os.path.join(save_dir, 'final_test.bvh'), anim, names)

    # get final 3d joints
    final_pos = Animation.positions_global(anim)
    newPose3D = 0 * final_pos
    for j in range(final_pos.shape[1]):
        newPose3D[:, j, :] = final_pos[:, BACKWARD_MAPPING[j], :]

    # get final 2d joints
    for frameNum in range(num_frames):
        varIndex = frameNum * num_joints * 3

        pts2D = poses2D[frameNum, :]

        projPt = np.zeros(pts2D.shape)
        for j in range(0, num_joints):
            corr2DPt = j 
            corr3DPt = j
            projPt[corr2DPt, 0] = camFocal[0] * (
                    newPose3D[frameNum, corr3DPt, 0] / newPose3D[frameNum, corr3DPt, 2]) + cam_center[0]
            projPt[corr2DPt, 1] = camFocal[1] * (
                    newPose3D[frameNum, corr3DPt, 1] / newPose3D[frameNum, corr3DPt, 2]) + cam_center[1]

        projPose2D.append(projPt)

    projPose2D = np.array(projPose2D)

    return (anim, newPose3D, projPose2D, plane_normal, plane_point, velConstraints)