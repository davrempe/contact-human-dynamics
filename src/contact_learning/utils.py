import os, sys
from copy import copy
from pathlib import Path
import shutil, subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from models.openpose_only import OpenPoseModel
import data.openpose_dataset as openpose_dataset

def val_epoch(dataloader, model, device, classify_thresh, pred_size):
    ''' Evaluates the model on the entirety of the given dataloader '''
    model.eval()

    loss_sum = 0.0
    loss_count = 0
    # get confusion matrix for all predicted frames
    confusion_count = np.zeros((pred_size, 4), dtype=int)
    batch_size = -1
    target_idx = -1
    for batch_idx, batch_data in enumerate(dataloader):
        # prepare the data for this batch
        input_data = batch_data['joint2d']
        if batch_size == -1:
            batch_size = input_data.size()[0]
        input_data = input_data.to(device)
        label_data = batch_data['contacts'].to(device)
        if target_idx == -1:
            target_idx = label_data.size()[1] // 2
        # run model
        output_data = model(input_data)
        loss = model.loss(output_data, label_data)
        for target_frame_idx in range(pred_size):
            n_tp, n_fp, n_fn, n_tn = model.accuracy(output_data, label_data, thresh=classify_thresh, tgt_frame=target_frame_idx)
            confusion_count[target_frame_idx] += np.array([n_tp, n_fp, n_fn, n_tn], dtype=int)
        # save loss
        loss_sum += torch.sum(loss).to('cpu').item()
        loss_count += loss.size()[0] * loss.size()[1] * loss.size()[2]

    mean_loss = loss_sum / loss_count
    metrics = []
    for target_frame_idx in range(pred_size):
        metrics.append(calculate_metrics(confusion_count[target_frame_idx]))
    return mean_loss, metrics

def get_device(device_str=None):
    if device_str == 'cpu':
        print('Using CPU as requested...')
    else:
        device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device_str == 'cpu':
            print('No detected GPUs, using CPU...')
        else:
            print('Detected GPU(s)!')
    device = torch.device(device_str)
    return device


def create_model(window_size, num_joints, pred_size, device, use_confidence=True, use_random=False):
    # create the model and move to GPU(s) if needed
    feat_size = 3
    if not use_confidence:
        feat_size = 2
    op_model = OpenPoseModel(window_size, num_joints, pred_size, feat_size)
    if torch.cuda.device_count() > 1:
        print("Detected ", torch.cuda.device_count(), " available GPUs...using all of them.")
        op_model = nn.DataParallel(op_model)
    op_model.to(device)
    return op_model

def calculate_metrics(confusion_count):
    ''' 
    Calculates accuracy, precision, recall, F1 score, and confusion matrix for the given confusion
    count which should be an np.array([n_tp, n_fp, n_fn, n_tn]).
    '''
    n_tp, n_fp, n_fn, n_tn = confusion_count
    n_total = np.sum(confusion_count)

    accuracy = (n_tp + n_tn) / n_total
    if n_tp + n_fp == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = n_tp / (n_tp + n_fp)
        recall = n_tp / (n_tp + n_fn)
        f1 = 2 * ((precision*recall) / (precision+recall))

    # print(precision)
    # print(recall)

    confusion_mat = confusion_count.reshape((2,2)) / n_total

    return accuracy, precision, recall, f1, confusion_mat

def print_metrics(metrics):
    ''' Metrics is (accuracy, precision, recall, f1, confusion_mat), '''
    accuracy, precision, recall, f1, confusion_mat = metrics
    print('Accuracy: %.3f' % (accuracy))
    print('Precision: %.3f' % (precision))
    print('Recall: %.3f' % (recall))
    print('F1 Score: %.3f' % (f1))
    print('Confusion Matrix:')
    print('                 + Actual -   ')
    print(' Predicted  + | %.3f  %.3f |' % (confusion_mat[0,0], confusion_mat[0,1]))
    print('            - | %.3f  %.3f |' % (confusion_mat[1,0], confusion_mat[1,1]))

def plot_accuracy_hist(accuracies, out_path):
    fig = plt.figure()
    plt.bar(np.arange(len(accuracies), dtype=int), np.array(accuracies))
    plt.title('Predicted Frame Accuracy')
    plt.xlabel('Frame Idx')
    plt.ylabel('Accuracy')
    plt.ylim((0.8, 1.0))

    plt.savefig(out_path)
    # plt.show()
    plt.close(fig)

def plot_confusion_mat(cm, out_path):
    accuracy = cm[0, 0] + cm[1, 1]
    title = 'Normalized Confusion Matrix (acc=%.3f)' % (accuracy)
    x_classes = ['in contact', 'no contact']
    y_classes = copy(x_classes)
    y_classes.reverse()
    cmap = plt.cm.Blues

    centers = [0.5, 1.5, 0.5, 1.5]
    dx, = np.diff(centers[:2])/(cm.shape[1]-1)
    dy, = -np.diff(centers[2:])/(cm.shape[0]-1)
    extent = [centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]

    fig, ax = plt.subplots(figsize=(9,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, extent=extent)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(centers[0], centers[1]+dx, dx),
           yticks=np.arange(centers[0], centers[1]+dx, dx),
           # ... and label them with the respective list entries
           xticklabels=x_classes, yticklabels=y_classes,
           title=title,
           ylabel='Predicted',
           xlabel='Actual')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(centers[0] + j*dx, centers[1] - i*dx, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    plt.savefig(out_path)
    # plt.show()
    plt.close(fig)


def plot_train_stats(train_progress, val_progress, out_dir, accuracy_metrics=None):
    ''' plots and saves the current training curve and other related statistics '''
    # plot training and validation curves
    train_steps, train_losses, train_accs = train_progress
    val_steps, val_losses, val_accs = val_progress

    fig = plt.figure(figsize=(12, 8))

    plt.plot(np.array(train_steps), np.array(train_losses), '-', label='train loss')
    plt.plot(np.array(train_steps), np.array(train_accs), '-', label='train acc')

    plt.plot(np.array(val_steps), np.array(val_losses), '-', label='val loss')
    plt.plot(np.array(val_steps), np.array(val_accs), '-', label='val acc')

    plt.xlabel('optim steps')

    plt.legend()
    plt.title('OpenPose-Only Training')
    plt.savefig(os.path.join(out_dir, 'train_curve.png'))
    plt.close(fig)

    if accuracy_metrics is None:
        return

    # plot confusion matrix
    accuracy, precision, recall, f1, cm = accuracy_metrics
    plot_confusion_mat(cm, os.path.join(out_dir, 'train_confusion_matrix.png'))

def viz_video_opencv(frame_seq, joint2d_seq, contact_preds, contact_labels, show=False, save_path=None, fps=30):
    '''
    Visualize video and contacts by opencv
    '''
    lheel_joints = [openpose_dataset.OP_LOWER_JOINTS_MAP['LHeel'], openpose_dataset.OP_LOWER_JOINTS_MAP['LAnkle']]
    ltoe_joints = [openpose_dataset.OP_LOWER_JOINTS_MAP['LBigToe'], openpose_dataset.OP_LOWER_JOINTS_MAP['LSmallToe']]
    rheel_joints = [openpose_dataset.OP_LOWER_JOINTS_MAP['RHeel'], openpose_dataset.OP_LOWER_JOINTS_MAP['RAnkle']]
    rtoe_joints = [openpose_dataset.OP_LOWER_JOINTS_MAP['RBigToe'], openpose_dataset.OP_LOWER_JOINTS_MAP['RSmallToe']]
    vis_joints = lheel_joints + ltoe_joints +  rheel_joints +  rtoe_joints

    # mp4v cannot render in webpage, convert to h264 later
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_height, frame_width = frame_seq.shape[1:3]
    temp_path = str(Path(save_path).with_name(Path(save_path).stem + "_temp.mp4"))
    out = cv2.VideoWriter(temp_path, fourcc, fps, (frame_width, frame_height))
    # convert to cv2
    frame_seq = (frame_seq*255).astype(np.uint8)
    frame_seq = frame_seq[:, :, :, ::-1].copy()

    frame_idx = 0
    while frame_idx < frame_seq.shape[0]:
        frame = frame_seq[frame_idx]
        for idx, joint in enumerate(vis_joints):
            joint_pos = joint2d_seq[frame_idx, joint].astype(int)
            if contact_preds[frame_idx, idx//2] == 1:
                cv2.circle(frame, (joint_pos[0], joint_pos[1]), 3, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (joint_pos[0], joint_pos[1]), 3, (0, 255, 0), -1)
        out.write(frame)
        frame_idx += 1
    out.release()
    subprocess.run(f"ffmpeg -i {temp_path} -vcodec libx264 -f mp4 {save_path}", shell=True)

def viz_full_video_simple(frame_seq, joint2d_seq, contact_preds, contact_labels, show=False, save_path=None, fps=30):
    '''
    Viz full video and contacts.
    '''
    fig = plt.figure(figsize=(8, 4.5), dpi=100)
    ax = []
    ax.append(plt.subplot(111))

    # first video frames
    frame_height, frame_width = frame_seq.shape[1:3]
    ax[0].set_xlim(0, frame_width)
    ax[0].set_ylim(frame_height, 0)
    ax[0].axis('off')

    lheel_joints = [openpose_dataset.OP_LOWER_JOINTS_MAP['LHeel'], openpose_dataset.OP_LOWER_JOINTS_MAP['LAnkle']]
    rheel_joints = [openpose_dataset.OP_LOWER_JOINTS_MAP['RHeel'], openpose_dataset.OP_LOWER_JOINTS_MAP['RAnkle']]
    ltoe_joints = [openpose_dataset.OP_LOWER_JOINTS_MAP['LBigToe'], openpose_dataset.OP_LOWER_JOINTS_MAP['LSmallToe']]
    rtoe_joints = [openpose_dataset.OP_LOWER_JOINTS_MAP['RBigToe'], openpose_dataset.OP_LOWER_JOINTS_MAP['RSmallToe']]

    joint_color = 'lime'
    bone_color = 'blue'
    contact_color = 'red'

    num_joints = joint2d_seq.shape[1]

    tgt_bones = []
    tgt_bones_overlay = []
    pred_bones = []
    pred_bones_overlay = []
    for i in range(num_joints - 1):
        pred_bones_overlay.append(ax[0].plot([0,0], [0,0], color=bone_color, lw=2)[0])

    tgt_joints = []
    tgt_joints_overlay = []
    pred_joints = []
    pred_joints_overlay = []
    for i in range(num_joints):
        pred_joints_overlay.append(ax[0].plot([0] ,[0], 'o', color=joint_color)[0])
    
    parents = openpose_dataset.OP_LOWER_PARENTS

    def animate(i):
        # video
        ax[0].imshow(frame_seq[i])
        # bones
        for j in range(1,num_joints):
            cur_joint = joint2d_seq[i, j]
            cur_par_joint = joint2d_seq[i, parents[j]]

            pred_bones_overlay[j-1].set_data(
                [ cur_joint[0],     cur_par_joint[0]],
                [cur_joint[1],       cur_par_joint[1]])
        # joints
        for j in range(num_joints):
            cur_joint = joint2d_seq[i, j]
            pred_joints_overlay[j].set_data([cur_joint[0]], [cur_joint[1]])

            # we have a contact label
            contact_idx = i
            # color based on contacts [left_heel, left_toes, right_heel, right_toes]
            if contact_preds[contact_idx, 0] and j in lheel_joints:
                pred_joints_overlay[j].set_color(contact_color)
            elif contact_preds[contact_idx, 1] and j in ltoe_joints:
                pred_joints_overlay[j].set_color(contact_color)
            elif contact_preds[contact_idx, 2] and j in rheel_joints:
                pred_joints_overlay[j].set_color(contact_color)
            elif contact_preds[contact_idx, 3] and j in rtoe_joints:
                pred_joints_overlay[j].set_color(contact_color)
            else:
                pred_joints_overlay[j].set_color(joint_color)

        return []

    plt.tight_layout()

    if save_path != None:
        base_path = '.'.join(save_path.split('.')[:-1])
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        print('Rendering frames...')
        for i in range(frame_seq.shape[0]):
            animate(i)
            plt.savefig(os.path.join(base_path, 'frame_%06d.png' % (i)))

        plt.close(fig)

        print('Processing video...')
        subprocess.run(['ffmpeg', '-r', str(fps), '-i', base_path+'/frame_%06d.png', \
                            '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', save_path])
        # clean up frames
        shutil.rmtree(base_path)

    if show:
        ani = animation.FuncAnimation(fig, animate, np.arange(frame_seq.shape[0]), interval=33.33)
        plt.show()

    plt.close(fig)
