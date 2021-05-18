import os, sys, gc
import argparse
from copy import copy

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from skimage import io, transform

import data.openpose_dataset as openpose_dataset
from data.openpose_dataset import OpenPoseDataset
from data.real_video_dataset import RealVideoDataset, TRAIN_DIM
from data.contact_data_utils import get_frame_paths

from utils import get_device, create_model, calculate_metrics, print_metrics, val_epoch, plot_confusion_mat, viz_full_video_simple, plot_accuracy_hist

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, required=True, help='Path to root of data directory.')
    parser.add_argument('--out', type=str, required=True, help='Directory to save outputs in.')
    parser.add_argument('--weights-path', type=str, help='Path to the weights file to use')

    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--window-size', type=int, default=9, help='Number of frames to use when predicting contact.')
    parser.add_argument('--pred-size', type=int, default=5, help='Will make contact predictions for this many of the middle frames within the window.')
    parser.add_argument('--classify-thresh', type=int, default=0.5, help='If above this threshold, will classify as in contact')
    parser.add_argument('--no-confidence', dest='use_confidence', action='store_false', help='If given, network will not use OpenPose confidence as input.')
    parser.set_defaults(use_confidence=True)
    parser.add_argument('--joint-set', type=str, default='lower', help='The OpenPose joint subset to give to network. From [lower, lower_knees, lower_ankles, lower_feet, upper, upper_hips, upper_knees, upper_ankles, full]')

    parser.add_argument('--cpu', dest='cpu', action='store_true', help='Force using the CPU rather than any detected GPUs.')
    parser.set_defaults(cpu=False)

    parser.add_argument('--viz', dest='viz', action='store_true', help='Visualize every result.')
    parser.set_defaults(viz=False)

    parser.add_argument('--full-video', dest='full_vid_eval', action='store_true', help='Applies the trained model in a sliding model over entire videos in the test set. Uses majority voting for final output.')
    parser.set_defaults(full_vid_eval=False)
    parser.add_argument('--real-data', dest='use_real_data', action='store_true', help='Indicates that the data path (--data) given is a dataset of real videos (rather than the synthetic dataset).')
    parser.set_defaults(use_real_data=False)
    parser.add_argument('--save-contacts', dest='save_contacts', action='store_true', help='Saves all predicted contact sequences if doing full video evaluation.')
    parser.set_defaults(save_contacts=False)

    flags = parser.parse_known_args()
    flags = flags[0]
    return flags


def val_full_video(dataloader, dataset, model, device, classify_thresh, pred_size, contacts_out_path=None, viz_out_path=None, fps=30):
    ''' 
    Evaluates each batch as if it contains the windows for every frame of a full video.
    Visualizes result if desired.
    '''
    model.eval()

    loss_sum = 0.0
    loss_count = 0
    confusion_count = np.zeros((pred_size, 4), dtype=int)
    merged_confusion = np.zeros((4), dtype=int)
    batch_size = -1
    target_idx = -1
    for batch_idx, batch_data in enumerate(dataloader):
        # prepare the data for this batch
        input_data = batch_data['joint2d']
        if batch_size == -1:
            batch_size = input_data.size()[0]
        input_data = input_data.to(device)
        have_contacts = False
        if 'contacts' in batch_data.keys():
            label_data = batch_data['contacts'].to(device)
            if target_idx == -1:
                target_idx = label_data.size()[1] // 2
            have_contacts = True
        # run model
        output_data = model(input_data) # B x contact_size x 4
        if have_contacts:
            loss = model.loss(output_data, label_data)
            for target_frame_idx in range(pred_size):
                n_tp, n_fp, n_fn, n_tn = model.accuracy(output_data, label_data, thresh=classify_thresh, tgt_frame=target_frame_idx)
                confusion_count[target_frame_idx] += np.array([n_tp, n_fp, n_fn, n_tn], dtype=int)
            # save loss
            loss_sum += torch.sum(loss).to('cpu').item()
            loss_count += loss.size()[0] * loss.size()[1] * loss.size()[2]

        # merge together to get full video labels
        model_predictions, model_probs = model.prediction(output_data)
        model_predictions = model_predictions.to('cpu').numpy() # B x contact_size x 4

        # sliding window through entire video to aggregate votes
        window_size = input_data.size()[1]
        vote_aggregation = np.zeros((batch_size + 2*(pred_size // 2), 4)) # only collect for frames we directly predict
        for window_start_idx in range(model_predictions.shape[0]):
            window_end_idx = window_start_idx + pred_size
            vote_aggregation[window_start_idx:window_end_idx] += model_predictions[window_start_idx]
        # define threshold for considering in contact
        # on edges there are less possible votes b/c were never a target frame
        # must account for this
        # don't need majority in order to be considered in contact (this pushes towards more false positives than negatives)
        vote_thresh = np.ones((vote_aggregation.shape[0])) * (((pred_size+1) / 2))
        for edge_offset in range(pred_size - 1):
            vote_thresh[edge_offset] = (edge_offset // 2) + 1
            vote_thresh[(-1-edge_offset)] = (edge_offset // 2) + 1
        # print(vote_thresh)
        vote_predictions = vote_aggregation >= vote_thresh.reshape((-1, 1))
        contact_preds = vote_predictions.astype(np.int)

        # # NOTE: uncomment this to turn off vote merging (majority voting)
        # # want contact predictions 
        # contact_preds = model_predictions[:,target_idx,:].copy() # B x 4
        # # still need edges
        # # take as much as we can from predictions
        # leading_preds = model_predictions[0,:target_idx,:].reshape((-1, 4))
        # tailing_preds = model_predictions[batch_size-1,target_idx+1:,:].reshape((-1, 4))
        # contact_preds = np.concatenate([leading_preds, contact_preds, tailing_preds], axis=0)

        # fill in the rest with copies
        contact_offset = (window_size - pred_size) // 2
        leading_pad = np.repeat(contact_preds[0].reshape((1, 4)), contact_offset, axis=0)
        tailing_pad = np.repeat(contact_preds[-1].reshape((1, 4)), contact_offset, axis=0)
        contact_preds = np.concatenate([leading_pad, contact_preds, tailing_pad], axis=0) # F x 4

        if have_contacts:
            # same thing for labels
            contact_label_data = label_data.to('cpu').numpy() # B x contact_size x 4
            contact_labels = contact_label_data[:,target_idx,:].copy() # B x 4
            leading_labels = contact_label_data[0,:target_idx,:].reshape((-1, 4))
            tailing_labels = contact_label_data[batch_size-1,target_idx+1:,:].reshape((-1, 4))
            contact_labels = np.concatenate([leading_labels, contact_labels, tailing_labels], axis=0)
            # fill in the rest with copies
            leading_pad = np.repeat(contact_labels[0].reshape((1, 4)), contact_offset, axis=0)
            tailing_pad = np.repeat(contact_labels[-1].reshape((1, 4)), contact_offset, axis=0)
            contact_labels = np.concatenate([leading_pad, contact_labels, tailing_pad], axis=0) # F x 4

            # evaluate accuracy after merging
            n_tp, n_fp, n_fn, n_tn = model.accuracy(torch.from_numpy(contact_preds.reshape((-1, 1, 4))).to(torch.float),\
                                                    torch.from_numpy(contact_labels.reshape((-1, 1, 4))).to(torch.float), \
                                                    thresh=0.5, tgt_frame=0)
            merged_confusion += np.array([n_tp, n_fp, n_fn, n_tn], dtype=int)

        # save predictions
        if contacts_out_path:
            video_name = batch_data['name'][0]
            contact_dir_out = os.path.join(contacts_out_path, video_name)
            if not os.path.exists(contact_dir_out):
                os.makedirs(contact_dir_out, exist_ok=True)
            contact_path_out = os.path.join(contact_dir_out, 'foot_contacts')
            true_seq_len = batch_data['seq_len'][0]
            # trim to actual seq_len
            save_contact_preds = contact_preds.astype(np.int)[:true_seq_len]
            np.save(contact_path_out, save_contact_preds)

        # visualization
        if have_contacts and viz_out_path:
            video_name = batch_data['name'][0]
            result_vid_path = os.path.join(viz_out_path, video_name.replace('/', '-') + '.mp4')
            print('Saving video to %s...' % (result_vid_path))

            frame_data = batch_data['frames'].to('cpu').numpy() # B x x H x W x 3
            _, H, W, _ = frame_data.shape
            # frame data is only target frames, need leading and trailing frames for whole first and last window
            leading_frames = []
            trailing_frames = []
            for frame_idx in range(model.window_size // 2):
                cur_frame_paths = None
                if isinstance(dataset, RealVideoDataset):
                    cur_frame_paths = dataset.frame_paths[batch_idx]
                else:
                    # synthetic dataset
                    cur_frame_paths = get_frame_paths(dataset.view_dirs[batch_idx])
                cur_lead_frame = io.imread(cur_frame_paths[frame_idx])[:,:,:3] # remove alpha
                cur_trail_frame = io.imread(cur_frame_paths[-(frame_idx+1)])[:,:,:3]
                if cur_lead_frame.shape[0] != H or cur_lead_frame.shape[1] != W:
                    cur_lead_frame = transform.resize(cur_lead_frame, (TRAIN_DIM[1], TRAIN_DIM[0]))
                    cur_trail_frame = transform.resize(cur_trail_frame, (TRAIN_DIM[1], TRAIN_DIM[0]))
                leading_frames.append(cur_lead_frame)
                trailing_frames.append(cur_trail_frame)
            leading_frames = np.stack(leading_frames, axis=0)
            trailing_frames = np.stack(trailing_frames[::-1], axis=0)
            frame_seq = np.concatenate([leading_frames, frame_data, trailing_frames], axis=0)

            # need 2d joint sequence
            joint_data = input_data.to('cpu').numpy() # B x window_size x J x 3
            window_tgt = joint_data.shape[1] // 2
            joint2d_seq = joint_data[:,window_tgt,:,:2].copy() # B x J x 2 (x,y)
            # unnormalize
            root_idx = openpose_dataset.OP_LOWER_JOINTS_MAP['MidHip']
            joint_trans_normalization = joint2d_seq[:,root_idx,:].copy()
            joint2d_seq[:,root_idx,:] -= joint_trans_normalization # zero it out so it's correct when added back in
            joint_trans_normalization = joint_trans_normalization.reshape((batch_size, 1, 2))
            joint2d_seq += joint_trans_normalization
            # need all frames
            num_joints = input_data.size()[2]
            leading_joint2d = joint_data[0,:window_tgt,:,:2].reshape((-1,num_joints,2))
            leading_joint2d += joint_data[0, window_tgt, root_idx, :2].reshape((1,1,2)) # unnormalize
            tailing_joint2d = joint_data[batch_size-1,window_tgt+1:,:,:2].reshape((-1,num_joints,2))
            tailing_joint2d += joint_data[batch_size-1, window_tgt, root_idx, :2].reshape((1,1,2)) # unnormalize
            joint2d_seq = np.concatenate([leading_joint2d, joint2d_seq, tailing_joint2d], axis=0)
            joint2d_seq *= dataset.get_joint_scaling()

            # now visualize
            viz_full_video_simple(frame_seq, joint2d_seq, contact_preds, contact_labels, show=False, save_path=result_vid_path, fps=fps)
            # viz_full_video_simple(frame_seq, joint2d_seq, contact_preds, contact_labels, show=True, save_path=None, fps=fps)

            frame_data = None
            frame_seq = None
            gc.collect()

    mean_loss = 0.0
    metrics = []
    merged_metrics = None
    if have_contacts:
        mean_loss = loss_sum / loss_count
        metrics = []
        for target_frame_idx in range(pred_size):
            metrics.append(calculate_metrics(confusion_count[target_frame_idx]))
        merged_metrics = calculate_metrics(merged_confusion)
    return mean_loss, metrics, merged_metrics

def test(flags, op_model=None, weights_path=None):
    data_root = flags.data
    window_size = flags.window_size
    pred_size = flags.pred_size
    batch_size = flags.batch_size
    out_dir = flags.out
    viz = flags.viz
    classify_thresh = flags.classify_thresh
    full_vid_eval = flags.full_vid_eval
    use_real_data = flags.use_real_data
    use_confidence = flags.use_confidence
    joint_set = flags.joint_set

    if not os.path.exists(data_root):
        print('Could not find test data at ' + data_root)
        return
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if viz and full_vid_eval:
        viz_path_out = os.path.join(out_dir, 'full_video_results')
        if not os.path.exists(viz_path_out):
            os.mkdir(viz_path_out)
    else:
        viz_path_out = None

    contacts_out_path = None
    if flags.save_contacts and full_vid_eval:
        contacts_out_path = os.path.join(out_dir, 'contact_results')
        if not os.path.exists(contacts_out_path):
            os.mkdir(contacts_out_path)

    # load training and validation data
    if use_real_data:
        test_dataset = RealVideoDataset(data_root, split='test', 
                                                   window_size=window_size,
                                                   contact_size=pred_size,
                                                   load_img=viz,
                                                   use_confidence=use_confidence,
                                                   joint_set=joint_set)
        fps = 24 # only affects visualization
    else:
        test_dataset = OpenPoseDataset(data_root, split='test',
                                                  window_size=window_size,
                                                  contact_size=pred_size,
                                                  load_img=viz,
                                                  overlap_test=full_vid_eval,
                                                  use_confidence=use_confidence,
                                                  joint_set=joint_set)
        fps = 30 # only affects visualization
    if full_vid_eval:
        batch_size =  test_dataset.get_num_test_windows_per_seq()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_joints = len(openpose_dataset.OP_JOINT_SUBSETS[test_dataset.joint_set])

    # create the model and load weights if need be
    device_str = 'cpu' if flags.cpu else None
    device = get_device(device_str)
    if not op_model:
        op_model = create_model(window_size, num_joints, pred_size, device, use_confidence=use_confidence)
        print('Loading weights...')
        if not os.path.exists(weights_path):
            print('Could not find weights at ' + weights_path + '!')
            return
        if device.type == 'cpu':
            saved_weights = torch.load(weights_path, map_location=torch.device('cpu'))
        else:
            saved_weights = torch.load(weights_path)
        op_model.load_state_dict(saved_weights)
        print('Done loading weights!')

    # run on test data
    if full_vid_eval:
        test_loss, test_metrics, merged_metrics = val_full_video(test_loader, test_dataset, op_model, device, classify_thresh, pred_size, 
                                                                  contacts_out_path=contacts_out_path,
                                                                  viz_out_path=viz_path_out,
                                                                  fps=fps)
    else:
        test_loss, test_metrics = val_epoch(test_loader, op_model, device, classify_thresh, pred_size)
    print('==================== TEST RESULTS ===========================================')
    print('Mean Loss: %0.3f' % (test_loss))
    if len(test_metrics) > 0:
        for tgt_frame_idx in range(pred_size):
            print('----- Pred Frame ' + str(tgt_frame_idx) + ' ------')
            print_metrics(test_metrics[tgt_frame_idx])
    print('=======================================================')
    if full_vid_eval and merged_metrics is not None:
        print('============== FULL VIDEO MERGED RESULTS ======================')
        print_metrics(merged_metrics)
        print('===============================================================')

    if len(test_metrics) > 0:
        all_accuracies = []
        for tgt_frame_idx in range(pred_size):
            accuracy, precision, recall, f1, cm = test_metrics[tgt_frame_idx]
            all_accuracies.append(accuracy)
            plot_confusion_mat(cm, os.path.join(out_dir, 'test_confusion_matrix%d.png' % (tgt_frame_idx)))
        plot_accuracy_hist(all_accuracies, os.path.join(out_dir, 'test_accuracy_hist.png'))
        if full_vid_eval:
            accuracy, precision, recall, f1, cm = merged_metrics
            plot_confusion_mat(cm, os.path.join(out_dir, 'test_confusion_matrix_merged.png'))


def main(flags):
    test(flags, weights_path=flags.weights_path)

if __name__=='__main__':
    flags = parse_args(sys.argv[1:])
    print(flags)
    main(flags)