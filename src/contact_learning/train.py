import os, sys
import argparse
from copy import copy

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import data.openpose_dataset as openpose_dataset
from data.openpose_dataset import OpenPoseDataset

from utils import get_device, create_model, calculate_metrics, print_metrics, plot_train_stats, val_epoch, plot_confusion_mat

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, required=True, help='Path to root of contact data directory.')
    parser.add_argument('--out', type=str, required=True, help='Directory to save outputs in (i.e. weights, training plots, etc..).')

    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--window-size', type=int, default=9, help='Number of frames to use when predicting contact.')
    parser.add_argument('--pred-size', type=int, default=5, help='Will make contact predictions for this many of the middle frames within the window.')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs for training.')
    parser.add_argument('--val-every', type=int, default=20, help='Number of epochs between validations for early stopping')
    parser.add_argument('--classify-thresh', type=int, default=0.5, help='If above this threshold, will classify as in contact')
    parser.add_argument('--no-confidence', dest='use_confidence', action='store_false', help='If given, network will not use OpenPose confidence as input.')
    parser.set_defaults(use_confidence=True)
    parser.add_argument('--joint-set', type=str, default='lower', help='The OpenPose joint subset to give to network. From [lower, lower_knees, lower_ankles, lower_feet, upper, upper_hips, upper_knees, upper_ankles, full]')

    parser.add_argument('--cpu', dest='cpu', action='store_true', help='Force using the CPU rather than any detected GPUs.')
    parser.set_defaults(cpu=False)

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for ADAM')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for ADAM')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for ADAM')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon rate for ADAM')
    parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay on params.')

    flags = parser.parse_known_args()
    flags = flags[0]
    return flags


def train(flags):
    data_root = flags.data
    window_size = flags.window_size
    pred_size = flags.pred_size
    batch_size = flags.batch_size
    out_dir = flags.out
    num_epochs = flags.epochs
    val_every = flags.val_every
    classify_thresh = flags.classify_thresh
    # optim args
    lr = flags.lr
    betas = (flags.beta1, flags.beta2)
    eps = flags.eps
    weight_decay = flags.decay
    use_confidence = flags.use_confidence
    joint_set = flags.joint_set

    if not os.path.exists(data_root):
        print('Could not find training data at ' + data_root)
        return
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    weights_out_path = os.path.join(out_dir, 'op_only_weights.pth')
    best_weights_out_path = os.path.join(out_dir, 'op_only_weights_BEST.pth')

    # load training and validation data
    train_dataset = OpenPoseDataset(data_root, split='train', window_size=window_size, contact_size=pred_size, use_confidence=use_confidence, joint_set=joint_set)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataset = OpenPoseDataset(data_root, split='val', window_size=window_size, contact_size=pred_size, use_confidence=use_confidence, joint_set=joint_set)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    num_joints = len(openpose_dataset.OP_JOINT_SUBSETS[train_dataset.joint_set])

    # create the model and optimizer
    device_str = 'cpu' if flags.cpu else None
    device = get_device(device_str)
    op_model = create_model(window_size, num_joints, pred_size, device, use_confidence=use_confidence)
    op_optim = optim.Adam(op_model.parameters(), lr=lr, betas=betas, \
                                    eps=eps, weight_decay=weight_decay)

    model_parameters = filter(lambda p: p.requires_grad, op_model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Num model params: ' + str(params))

    # viz stats
    train_steps = []
    train_losses = []
    train_accs = []
    val_steps = []
    val_losses = []
    val_accs = []

    # train
    loss_sum = 0.0
    loss_count = 0
    best_val_f1 = -float('inf')
    confusion_count = np.zeros((4), dtype=int)
    for epoch_idx in range(num_epochs): 
        for batch_idx, batch_data in enumerate(train_loader):
            # prepere the data for this batch
            input_data = batch_data['joint2d'].to(device)
            label_data = batch_data['contacts'].to(device)

            # zero the gradients
            op_optim.zero_grad()
            # forward + backward + optimize
            output_data = op_model(input_data)
            loss = op_model.loss(output_data, label_data)
            n_tp, n_fp, n_fn, n_tn = op_model.accuracy(output_data, label_data, thresh=classify_thresh)
            loss = torch.mean(loss)
            loss.backward()
            op_optim.step()

            loss_sum += loss.to('cpu').item()
            loss_count += 1
            confusion_count += np.array([n_tp, n_fp, n_fn, n_tn], dtype=int)

        if epoch_idx % 5 == 0:
            print('=================== TRAIN (' + str(epoch_idx+1) + ' epochs) ================================================')
            mean_loss = loss_sum / loss_count
            print('Mean loss: %0.3f' % (mean_loss))
            loss_sum = 0.0
            loss_count = 0

            metrics = calculate_metrics(confusion_count)
            cur_acc, _, _, _, _ = metrics
            print_metrics(metrics)
            confusion_count = np.zeros((4), dtype=int)
            print('======================================================================================')

            train_steps.append(epoch_idx * len(train_loader) + batch_idx)
            train_losses.append(mean_loss)
            train_accs.append(cur_acc)

            # save plot
            plot_train_stats((train_steps, train_losses, train_accs), \
                             (val_steps, val_losses, val_accs), \
                             out_dir, accuracy_metrics=metrics)


        if epoch_idx % val_every == 0:
            # run on the validation data
            print('==================== VALIDATION (' + str(epoch_idx+1) + ' epochs) ===========================================')
            val_loss, val_metrics = val_epoch(val_loader, op_model, device, classify_thresh, pred_size)
            print('Mean Loss: %0.3f' % (val_loss))
            
            for tgt_frame_idx in range(pred_size):
                print('----- Pred Frame ' + str(tgt_frame_idx) + ' ------')
                print_metrics(val_metrics[tgt_frame_idx])
            val_acc, _, _, _, _ = val_metrics[pred_size // 2] # only want accuracy for middle target
            print('======================================================================================')
            op_model.train()

            val_steps.append(epoch_idx * len(train_loader) + batch_idx)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # save confusion matrix
            for tgt_frame_idx in range(pred_size):
                accuracy, precision, recall, f1, cm = val_metrics[tgt_frame_idx]
                plot_confusion_mat(cm, os.path.join(out_dir, 'val_confusion_matrix_%d.png' % (tgt_frame_idx)))

            # also save model weights
            print('Saving checkpoint...')
            torch.save(op_model.state_dict(), weights_out_path)

            # check if this is the best so far and save (in terms of f1 score)
            if f1 > best_val_f1:
                best_val_f1 = f1
                print('Saving best model so far...')
                torch.save(op_model.state_dict(), best_weights_out_path)

    # save final model
    print('Saving final checkpoint...')
    torch.save(op_model.state_dict(), os.path.join(out_dir, 'op_only_weights_FINAL.pth'))
    # save plot
    metrics = calculate_metrics(confusion_count)
    plot_train_stats((train_steps, train_losses, train_accs), \
                        (val_steps, val_losses, val_accs), \
                        out_dir, accuracy_metrics=metrics)
    print('FINISHED Training!')


def main(flags):
    train(flags)

if __name__=='__main__':
    flags = parse_args(sys.argv[1:])
    print(flags)
    main(flags)