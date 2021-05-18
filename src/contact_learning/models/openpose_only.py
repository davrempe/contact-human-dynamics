import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

sys.path.extend(['contact_learning'])
from data.openpose_dataset import OP_LOWER_JOINTS_MAP

TORCH_VER = torch.__version__

class OpenPoseModel(nn.Module):
    def __init__(self, window_size, joints, pred_size, feat_size):
        super(OpenPoseModel, self).__init__()
        self.window_size = window_size
        self.contact_size = pred_size
        self.feat_size = feat_size
        #
        # Losses
        #
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.sigmoid = nn.Sigmoid()

        #
        # Create the model
        #
        self.model = nn.Sequential(
                        nn.Linear(window_size*joints*self.feat_size, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Linear(512, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Dropout(p=0.3),
                        nn.Linear(128, 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Linear(32, 4*pred_size)
                    )
        # initialize weights
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        # data_batch is B x N x J x 3(or 2 if no confidence)
        B, N, J, F = x.size()
        # flatten to single data vector
        x = x.view(B, N*J*F)
        # run model
        x = self.model(x)
        return x.view(B, self.contact_size, 4)

    def loss(self, outputs, labels):
        ''' Returns the loss value for the given network output '''
        B, N, _ = outputs.size()
        outputs = outputs.view(B, N*4)

        B, N, _ = labels.size()
        labels = labels.view(B, N*4)

        loss_flat = self.bce_loss(outputs, labels)
        loss = loss_flat.view(B, N, 4)

        return loss

    def prediction(self, outputs, thresh=0.5):
        probs = self.sigmoid(outputs)
        pred = probs > thresh
        return pred, probs

    def accuracy(self, outputs, labels, thresh=0.5, tgt_frame=None):
        ''' Calculates confusion matrix counts for TARGET (middle) FRAME ONLY'''
        # threshold to classify
        pred, _ = self.prediction(outputs, thresh)

        if tgt_frame is None:
            tgt_frame = self.contact_size // 2

        # only want to evaluate accuracy of middle frame
        pred = pred[:, tgt_frame, :]
        if TORCH_VER == '1.0.0' or TORCH_VER == '1.1.0':
            pred = pred.byte()
        else:
            # 1.2.0
            pred = pred.to(torch.bool)
        labels = labels[:, tgt_frame, :]
        if TORCH_VER == '1.0.0' or TORCH_VER == '1.1.0':
            labels = labels.byte()
        else:
            labels = labels.to(torch.bool)

        # counts for confusion matrix
        # true positive (pred contact, labeled contact)
        true_pos = pred & labels
        true_pos_cnt = torch.sum(true_pos).to('cpu').item()
        # false positive (pred contact, not lebeled contact)
        false_pos = pred & ~(labels)
        false_pos_cnt = torch.sum(false_pos).to('cpu').item()
        # false negative (pred no contact, labeled contact)
        false_neg = ~(pred) & labels
        false_neg_cnt = torch.sum(false_neg).to('cpu').item()
        # true negative (pred no contact, no labeled contact)
        true_neg = (~pred) & (~labels)
        true_neg_cnt = torch.sum(true_neg).to('cpu').item()

        return true_pos_cnt, false_pos_cnt, false_neg_cnt, true_neg_cnt