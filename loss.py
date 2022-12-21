import torch as t
from torch import nn


class Loss(nn.Module):

    def __init__(self, lamda_size, lamda_off, num_classes, alpha, beta):
        super(Loss, self).__init__()
        self.lamda_size = lamda_size
        self.lamda_off = lamda_off
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

    def forward(self, model_output, target):
        heatmap_output = model_output[:, :self.num_classes, :, :]
        heatmap_gt = target[:, :self.num_classes, :, :]
        offset_output = model_output[:, self.num_classes:self.num_classes + 2, :, :]
        offset_gt = target[:, self.num_classes:self.num_classes + 2, :, :]
        size_output = model_output[:, self.num_classes + 2:self.num_classes + 4, :, :]
        size_gt = target[:, self.num_classes + 2:self.num_classes + 4, :, :]
        heatmap_loss = self.get_heatmap_loss(heatmap_output, heatmap_gt)
        offset_loss = self.get_offset_loss(offset_output, offset_gt)
        size_loss = self.get_size_loss(size_output, size_gt)
        loss = heatmap_loss + self.lamda_size * size_loss + self.lamda_off * offset_loss
        return loss, heatmap_loss, size_loss, offset_loss

    def get_heatmap_loss(self, heatmap_output, heatmap_gt):
        pos_idxs = heatmap_gt == 1
        neg_idxs = heatmap_gt != 1
        key_point_count_of_every_image = t.sum(pos_idxs, dim=[1, 2, 3])
        total_loss = 0
        for i in range(heatmap_output.size()[0]):
            pos_idx = pos_idxs[i]
            neg_idx = neg_idxs[i]
            key_point_count = key_point_count_of_every_image[i]
            heatmap_output_sample = heatmap_output[i]
            heatmap_gt_sample = heatmap_gt[i]
            loss_pos = -t.sum(t.pow((1 - heatmap_output_sample[pos_idx]), self.alpha) * t.log(heatmap_output_sample[pos_idx] + 0.00001))
            loss_neg = -t.sum(t.pow((1 - heatmap_gt_sample[neg_idx]), self.beta) * t.pow(heatmap_output_sample[neg_idx], self.alpha) * t.log(1 - heatmap_output_sample[neg_idx] + 0.00001))
            loss = (loss_pos + loss_neg) / key_point_count
            total_loss += loss
        heatmap_loss = total_loss / heatmap_output.size()[0]
        return heatmap_loss

    def get_offset_loss(self, offset_output, offset_gt):
        total_loss = 0
        for i in range(offset_output.size()[0]):
            offset_output_sample = offset_output[i]
            offset_gt_sample = offset_gt[i]
            idx = t.norm(offset_gt_sample, p=1, dim=0) != 0
            l1_of_every_pix = t.norm(offset_output_sample - offset_gt_sample, p=1, dim=0)[idx]
            N = idx.sum()
            total_loss += (l1_of_every_pix.sum() / (N + 0.0001))
        offset_loss = total_loss / offset_output.size()[0]
        return offset_loss

    def get_size_loss(self, size_output, size_gt):
        total_loss = 0
        for i in range(size_output.size()[0]):
            size_output_sample = size_output[i]
            size_gt_sample = size_gt[i]
            idx = t.norm(size_gt_sample, p=1, dim=0) != 0
            l1_of_every_pix = t.norm(size_output_sample - size_gt_sample, p=1, dim=0)[idx]
            N = idx.sum()
            total_loss += (l1_of_every_pix.sum() / (N + 0.0001))
        size_loss = total_loss / size_output.size()[0]
        return size_loss