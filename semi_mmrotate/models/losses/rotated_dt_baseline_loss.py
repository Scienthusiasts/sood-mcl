#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/18 21:01
# @Author : WeiHua
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models import ROTATED_LOSSES, build_loss
from mmrotate.core import build_bbox_coder
from mmdet.core.anchor.point_generator import MlvlPointGenerator
import numpy as np
from mmrotate.core import poly2obb_np
import cv2
import mmcv

INF = 1e8
CLASSES = ('large-vehicle', 'swimming-pool', 'helicopter', 'bridge', 'plane', 
               'ship', 'soccer-ball-field', 'basketball-court', 'ground-track-field', 'small-vehicle', 
               'baseball-diamond', 'tennis-court', 'roundabout', 'storage-tank', 'harbor', 'container-crane')


@ROTATED_LOSSES.register_module()
class RotatedDTBLLoss(nn.Module):
    def __init__(self, cls_channels=16, loss_type='origin', bbox_loss_type='l1'):
        super(RotatedDTBLLoss, self).__init__()
        self.cls_channels = cls_channels
        assert bbox_loss_type in ['l1', 'iou']
        self.bbox_loss_type = bbox_loss_type
        self.bbox_coder = build_bbox_coder(dict(type='DistanceAnglePointCoder', angle_version='le90'))
        self.prior_generator = MlvlPointGenerator([8, 16, 32, 64, 128])
        if self.bbox_loss_type == 'l1':
            self.bbox_loss = nn.SmoothL1Loss(reduction='none')
        else:
            self.bbox_coder = build_bbox_coder(dict(type='DistanceAnglePointCoder', angle_version='le90'))
            self.prior_generator = MlvlPointGenerator([8, 16, 32, 64, 128])
            self.bbox_loss = build_loss(dict(type='RotatedIoULoss', reduction='none'))
        self.loss_type = loss_type


    def convert_shape(self, logits):
        cls_scores, bbox_preds, angle_preds, centernesses = logits
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses)

        batch_size = cls_scores[0].shape[0]   
        cls_scores = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_channels) for x in cls_scores
        ], dim=1).view(-1, self.cls_channels)
        bbox_preds = torch.cat([
            torch.cat([x, y], dim=1).permute(0, 2, 3, 1).reshape(batch_size, -1, 5) for x, y in
            zip(bbox_preds, angle_preds)
        ], dim=1).view(-1, 5)
        centernesses = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1, 1) for x in centernesses
        ], dim=1).view(-1, 1)
        return cls_scores, bbox_preds, centernesses


    def pseudoLabelSelection(self, mode:str, t_cls_scores, k:float):
        '''伪标签筛选
        '''
        with torch.no_grad():
            # Region Selection
            topk = int(t_cls_scores.size(0) * k)
            teacher_probs = t_cls_scores.sigmoid()
            # NOTE:yan add S_dps to tensorboard
            S_dps = teacher_probs.mean()
            # max_vals提取最大的类别置信度
            max_vals = torch.max(teacher_probs, 1)[0]
            # 从大到小排序
            sorted_vals, sorted_inds = torch.topk(max_vals, t_cls_scores.size(0))
            mask = torch.zeros_like(max_vals)
            # 前topk个元素为正样本
            mask[sorted_inds[:topk]] = 1.
            fg_num = sorted_vals[:topk].sum()
            pos_mask = mask > 0.
        
        return pos_mask, fg_num, S_dps


    def forward(self, teacher_logits, student_logits, ratio=0.01, img_metas=None, **kwargs):

        t_cls_scores, t_bbox_preds, t_centernesses = self.convert_shape(teacher_logits)
        s_cls_scores, s_bbox_preds, s_centernesses = self.convert_shape(student_logits)
        '''伪标签筛选'''
        pos_mask, fg_num, S_dps = self.pseudoLabelSelection('topk', t_cls_scores, ratio)

        '''损失'''
        loss_cls = QFLv2(
            s_cls_scores.sigmoid(),
            t_cls_scores.sigmoid(),
            weight=pos_mask,
            reduction="sum",
        ) / fg_num
        if self.bbox_loss_type == 'l1':
            loss_bbox = (self.bbox_loss(
                s_bbox_preds[pos_mask],
                t_bbox_preds[pos_mask],
            ) * t_centernesses.sigmoid()[pos_mask]).mean()
        else:
            all_level_points = self.prior_generator.grid_priors(
                [featmap.size()[-2:] for featmap in teacher_logits[0]],
                dtype=s_bbox_preds.dtype,
                device=s_bbox_preds.device)
            flatten_points = torch.cat(
                [points.repeat(len(teacher_logits[0][0]), 1) for points in all_level_points])
            s_bbox_preds = self.bbox_coder.decode(flatten_points, s_bbox_preds)[pos_mask]
            t_bbox_preds = self.bbox_coder.decode(flatten_points, t_bbox_preds)[pos_mask]
            loss_bbox = self.bbox_loss(
                s_bbox_preds,
                t_bbox_preds,
            ) * t_centernesses.sigmoid()[pos_mask]
            nan_indexes = ~torch.isnan(loss_bbox)
            if nan_indexes.sum() == 0:
                loss_bbox = torch.zeros(1, device=s_cls_scores.device).sum()
            else:
                loss_bbox = loss_bbox[nan_indexes].mean()

        loss_centerness = F.binary_cross_entropy(
            s_centernesses[pos_mask].sigmoid(),
            t_centernesses[pos_mask].sigmoid(),
            reduction='mean'
        )

        unsup_losses = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            # NOTE:yan add S_dps to tensorboard
            S_dps=S_dps
        )

        return unsup_losses


def QFLv2(pred_sigmoid, teacher_sigmoid, weight=None, beta=2.0, reduction='mean'):
    # all goes to 0
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    # 一开始假设所有样本都是负样本, 因此实际上有对负样本计算损失, 对应的标签是全0
    loss = F.binary_cross_entropy(pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    # positive goes to bbox quality

    # 这句话有时候会报错, 不知道为啥(内容如下):
    # ... ...
    # ../aten/src/ATen/native/cuda/Loss.cu:92: operator(): block: [79,0,0], thread: [118,0,0] Assertion `input_val >= zero && input_val <= one` failed.
    # RuntimeError: numel: integer multiplication overflow 
    try:
        pt = teacher_sigmoid[weight] - pred_sigmoid[weight]
    except:
        print(weight.shape, teacher_sigmoid.shape)
        print(torch.isnan(weight).any(), torch.isnan(teacher_sigmoid).any(), torch.isnan(pred_sigmoid).any())
        pt = teacher_sigmoid[weight] - pred_sigmoid[weight]
    # 在所有样本都是负样本的基础上更新那些正样本对应位置为正样本损失
    loss[weight] = F.binary_cross_entropy(pred_sigmoid[weight], teacher_sigmoid[weight], reduction='none') * pt.pow(beta)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss