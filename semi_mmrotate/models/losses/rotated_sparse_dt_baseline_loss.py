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
from mmrotate.core import poly2obb_np, obb2poly, poly2obb, obb2xyxy
import cv2
import mmcv
# yan
import copy
import os
import matplotlib.pyplot as plt
import time
from custom.visualize import OpenCVDrawBox
from custom.utils import *
from custom.visualize import *
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from mmcv.ops import box_iou_quadri, box_iou_rotated
from custom.loss import QFLv2



@ROTATED_LOSSES.register_module()
class RotatedSparseDTBLLoss(nn.Module):
    def __init__(self, p_selection:dict, cls_channels=16, loss_type='origin', bbox_loss_type='l1'):
        super(RotatedSparseDTBLLoss, self).__init__()
        self.nc = cls_channels
        assert bbox_loss_type in ['l1', 'iou']
        self.bbox_loss_type = bbox_loss_type
        self.bbox_coder = build_bbox_coder(dict(type='DistanceAnglePointCoder', angle_version='le90'))
        self.prior_generator = MlvlPointGenerator([8, 16, 32, 64, 128])
        if self.bbox_loss_type == 'l1':
            self.bbox_loss = nn.SmoothL1Loss(reduction='none')
        # NOTE: modified by yan:
        if p_selection['mode'] == 'global_w':
            self.iou_loss = build_loss(dict(type='RotatedIoULoss', loss_weight=1.0, linear=True, reduction='none'))
        self.loss_type = loss_type
        # 伪标签筛选策略超参, added by yan
        self.p_selection = p_selection
        # QFLv2损失
        self.QFLv2 = QFLv2()
        self.riou_loss = build_loss(dict(type='RotatedIoULoss', reduction='none'))



    def pseudoLabelSelection(self, mode:str, t_cls_scores, t_centernesses, ratio:float, beta:float):
        '''伪标签筛选 added by yan
            Args:
                mode:                 伪标签筛选策略 ('topk', 'top_dps', 'catwise_top_dps')
                t_cls_scores:         网络预测整张特征图的分类置信度 [bs * h * w, cat_num] (经过refine)
                t_centernesses:
                ratio:                当mode=='topk'时, 这个参数代表top k% 的 k%
                beta:                 当mode=='top_dps'时, beta为S_pds的权重系数
            Returns:
                pos_mask: 正样本mask 
                fg_num:   正样本的数量
                S_pds:    当前batch的平均联合置信度
        '''
        # 分类置信度(refine)
        teacher_probs = t_cls_scores.sigmoid()
        # t_scores, t_pred提取最大的类别置信度和对应的类别索引 [bs * h * w, cat_num] -> [bs * h * w], [bs * h * w]
        t_scores, t_pred = torch.max(teacher_probs, 1)
        # 联合置信度(normal)
        t_joint_scores = t_centernesses.sigmoid().reshape(-1) * t_scores
        # S_dps是最大的类别置信度特征图的期望
        S_dps = t_scores.mean()
        # weight_mask
        weight_mask = t_joint_scores

        '''根据伪标签筛选策略确定k'''
        if mode in ['topk', 'top_dps']:
            if mode == 'topk':
                ratio = ratio
            if mode == 'top_dps':
                ratio = S_dps * beta
            # 确定topk的正样本数量(有一个最小值为2)
            topk_num = max(int(t_cls_scores.size(0) * ratio), 2)
            # 从大到小排序
            pos_sorted_vals, pos_sorted_inds = torch.topk(t_scores, t_cls_scores.size(0))
            neg_sorted_vals, neg_sorted_inds = torch.topk(t_scores, t_cls_scores.size(0), largest=False)
            # 创建mask, 指定哪些样本为正样本, 哪些样本为负样本
            mask = torch.zeros_like(t_scores)
            # 前topk个元素为正样本 / 后topk个元素为负样本
            mask[pos_sorted_inds[:topk_num]] = 1.
            mask[neg_sorted_inds[:topk_num]] = -1.
            # 正样本数量(根据置信度加权)
            fg_num = pos_sorted_vals[:topk_num].sum()
        if mode == 'global_w':
            # 设置所有样本都为正样本, 均参与计算损失
            mask = torch.ones_like(t_scores, dtype=torch.bool)
            if beta==-1.:
                # weight_mask基于sigmoid(联合置信度)
                weight_mask = 1 / (1 + torch.exp(-10 * t_joint_scores)).pow(10) - 1/1024. 
            else:
                # weight_mask基于联合置信度^beta
                weight_mask = t_joint_scores.pow(beta)
            fg_num = weight_mask.sum()

        # 获得正负样本mask
        pos_mask = mask > 0.
        # weight_mask默认为联合置信度
        return pos_mask, weight_mask, fg_num, S_dps, t_joint_scores



    def forward(self, reshape_t_logits, reshape_s_logits, **kwargs):
        unsup_losses = {}
        # 注意cls_scores和centernesses都是未经过sigmoid()的logits
        t_cls_scores, t_bbox_preds, t_centernesses = reshape_t_logits
        s_cls_scores, s_bbox_preds, s_centernesses = reshape_s_logits


        '''伪标签筛选'''
        # 读取伪标签筛选超参
        mode = self.p_selection.get('mode', 'topk')
        k = self.p_selection.get('k', 0.01)
        beta = self.p_selection.get('beta', 1.0)
        with torch.no_grad():
            pos_mask, weight_mask, fg_num, S_dps, t_joint_scores = self.pseudoLabelSelection(mode, t_cls_scores, t_centernesses, k, beta)

        '''损失'''
        # 无监督分类损失QFLv2 (without ignore region)
        loss_cls = self.QFLv2(
            s_cls_scores.sigmoid(),
            t_cls_scores.sigmoid(),
            weight=pos_mask,
            reduction="none",
        )
        # 无监督回归损失
        # if self.bbox_loss_type == 'l1':
        #     loss_bbox = self.bbox_loss(
        #         s_bbox_preds[pos_mask],
        #         t_bbox_preds[pos_mask],
        #     )
        loss_bbox = self.riou_loss(s_bbox_preds[pos_mask], t_bbox_preds[pos_mask])
        print(loss_bbox.min(), loss_bbox.max())
        # 无监督centerness损失
        loss_centerness = F.binary_cross_entropy(
            s_centernesses[pos_mask].sigmoid(),
            t_centernesses[pos_mask].sigmoid(),
            reduction='none'
        ) 

        # 最终loss采用不同的组织形式进行加权平均:
        if mode in ['topk', 'top_dps', 'catwise_top_dps']:
            unsup_loss_cls = loss_cls.sum() / fg_num
            unsup_loss_bbox = (loss_bbox * t_centernesses[pos_mask].sigmoid()).mean()
            unsup_loss_centerness = loss_centerness.mean()
        if mode == 'global_w':
            # NOTE: 这里的分类损失不*weight_mask是因为分类会算负样本, 如果*weight_mask那么负样本的权重就很低(相当于负样本权重也应该高)
            # 因此 'global_w'的unsup_loss_cls的不同之处就在于, 不会将负样本的label强制置为0, 且fg_num是所有样本的score之和而不单单只有正样本
            unsup_loss_cls = loss_cls.sum() / fg_num
            unsup_loss_bbox = (weight_mask * loss_bbox).sum() / fg_num
            unsup_loss_centerness = (weight_mask * loss_centerness.reshape(-1)).sum() / fg_num


        # 无监督dense损失(denseteacher)
        unsup_losses['loss_cls'] = unsup_loss_cls
        unsup_losses['loss_bbox'] = unsup_loss_bbox
        unsup_losses['loss_centerness'] = unsup_loss_centerness
        unsup_losses['S_dps'] = S_dps

        # print(unsup_losses)
        return unsup_losses, t_joint_scores