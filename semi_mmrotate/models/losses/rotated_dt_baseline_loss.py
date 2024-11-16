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
    def __init__(self, p_selection:dict, cls_channels=16, loss_type='origin', bbox_loss_type='l1'):
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
        # 伪标签筛选策略超参, added by yan
        self.p_selection = p_selection


    def convert_shape(self, logits):
        '''将模型输出logit reshape
        '''
        cls_scores, bbox_preds, angle_preds, centernesses, fpn_feat = logits
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses) == len(fpn_feat)
        bs = cls_scores[0].shape[0]   

        # [[bs, cat_num, h1, w1], ...[bs, cat_num, h5, w5]] -> [total_grid_num, cat_num]
        cls_score = [x.permute(0, 2, 3, 1).reshape(bs, -1, self.cls_channels) for x in cls_scores]
        cls_scores = torch.cat(cls_score, dim=1).view(-1, self.cls_channels)
        # [[bs, 4+1, h1, w1], ...[bs, 4+1, h5, w5]] -> [total_grid_num, 5]
        bbox_preds = [torch.cat([x, y], dim=1).permute(0, 2, 3, 1).reshape(bs, -1, 5) for x, y in zip(bbox_preds, angle_preds)]
        bbox_preds = torch.cat(bbox_preds, dim=1).view(-1, 5)
        # [[bs, 1, h1, w1], ...[bs, 1, h5, w5]] -> [total_grid_num, 1]
        centernesses = [x.permute(0, 2, 3, 1).reshape(bs, -1, 1) for x in centernesses]
        centernesses = torch.cat(centernesses, dim=1).view(-1, 1)
        # [[bs, 256, h1, w1], ...[bs, 256, h5, w5]] -> [total_grid_num, 256]
        fpn_feat = [x.permute(0, 2, 3, 1).reshape(bs, -1, 256) for x in fpn_feat]
        fpn_feat = torch.cat(fpn_feat, dim=1).view(-1, 256)
        return cls_scores, bbox_preds, centernesses, fpn_feat


    def pseudoLabelSelection(self, mode:str, t_cls_scores, t_centernesses, ratio:float, beta:float):
        '''伪标签筛选 added by yan
            Args:
                mode:         伪标签筛选策略 ('topk', 'top_dps', 'catwise_top_dps')
                t_cls_scores: 网络预测整张特征图的分类置信度 [bs * h * w, cat_num]
                ratio:        当mode=='topk'时, 这个参数代表top k% 的 k%
                beta:         当mode=='top_dps'时, beta为S_pds的权重系数
            Returns:
                pos_mask: 正样本mask 
                fg_num:   正样本的数量
                S_pds:    当前batch的平均联合置信度
        '''
        teacher_probs = t_cls_scores.sigmoid()
        # t_scores, t_pred提取最大的类别置信度和对应的类别索引 [bs * h * w, cat_num] -> [bs * h * w], [bs * h * w]
        t_scores, t_pred = torch.max(teacher_probs, 1)
        # 联合置信度
        t_joint_scores = t_centernesses.sigmoid().reshape(-1) * t_scores
        # S_dps是最大的类别置信度特征图的期望
        S_dps = t_scores.mean()
        # weight_mask只有当mode='global_w'用到
        weight_mask = None

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
        if mode == 'catwise_top_dps':
            mask = torch.zeros_like(t_scores, dtype=torch.bool)
            fg_num = 0
            for cat_id in range(self.cls_channels):
                cat_id_mask = t_pred==cat_id
                cat_scores = t_scores[cat_id_mask]
                # 当前类别有正样本才继续:
                if cat_scores.shape[0] > 0:
                    # cat_S_dps = S_dps * beta
                    cat_S_dps = cat_scores.mean() * beta
                    # cat_score_mask需要基于t_scores的索引而不是cat_scores索引
                    # 如果cat_S_dps > cat_scores.max(), 则至少保留最大的那个样本
                    catwise_pos_mask = (t_scores >= min(cat_S_dps, cat_scores.max())) & cat_id_mask
                    # 每个类别的正样本mask或就得到所有类别上的正样本mask
                    mask = mask | catwise_pos_mask
                    fg_num += t_scores[catwise_pos_mask].sum()
        if mode == 'global_w':
            # 设置所有样本都为正样本, 均参与计算损失
            mask = torch.ones_like(t_scores, dtype=torch.bool)
            # weight_mask基于分类置信度, 可以再试一下联合置信度
            weight_mask = t_scores
            # weight_mask = t_joint_scores
            fg_num = weight_mask.sum()

        # 获得正负样本mask
        pos_mask = mask > 0.
        neg_mask = mask < 0.
        return pos_mask, neg_mask, weight_mask, fg_num, S_dps



    def forward(self, teacher_logits, student_logits, img_metas=None, **kwargs):
        # 对输出的特征进行reshape
        # [total_grid_num, cat_num], [total_grid_num, 4+1], [total_grid_num, 1], [total_grid_num, 256]
        t_cls_scores, t_bbox_preds, t_centernesses, t_fpn_feat = self.convert_shape(teacher_logits)
        s_cls_scores, s_bbox_preds, s_centernesses, s_fpn_feat = self.convert_shape(student_logits)

        '''伪标签筛选'''
        # 读取伪标签筛选超参
        mode = self.p_selection.get('mode', 'topk')
        k = self.p_selection.get('k', 0.01)
        beta = self.p_selection.get('beta', 1.0)
        with torch.no_grad():
            pos_mask, neg_mask, weight_mask, fg_num, S_dps = self.pseudoLabelSelection(mode, t_cls_scores, t_centernesses, k, beta)
        '''损失'''
        # 无监督分类损失 (with ignore region)
        # clear_mask = pos_mask | neg_mask
        # loss_cls = QFLv2(
        #     s_cls_scores.sigmoid()[clear_mask],
        #     t_cls_scores.sigmoid()[clear_mask],
        #     weight=pos_mask[clear_mask],
        #     reduction="sum",
        # ) / fg_num
        # 无监督分类损失 (without ignore region)
        loss_cls = QFLv2(
            s_cls_scores.sigmoid(),
            t_cls_scores.sigmoid(),
            weight=pos_mask,
            reduction="none",
        )
        # 无监督回归损失(删除了else的IoU的部分)
        if self.bbox_loss_type == 'l1':
            loss_bbox = self.bbox_loss(
                s_bbox_preds[pos_mask],
                t_bbox_preds[pos_mask],
            )

        # 无监督centerness损失
        loss_centerness = F.binary_cross_entropy(
            s_centernesses[pos_mask].sigmoid(),
            t_centernesses[pos_mask].sigmoid(),
            reduction='none'
        )
        # 最终loss采用不同的组织形式进行加权平均:
        if mode in ['topk', 'top_dps', 'catwise_top_dps']:
            unsup_loss_cls = loss_cls.sum() / fg_num
            unsup_loss_bbox = (loss_bbox * t_centernesses.sigmoid()[pos_mask]).mean()
            unsup_loss_centerness = loss_centerness.mean()
        if mode == 'global_w':
            # NOTE: 这里的分类损失不*weight_mask是因为分类会算负样本, 如果*weight_mask那么负样本的权重就很低(相当于负样本权重也应该高)
            # 因此 'global_w'的unsup_loss_cls的不同之处就在于, 不会将负样本的label强制置为0, 且fg_num是所有样本的score之和而不单单只有正样本
            unsup_loss_cls = loss_cls.sum() / fg_num
            unsup_loss_bbox = (weight_mask * loss_bbox.sum(dim=1)).sum() / fg_num
            unsup_loss_centerness = (weight_mask * loss_centerness.reshape(-1)).sum() / fg_num

        # 总损失采用字典形式组织
        unsup_losses = dict(
            loss_cls=unsup_loss_cls,
            loss_bbox=unsup_loss_bbox,
            loss_centerness=unsup_loss_centerness,
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
    # 在所有样本都是负样本的基础上更新那些正样本对应位置为正样本损失(当teacher_sigmoid足够低时,也相当于计算负样本损失)
    loss[weight] = F.binary_cross_entropy(pred_sigmoid[weight], teacher_sigmoid[weight], reduction='none') * pt.pow(beta)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss