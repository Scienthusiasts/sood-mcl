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

INF = 1e8
CLASSES = ('large-vehicle', 'swimming-pool', 'helicopter', 'bridge', 'plane', 
               'ship', 'soccer-ball-field', 'basketball-court', 'ground-track-field', 'small-vehicle', 
               'baseball-diamond', 'tennis-court', 'roundabout', 'storage-tank', 'harbor', 'container-crane')


@ROTATED_LOSSES.register_module()
class RotatedDTBLGIHeadLoss(nn.Module):
    def __init__(self, p_selection:dict, cls_channels=16, loss_type='origin', bbox_loss_type='l1'):
        super(RotatedDTBLGIHeadLoss, self).__init__()
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



    def pseudoLabelSelection(self, mode:str, teacher_logits, t_cls_scores, t_bbox_preds, t_centernesses, ratio:float, beta:float, refine_t_joint_score=None):
        '''伪标签筛选 added by yan
            Args:
                mode:                 伪标签筛选策略 ('topk', 'top_dps', 'catwise_top_dps')
                t_cls_scores:         网络预测整张特征图的分类置信度 [bs * h * w, cat_num] (经过refine)
                t_bbox_preds: 
                t_centernesses:
                ratio:                当mode=='topk'时, 这个参数代表top k% 的 k%
                beta:                 当mode=='top_dps'时, beta为S_pds的权重系数
                refine_t_joint_score: =None
            Returns:
                pos_mask: 正样本mask 
                fg_num:   正样本的数量
                S_pds:    当前batch的平均联合置信度
        '''
        # 分类置信度(refine)
        teacher_probs = t_cls_scores.sigmoid()
        # t_scores, t_pred提取最大的类别置信度和对应的类别索引 [bs * h * w, cat_num] -> [bs * h * w], [bs * h * w]
        t_scores, t_pred = torch.max(teacher_probs, 1)
        # 联合置信度(prototype refine)
        # t_joint_scores = refine_t_joint_score.max(dim=1)[0]
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

        if mode == 'sla':
            teacher_bboxes = t_bbox_preds
            teacher_centernesses = t_centernesses.sigmoid()

            level_inds = [16384, 20480]
            # P3
            teacher_probs_p3 = teacher_probs[:level_inds[0]]
            teacher_centernesses_p3 = teacher_centernesses[:level_inds[0]]
            joint_confidences_p3 = teacher_probs_p3 * teacher_centernesses_p3
            max_vals_p3 = torch.max(joint_confidences_p3, 1)[0]
            selected_inds_p3 = torch.topk(max_vals_p3, joint_confidences_p3.size(0))[1][:2000]
            # P4
            teacher_probs_p4 = teacher_probs[level_inds[0]:level_inds[1]]
            teacher_centernesses_p4 = teacher_centernesses[level_inds[0]:level_inds[1]]
            joint_confidences_p4 = teacher_probs_p4 * teacher_centernesses_p4
            select_inds_p4 = torch.arange(level_inds[0], level_inds[1]).to(joint_confidences_p4.device)
            max_vals_p4 = torch.max(joint_confidences_p4, 1)[0]
            selected_inds_p4 = select_inds_p4[torch.topk(max_vals_p4, joint_confidences_p4.size(0))[1][:2000]]
            # P5, P6, P7
            confidences_rest = teacher_probs[level_inds[1]:]
            selected_inds_rest = torch.arange(level_inds[1], teacher_probs.shape[0]).to(confidences_rest.device)

            # coarse_inds
            selected_inds_coarse = torch.cat([selected_inds_p3, selected_inds_p4, selected_inds_rest], 0)
            # fine_inds
            all_confidences = torch.cat([joint_confidences_p3, joint_confidences_p4, confidences_rest], 0)
            max_vals = torch.max(all_confidences, 1)[0]
            selected_inds = torch.nonzero(max_vals > 0.02).squeeze(-1)

            selected_inds, counts = torch.cat([selected_inds_coarse, selected_inds], 0).unique(return_counts=True)
            selected_inds = selected_inds[counts>1]
            # weight_mask其实就是 max_vals 把小于0.02那部分的score置为0
            weight_mask = torch.zeros_like(max_vals)
            weight_mask[selected_inds] = max_vals[selected_inds] 

            mask = weight_mask
            pos_mask = weight_mask > 0.
            if pos_mask.sum() == 0:
                fg_num = max_vals.sum()
            else:
                fg_num = weight_mask.sum()

        # 获得正负样本mask
        pos_mask = mask > 0.
        neg_mask = mask < 0.
        # weight_mask默认为联合置信度
        return pos_mask, neg_mask, weight_mask, fg_num, S_dps, t_joint_scores



    def forward(self, student, teacher, reshape_t_logits, reshape_s_logits, student_logits, teacher_logits, bs, sup_img_metas=None, unsup_img_metas=None, use_refine_head=True, **kwargs):
        unsup_losses = {}
        # 注意cls_scores和centernesses都是未经过sigmoid()的logits
        t_cls_scores, t_bbox_preds, t_centernesses, _ = reshape_t_logits
        s_cls_scores, s_bbox_preds, s_centernesses, _ = reshape_s_logits
        # refine head 相关:
        if use_refine_head:
            '''对stundent的预测调整为gi_head接受的格式'''
            s_cls_labels = torch.argmax(s_cls_scores, dim=1, keepdim=True)
            s_joint_score = torch.einsum('ij, i -> ij', s_cls_scores.sigmoid(), s_centernesses.sigmoid().squeeze(1)).max(dim=-1)[0].unsqueeze(1)
            # [bs, total_anchor_num, 7=(cx, cy, w, h, θ, joint_score, label)] 这里的 cx, cy, w, h, θ格式还不对, 还需要解码
            s_rbb_preds = torch.cat([s_bbox_preds, s_joint_score, s_cls_labels], dim=-1).reshape(bs, -1, 7)
            # NOTE:Ablation1: 断开refine-head与主体检测器的梯度
            # stu_fpn_feat = [fpn_feat.detach() for fpn_feat in student_logits[4]]
            # NOTE:Ablation2: 维持refine-head与主体检测器的梯度
            stu_fpn_feat = [fpn_feat for fpn_feat in student_logits[4]]
            # 对原始预测解码
            # 这里rbb_preds不加.detach() 会报inplace op的错
            s_rbb_preds = self.rbb_decode(bs, stu_fpn_feat, s_rbb_preds.detach())

            '''对teacher的预测进行nms转化为pgt'''
            # NOTE:注意这里传参共享内存, 所以得.clone()
            t_nms_bboxes, t_nms_labels, t_all_bboxes = self.decode_and_nms(t_bbox_preds.clone(), t_cls_scores.sigmoid(), t_centernesses.sigmoid())
            # 默认bs=1:
            nms_t_bboxes_list = [t_nms_bboxes[:, :5]] if t_nms_bboxes.shape[0]!=0 else []
            nms_t_labels_list = [t_nms_labels] if t_nms_bboxes.shape[0]!=0 else []
            t_rbb_preds = t_all_bboxes.reshape(bs, -1, 7)
            '''teacher 一阶段的结果给student roihead学习'''
            # 注意 roi_head.forward_train接受的回归框坐标的格式是[cx, cy, w, h, a]
            roi_losses = student.roi_head.loss(
                stu_fpn_feat, 
                s_rbb_preds, s_cls_scores.sigmoid().reshape(bs, -1, self.nc), s_centernesses.sigmoid().reshape(bs, -1),
                # teacher的结果作为gt
                nms_t_bboxes_list, nms_t_labels_list,
                unsup_img_metas,
                train_mode='train_unsup'
                )
            '''teacher roihead的微调结果给student一阶段学习'''
            with torch.no_grad():
                batch_preds = teacher.roi_head.infer(
                    # t_fpn_feat:
                    teacher_logits[4], 
                    t_rbb_preds, t_cls_scores.sigmoid().reshape(bs, -1, self.nc), t_centernesses.sigmoid().reshape(bs, -1),
                    unsup_img_metas,
                    )
                batch_t_res_bboxes, batch_t_res_labels = [], []
                for preds in batch_preds:
                    # [nms_boxes_num, 7=(cx, cy, w, h, θ, score, label)] -> [nms_boxes_num, 5=(cx, cy, w, h, θ)], [nms_boxes_num]
                    batch_t_res_bboxes.append(preds[:, :5])
                    batch_t_res_labels.append(preds[:, 6])

            # 可视化(一般情况下注释)
            # vis_unsup_bboxes_batch(unsup_img_metas, bs, nms_t_bboxes_list, t_rbb_preds, batch_t_res_bboxes, './vis_unsup_bboxes')
            # 2.送到head进行正负样本分配(nms后的结果作为gt) + 计算损失
            # cls_scores, bbox_preds, angle_preds, centernesses, _ = student_logits
            sup_losses, _, _, _, _, _, = student.bbox_head.loss(student_logits[0], student_logits[1], student_logits[2], student_logits[3], batch_t_res_bboxes, batch_t_res_labels,  None, None)
            # 获取对应损失
            denoise_cls_loss, denoise_cnt_loss, denoise_box_loss = sup_losses['loss_cls'], sup_losses['loss_centerness'], sup_losses['loss_bbox']
            # refinehead结果微调student一阶段结果的损失 (无监督多对一sparse损失(roi-head))
            unsup_losses['loss_denoise_box'] = denoise_box_loss
            # refinehead自己的损失 (无监督 refine head 损失)
            unsup_losses['loss_bbox_refine'] = roi_losses['gi_reg_loss']
            unsup_losses['loss_cls_refine'] = roi_losses['gi_cls_loss']
            unsup_losses['iou_improve'] = roi_losses['iou_improve']




        '''伪标签筛选'''
        # 读取伪标签筛选超参
        mode = self.p_selection.get('mode', 'topk')
        k = self.p_selection.get('k', 0.01)
        beta = self.p_selection.get('beta', 1.0)
        with torch.no_grad():
            pos_mask, neg_mask, weight_mask, fg_num, S_dps, t_joint_scores = self.pseudoLabelSelection(mode, teacher_logits, t_cls_scores, t_bbox_preds, t_centernesses, k, beta)

        '''损失'''
        # 无监督分类损失QFLv2 (without ignore region)
        loss_cls = self.QFLv2(
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
            unsup_loss_bbox = (loss_bbox * t_centernesses[pos_mask].sigmoid()).mean()
            # unsup_loss_bbox = loss_bbox.mean()
            unsup_loss_centerness = loss_centerness.mean()
        if mode == 'global_w':
            # NOTE: 这里的分类损失不*weight_mask是因为分类会算负样本, 如果*weight_mask那么负样本的权重就很低(相当于负样本权重也应该高)
            # 因此 'global_w'的unsup_loss_cls的不同之处就在于, 不会将负样本的label强制置为0, 且fg_num是所有样本的score之和而不单单只有正样本
            unsup_loss_cls = loss_cls.sum() / fg_num
            unsup_loss_bbox = (weight_mask * loss_bbox.sum(dim=1)).sum() / fg_num
            unsup_loss_centerness = (weight_mask * loss_centerness.reshape(-1)).sum() / fg_num
        if mode == 'sla':
            unsup_loss_cls = loss_cls.sum() / fg_num
            # pos_mask可能全为False, 此时loss为空, 空tensor计算.mean()时会出现nan. 下面这个分支防止损失出现nan
            if loss_bbox.shape[0]==0:
                unsup_loss_bbox = torch.tensor(0., device=loss_bbox.device)
                unsup_loss_centerness = torch.tensor(0., device=loss_centerness.device)
            else:
                unsup_loss_bbox = (weight_mask[:, None][pos_mask] * loss_bbox).mean() * 10
                unsup_loss_centerness = (weight_mask[:, None][pos_mask] * loss_centerness).mean() * 10

        # 无监督dense损失(denseteacher)
        unsup_losses['loss_cls'] = unsup_loss_cls
        unsup_losses['loss_bbox'] = unsup_loss_bbox
        unsup_losses['loss_centerness'] = unsup_loss_centerness
        unsup_losses['S_dps'] = S_dps

        # print(unsup_losses)
        return unsup_losses, t_joint_scores







    def rbb_decode(self, bs, fpn_feat, rbb_preds):
        '''对网络得到的框的回归值解码
        '''
        # 0.对角度再乘一个可学习的尺度(这里不加with torch.no_grad()显存会持续增加直到OOM, 不知道为啥)
        # NOTE: 这里似乎不需要了(加上之后可视化角度不对, 去掉后角度就正常了), 很奇怪:
        # NOTE: 所以之前训练其实角度都是不太对的? TvT(25-1-15)
        # with torch.no_grad():
        #     rbb_preds[:, :, 4] = self.student.bbox_head.scale_angle(rbb_preds[:, :, 4])
        # 1. 获得grid网格点坐标
        all_level_points = self.prior_generator.grid_priors(
            [featmap.size()[-2:] for featmap in fpn_feat],
            dtype=rbb_preds.dtype,
            device=rbb_preds.device
            )
        # [[h1*w1, 2], ..., [h5*w5, 2]] -> [total_anchor_num, 2]
        concat_points = torch.cat(all_level_points, dim=0)
        # 2. 对bbox的乘上对应的尺度
        lvl_range  = [0, 16384, 20480, 21504, 21760, 21824]
        lvl_stride = [8, 16, 32, 64, 128]
        for i in range(bs):
            for lvl in range(5):
                rbb_preds[i, lvl_range[lvl]:lvl_range[lvl+1], :4] *= lvl_stride[lvl]
            # 3. 对预测的bbox解码得到最终的结果, 并得到联合置信度作为类别置信度
            rbb_preds[i, :, :5] = self.bbox_coder.decode(concat_points, rbb_preds[i, :, :5])
        return rbb_preds





    def decode_and_nms(self, t_bbox_preds, t_cls_scores, t_centernesses):
        '''decode + nms
            Args: 

            Returns:
                None
        '''
        # 1. 获得grid网格点坐标
        all_level_points = self.prior_generator.grid_priors(
            [[128, 128], [64,64], [32,32], [16,16], [8,8]],
            dtype=t_bbox_preds.dtype,
            device=t_bbox_preds.device
            )
        # [[h1*w1, 2], ..., [h5*w5, 2]] -> [total_anchor_num, 2]
        concat_points = torch.cat(all_level_points, dim=0)
        # 2. 对bbox的乘上对应的尺度
        lvl_range  = [0, 16384, 20480, 21504, 21760, 21824]
        lvl_stride = [8, 16, 32, 64, 128]
        for i in range(5):
            t_bbox_preds[lvl_range[i]:lvl_range[i+1], :4] *= lvl_stride[i]
        # 3. 对预测的bbox解码得到最终的结果, 并得到联合置信度作为类别置信度
        t_bbox_preds = self.bbox_coder.decode(concat_points, t_bbox_preds)
        t_scores, t_pred_labels = torch.max(t_cls_scores, 1)
        t_joint_score = t_scores * t_centernesses.reshape(-1)
        # 把所有信息concat在一起 [total_anchor_num, 7=(cx, cy, w, h, θ, score, label)]
        t_results = torch.cat([t_bbox_preds, t_joint_score.reshape(-1, 1), t_pred_labels.reshape(-1, 1)], dim=1)

        '''nms'''
        # nms(至少会保留一个置信度最大的pgt):
        batch_nms_bboxes, batch_nms_labels, batch_nms_scores = batch_nms(t_results.unsqueeze(0), t_cls_scores.unsqueeze(0), t_joint_score.unsqueeze(0))
        # 这一步偷懒, 因为只有一个batch
        nms_bboxes, nms_labels = batch_nms_bboxes[0], batch_nms_labels[0]

        return nms_bboxes, nms_labels, t_results