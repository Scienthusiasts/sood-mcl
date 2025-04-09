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
class RotatedDTBLORCNNHeadLoss(nn.Module):
    def __init__(self, p_selection:dict, distill:dict, cls_channels=16, loss_type='origin', bbox_loss_type='l1'):
        super(RotatedDTBLORCNNHeadLoss, self).__init__()
        self.cls_channels = cls_channels
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
        # 特征蒸馏超参, added by yan
        self.distill = distill
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
        # weight_mask只有当mode='global_w'用到 (或自蒸馏损失用得到)
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
            # weight_mask基于联合置信度^beta
            # weight_mask = t_joint_scores.pow(beta)
            # weight_mask基于sigmoid(联合置信度)
            weight_mask = 1 / (1 + torch.exp(-10 * t_joint_scores)).pow(10) - 1/1024. 
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
        return pos_mask, neg_mask, weight_mask, fg_num, S_dps



    def forward(self, student, teacher, reshape_t_logits, reshape_s_logits, student_logits, teacher_logits, bs, img_metas=None, stu_img_metas=None, **kwargs):
        self.img_metas = img_metas

        # 注意cls_scores和centernesses都是未经过sigmoid()的logits
        t_cls_scores, t_bbox_preds, t_centernesses, _ = reshape_t_logits
        s_cls_scores, s_bbox_preds, s_centernesses, _ = reshape_s_logits
        refine_t_joint_score = teacher_logits[-1]






        """多对一匹配进行伪标签学习"""
        # 0.decode + nms + 分组
        # NOTE:注意这里传参共享内存, 所以得.clone()
        t_nms_bboxes, t_nms_labels, t_all_bboxes = self.decode_and_nms(t_bbox_preds.clone(), t_cls_scores.sigmoid(), t_centernesses.sigmoid())
        # 调整格式(注意, 这里的proposal_list拿的就是分组之后的框而不是所有的框了)
        # TODO: 存在的问题:分组的框依然会存在漏检的问题
        # proposal_list本来只包括坐标, 现在连score也加进去:
        all_t_proposal_list = [t_all_bboxes[:, :6]] if t_nms_bboxes.shape[0]!=0 else []
        nms_t_bboxes_list = [t_nms_bboxes[:, :5]] if t_nms_bboxes.shape[0]!=0 else []
        nms_t_labels_list = [t_nms_labels] if t_nms_bboxes.shape[0]!=0 else []


        '''teacher roihead的微调结果给student一阶段网络学习'''
        # 1.送入refine head微调(其实相当于一个二阶段检测器的检测头)
        batch_t_roihead_bboxes, batch_t_roihead_labels = [], []
        if len(nms_t_bboxes_list)!=0:
            with torch.no_grad():
                p = copy.deepcopy(nms_t_bboxes_list)
                refine_bbox = teacher.roi_head.simple_test(teacher_logits[4], p, img_metas['img_metas'], rescale=True)
                # 调整输出格式, 把所有类别下的预测结果拼在一起
                for i in range(bs):
                    t_res_bboxes, t_res_labels = [], []
                    for cls_id in range(len(refine_bbox[i])):
                        if(len(refine_bbox[i][cls_id])!=0):
                            t_res_bboxes.append(refine_bbox[i][cls_id])
                            t_res_labels += [cls_id] * len(refine_bbox[i][cls_id])
                    
                    t_res_bboxes = torch.tensor(np.concatenate(t_res_bboxes, axis=0)[:, :5], device=t_nms_labels.device) if len(t_res_bboxes) > 0 else t_nms_bboxes[:, :5]
                    t_res_labels = torch.tensor(t_res_labels, device=t_nms_labels.device) if len(t_res_labels) > 0 else t_nms_labels
                    batch_t_roihead_bboxes.append(t_res_bboxes)
                    batch_t_roihead_labels.append(t_res_labels)

        # 可视化(一般情况下注释)
        # vis_unsup_bboxes_batch(img_metas, bs, nms_bboxes_list, proposal_list, batch_res_bboxes, './vis_unsup_bboxes')
        # 2.送到head进行正负样本分配(nms后的结果作为gt) + 计算损失
        # cls_scores, bbox_preds, angle_preds, centernesses, _ = student_logits
        sup_losses, _, _, _, _, _, = student.bbox_head.loss(student_logits[0], student_logits[1], student_logits[2], student_logits[3], batch_t_roihead_bboxes, batch_t_roihead_labels,  None, None)
        # 获取对应损失
        denoise_cls_loss, denoise_cnt_loss, denoise_box_loss = sup_losses['loss_cls'], sup_losses['loss_centerness'], sup_losses['loss_bbox']
        

        '''teacher roihead的微调结果给student roihead学习'''
        # 1.送入roi head进行微调(其实相当于一个二阶段检测器的检测头)
        # TODO: 有一个问题, 调用forward_train函数时, 会再执行一次正负样本分配(orcnnhead)
        # [[bs, 256, h1, w1], ...], [[roi_nums, 5], ...], [[gt_nums], ...], [[gt_nums, 5], ...]

        # NOTE:Ablation1: 断开refine-head与主体检测器的梯度
        stu_fpn_feat = [fpn_feat.detach() for fpn_feat in student_logits[4]]
        # NOTE:Ablation2: 维持refine-head与主体检测器的梯度
        # stu_fpn_feat = [fpn_feat for fpn_feat in student_logits[4]]

        roi_losses = student.roi_head.forward_train(stu_fpn_feat, batch_t_roihead_labels, all_t_proposal_list, batch_t_roihead_bboxes, batch_t_roihead_labels)
        '''这里用all_s_proposal_list说明是s的proposal给s二阶段头做微调训练'''
        # # decode + nms + 分组;         NOTE:注意这里传参共享内存, 所以得.clone()
        # s_nms_bboxes, s_nms_labels, s_all_bboxes = self.decode_and_nms(s_bbox_preds.clone(), s_cls_scores.sigmoid(), s_centernesses.sigmoid()) 
        # # proposal_list本来只包括坐标, 现在连score也加进去:
        # all_s_proposal_list = [s_all_bboxes[:, :6].detach()] if s_nms_bboxes.shape[0]!=0 else []
        # # 注意 roi_head.forward_train接受的回归框坐标的格式是[cx, cy, w, h, a]
        # roi_losses = student.roi_head.forward_train(stu_fpn_feat, batch_t_roihead_labels, all_s_proposal_list, batch_t_roihead_bboxes, batch_t_roihead_labels)
        # 2.组织微调模块的损失
        loss_bbox_refine_unsup, loss_cls_refine_unsup, acc_unsup = roi_losses['loss_bbox'], roi_losses['loss_cls'], roi_losses['acc']













        '''伪标签筛选'''
        # 读取伪标签筛选超参
        mode = self.p_selection.get('mode', 'topk')
        k = self.p_selection.get('k', 0.01)
        beta = self.p_selection.get('beta', 1.0)
        with torch.no_grad():
            pos_mask, neg_mask, weight_mask, fg_num, S_dps = self.pseudoLabelSelection(mode, teacher_logits, t_cls_scores, t_bbox_preds, t_centernesses, k, beta, refine_t_joint_score)

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


        # 总损失采用字典形式组织
        unsup_losses = dict(
            # 无监督dense损失(denseteacher)
            loss_cls=unsup_loss_cls,
            loss_bbox=unsup_loss_bbox,
            loss_centerness=unsup_loss_centerness,

            # sdps
            # NOTE:yan add S_dps to tensorboard
            S_dps=S_dps,

            # 无监督 refine head 损失
            loss_bbox_refine = loss_bbox_refine_unsup, 
            loss_cls_refine = loss_cls_refine_unsup, 
            acc = acc_unsup,

            # 无监督多对一sparse损失(roi-head)
            loss_denoise_box=denoise_box_loss,
            # loss_denoise_cls=denoise_cls_loss,
            # loss_denoise_cnt=denoise_cnt_loss,
        )
        # print(unsup_losses)
        return unsup_losses, weight_mask







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








def wbf_aggregating(group_ids_mask, prenms_bboxes, pbox_nums):
    '''类似WBF的方法进行融合
    '''
    # group里的框进行聚类每个group都得到一个聚类中心(貌似聚类后的框也不比单纯nms后的框效果好)
    cluster_center_bboxes = []
    for group_id in range(pbox_nums):
        group_mask = group_ids_mask == group_id
        group_bboxes = prenms_bboxes[group_mask]
        # 5参转8参表示法(group)
        group_poly_boxes = obb2poly(group_bboxes[:, :5])
        group_score = group_bboxes[:, 5]
        # WBF后的框
        weighted_group_bboxes = group_poly_boxes * group_score.unsqueeze(1) / group_score.sum()
        weighted_mean_bboxes = weighted_group_bboxes.sum(dim=0).unsqueeze(0)
        cluster_center_bboxes.append(weighted_mean_bboxes)
    cluster_center_bboxes = torch.cat(cluster_center_bboxes, dim=0)
    # 8参转5参表示法(group)
    cluster_center_bboxes = poly2obb(cluster_center_bboxes)
    return cluster_center_bboxes




