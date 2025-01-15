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
from mmrotate.core import poly2obb_np, obb2poly
import cv2
import mmcv
# yan
import os
import matplotlib.pyplot as plt
import time
from custom.utils import OpenCVDrawBox
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from mmcv.ops import box_iou_quadri, box_iou_rotated


INF = 1e8
CLASSES = ('large-vehicle', 'swimming-pool', 'helicopter', 'bridge', 'plane', 
               'ship', 'soccer-ball-field', 'basketball-court', 'ground-track-field', 'small-vehicle', 
               'baseball-diamond', 'tennis-court', 'roundabout', 'storage-tank', 'harbor', 'container-crane')


@ROTATED_LOSSES.register_module()
class RotatedDTBLLoss(nn.Module):
    def __init__(self, p_selection:dict, distill:dict, cls_channels=16, loss_type='origin', bbox_loss_type='l1'):
        super(RotatedDTBLLoss, self).__init__()
        self.cls_channels = cls_channels
        assert bbox_loss_type in ['l1', 'iou']
        self.bbox_loss_type = bbox_loss_type
        self.bbox_coder = build_bbox_coder(dict(type='DistanceAnglePointCoder', angle_version='le90'))
        self.prior_generator = MlvlPointGenerator([8, 16, 32, 64, 128])
        if self.bbox_loss_type == 'l1':
            self.bbox_loss = nn.SmoothL1Loss(reduction='none')
        # NOTE: modified by yan:
        if p_selection['mode'] == 'global_w':
            self.bbox_coder = build_bbox_coder(dict(type='DistanceAnglePointCoder', angle_version='le90'))
            self.prior_generator = MlvlPointGenerator([8, 16, 32, 64, 128])
            self.iou_loss = build_loss(dict(type='RotatedIoULoss', loss_weight=1.0, linear=True, reduction='none'))
        self.loss_type = loss_type
        # 伪标签筛选策略超参, added by yan
        self.p_selection = p_selection
        # 特征蒸馏超参, added by yan
        self.distill = distill


    def convert_shape(self, logits, wo_cls_score=False):
        '''将模型输出logit reshape
        '''
        cls_scores, bbox_preds, angle_preds, centernesses, fpn_feat = logits
        bs = bbox_preds[0].shape[0]
        # wo_cls_score=True时表示cls_scores已经是prototype refine过的
        if wo_cls_score==False:
            assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses) == len(fpn_feat)
            # [[bs, cat_num, h1, w1], ...[bs, cat_num, h5, w5]] -> [total_grid_num, cat_num]
            cls_score = [x.permute(0, 2, 3, 1).reshape(bs, -1, self.cls_channels) for x in cls_scores]
            cls_scores = torch.cat(cls_score, dim=1).view(-1, self.cls_channels)
        else:
            assert len(bbox_preds) == len(angle_preds) == len(centernesses) == len(fpn_feat)

        # [[bs, 4+1, h1, w1], ...[bs, 4+1, h5, w5]] -> [total_grid_num, 5]
        bbox_preds = [torch.cat([x, y], dim=1).permute(0, 2, 3, 1).reshape(bs, -1, 5) for x, y in zip(bbox_preds, angle_preds)]
        bbox_preds = torch.cat(bbox_preds, dim=1).view(-1, 5)
        # [[bs, 1, h1, w1], ...[bs, 1, h5, w5]] -> [total_grid_num, 1]
        centernesses = [x.permute(0, 2, 3, 1).reshape(bs, -1, 1) for x in centernesses]
        centernesses = torch.cat(centernesses, dim=1).view(-1, 1)
        # [[bs, 256, h1, w1], ...[bs, 256, h5, w5]] -> [total_grid_num, 256]
        fpn_feat = [x.permute(0, 2, 3, 1).reshape(bs, -1, 256) for x in fpn_feat]
        fpn_feat = torch.cat(fpn_feat, dim=1).view(-1, 256)
        return cls_scores, bbox_preds, centernesses, fpn_feat, bs


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
            # weight_mask基于联合置信度^beta
            weight_mask = t_joint_scores.pow(beta)
            
            # weight_mask基于sigmoid(联合置信度)
            # weight_mask = 1 / (1 + torch.exp(-10 * t_joint_scores - 0.5))
            # weight_mask = 1 / (1 + torch.exp(-10 * t_joint_scores)).pow(10) - 1/1024. # 目前这个曲线在DOTA1.5, burn-in-12800 效果最好

            # 用SGC替代joint_score作为weight_mask(效果不好)
            # sgc = self.gen_SGC_mask(teacher_logits, t_bbox_preds, t_cls_scores, t_centernesses)
            # weight_mask = sgc.pow(beta)
            # weight_mask = 1 / (1 + torch.exp(-10 * sgc)).pow(10) - 1/1024.
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



    def forward(self, student, teacher_logits, student_logits, img_metas=None, stu_img_metas=None, **kwargs):
        self.img_metas = img_metas
        # 对输出的特征进行reshape
        # [total_grid_num, cat_num], [total_grid_num, 4+1], [total_grid_num, 1], [total_grid_num, 256]
        t_cls_scores, t_bbox_preds, t_centernesses, t_fpn_feat, bs = self.convert_shape(teacher_logits)
        s_cls_scores, s_bbox_preds, s_centernesses, s_fpn_feat, bs = self.convert_shape(student_logits)
        refine_t_joint_score = teacher_logits[-1]

        '''多对一匹配进行伪标签学习'''
        # decode得到最终结果(before nms)
        # NOTE:注意这里传参共享内存, 所以得.clone()
        nms_bboxes, nms_labels = self.decode_and_clustering(teacher_logits, t_bbox_preds.clone(), t_cls_scores, t_centernesses, img_metas)
        cls_scores, bbox_preds, angle_preds, centernesses, _ = student_logits
        # 送到head进行正负样本分配(nms后的结果作为gt) + 计算损失
        sup_losses, _, _, _ = \
            student.bbox_head.loss(cls_scores, bbox_preds, angle_preds, centernesses, [nms_bboxes[:, :5]], [nms_labels],  None, None)
        # 获取对应损失
        denoise_cls_loss, denoise_cnt_loss, denoise_box_loss = sup_losses['loss_cls'], sup_losses['loss_centerness'], sup_losses['loss_bbox']



        '''伪标签筛选'''
        # 读取伪标签筛选超参
        mode = self.p_selection.get('mode', 'topk')
        k = self.p_selection.get('k', 0.01)
        beta = self.p_selection.get('beta', 1.0)
        with torch.no_grad():
            pos_mask, neg_mask, weight_mask, fg_num, S_dps = self.pseudoLabelSelection(mode, teacher_logits, t_cls_scores, t_bbox_preds, t_centernesses, k, beta, refine_t_joint_score)

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

        '''特征蒸馏'''
        # distill_mode = self.distill.get('mode', 'l2')
        # distill_weight = self.distill.get('loss_weight', 1.0)
        # distill_beta = self.distill.get('beta', 1.0)
        # # 计算自蒸馏损失
        # unsup_loss_self_distill = selfDistillLoss(s_fpn_feat, t_fpn_feat, distill_mode, distill_beta, weight_mask) * distill_weight




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
            unsup_loss_bbox = (weight_mask[:, None][pos_mask] * loss_bbox).mean() * 10
            unsup_loss_centerness = (weight_mask[:, None][pos_mask] * loss_centerness).mean() * 10


        # 总损失采用字典形式组织
        unsup_losses = dict(
            # '''无监督dense损失'''
            loss_cls=unsup_loss_cls,
            loss_bbox=unsup_loss_bbox,
            loss_centerness=unsup_loss_centerness,

            # '''sdps'''
            # NOTE:yan add S_dps to tensorboard
            S_dps=S_dps,

            # '''蒸馏损失'''
            # loss_self_distill=unsup_loss_self_distill

            # '''无监督多对一sparse损失'''
            # 多对一标签分配去噪损失
            loss_denoise_box=denoise_box_loss,
            # loss_denoise_cls=denoise_cls_loss,
            # loss_denoise_cnt=denoise_cnt_loss,
        )
        return unsup_losses




    def gen_SGC_mask(self, teacher_logits, t_bbox_preds, t_cls_scores, t_centernesses):
        '''生成基于联合置信度加权的正样本高斯热力图, 用于加权global_w蒸馏损失
        '''
        # 1. 获得grid网格点坐标
        all_level_points = self.prior_generator.grid_priors(
            [featmap.size()[-2:] for featmap in teacher_logits[0]],
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
        # 3. 对预测的bbox解码得到最终的结果
        t_bbox_preds = self.bbox_coder.decode(concat_points, t_bbox_preds)

        # 4. 只取top0.03的样本参与高斯mask的计算(全部参与会爆显存)
        t_cls_scores = torch.max(t_cls_scores, 1)[0].sigmoid()
        t_joint_score = t_cls_scores * t_centernesses.sigmoid().reshape(-1)
        # 确定topk的正样本数量(有一个最小值为2)
        topk_num = int(t_cls_scores.size(0) * 0.03)
        # 从大到小排序
        sorted_vals, sorted_inds = torch.topk(t_joint_score, t_cls_scores.size(0))
        # 创建mask, 指定哪些样本为正样本, 哪些样本为负样本
        keep_mask = torch.zeros_like(t_cls_scores).type(torch.bool)
        # 前topk个元素为正样本 / 后topk个元素为负样本
        keep_mask[sorted_inds[:topk_num]] = True

        # 5.按所在尺度拆分
        sizes = [128, 64, 32, 16, 8]
        t_bbox_preds = torch.split(t_bbox_preds, [size * size for size in sizes], dim=0)
        concat_points = torch.split(concat_points, [size * size for size in sizes], dim=0)
        t_joint_score = torch.split(t_joint_score, [size * size for size in sizes], dim=0)
        keep_mask = torch.split(keep_mask, [size * size for size in sizes], dim=0)
        # 6. 逐尺度计算soft GCA 
        soft_gaussian_center = []
        # TODO:循环改成用multi_apply?
        for lvl in range(5):
            lvl_t_bbox_preds = t_bbox_preds[lvl]
            lvl_concat_points = concat_points[lvl]
            lvl_t_joint_score = t_joint_score[lvl]
            lvl_keep_mask = keep_mask[lvl]
            # 计算soft GCA (这玩意很吃显存) [total_anchor_num]
            # TODO: 这里soft_GCA_single默认bs_per_gpu=1, 因此bs_per_gpu>1会报错, 后续还需修改
            soft_gaussian_center.append(soft_GCA_single(lvl_t_bbox_preds[lvl_keep_mask], lvl_concat_points, lvl_t_joint_score[lvl_keep_mask]))

        soft_gaussian_center = torch.cat(soft_gaussian_center, dim=0)
        return soft_gaussian_center












    # for test only:
    def decode_and_clustering(self, teacher_logits, t_bbox_preds, t_cls_scores, t_centernesses, img_metas):
        '''decode'''
        # 1. 获得grid网格点坐标
        all_level_points = self.prior_generator.grid_priors(
            [featmap.size()[-2:] for featmap in teacher_logits[0]],
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
        t_joint_score = t_scores.sigmoid() * t_centernesses.sigmoid().reshape(-1)
        # 把所有信息concat在一起 [total_anchor_num, 7=(cx, cy, w, h, θ, score, label)]
        t_results = torch.cat([t_bbox_preds, t_joint_score.reshape(-1, 1), t_pred_labels.reshape(-1, 1)], dim=1)
        # 采用联合置信度卡正样本
        pos_mask = t_joint_score > 1e-4

        '''nms'''
        # det_bboxes.shape=[nms_num, 6]   det_labels.shape=[nms_num]
        det_bboxes, det_labels = multiclass_nms_rotated(
            multi_bboxes=t_results[:, :5],              # [total_box_num, 5]
            multi_scores=t_cls_scores,                  # [total_box_num, 16]
            score_thr=1e-4,                             # 0.05
            nms={'iou_thr': 0.1},                       # {'iou_thr': 0.1}
            max_num=2000,                               # 2000
            score_factors=t_centernesses.sigmoid().reshape(-1)  # [total_box_num]
            )


        '''根据nms结果进行分组+聚类'''
        # nms后还有正样本, 才继续
        if det_bboxes.shape[0]==0: return det_bboxes, det_labels 

        pbox_nums = det_bboxes.shape[0]
        # nms_post和nms_pre的样本计算riou [pre_num, post_num]
        post_pre_riou = box_iou_rotated(t_results[:, :5][pos_mask], det_bboxes[:, :5])
        group_max_iou, group_ids_mask = post_pre_riou.max(dim=1)
        # 过滤掉iou太小的框
        fgd_mask = group_max_iou > 0.1
        # 得到最终结果
        group_max_iou, group_ids_mask, prenms_bboxes = group_max_iou[fgd_mask], group_ids_mask[fgd_mask], t_results[pos_mask][fgd_mask]

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

        # TODO: return的 det_bboxes, det_labels 只是基于nms后的框, 并没有对聚类的框进行微调, 
        # 因此还需要加上一个微调模块, 看怎么把group_ids_mask对应的分组的伪框进行聚类+微调去噪得到更准确的伪框
        # TODO: 还需要考虑的问题: 微调模块在哪里定义(需要独立于教师和学生模型), 在哪里调用, 采用何种正负样本分配方式(采用类似二阶段的roialign+正负样本分配?) 
        return det_bboxes, det_labels















def selfDistillLoss(s_fpn_feat, t_fpn_feat, mode, beta, weight_mask):
    '''自蒸馏损失
        Args:
            s_fpn_feat:  学生模型的fpn特征(保留梯度) [total_grid_num, 256]
            t_fpn_feat:  教师模型的fpn特征(没有梯度) [total_grid_num, 256]
            mode:        蒸馏损失 'kld', 'l2'
            weight_mask: 默认为pseudoLabelSelection()传来的联合置信度mask

        Returns:
            unsup_loss_self_distill: 自蒸馏损失
    '''
    # 使用KLD计算自蒸馏损失
    if mode=='kld':
        s_fpn_sigmoid = s_fpn_feat.sigmoid()
        t_fpn_sigmoid = t_fpn_feat.sigmoid()
        s_t_fpn_kld = KLD(s_fpn_sigmoid, t_fpn_sigmoid) 
        # 平衡损失的系数
        # sigma = (t_fpn_sigmoid - s_fpn_sigmoid).pow(2).mean()
        # unsup_loss_self_distill = s_t_fpn_kld * sigma
        unsup_loss_self_distill = s_t_fpn_kld * beta
    if mode=='cosine':
        # 特征向量归一化
        s_fpn_feat_norm = s_fpn_feat / s_fpn_feat.norm(dim=-1, keepdim=True)
        t_fpn_feat_norm = t_fpn_feat / t_fpn_feat.norm(dim=-1, keepdim=True)
        # 计算余弦相似度
        ts_feat_sim = torch.einsum('ij,ij->i', s_fpn_feat_norm, t_fpn_feat_norm)
        unsup_loss_self_distill = -torch.log((ts_feat_sim + 1) * 0.5 + 1e-7).mean() 
    # 使用l2计算自蒸馏损失
    if mode=='l2':
        s_t_fpn_l2 = (s_fpn_feat - t_fpn_feat).pow(2)
        # weight_mask默认为联合置信度
        # (当伪标签筛选为global_w时, beta应为1, 因为伪标签筛选时就已经对weight_mask取pow操作)
        # sigma = weight_mask.pow(beta)
        # unsup_loss_self_distill = (s_t_fpn_l2.mean(dim=1) * sigma).sum() / sigma.sum()
        # 不采用加权(还没试过):
        unsup_loss_self_distill = s_t_fpn_l2.mean()
    if mode=='qflv2':
        s_fpn_sigmoid = s_fpn_feat.sigmoid()
        t_fpn_sigmoid = t_fpn_feat.sigmoid()
        sigma = (s_fpn_sigmoid - t_fpn_sigmoid).pow(2)
        s_t_fpn_qflv2 = F.binary_cross_entropy(s_fpn_sigmoid, t_fpn_sigmoid, reduction='none') * sigma
        # weight_mask默认为联合置信度
        unsup_loss_self_distill = s_t_fpn_qflv2.sum() / 256. / weight_mask.sum()
    
    return unsup_loss_self_distill



def QFLv2(pred_sigmoid, teacher_sigmoid, weight=None, beta=2.0, reduction='mean'):
    # all goes to 0
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    # 一开始假设所有样本都是负样本, 因此实际上有对负样本计算损失, 对应的标签是全0
    loss = F.binary_cross_entropy(pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    # positive goes to bbox quality

    pt = teacher_sigmoid[weight] - pred_sigmoid[weight]
    # 在所有样本都是负样本的基础上更新那些正样本对应位置为正样本损失(当teacher_sigmoid足够低时,也相当于计算负样本损失)
    loss[weight] = F.binary_cross_entropy(pred_sigmoid[weight], teacher_sigmoid[weight], reduction='none') * pt.pow(beta)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss



def KLD(s, t):
    # 将s和t第一维度转化为频率(和=1)
    s = s / s.sum(dim=1, keepdim=True)
    t = t / t.sum(dim=1, keepdim=True)
    # 计算 KL 散度, 添加一个小的 epsilon 防止 log(0)
    eps = 1e-10
    kl_div = s * torch.log((s + eps) / (t + eps))
    return kl_div.sum()





def soft_GCA_single(gt_bboxes, points, gt_scores):
    '''基于高斯椭圆的方法将伪标签转化为global_w的weight_mask
        Args:
            - gt_bboxes:          [gt_num, 5]
            - points:             [total_anchor_num, 2] (int型, 为box中心点坐标)
            - regress_ranges:     [total_anchor_num, 2] ((-1, 64), (64, 128), (128, 256), (256, 512), (512, 100000000.0))
        Return:
    '''
    # 获取检测点的数量
    num_points = points.size(0)
    # 获取该图像中的目标（ground truth, gt）的数量
    num_gts = gt_bboxes.size(0)
    if num_gts == 0: 
        return torch.zeros(num_points, device=points.device)
    # 计算每个目标的面积（宽度 * 高度）
    areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
    # 将面积扩展为与检测点数量匹配的形状，使得每个检测点都能计算对应目标的面积
    areas = areas[None].repeat(num_points, 1)
    # 扩展检测点坐标，使其与每个目标一一对应
    points = points[:, None, :].expand(num_points, num_gts, 2)
    # 将目标框扩展为与检测点数量相匹配的形状
    gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
    # 将目标框拆分为中心点 (gt_ctr)，宽高 (gt_wh)，以及角度 (gt_angle)
    gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)
    # 计算旋转角度的余弦值和正弦值
    cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
    # 构建旋转矩阵，用于将检测点的偏移应用于目标框的角度旋转
    rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle], dim=-1).reshape(num_points, num_gts, 2, 2)
    # 计算检测点相对于目标框中心的偏移量
    offset = points - gt_ctr
    # 应用旋转矩阵，将偏移量旋转到目标框的角度
    offset = torch.matmul(rot_matrix, offset[..., None])
    # 去掉多余的维度，得到旋转后的偏移量
    offset = offset.squeeze(-1)
    # 提取目标框的宽度和高度
    w, h = gt_wh[..., 0], gt_wh[..., 1]
    # 提取偏移量的x和y方向的分量
    offset_x, offset_y = offset[..., 0], offset[..., 1]
    # 根据偏移量计算出目标框的四个边界：左、右、上、下
    left = w / 2 + offset_x
    right = w / 2 - offset_x
    top = h / 2 + offset_y
    bottom = h / 2 - offset_y
    # 将目标框的四个边界合并为 (left, top, right, bottom) 的形式
    bbox_targets = torch.stack((left, top, right, bottom), -1)
    # 计算检测点在目标框中的高斯分布中心偏移，归一化到宽高的范围
    gaussian_center = 1 - (offset_x.pow(2) / (w / 2).pow(2) + offset_y.pow(2) / (h / 2).pow(2))
    # 类别置信度对gaussian_center加权
    weighted_gaussian_center = torch.einsum('ij,j->ij', gaussian_center, gt_scores)
    # 取gaussian置信度的最大值
    soft_gaussian_center = weighted_gaussian_center.max(dim=1)[0]
    soft_gaussian_center = torch.clamp(soft_gaussian_center, 0, 1)

    return soft_gaussian_center