#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/18 17:03
# @Author : WeiHua
import torch
import numpy as np
from .rotated_semi_detector import RotatedSemiDetector
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models import build_detector
# yan 
import copy
import os
import cv2
from mmrotate.models import ROTATED_LOSSES, build_loss
from mmrotate.core import rbbox2result, poly2obb_np, obb2poly, poly2obb, build_bbox_coder, obb2xyxy
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from  mmdet.core.bbox.samplers.sampling_result import SamplingResult
from custom.utils import *
from custom.visualize import *
from custom.ss_branch import SSBranch
from custom.fn_mining import FNMining, vis_sparse_data
# 计算IoU Loss
from mmcv.ops import diff_iou_rotated_2d




@ROTATED_DETECTORS.register_module()
# GI的意思是group interactive, 即将之前的二阶段orcnn-roihead换成group proposals之间存在交互的roihead
class RotatedDTBaselineGISSOnlySparse(RotatedSemiDetector):
    def __init__(self, nc, use_ss_branch, ss_branch:dict, use_refine_head, model: dict, semi_loss, train_cfg=None, test_cfg=None, symmetry_aware=False, pretrained=None):
        super(RotatedDTBaselineGISSOnlySparse, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            semi_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        # 对回归的bbox解码会用到
        self.bbox_coder = build_bbox_coder(dict(type='DistanceAnglePointCoder', angle_version='le90'))
        self.prior_generator = MlvlPointGenerator([8, 16, 32, 64, 128])

        if train_cfg is not None:
            self.freeze("teacher")
            # ugly manner to get start iteration, to fit resume mode
            self.iter_count = train_cfg.get("iter_count", 0)
            # Prepare semi-training config
            # step to start training student (not include EMA update)
            self.burn_in_steps = train_cfg.get("burn_in_steps", 5000)
            # prepare super & un-super weight
            self.sup_weight = train_cfg.get("sup_weight", 1.0)
            self.unsup_weight = train_cfg.get("unsup_weight", 1.0)
            self.weight_suppress = train_cfg.get("weight_suppress", "linear")
            self.logit_specific_weights = train_cfg.get("logit_specific_weights")
        self.symmetry_aware = symmetry_aware
        # 数据集类别数
        self.nc = nc
        # 是否开启自监督分支(旋转一致性自监督)
        self.use_ss_branch = use_ss_branch
        if self.use_ss_branch:
            self.SSBranch = SSBranch(**ss_branch)
        # 是否开启refine-roihead
        self.use_refine_head = use_refine_head





    def forward_train(self, imgs, img_metas, **kwargs):
        super(RotatedDTBaselineGISSOnlySparse, self).forward_train(imgs, img_metas, **kwargs)
        losses = dict()

        '''数据读取'''
        format_data = self.wrap_datas(imgs, img_metas, **kwargs)
        aug_orders = ['unsup_strong', 'unsup_weak']
        # 可视化稀疏数据和标签(通常注释)
        # vis_sparse_data(format_data, save_dir='./vis_strong_weak_img')

        '''对无监督分支的图像进行旋转增强处理(旋转一致性自监督学习)'''
        if self.use_ss_branch:
            # format_data参数共享内存, 不返回也会同步修改
            format_data, rand_angle, isflip = self.SSBranch.gen_aug_data(format_data, aug_orders)


        '''稀疏监督分支(burn-in阶段)'''
        if self.iter_count <= self.burn_in_steps:

            # student部分前向+计算损失
            # NOTE:这里会和稀疏GT也计算损失, 返回s_losses
            s_feat_losses = self.student.forward_train(return_fpn_feat=False, get_data=False, **format_data[aug_orders[0]])
            s_losses, _,_,_,_,_ = s_feat_losses


            





        '''burn-in结束, 开始t-s自标注+自蒸馏训练'''
        if self.iter_count > self.burn_in_steps:

            # burn_in之后的慢启动, 慢慢增加无监督分支损失的权重
            unsup_weight = self.ema_weight()

            '''得到teacher预测结果'''
            with torch.no_grad():
                # NOTE:yan, 这里额外回传类别GT, 正样本索引, fpn的多尺度特征
                t_feat_losses, t_fpn_feat = self.teacher.forward_train(return_fpn_feat=True, get_data=False, **format_data[aug_orders[1]])
                bs = t_fpn_feat[0].shape[0]
                #              [bs*21824]        [bs*21824]     [bs*21824, self.nc]   [bs*21824, 4]      [bs*21824, 1]
                t_losses, t_flat_label_preds, t_flat_cnt_logits, t_flat_cls_logits, t_flat_bbox_preds, t_flat_angle_preds = t_feat_losses
                # 调整teacher feature拼接顺序
                t_flat_cls_logits = rearrange_order(bs, t_flat_cls_logits)
                t_flat_label_preds = rearrange_order(bs, t_flat_label_preds)
                t_flat_cnt_logits = rearrange_order(bs, t_flat_cnt_logits)
                t_flat_bbox_preds = rearrange_order(bs, t_flat_bbox_preds)
                t_flat_angle_preds = rearrange_order(bs, t_flat_angle_preds)
                t_flat_rbbox_preds = torch.cat([t_flat_bbox_preds, t_flat_angle_preds], dim=-1)

                # 给特征加上batch维度(dim=0) -> [bs, 21824, dim]
                batch_t_flat_labels, batch_t_flat_cnt_preds, batch_t_flat_cls_preds, batch_t_flat_rbbox_preds = \
                    t_flat_label_preds.reshape(bs, -1, 1), t_flat_cnt_logits.reshape(bs, -1, 1), \
                    t_flat_cls_logits.reshape(bs, -1, self.nc), t_flat_rbbox_preds.reshape(bs, -1, 5)
                # 对teacher的batch dense预测进行解码+nms转化为pgt
                batch_t_nms_bboxes, batch_t_nms_scores, batch_t_nms_labels, batch_t_all_results = \
                    self.batch_decode_and_nms(bs, batch_t_flat_rbbox_preds.clone(), batch_t_flat_cls_preds.sigmoid(), batch_t_flat_cnt_preds.sigmoid(), nms_score_thres=0.1)
                '''teacher正样本挖掘(sparse-level)'''
                format_data = FNMining.fp_mining(bs, batch_t_nms_bboxes, batch_t_nms_scores, format_data, aug_orders)

            '''student稀疏监督训练(sparse-level)'''
            # student部分前向+计算损失
            s_feat_losses, s_fpn_feat = self.student.forward_train(return_fpn_feat=True, get_data=False, **format_data[aug_orders[0]])
            #              [bs*21824]        [bs*21824]     [bs*21824, self.nc]   [bs*21824, 4]      [bs*21824, 1]
            s_losses, s_flat_label_preds, s_flat_cnt_logits, s_flat_cls_logits, s_flat_bbox_preds, s_flat_angle_preds = s_feat_losses


            # TODO: 旋转一致性自监督分支, 把图像和gt都旋转(额外产生旋转图像+旋转标签)











            '''student稀疏监督训练(dense-level), 基于半监督的t监督s'''
            # # TODO: 要不要把sgt那部分mask掉, 只对其余部分自蒸馏?
            # # 调整student feature拼接顺序
            # s_flat_cls_logits = rearrange_order(bs, s_flat_cls_logits)
            # s_flat_label_preds = rearrange_order(bs, s_flat_label_preds)
            # s_flat_cnt_logits = rearrange_order(bs, s_flat_cnt_logits)
            # s_flat_bbox_preds = rearrange_order(bs, s_flat_bbox_preds)
            # s_flat_angle_preds = rearrange_order(bs, s_flat_angle_preds)
            # s_flat_rbbox_preds = torch.cat([s_flat_bbox_preds, s_flat_angle_preds], dim=-1)

            # # 给特征加上batch维度(dim=0) -> [bs, 21824, dim]
            # batch_s_flat_labels, batch_s_flat_cnt_preds, batch_s_flat_cls_preds, batch_s_flat_rbbox_preds = \
            #     s_flat_label_preds.reshape(bs, -1, 1), s_flat_cnt_logits.reshape(bs, -1, 1), \
            #     s_flat_cls_logits.reshape(bs, -1, self.nc), s_flat_rbbox_preds.reshape(bs, -1, 5)
            # # 对student的batch dense预测进行解码
            # batch_s_all_results = \
            #     self.batch_decode_and_nms(bs, batch_s_flat_rbbox_preds.clone(), batch_s_flat_cls_preds.sigmoid(), batch_s_flat_cnt_preds.sigmoid(), use_nms=False)
            # # [bs*21824, 5]
            # s_flat_rbbox_preds = batch_s_all_results[..., :5].reshape(-1, 5)
            # t_flat_rbbox_preds = batch_t_all_results[..., :5].reshape(-1, 5)

            # reshape_t_logits = [t_flat_cls_logits, t_flat_rbbox_preds, t_flat_cnt_logits]
            # reshape_s_logits = [s_flat_cls_logits, s_flat_rbbox_preds, s_flat_cnt_logits]
            # # teacher-student自蒸馏损失:
            # unsup_losses, t_joint_scores = self.semi_loss(reshape_t_logits, reshape_s_logits)

            # # 组织常规的无监督损失
            # for key, val in unsup_losses.items():
            #     if key[:4] == 'loss':
            #         losses[f"{key}_unsup"] = unsup_weight * val
            #     else:
            #         losses[key] = val





        # 组织常规稀疏监督损失
        for key, val in s_losses.items():
            if key[:4] == 'loss':
                if isinstance(val, list):
                    losses[f"{key}_sup"] = [self.sup_weight * x for x in val]
                else:
                    losses[f"{key}_sup"] = self.sup_weight * val
            else:
                losses[key] = val

        self.iter_count += 1
        return losses








    def wrap_datas(self, imgs, img_metas, **kwargs):
        """数据读取并组织数据格式
        """
        gt_bboxes = kwargs.get('gt_bboxes')
        gt_labels = kwargs.get('gt_labels')
        # preprocess
        format_data = dict()
        for idx, img_meta in enumerate(img_metas):
            tag = img_meta['tag']
            if tag not in format_data.keys():
                format_data[tag] = dict()
                format_data[tag]['img'] = [imgs[idx]]
                # 'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'tag', 'batch_input_shape'
                format_data[tag]['img_metas'] = [img_metas[idx]]
                format_data[tag]['gt_bboxes'] = [gt_bboxes[idx]]
                format_data[tag]['gt_labels'] = [gt_labels[idx]]
            else:
                format_data[tag]['img'].append(imgs[idx])
                format_data[tag]['img_metas'].append(img_metas[idx])
                format_data[tag]['gt_bboxes'].append(gt_bboxes[idx])
                format_data[tag]['gt_labels'].append(gt_labels[idx])
        for key in format_data.keys():
            format_data[key]['img'] = torch.stack(format_data[key]['img'], dim=0)
        
        return format_data









    def ema_weight(self, ):
        """burn_in之后的慢启动, 慢慢增加无监督分支损失的权重
        """
        unsup_weight = self.unsup_weight
        if self.weight_suppress == 'exp':
            target = self.burn_in_steps + 2000
            if self.iter_count <= target:
                scale = np.exp((self.iter_count - target) / 1000)
                unsup_weight *= scale
        elif self.weight_suppress == 'step':
            target = self.burn_in_steps * 2
            if self.iter_count <= target:
                unsup_weight *= 0.25
        elif self.weight_suppress == 'linear':
            target = self.burn_in_steps * 2
            if self.iter_count <= target:
                unsup_weight *= (self.iter_count - self.burn_in_steps) / self.burn_in_steps
        return unsup_weight









    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,)









    # utils: ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


    def convert_shape(self, logits, nc):
        '''将模型输出logit reshape
        '''
        cls_scores, bbox_preds, angle_preds, centernesses, fpn_feat = logits
        bs = bbox_preds[0].shape[0]

        # [[bs, cat_num, h1, w1], ...[bs, cat_num, h5, w5]] -> [total_grid_num, cat_num]
        cls_scores = convert_shape_single(cls_scores, nc)
        # [[bs, 4+1, h1, w1], ...[bs, 4+1, h5, w5]] -> [total_grid_num, 5] box是坐标和角度拼在一起
        bbox_preds = [torch.cat([x, y], dim=1).permute(0, 2, 3, 1).reshape(bs, -1, 5) for x, y in zip(bbox_preds, angle_preds)]
        bbox_preds = torch.cat(bbox_preds, dim=1).view(-1, 5)
        # [[bs, 1, h1, w1], ...[bs, 1, h5, w5]] -> [total_grid_num, 1]
        centernesses = convert_shape_single(centernesses, 1)
        # [[bs, 256, h1, w1], ...[bs, 256, h5, w5]] -> [total_grid_num, 256]
        fpn_feat = convert_shape_single(fpn_feat, 256)

        return [cls_scores, bbox_preds, centernesses, fpn_feat], bs
    



    def batch_decode_and_nms(self, bs, t_bbox_preds, t_cls_scores, t_centernesses, use_nms=True, nms_score_thres=0.1):
        '''对网络预测的dense预测框进行decode + nms
        '''
        # lvl_range默认输入图像尺寸为1024x1024,否则会有问题
        lvl_range  = [0, 16384, 20480, 21504, 21760, 21824]
        lvl_stride = [8, 16, 32, 64, 128]
        # 1. 获得grid网格点坐标 
        all_level_points = self.prior_generator.grid_priors(
            [[128, 128], [64,64], [32,32], [16,16], [8,8]],
            dtype=t_bbox_preds.dtype,
            device=t_bbox_preds.device
            )
        # [[h1*w1, 2], ..., [h5*w5, 2]] -> [total_anchor_num, 2]
        concat_points = torch.cat(all_level_points, dim=0)
        t_decode_bbox_preds = []
        # 每张图像里的框分别解码:
        for t_b_bbox_pred in t_bbox_preds:
            # 创建原始张量的副本，避免修改视图
            modified_bbox_pred = t_b_bbox_pred.clone()
            # 2. 对bbox的乘上对应的尺度
            for i in range(5):
                start, end = lvl_range[i], lvl_range[i+1]
                # 非原地操作：先计算再赋值
                modified_bbox_pred[start:end, :4] = t_b_bbox_pred[start:end, :4] * lvl_stride[i]
            # 3. 对预测的bbox解码得到最终的结果, 并得到联合置信度作为类别置信度(batch)
            t_decode_bbox_preds.append(self.bbox_coder.decode(concat_points, modified_bbox_pred).unsqueeze(0))
        t_decode_bbox_preds = torch.cat(t_decode_bbox_preds, dim=0)
        # [bs, total_anchor_num, nc] -> [bs, total_anchor_num, 1], [bs, total_anchor_num, 1]
        t_scores, t_pred_labels = torch.max(t_cls_scores, dim=-1, keepdim=True)
        t_joint_score = t_scores * t_centernesses
        # 把所有信息concat在一起(pre nms的所有框) [total_anchor_num, 7=(cx, cy, w, h, θ, score, label)]
        t_all_results = torch.cat([t_decode_bbox_preds, t_joint_score, t_pred_labels], dim=-1)

        '''nms'''
        if use_nms:
            # nms(至少会保留一个置信度最大的pgt):
            batch_nms_bboxes, batch_nms_labels, batch_nms_scores = batch_nms(t_all_results, t_cls_scores, t_joint_score, score_thr=nms_score_thres)
            return batch_nms_bboxes, batch_nms_scores, batch_nms_labels, t_all_results
        else:
            return t_all_results
    

