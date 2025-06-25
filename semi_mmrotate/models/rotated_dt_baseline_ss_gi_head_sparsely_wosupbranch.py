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
from custom.fgclip_distill import FGCLIPDistillBranch
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

            # for i in format_data[aug_orders[0]]['img_metas']:
            #     print(i['ori_filename'])
            # print('='*100)

            





        '''burn-in结束, 开始t-s自标注+自蒸馏训练'''
        if self.iter_count > self.burn_in_steps:

            # burn_in之后的慢启动, 慢慢增加无监督分支损失的权重
            unsup_weight = self.ema_weight()

            with torch.no_grad():
                # get teacher data
                # NOTE:yan, 这里额外回传类别GT, 正样本索引, fpn的多尺度特征
                # TODO: 对teacher的预测结果执行正样本挖掘
                teacher_logits, t_fpn_feat = self.teacher.forward_train(return_fpn_feat=True, get_data=True, **format_data[aug_orders[1]])
                teacher_logits = list(teacher_logits)
                teacher_logits.append(t_fpn_feat)
            
            


            # student部分前向+计算损失(自标注)
            # NOTE:这里会和稀疏GT也计算损失, 返回s_losses
            # TODO: 稀疏标签+teacher挖掘的正样本 -> student训练自标注损失(sparse-level). 此外看看这里能否返回正负样本分配的dense结果(基于稀疏标签)
            s_feat_losses, s_fpn_feat = self.student.forward_train(return_fpn_feat=True, get_data=False, **format_data[aug_orders[0]])
            #        torch.Size([43648]) torch.Size([43648]) torch.Size([43648, 16]) torch.Size([43648, 4]) torch.Size([43648, 1])
            s_losses, s_flatten_labels, s_flatten_centerness, s_flatten_cls_scores, s_flatten_bbox_preds, s_flatten_angle_preds = s_feat_losses
            bs=s_fpn_feat[0].shape[0]
            # 将flatten后的特征reshape成多尺度列表的形式
            s_cls_scores = revert_shape_single(s_flatten_cls_scores, [128, 64, 32, 16, 8], bs=bs)
            s_bbox_preds = revert_shape_single(s_flatten_bbox_preds, [128, 64, 32, 16, 8], bs=bs)
            s_angle_preds = revert_shape_single(s_flatten_angle_preds, [128, 64, 32, 16, 8], bs=bs)
            s_centerness_preds = revert_shape_single(s_flatten_centerness, [128, 64, 32, 16, 8], bs=bs)
            student_logits = [s_cls_scores, s_bbox_preds, s_angle_preds, s_centerness_preds]
            student_logits.append(s_fpn_feat)



            '''格式调整(旋转一致性自监督学习)'''
            if self.use_ss_branch:
                # 将原始图像的推理结果与旋转图像的推理结果分离开
                s_ori_logits,  s_rot_logits = [], []
                for pred in student_logits:
                    ori_logits, rot_logits = [], [] 
                    ori_logits = [x[0].unsqueeze(0) for x in pred]
                    rot_logits = [x[1].unsqueeze(0) for x in pred]
                    s_ori_logits.append(ori_logits)
                    s_rot_logits.append(rot_logits)
                # 对输出的特征进行reshape
                # [total_grid_num, cat_num], [total_grid_num, 4+1], [total_grid_num, 1], [total_grid_num, 256]
                reshape_s_ori_logits, bs = convert_shape(s_ori_logits, self.nc)
                reshape_s_rot_logits, bs = convert_shape(s_rot_logits, self.nc)
                reshape_t_logits, bs = convert_shape(teacher_logits, self.nc)
            else:
                '''去除旋转一致性自监督学习的格式调整'''
                s_ori_logits = student_logits
                reshape_s_ori_logits, bs = convert_shape(s_ori_logits, self.nc)
                reshape_t_logits, bs = convert_shape(teacher_logits, self.nc)






            '''无监督分支损失'''
            # weight_mask旋转自监督分支会用到
            # TODO: 正负样本分配的dense结果(基于稀疏标签) 对teacher的dense pred进行修正得到修正后的dense-pred -> student训练自蒸馏损失(dense-level)
            unsup_losses, t_joint_scores = self.semi_loss(
                self.student, self.teacher, reshape_t_logits, reshape_s_ori_logits, s_ori_logits, teacher_logits, bs, 
                sup_img_metas=format_data[aug_orders[1]], unsup_img_metas=format_data[aug_orders[0]], 
                # gihead微调伪框
                use_refine_head=self.use_refine_head
            )

            # 组织常规的无监督损失
            for key, val in self.logit_specific_weights.items():
                if key in unsup_losses.keys():
                    unsup_losses[key] *= val
            for key, val in unsup_losses.items():
                if key[:4] == 'loss':
                    losses[f"{key}_unsup"] = unsup_weight * val
                else:
                    losses[key] = val


            '''旋转一致性自监督分支(旋转一致性自监督学习)'''
            if self.use_ss_branch:
                # beta = 2
                # weight_mask = t_joint_scores.pow(beta)
                weight_mask = 1 / (1 + torch.exp(-10 * t_joint_scores)).pow(10) - 1/1024. 
                ss_loss_joint_score, ss_loss_box = self.SSBranch.forward(format_data, aug_orders, reshape_s_ori_logits, reshape_s_rot_logits, weight_mask, rand_angle, isflip)

                losses['ss_loss_joint_score'] = ss_loss_joint_score
                losses['ss_loss_box'] = ss_loss_box 



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



