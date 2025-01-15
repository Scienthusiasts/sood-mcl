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
from mmrotate.core import rbbox2result, poly2obb_np, obb2poly, poly2obb, build_bbox_coder, obb2xyxy
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from  mmdet.core.bbox.samplers.sampling_result import SamplingResult
from custom.utils import *
from custom.visualize import *
# prototype实例
# from .prototype.prototype_scalewise import FCOSPrototype
# from .prototype.prototype import FCOSPrototype

@ROTATED_DETECTORS.register_module()
class RotatedDTBaseline(RotatedSemiDetector):
    def __init__(self, model: dict, prototype:dict, semi_loss, train_cfg=None, test_cfg=None, symmetry_aware=False, pretrained=None):
        super(RotatedDTBaseline, self).__init__(
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
        # NOTE: prototype, added by yan
        # self.prototype = FCOSPrototype(**prototype)
        self.nc = prototype['cat_nums']


    def convert_shape(self, logits, dim):
        '''将模型输出logit reshape, added by yan
        '''
        bs = logits[0].shape[0]   
        # [[bs, dim, h1, w1], ...[bs, dim, h5, w5]] -> [total_grid_num, dim]
        reshape_logits = [x.permute(0, 2, 3, 1).reshape(bs, -1, dim) for x in logits]
        reshape_logits = torch.cat(reshape_logits, dim=1).view(-1, dim)
        return reshape_logits
    

    def rearrange_order(self, bs, flatten_tensor):
        '''调整flatten_tensor的拼接顺序
           flatten_tensor: ============= ============= ---------- ---------- ~~~~~ ~~~~~ ·· ··
           rearrange:      ============= ---------- ~~~~~ ·· ============= ---------- ~~~~~ ··
        '''
        scale_num = 5
        lvl_range = [0, 16384, 20480, 21504, 21760, 21824]
        sizes = [16384, 4096, 1024, 256, 64]
        total_anchor_num = 21824
        rearrange_flatten_tensor = torch.zeros_like(flatten_tensor)
        for b in range(bs):
            for lvl in range(scale_num):
                rearrange_flatten_tensor[b * total_anchor_num + lvl_range[lvl]: b * total_anchor_num + lvl_range[lvl+1]] = \
                flatten_tensor[lvl_range[lvl]*2+b*sizes[lvl]:lvl_range[lvl]*2+(b+1)*sizes[lvl]]
        return rearrange_flatten_tensor


    def extract_scale_order(self, bs):
        scale_num = 5
        lvl_range = [0, 16384, 20480, 21504, 21760, 21824]
        total_anchor_num = 21824
        lvl_idx = [[] for _ in range(scale_num)]
        for b in range(bs):
            for lvl in range(scale_num):
                lvl_idx[lvl] += list(range(b * total_anchor_num + lvl_range[lvl], b * total_anchor_num + lvl_range[lvl+1]))
        return lvl_idx



    def forward_train(self, imgs, img_metas, **kwargs):
        super(RotatedDTBaseline, self).forward_train(imgs, img_metas, **kwargs)
        gt_bboxes = kwargs.get('gt_bboxes')
        gt_labels = kwargs.get('gt_labels')
        # preprocess
        format_data = dict()
        for idx, img_meta in enumerate(img_metas):
            tag = img_meta['tag']
            if tag in ['sup_strong', 'sup_weak']:
                tag = 'sup'
            if tag not in format_data.keys():
                format_data[tag] = dict()
                format_data[tag]['img'] = [imgs[idx]]
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
            
        '''全监督分支'''
        losses = dict()
        # supervised forward and loss
        # NOTE:yan, 这里额外回传类别GT, 正样本索引, fpn的多尺度特征, 用于计算prototype
        # print(format_data['sup']['gt_bboxes'][0])
        sup_losses_and_data, sup_fpn_feat = self.student.forward_train(return_fpn_feat=True, get_data=False, **format_data['sup'])
        bs = sup_fpn_feat[0].shape[0]

        sup_losses, flatten_labels, flatten_centerness, flatten_cls_scores, flatten_bbox_preds, flatten_angle_preds = sup_losses_and_data
        # 调整拼接顺序
        flatten_cls_scores = self.rearrange_order(bs, flatten_cls_scores).sigmoid()
        flatten_centerness = self.rearrange_order(bs, flatten_centerness).sigmoid()
        flatten_bbox_preds = self.rearrange_order(bs, flatten_bbox_preds)
        flatten_angle_preds = self.rearrange_order(bs, flatten_angle_preds)
        # 获得联合置信度
        flatten_joint_score = torch.einsum('ij, i -> ij', flatten_cls_scores, flatten_centerness).max(dim=-1)[0].unsqueeze(1)
        # [bs, total_anchor_num, 6=(cx, cy, w, h, θ, joint_score)] 这里的 cx, cy, w, h, θ格式还不对, 还需要解码
        rbb_preds = torch.cat([flatten_bbox_preds, flatten_angle_preds, flatten_joint_score], dim=-1).reshape(bs, -1, 6)

        '''有监督分支进行预测框去噪微调'''
        # 断开refine-head与主体检测器的梯度
        sup_fpn_feat = [fpn_feat.detach() for fpn_feat in sup_fpn_feat]
        # 0.对原始预测解码
        # 这里rbb_preds不加.detach() 会报inplace op的错
        rbb_preds = self.rbb_decode(bs, sup_fpn_feat, rbb_preds.detach())
        # 1.将batch拆开, 变为list, 符合roi_head.forward_train的输入格式
        proposal_list = []
        for i in range(bs):
            proposal_list.append(rbb_preds[i, :, :5])
        # 2.送入roi head进行微调
        # TODO: 有一个问题, 调用forward_train函数时, 会再执行一次正负样本分配而不是采用已经分好的组
        # [[bs, 256, h1, w1], ...], [[roi_nums, 5], ...], [[gt_nums], ...], [[gt_nums, 5], ...]
        # 注意 roi_head.forward_train接受的回归框坐标的格式是[cx, cy, w, h, a]
        # hproposal_list = copy.deepcopy(proposal_list)
        roi_losses = self.student.roi_head.forward_train(sup_fpn_feat, format_data['sup']['gt_labels'], proposal_list, format_data['sup']['gt_bboxes'], format_data['sup']['gt_labels'])
        # 3.组织微调模块的损失
        for key, val in roi_losses.items():
            if key[:4] == 'loss':
                losses[f"{key}_refine_sup"] = self.sup_weight * val
            else:
                losses[key] = val
                
        # 根据GT框对网络输出的densebbox进行分组(还在考虑有监督分支是否要分组, 还是直接基于GT训练):
        # batch_group_iou, batch_group_id_mask, batch_keep_dense_bboxes = grouping_by_gts(rbb_preds, format_data['sup']['gt_bboxes'], iou_thres=0.3, score_thres=1e-6)
        # vis_grouping_batch(format_data['sup']['gt_bboxes'], batch_group_iou, batch_group_id_mask, batch_keep_dense_bboxes, format_data['sup'], './vis_sup_gt_grouping')
        # 可视化(一般情况下注释)
        vis_sup_bboxes_batch(self.teacher, format_data['sup'], bs, self.nc, flatten_cls_scores.reshape(bs, -1, self.nc).detach(), sup_fpn_feat, rbb_preds, './vis_res_wo_nms')





        '''有监督分支更新prototypes'''
        # # 对输出的特征进行reshape [bs * total_anchor_num, dim]
        # sup_fpn_feat = self.convert_shape(sup_fpn_feat, dim=256)
        # # 调整拼接顺序
        # flatten_centerness = self.rearrange_order(bs, flatten_centerness).sigmoid()
        # flatten_cls_scores = self.rearrange_order(bs, flatten_cls_scores).sigmoid()
        # flatten_joint_score = torch.einsum('ij, i -> ij', flatten_cls_scores, flatten_centerness)
        # cat_mask = self.rearrange_order(bs, flatten_labels)
        # # 获取每一个尺度的特征索引(batch无关)
        # # lvl_idx = self.extract_scale_order(bs)
        # # 更新prototype
        # # prototype_loss = self.prototype(sup_fpn_feat.clone().detach(), cat_mask, lvl_idx)
        # prototype_loss = self.prototype(sup_fpn_feat, cat_mask, flatten_cls_scores, flatten_centerness, 'sup')
        # losses["loss_prototype_sup"] = self.sup_weight * prototype_loss

        # # # 可视化(only for experimental validation)
        # # with torch.no_grad():
        # #     pos_mask = (cat_mask >= 0) & (cat_mask < self.nc) 
        # #     # 获取图像文件名和图像
        # #     batch_img_names = [img_metas['ori_filename'] for img_metas in format_data['sup']['img_metas']]
        # #     batch_imgs = format_data['sup']['img']
        # #     # 可视化
        # #     # self.prototype.vis_heatmap(sup_fpn_feat.data, batch_imgs, batch_img_names, pos_mask, lvl_idx, flatten_centerness, flatten_cls_scores)
        # #     self.prototype.vis_heatmap(sup_fpn_feat.data, batch_imgs, batch_img_names, pos_mask, flatten_centerness, flatten_cls_scores)


        # 组织全监督损失
        for key, val in sup_losses.items():
            if key[:4] == 'loss':
                if isinstance(val, list):
                    losses[f"{key}_sup"] = [self.sup_weight * x for x in val]
                else:
                    losses[f"{key}_sup"] = self.sup_weight * val
            else:
                losses[key] = val

        '''无监督分支'''
        if self.iter_count > self.burn_in_steps:
            # Train Logic
            # unsupervised forward
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

            # get student data
            # NOTE: yan, 这里可以调整教师和学生增强的顺序
            aug_orders = ['unsup_strong', 'unsup_weak']     # 1.正常
            # aug_orders = ['unsup_weak', 'unsup_strong']   # 2.调换
            # aug_orders = ['unsup_strong', 'unsup_strong'] # 3.一致(强)
            # aug_orders = ['unsup_weak', 'unsup_weak']     # 4.一致(弱)
            with torch.no_grad():
                # get teacher data
                # NOTE:yan, 这里额外回传类别GT, 正样本索引, fpn的多尺度特征, 用于计算prototype
                teacher_logits, t_fpn_feat = self.teacher.forward_train(return_fpn_feat=True, get_data=True, **format_data[aug_orders[1]])
                teacher_logits = list(teacher_logits)
                teacher_logits.append(t_fpn_feat)
            # NOTE:yan, 这里额外回传类别GT, 正样本索引, fpn的多尺度特征, 用于计算prototype(这里保留s_fpn_feat, 后续可以和t_fpn_feat做自蒸馏)
            student_logits, s_fpn_feat = self.student.forward_train(return_fpn_feat=True, fpn_feat_grad=True, get_data=True, **format_data[aug_orders[0]])
            student_logits = list(student_logits)
            student_logits.append(s_fpn_feat)

            '''无监督分支更新prototypes'''
            # t_cls_scores, t_cnt_scores = teacher_logits[0], teacher_logits[3]
            # # t_cls_scores, t_cnt_scores = student_logits[0], student_logits[3]
            # # 对输出的特征进行reshape [bs * total_anchor_num, dim]
            # t_cls_scores = self.convert_shape(t_cls_scores, dim=self.nc).sigmoid()
            # t_cnt_scores = self.convert_shape(t_cnt_scores, dim=1).sigmoid()
            # t_fpn_feat = self.convert_shape(t_fpn_feat, dim=256)

            # # 返回prototype loss, 以及refine的score shape均为 [bs*total_anchor_num, cat_num]
            # prototype_loss, refine_t_joint_score, refine_t_cls_score = self.prototype(t_fpn_feat, None, t_cls_scores, t_cnt_scores, 'unsup')
            # # prototype损失(如果有的话)
            # losses["loss_prototype_unsup"] = unsup_weight * prototype_loss
            # # 替换teacher的score为refine的特征
            # # teacher_logits[0] = refine_t_cls_score
            # teacher_logits.append(refine_t_joint_score)

            # # # 可视化(only for experimental validation)
            # # with torch.no_grad():
            # #     # 获取图像文件名和图像
            # #     batch_img_names = [img_metas['ori_filename'] for img_metas in format_data['unsup_weak']['img_metas']]
            # #     batch_imgs = format_data['unsup_weak']['img']
            # #     # 可视化
            # #     self.prototype.vis_heatmap_unsup(t_fpn_feat.data, batch_imgs, batch_img_names, t_cnt_scores, t_cls_scores)



            '''半监督分支计算损失'''
            unsup_losses = self.semi_loss(self.student, self.teacher, teacher_logits, student_logits, img_metas=format_data[aug_orders[1]], stu_img_metas=format_data[aug_orders[0]])

            # 组织无监督损失
            for key, val in self.logit_specific_weights.items():
                if key in unsup_losses.keys():
                    unsup_losses[key] *= val
            for key, val in unsup_losses.items():
                if key[:4] == 'loss':
                    losses[f"{key}_unsup"] = unsup_weight * val
                    # losses[f"{key}_unsup"] = val
                else:
                    losses[key] = val
        self.iter_count += 1

        return losses


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,)




    def rbb_decode(self, bs, sup_fpn_feat, rbb_preds):
        '''对网络得到的框的回归值解码
        '''
        # 0.对角度再乘一个可学习的尺度(这里不加with torch.no_grad()显存会持续增加直到OOM, 不知道为啥)
        # NOTE: 这里改成ORCNN ROI Head(或许其实是config的问题?)似乎就不需要了(加上之后可视化角度不对, 去掉后角度就正常了), 很奇怪:
        # NOTE: 所以之前训练其实角度都是不太对的? TvT(25-1-15)
        # with torch.no_grad():
        #     rbb_preds[:, :, 4] = self.student.bbox_head.scale_angle(rbb_preds[:, :, 4])
        # 1. 获得grid网格点坐标
        all_level_points = self.prior_generator.grid_priors(
            [featmap.size()[-2:] for featmap in sup_fpn_feat],
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