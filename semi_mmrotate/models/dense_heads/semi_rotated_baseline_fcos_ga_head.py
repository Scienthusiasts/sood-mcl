#!/usr/bin/env python
# -*- coding: utf-8 -*-
# modified from /data/yht/code/sood-mcl/semi_mmrotate/models/dense_heads/semi_rotated_baseline_fcos_head.py 
# and /data/yht/code/sood-mcl/semi_mmrotate/models/dense_heads/semi_rotated_fcos_head_mcl.py
import torch
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmrotate.models.builder import ROTATED_HEADS, build_loss
# 继承自这个(核心就是在标签分配时改成了Gaussian标签分配):
from .semi_rotated_baseline_fcos_head import SemiRotatedBLFCOSHead
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.distributed as dist

INF = 1e8

@ROTATED_HEADS.register_module()
class SemiRotatedBLFCOSGAHead(SemiRotatedBLFCOSHead):

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        # added by yan:
        img_shape = centernesses[0].shape[2:-1] * 8

        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets, angle_targets, centerness_targets = self.get_targets(
            all_level_points, gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        flatten_centerness_targets = torch.cat(centerness_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
            
        num_pos = max(reduce_mean(num_pos), 1.0)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]

        # NOTE: 使用dist.all_reduce同步操作, 当某张卡上不存在正样本时，所有卡都采用无正样本的loss计算方式
        # has_pos 判断当前gpu上是否有正样本
        has_pos = torch.tensor(len(pos_inds)>0, dtype=torch.int32, device=pos_bbox_preds.device)
        # if len(pos_inds) < 10: has_pos = torch.tensor(0, dtype=torch.int32, device=pos_bbox_preds.device)
        local_has_pos = has_pos.clone()
        # 进行多卡之间通信, 此时has_pos数值为所有gpu上has_pos的值之和, 当所有卡上都有正样本时has_pos == dist.get_world_size(), 否则不等
        dist.all_reduce(has_pos, op=dist.ReduceOp.SUM)

        if has_pos == dist.get_world_size():
            pos_points = flatten_points[pos_inds]
            if self.separate_angle:
                bbox_coder = self.h_bbox_coder
            else:
                bbox_coder = self.bbox_coder
                pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
                                           dim=-1)
                pos_bbox_targets = torch.cat(
                    [pos_bbox_targets, pos_angle_targets], dim=-1)
            pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                       pos_bbox_preds)
            pos_decoded_target_preds = bbox_coder.decode(
                pos_points, pos_bbox_targets)
            
            # smooth the centerness based on relative scale (mcl:aaai25)
            img_scale = img_shape[0] * img_shape[1]
            scale_factor = ((flatten_bbox_targets[:, 2] * flatten_bbox_targets[:, 3]) / img_scale).pow(0.2)
            flatten_centerness_targets = flatten_centerness_targets ** scale_factor
            pos_centerness_targets = flatten_centerness_targets[pos_inds]
            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
            centerness_denorm = max(
                    reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
            
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            if self.separate_angle:
                loss_angle = self.loss_angle(
                    pos_angle_preds, pos_angle_targets, avg_factor=num_pos)
        else:
            print(f"pos_inds:{len(pos_inds)}, gt_labels:{gt_labels}")
            print(gt_bboxes)
            # weight当某张卡上无正样本时, 保证其他卡上的损失为0
            weight = 1 - local_has_pos
            loss_bbox = pos_bbox_preds.sum() * weight
            loss_centerness = pos_centerness.sum() * weight
            if self.separate_angle:
                loss_angle = pos_angle_preds.sum() * weight
            print(f"local_has_pos:{local_has_pos}, loss_bbox:{loss_bbox}")

        joint_confidence_scores = flatten_cls_scores.sigmoid() * flatten_centerness.sigmoid()[:, None]
        loss_cls = self.loss_cls(
                    joint_confidence_scores, (flatten_labels, flatten_centerness_targets), avg_factor=num_pos)

        # loss以字典形式返回 
        # NOTE: added by yan, 返回flatten_labels, 计算prototype会用到
        if self.separate_angle:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_angle=loss_angle,
                loss_centerness=loss_centerness), flatten_labels, flatten_centerness, flatten_cls_scores, flatten_bbox_preds, flatten_angle_preds
        else:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_centerness=loss_centerness), flatten_labels, flatten_centerness, flatten_cls_scores, flatten_bbox_preds, flatten_angle_preds







    # @force_fp32(
    #     apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    # def loss(self,
    #          cls_scores,
    #          bbox_preds,
    #          angle_preds,
    #          centernesses,
    #          gt_bboxes,
    #          gt_labels,
    #          img_metas,
    #          gt_bboxes_ignore=None):
    #     # added by yan:
    #     img_shape = centernesses[0].shape[2:-1] * 8

    #     assert len(cls_scores) == len(bbox_preds) \
    #            == len(angle_preds) == len(centernesses)
    #     featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    #     all_level_points = self.prior_generator.grid_priors(
    #         featmap_sizes,
    #         dtype=bbox_preds[0].dtype,
    #         device=bbox_preds[0].device)
    #     labels, bbox_targets, angle_targets, centerness_targets = self.get_targets(
    #         all_level_points, gt_bboxes, gt_labels)

    #     num_imgs = cls_scores[0].size(0)
    #     # flatten cls_scores, bbox_preds and centerness
    #     flatten_cls_scores = [
    #         cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
    #         for cls_score in cls_scores
    #     ]
    #     flatten_bbox_preds = [
    #         bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
    #         for bbox_pred in bbox_preds
    #     ]
    #     flatten_angle_preds = [
    #         angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
    #         for angle_pred in angle_preds
    #     ]
    #     flatten_centerness = [
    #         centerness.permute(0, 2, 3, 1).reshape(-1)
    #         for centerness in centernesses
    #     ]
    #     flatten_cls_scores = torch.cat(flatten_cls_scores)
    #     flatten_bbox_preds = torch.cat(flatten_bbox_preds)
    #     flatten_angle_preds = torch.cat(flatten_angle_preds)
    #     flatten_centerness = torch.cat(flatten_centerness)
    #     flatten_labels = torch.cat(labels)
    #     flatten_bbox_targets = torch.cat(bbox_targets)
    #     flatten_angle_targets = torch.cat(angle_targets)
    #     flatten_centerness_targets = torch.cat(centerness_targets)
    #     # repeat points to align with bbox_preds
    #     flatten_points = torch.cat(
    #         [points.repeat(num_imgs, 1) for points in all_level_points])

    #     # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    #     bg_class_ind = self.num_classes
    #     pos_inds = ((flatten_labels >= 0)
    #                 & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        
    #     num_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
    #     num_pos = max(reduce_mean(num_pos), 1.0)

    #     pos_bbox_preds = flatten_bbox_preds[pos_inds]
    #     pos_angle_preds = flatten_angle_preds[pos_inds]
    #     pos_centerness = flatten_centerness[pos_inds]
    #     pos_bbox_targets = flatten_bbox_targets[pos_inds]
    #     pos_angle_targets = flatten_angle_targets[pos_inds]



    #     loss_bbox = pos_bbox_preds.sum() * 0
    #     loss_centerness = pos_centerness.sum() * 0
    #     if self.separate_angle:
    #         loss_angle = pos_angle_preds.sum() * 0
    #     # print(len(pos_inds))
    #     if len(pos_inds) > 0:
    #     # if False:
    #         pos_points = flatten_points[pos_inds]
    #         if self.separate_angle:
    #             bbox_coder = self.h_bbox_coder
    #         else:
    #             bbox_coder = self.bbox_coder
    #             pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
    #                                        dim=-1)
    #             pos_bbox_targets = torch.cat(
    #                 [pos_bbox_targets, pos_angle_targets], dim=-1)
    #         pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
    #                                                    pos_bbox_preds)
    #         pos_decoded_target_preds = bbox_coder.decode(
    #             pos_points, pos_bbox_targets)
            
    #         # smooth the centerness based on relative scale (mcl:aaai25)
    #         img_scale = img_shape[0] * img_shape[1]
    #         scale_factor = ((flatten_bbox_targets[:, 2] * flatten_bbox_targets[:, 3]) / img_scale).pow(0.2)
    #         flatten_centerness_targets = flatten_centerness_targets ** scale_factor
    #         pos_centerness_targets = flatten_centerness_targets[pos_inds]
    #         # centerness loss
    #         loss_centerness = self.loss_centerness(
    #             pos_centerness, pos_centerness_targets, avg_factor=num_pos)
    #         centerness_denorm = max(
    #                 reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
            
    #         loss_bbox = self.loss_bbox(
    #             pos_decoded_bbox_preds,
    #             pos_decoded_target_preds,
    #             weight=pos_centerness_targets,
    #             avg_factor=centerness_denorm)
    #         if self.separate_angle:
    #             loss_angle = self.loss_angle(
    #                 pos_angle_preds, pos_angle_targets, avg_factor=num_pos)

    #     joint_confidence_scores = flatten_cls_scores.sigmoid() * flatten_centerness.sigmoid()[:, None]
    #     loss_cls = self.loss_cls(joint_confidence_scores, (flatten_labels, flatten_centerness_targets), avg_factor=num_pos)
    #     # loss以字典形式返回 
    #     # NOTE: added by yan, 返回flatten_labels, 计算prototype会用到
    #     if self.separate_angle:
    #         return dict(
    #             loss_cls=loss_cls,
    #             loss_bbox=loss_bbox,
    #             loss_angle=loss_angle,
    #             loss_centerness=loss_centerness), flatten_labels, flatten_centerness, flatten_cls_scores, flatten_bbox_preds, flatten_angle_preds
    #     else:
    #         return dict(
    #             loss_cls=loss_cls,
    #             loss_bbox=loss_bbox,
    #             loss_centerness=loss_centerness), flatten_labels, flatten_centerness, flatten_cls_scores, flatten_bbox_preds, flatten_angle_preds






    def get_targets(self, points, gt_bboxes_list, gt_labels_list):

        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, angle_targets_list, centerness_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]
        centerness_targets_list = [centerness_targets.split(num_points, 0) for centerness_targets in centerness_targets_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        concat_lvl_centerness_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            centerness_targets = torch.cat(
                [centerness_targets[i] for centerness_targets in centerness_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
            concat_lvl_centerness_targets.append(centerness_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets, concat_lvl_centerness_targets)















    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression, classification and angle targets for a single
        image, the label assignment is GCA."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1)), \
                   gt_bboxes.new_zeros((num_points))

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        gaussian_center = offset_x.pow(2) / (w / 2).pow(2) + offset_y.pow(2) / (h / 2).pow(2)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = gaussian_center < 1

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG         
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]

        centerness_targets = 1 - gaussian_center[range(num_points), min_area_inds]
        # if (gt_labels.shape[0]==1):
        #     print(gt_labels.shape, (min_area!=INF).sum(), bbox_targets.shape, angle_targets.shape, centerness_targets.shape)

        '''可视化'''
        # root_dir = './soft_GA'
        # id = np.random.randint(0, 999999999)
        # if not os.path.exists(root_dir):os.makedirs(root_dir)
        # sizes = [128, 64, 32, 16, 8]
        # centerness_targets = torch.clamp(centerness_targets, 0, 1)
        # cnt_targets = torch.split(centerness_targets, [size * size for size in sizes], dim=0)
        # for lvl, lvl_t_gaussian_center in enumerate(cnt_targets):
        #     # 只可视化第1层特征
        #     if lvl!=0: continue
        #     lvl_t_gaussian_center = lvl_t_gaussian_center.reshape(sizes[lvl], sizes[lvl])
        #     plt.imshow(lvl_t_gaussian_center.cpu().numpy())
        #     plt.savefig(os.path.join(root_dir, f"id_{id}_lvl_{lvl}.jpg"), dpi=200)

        # print(labels.shape, bbox_targets.shape, angle_targets.shape, centerness_targets.shape)
        return labels, bbox_targets, angle_targets, centerness_targets