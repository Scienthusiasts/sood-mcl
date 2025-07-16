#!/usr/bin/env python
# -*- coding: utf-8 -*-
# modified from /data/yht/code/sood-mcl/semi_mmrotate/models/dense_heads/semi_rotated_baseline_fcos_head.py 
# and /data/yht/code/sood-mcl/semi_mmrotate/models/dense_heads/semi_rotated_fcos_head_mcl.py
import torch
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean, build_bbox_coder
from mmrotate.models.builder import ROTATED_HEADS, build_loss
# 继承自这个(核心就是在标签分配时改成了Gaussian标签分配):
from .sparse_rotated_baseline_fcos_head import SparseRotatedBLFCOSHead
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.distributed as dist
from custom.fn_mining import FNMining

INF = 1e8

@ROTATED_HEADS.register_module()
class SparseRotatedBLFCOSGAHeadWORegGT(SparseRotatedBLFCOSHead):



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
        use_fn_weight = True

        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device
        )
        labels, bbox_targets, angle_targets, centerness_targets, sample_pos_weights, sample_all_weights = \
            self.get_targets(all_level_points, gt_bboxes, gt_labels)

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
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])
        # 挖掘样本权重
        flatten_sample_pos_weights = torch.cat(sample_pos_weights)
        flatten_sample_all_weights = torch.cat(sample_all_weights)



        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_fn_weight_masks = flatten_sample_pos_weights[pos_inds]
        # print(f" | {pos_fn_weight_masks.min()}, {pos_fn_weight_masks.max()}, {pos_fn_weight_masks.mean()} | ")
        # NOTE: 使用dist.all_reduce同步操作, 当某张卡上不存在正样本时，所有卡都采用无正样本的loss计算方式
        # has_pos 判断当前gpu上是否有正样本
        has_pos = torch.tensor(len(pos_inds)>0, dtype=torch.int32, device=pos_bbox_preds.device)
        local_has_pos = has_pos.clone()
        # 进行多卡之间通信, 此时has_pos数值为所有gpu上has_pos的值之和, 当所有卡上都有正样本时has_pos == dist.get_world_size(), 否则不等
        dist.all_reduce(has_pos, op=dist.ReduceOp.SUM)

        if has_pos == dist.get_world_size():
            pos_points = flatten_points[pos_inds]
            if self.separate_angle:
                bbox_coder = self.h_bbox_coder
            else:
                bbox_coder = self.bbox_coder
                pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds], dim=-1)
                pos_bbox_targets = torch.cat([pos_bbox_targets, pos_angle_targets], dim=-1)
            pos_decoded_bbox_preds = bbox_coder.decode(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = bbox_coder.decode(pos_points, pos_bbox_targets)
            
            # smooth the centerness based on relative scale (mcl:aaai25)
            img_scale = img_shape[0] * img_shape[1]
            scale_factor = ((flatten_bbox_targets[:, 2] * flatten_bbox_targets[:, 3]) / img_scale).pow(0.2)
            flatten_centerness_targets = flatten_centerness_targets ** scale_factor
            pos_centerness_targets = flatten_centerness_targets[pos_inds]

            # 中心度损失 /home/yht/.conda/envs/sood-mcl/lib/python3.9/site-packages/mmdet/models/losses/cross_entropy_loss.py
            loss_centerness = self.loss_centerness(pos_centerness, pos_centerness_targets, avg_factor=num_pos)
            if use_fn_weight: loss_centerness *= pos_fn_weight_masks 
            loss_centerness = loss_centerness.sum() / num_pos
            centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

            # 回归损失
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            if use_fn_weight: loss_bbox *= pos_fn_weight_masks
            loss_bbox = loss_bbox.sum() / centerness_denorm


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

        # 类别损失 /home/yht/.conda/envs/sood-mcl/lib/python3.9/site-packages/mmdet/models/losses/gfocal_loss.py
        loss_cls = self.loss_cls(joint_confidence_scores, (flatten_labels, flatten_centerness_targets), avg_factor=num_pos)
        if use_fn_weight: loss_cls *= flatten_sample_all_weights
        loss_cls = loss_cls.sum() / num_pos

        # loss以字典形式返回 
        # NOTE: added by yan, 返回flatten_labels, 计算prototype会用到
        # print(f"{loss_cls.item()}, {loss_bbox.item()}, {loss_centerness.item()}")
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

        # 补充: fn_weight_mask_list
        labels_list, bbox_targets_list, angle_targets_list, centerness_targets_list, sample_pos_weight_list, sample_all_weight_list = \
            multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_labels_list,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points
            )

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [bbox_targets.split(num_points, 0)for bbox_targets in bbox_targets_list]
        angle_targets_list = [angle_targets.split(num_points, 0)for angle_targets in angle_targets_list]
        centerness_targets_list = [centerness_targets.split(num_points, 0) for centerness_targets in centerness_targets_list]
        # 加权挖掘样本
        sample_pos_weight_list = [sample_pos_weight.split(num_points, 0) for sample_pos_weight in sample_pos_weight_list]
        sample_all_weight_list = [sample_all_weight.split(num_points, 0) for sample_all_weight in sample_all_weight_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        concat_lvl_centerness_targets = []
        # 加权挖掘样本
        concat_lvl_sample_pos_weights = []
        concat_lvl_sample_all_weights = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            centerness_targets = torch.cat(
                [centerness_targets[i] for centerness_targets in centerness_targets_list])
            # 加权挖掘样本
            sample_pos_weights = torch.cat(
                [sample_pos_weight[i] for sample_pos_weight in sample_pos_weight_list])
            sample_all_weights = torch.cat(
                [sample_all_weight[i] for sample_all_weight in sample_all_weight_list])
            
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
            concat_lvl_centerness_targets.append(centerness_targets)
            # 加权挖掘样本
            concat_lvl_sample_pos_weights.append(sample_pos_weights)
            concat_lvl_sample_all_weights.append(sample_all_weights)

        return (concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_angle_targets, concat_lvl_centerness_targets, \
                concat_lvl_sample_pos_weights, concat_lvl_sample_all_weights)















    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges, num_points_per_lvl):
        """Compute regression, classification and angle targets for a single
        image, the label assignment is GCA."""
        fn_num = 0
        gt_scores = torch.ones_like(gt_labels)
        sgt_bboxes = gt_bboxes
        sgt_labels = gt_labels
        pos_thres = self.pos_thres
        pos_beta = self.pos_beta
        neg_beta = self.neg_beta
        fn_mining_flag = gt_bboxes.shape[1]==6
        # 把bbox坐标和其置信度区分开
        if fn_mining_flag:
            gt_scores = gt_bboxes[:, -1]
            gt_bboxes = gt_bboxes[:, :-1]
            fn_num = (gt_scores<1.0).sum()
            # 挖掘的样本中有一些会成为gt(sgt)，有一些则还是负样本(根据pos_thres划分), 有一些则是回归损失时的正样本(reg_gt)
            # 但是这些样本在计算损失时都会加权
            sgt_bboxes = gt_bboxes[gt_scores>=pos_thres]
            sgt_labels = gt_labels[gt_scores>=pos_thres]
            fn_bboxes = gt_bboxes[gt_scores<1.0]
            fn_scores = gt_scores[gt_scores<1.0]



        # 这里得到的targets只包含阈值不小于pos_thres的样本
        cls_targets, bbox_targets, angle_targets, centerness_targets = self.gen_target_feat_single(sgt_bboxes, sgt_labels, points, regress_ranges)
        if fn_num > 0:
            # 这里得到的targets只包含挖掘出的样本, 不包含sgt
            fn_target = FNMining.gen_fn_target_feat_single(fn_bboxes, points)
            # 将target转为mask, 用于后续对损失加权(减少哪些挖掘出的正样本对损失的贡献)
            # 这里会进一步根据pos_thres将挖掘出的样本再划分为正样本和负样本
            sample_pos_weight, sample_all_weight = FNMining.get_sample_weight(fn_target, fn_scores, pos_thres, pos_beta, neg_beta)
        else:
            sample_pos_weight, sample_all_weight = torch.ones_like(centerness_targets), torch.ones_like(centerness_targets)
            

        '''可视化'''
        # if fn_num > 0:
        #     # 准备特征数据
        #     sizes = [128, 64, 32, 16, 8]
        #     cnt_targets = torch.clamp(centerness_targets.clone(), 0, 1)
        #     reg_cnt_targets = torch.clamp(reg_centerness_targets.clone(), 0, 1)
            
        #     # 按尺度分割特征
        #     feature_data = {
        #         'pos_gau': [t.reshape(size, size) for t, size in zip(torch.split(cnt_targets, [s*s for s in sizes]), sizes)],
        #         'reg_pos_gau': [t.reshape(size, size) for t, size in zip(torch.split(reg_cnt_targets, [s*s for s in sizes]), sizes)],
        #         'pos_weight': [t.reshape(size, size) for t, size in zip(torch.split(sample_pos_weight, [s*s for s in sizes]), sizes)],
        #         'reg_pos_weight': [t.reshape(size, size) for t, size in zip(torch.split(sample_reg_pos_weight, [s*s for s in sizes]), sizes)],
        #         'all_weight': [t.reshape(size, size) for t, size in zip(torch.split(sample_all_weight, [s*s for s in sizes]), sizes)]
        #     }
        #     # 统一可视化
        #     visualize_all_layers(feature_data, save_dir='./vis_combined')


        #        [21824]     [21824, 4]    [21824, 1]         [21824]             [21824]            [21824] 
        return cls_targets, bbox_targets, angle_targets, centerness_targets, sample_pos_weight, sample_all_weight  



    def gen_target_feat_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        """得到fcos正负样本分配后的gt特征图(仅对sparse gt)
        """
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

        cls_targets = gt_labels[min_area_inds]
        cls_targets[min_area == INF] = self.num_classes  # set as BG         
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]
        centerness_targets = 1 - gaussian_center[range(num_points), min_area_inds]

        return cls_targets, bbox_targets, angle_targets, centerness_targets





# 可视化:================================================================================================================================================================


def visualize_all_layers(feature_data, save_dir='./vis_combined'):
    """
    紧凑型可视化所有层的特征图到一张5x5的画布中
    Args:
        feature_data: 包含所有特征数据的字典，结构如下：
            {
                'pos_gau': [tensor_lvl0, tensor_lvl1, ...],  # 5个尺度的特征
                'reg_pos_gau': [tensor_lvl0, tensor_lvl1, ...],
                'pos_weight': [tensor_lvl0, tensor_lvl1, ...],
                'reg_pos_weight': [tensor_lvl0, tensor_lvl1, ...],
                'all_weight': [tensor_lvl0, tensor_lvl1, ...]
            }
        save_dir: 保存路径
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 创建紧凑的5x5画布（去掉了colorbar的空间）
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))  # 缩小画布尺寸
    plt.subplots_adjust(wspace=0.05, hspace=0.15)     # 减少子图间距
    
    # 特征类型和对应的标题
    feature_types = [
        ('pos_gau', 'PosGau'),
        ('reg_pos_gau', 'RegPosGau'),
        ('pos_weight', 'PosW'),
        ('reg_pos_weight', 'RegPosW'),
        ('all_weight', 'AllW')
    ]
    
    # 遍历每种特征类型（每行）
    for row_idx, (feat_key, feat_title) in enumerate(feature_types):
        # 遍历每个尺度（每列）
        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            
            # 获取当前特征图数据
            feat_map = feature_data[feat_key][col_idx]
            if isinstance(feat_map, torch.Tensor):
                feat_map = feat_map.cpu().numpy()
            
            # 可视化（移除colorbar相关代码）
            ax.imshow(feat_map, vmin=0, vmax=1, cmap='viridis')
            
            # 仅在第一行显示列标题（尺度）
            if row_idx == 0:
                ax.set_title(f"L{col_idx}", fontsize=8, pad=2)
            
            # 仅在第一列显示行标题（特征类型）
            if col_idx == 0:
                ax.set_ylabel(feat_title, fontsize=8, labelpad=2)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
    
    # 保存图像
    id = np.random.randint(0, 999999999)
    plt.savefig(
        os.path.join(save_dir, f'{id}.jpg'), 
        dpi=200, 
        bbox_inches='tight',
        pad_inches=0.1  # 进一步减少边缘空白
    )
    plt.close()