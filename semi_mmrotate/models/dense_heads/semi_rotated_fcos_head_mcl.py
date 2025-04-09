import torch
from mmrotate.models.dense_heads import RotatedFCOSHead
from mmrotate.models import ROTATED_HEADS
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
# yan
import matplotlib.pyplot as plt

INF = 1e8

@ROTATED_HEADS.register_module()
class SemiRotatedFCOSHeadMCL(RotatedFCOSHead):
    def __init__(self, num_classes, in_channels, beta, **kwargs):
        super(SemiRotatedFCOSHeadMCL, self).__init__(
            num_classes,
            in_channels,
            **kwargs)
        self.beta = beta

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      get_data=False,
                      **kwargs):
        if get_data:
            return self(x)
        return super(SemiRotatedFCOSHeadMCL, self).forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg,
            **kwargs
        )
    
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

        if len(pos_inds) > 0:
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
            
            # smooth the centerness based on realative scale
            img_shape = img_metas[0]['img_shape'][0:-1]
            img_scale = img_shape[0] * img_shape[1]
            scale_factor = ((flatten_bbox_targets[:, 2] * flatten_bbox_targets[:, 3]) / img_scale).pow(self.beta)
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
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            if self.separate_angle:
                loss_angle = pos_angle_preds.sum()
        
        joint_confidence_scores = flatten_cls_scores.sigmoid() * flatten_centerness.sigmoid()[:, None]
        loss_cls = self.loss_cls(
                    joint_confidence_scores, (flatten_labels, flatten_centerness_targets), avg_factor=num_pos)

        if self.separate_angle:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_angle=loss_angle,
                loss_centerness=loss_centerness)
        else:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_centerness=loss_centerness)
        
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
            num_points_per_lvl=num_points
            )

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
                   gt_bboxes.new_zeros((num_points, 1))

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
        return labels, bbox_targets, angle_targets, centerness_targets





    # def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges, num_points_per_lvl):
    #     """为单张图像计算回归、分类和角度的目标, 标签分配使用高斯分配方法(GCA)
    #         (single是指对batch里每张图像逐一执行, 而不是每个尺度)
    #         Args:
    #             - gt_bboxes:          [gt_num, 5]
    #             - gt_labels:          [gt_num]
    #             - points:             [total_anchor_num, 2] (int型, 为box中心点坐标)
    #             - regress_ranges:     [total_anchor_num, 2] ((-1, 64), (64, 128), (128, 256), (256, 512), (512, 100000000.0))
    #             - num_points_per_lvl: no used
    #         Return:
    #     """
    #     num_points = points.size(0)
    #     # 获取检测点的数量

    #     num_gts = gt_labels.size(0)
    #     # 获取该图像中的目标（ground truth, gt）的数量

    #     if num_gts == 0:
    #         # 如果没有目标，则返回全背景标签、全零的bbox回归和角度预测
    #         return gt_labels.new_full((num_points,), self.num_classes), \
    #             gt_bboxes.new_zeros((num_points, 4)), \
    #             gt_bboxes.new_zeros((num_points, 1))

    #     areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
    #     # 计算每个目标的面积（宽度 * 高度）

    #     areas = areas[None].repeat(num_points, 1)
    #     # 将面积扩展为与检测点数量匹配的形状，使得每个检测点都能计算对应目标的面积

    #     regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
    #     # 将回归范围（regress_ranges）扩展为检测点和目标数量的匹配形状

    #     points = points[:, None, :].expand(num_points, num_gts, 2)
    #     # 扩展检测点坐标，使其与每个目标一一对应

    #     gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
    #     # 将目标框扩展为与检测点数量相匹配的形状

    #     gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)
    #     # 将目标框拆分为中心点 (gt_ctr)，宽高 (gt_wh)，以及角度 (gt_angle)

    #     cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
    #     # 计算旋转角度的余弦值和正弦值

    #     rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle], dim=-1).reshape(num_points, num_gts, 2, 2)
    #     # 构建旋转矩阵，用于将检测点的偏移应用于目标框的角度旋转

    #     offset = points - gt_ctr
    #     # 计算检测点相对于目标框中心的偏移量

    #     offset = torch.matmul(rot_matrix, offset[..., None])
    #     # 应用旋转矩阵，将偏移量旋转到目标框的角度

    #     offset = offset.squeeze(-1)
    #     # 去掉多余的维度，得到旋转后的偏移量

    #     w, h = gt_wh[..., 0], gt_wh[..., 1]
    #     # 提取目标框的宽度和高度

    #     offset_x, offset_y = offset[..., 0], offset[..., 1]
    #     # 提取偏移量的x和y方向的分量

    #     left = w / 2 + offset_x
    #     right = w / 2 - offset_x
    #     top = h / 2 + offset_y
    #     bottom = h / 2 - offset_y
    #     # 根据偏移量计算出目标框的四个边界：左、右、上、下

    #     bbox_targets = torch.stack((left, top, right, bottom), -1)
    #     # 将目标框的四个边界合并为 (left, top, right, bottom) 的形式

    #     gaussian_center = offset_x.pow(2) / (w / 2).pow(2) + offset_y.pow(2) / (h / 2).pow(2)
    #     # 计算检测点在目标框中的高斯分布中心偏移，归一化到宽高的范围

    #     inside_gt_bbox_mask = gaussian_center < 1
    #     # 判断检测点是否在目标框内部，高斯分布值小于1表示在框内

    #     max_regress_distance = bbox_targets.max(-1)[0]
    #     # 计算检测点到目标框四个边界的最大回归距离

    #     inside_regress_range = (
    #         (max_regress_distance >= regress_ranges[..., 0])
    #         & (max_regress_distance <= regress_ranges[..., 1]))
    #     # 判断检测点的回归距离是否在给定的回归范围内

    #     areas[inside_gt_bbox_mask == 0] = INF
    #     areas[inside_regress_range == 0] = INF
    #     # 如果检测点不在目标框内或不在回归范围内，将目标的面积设置为无穷大，表示忽略这些检测点

    #     min_area, min_area_inds = areas.min(dim=1)
    #     # 为每个检测点选择最小目标面积的索引，这个目标将作为该检测点的分配目标

    #     labels = gt_labels[min_area_inds]
    #     # 根据最小面积索引获取对应的标签

    #     labels[min_area == INF] = self.num_classes  # set as BG
    #     # 如果该检测点没有有效的目标（面积为无穷大），则将其标签设为背景

    #     bbox_targets = bbox_targets[range(num_points), min_area_inds]
    #     # 根据最小面积索引获取对应的bbox回归目标

    #     angle_targets = gt_angle[range(num_points), min_area_inds]
    #     # 获取对应的角度回归目标
        
    #     # gaussian_center.shape=[total_anchor_num, GT_num] min_area_inds.shape=[total_anchor_num]
    #     centerness_targets = 1 - gaussian_center[range(num_points), min_area_inds]
    #     # 计算centerness目标，越靠近高斯中心，centerness值越高


    #     # sizes = [128, 64, 32, 16, 8]
    #     # cnt_targets = torch.split(centerness_targets, [size * size for size in sizes], dim=0)
    #     # for lvl, lvl_t_gaussian_center in enumerate(cnt_targets):
    #     #     if lvl!=0:continue
    #     #     lvl_t_gaussian_center = lvl_t_gaussian_center.reshape(sizes[lvl], sizes[lvl])
    #     #     plt.imshow(lvl_t_gaussian_center.cpu().numpy())
    #     #     plt.savefig('./soft_GCA.jpg', dpi=200)



    #     return labels, bbox_targets, angle_targets, centerness_targets
    #     # 返回每个检测点的标签、bbox回归目标、角度回归目标和centerness目标
























def FCOSAssigner(gt_boxes, gt_angles, classes, input_shape, strides=[8, 16, 32, 64, 128],
                 limit_ranges=[[-1,64],[64,128],[128,256],[256,512],[512,999999]], sample_radiu_ratio=1.5):
    '''FCOS正负样本分配(优化显存使用)
        # Args:
            - gt_boxes:      GTbox  [bs, max_box_nums, 4]
            - classes:       类别gt [bs, max_box_nums]
            - input_shape:   网络输入的图像尺寸 默认[640, 640]
            - strides:    
            - limit_ranges:    
            - sample_radiu_ratio:   
        # Returns:
    '''
    cls_targets_all_level = []
    cnt_targets_all_level = []
    reg_targets_all_level = []
    angle_targets_all_level = []
    pos_mask_all_level, reg_pos_mask_all_level = [], []

    bs = gt_boxes.shape[0]  # 提前计算批次大小
    max_gt_nums = gt_boxes.size(-2)  # 提前获取最大GT数量

    # 遍历每一层特征层进行正负样本分配
    for level in range(len(strides)):
        feat_w = input_shape[0] // strides[level]
        feat_h = input_shape[1] // strides[level]
        h_mul_w = feat_w * feat_h
        stride = strides[level]
        limit_range = limit_ranges[level]

        '''获得网格'''
        grids = get_grids(feat_w, feat_h, stride).type_as(gt_boxes)
        x, y = grids[:, 0], grids[:, 1]

        '''计算ltrb偏移量'''
        left_off = x[None, :, None] - gt_boxes[..., 0][:, None, :]
        top_off = y[None, :, None] - gt_boxes[..., 1][:, None, :]
        right_off = gt_boxes[..., 2][:, None, :] - x[None, :, None]
        bottom_off = gt_boxes[..., 3][:, None, :] - y[None, :, None]

        ltrb_off = torch.stack([left_off, top_off, right_off, bottom_off], dim=-1)
        areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * (ltrb_off[..., 1] + ltrb_off[..., 3])
        off_min = torch.min(ltrb_off, dim=-1)[0]
        off_max = torch.max(ltrb_off, dim=-1)[0]

        '''过滤冗余，计算正样本'''
        mask_in_gtboxes = off_min > 0
        mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])
        radiu = stride * sample_radiu_ratio

        gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2
        gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2

        c_left_off = x[None, :, None] - gt_center_x[:, None, :]
        c_top_off = y[None, :, None] - gt_center_y[:, None, :]
        c_right_off = gt_center_x[:, None, :] - x[None, :, None]
        c_bottom_off = gt_center_y[:, None, :] - y[None, :, None]

        c_ltrb_off = torch.stack([c_left_off, c_top_off, c_right_off, c_bottom_off], dim=-1)
        c_off_max = torch.max(c_ltrb_off, dim=-1)[0]
        mask_center = c_off_max < radiu

        '''高斯分配'''
        # 只在需要时扩展 gt_boxes 的宽和高
        gt_w = (gt_boxes[..., 2] - gt_boxes[..., 0]).unsqueeze(1)
        gt_h = (gt_boxes[..., 3] - gt_boxes[..., 1]).unsqueeze(1)
        gt_w = gt_w.expand(bs, h_mul_w, max_gt_nums)
        gt_h = gt_h.expand(bs, h_mul_w, max_gt_nums)

        # 旋转角度
        cos_angle = torch.cos(torch.deg2rad(gt_angles)).unsqueeze(1)
        sin_angle = torch.sin(torch.deg2rad(gt_angles)).unsqueeze(1)
        cos_angle = cos_angle.expand(bs, h_mul_w, max_gt_nums)
        sin_angle = sin_angle.expand(bs, h_mul_w, max_gt_nums)

        # 计算旋转矩阵
        rotation_matrix = torch.stack([cos_angle, sin_angle, -sin_angle, cos_angle], dim=-1).reshape(bs, h_mul_w, max_gt_nums, 2, 2)
        offsets = torch.stack([c_left_off, c_top_off], dim=-1).unsqueeze(-1)
        rotated_offsets = torch.matmul(rotation_matrix, offsets).squeeze(-1)

        rotated_offset_x = rotated_offsets[..., 0]
        rotated_offset_y = rotated_offsets[..., 1]

        # 计算高斯中心
        gaussian_center = rotated_offset_x.pow(2) / (gt_w / 2).pow(2) + rotated_offset_y.pow(2) / (gt_h / 2).pow(2)
        mask_in_gaussian = gaussian_center < 1
        pos_mask = mask_in_gtboxes & mask_in_level & mask_in_gaussian
        reg_pos_mask = pos_mask & mask_center

        '''为每个grid分配最佳gt, 得到reg_targets, angle_targets, cls_targets'''
        areas[~pos_mask] = 99999999
        areas_min_idx = torch.min(areas, dim=-1)[1]
        match_mask = torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_idx.unsqueeze(dim=-1), 1)

        reg_targets = ltrb_off[match_mask].reshape(bs, -1, 4)

        _classes = torch.gather(classes[:, None, :].expand(bs, h_mul_w, max_gt_nums), 2, areas_min_idx.unsqueeze(-1)).squeeze(-1)
        cls_targets = _classes.reshape(bs, h_mul_w, 1)

        _angles = torch.gather(gt_angles[:, None, :].expand(bs, h_mul_w, max_gt_nums), 2, areas_min_idx.unsqueeze(-1)).squeeze(-1)
        angle_targets = _angles.reshape(bs, h_mul_w, 1)

        cnt_targets = 1 - gaussian_center[match_mask].reshape(bs, -1, 1)

        '''正负样本筛选'''
        pos_mask = pos_mask.long().sum(dim=-1) >= 1
        reg_pos_mask = reg_pos_mask.long().sum(dim=-1) >= 1
        cls_targets[~pos_mask] = -1
        cnt_targets[~pos_mask] = -1
        reg_targets[~pos_mask] = -1
        angle_targets[~pos_mask] = -1

        cls_targets_all_level.append(cls_targets)
        cnt_targets_all_level.append(cnt_targets)
        reg_targets_all_level.append(reg_targets)
        angle_targets_all_level.append(angle_targets)
        pos_mask_all_level.append(pos_mask)
        reg_pos_mask_all_level.append(reg_pos_mask)

        # 释放不再需要的中间张量，防止累积占用显存
        del ltrb_off, c_ltrb_off, areas, rotation_matrix, rotated_offsets

    return (
        torch.cat(cls_targets_all_level, dim=1).reshape(-1, 1),
        torch.cat(cnt_targets_all_level, dim=1).reshape(-1, 1),
        torch.cat(reg_targets_all_level, dim=1).reshape(-1, 4),
        torch.cat(angle_targets_all_level, dim=1).reshape(-1, 1),
        torch.cat(pos_mask_all_level, dim=1).reshape(-1, 1),
        torch.cat(reg_pos_mask_all_level, dim=1).reshape(-1, 1)
    )