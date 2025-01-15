import torch
import numpy as np
import cv2
import os
from mmrotate.core import rbbox2result, poly2obb_np, obb2poly, poly2obb, build_bbox_coder, obb2xyxy
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from mmcv.ops import box_iou_quadri, box_iou_rotated






def rbox2PolyNP(obboxes):
    """将5参旋转框表示法转换为8参四边形表示法
        Args:
            obboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-180, 0)

        Returns:
            polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
    """
    center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    cos, sin = np.cos(theta), np.sin(theta)
    # 旋转矩阵
    vector1 = np.concatenate([-w/2 * cos, -w/2 * sin], axis=-1)
    vector2 = np.concatenate([h/2 * sin, -h/2 * cos], axis=-1)

    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    order = obboxes.shape[:-1]
    return np.concatenate([point1, point2, point3, point4], axis=-1).reshape(*order, 8)






def batch_nms(t_results, t_cls_scores):
    '''rotated nms
        Args:
            t_results:      dense preds, [bs, total_anchor_num, 7=(cx, cy, w, h, θ, score, label)]
            t_cls_scores:   [bs, total_anchor_num, cls_num]

        Returns:
            det_bboxes:      [post_nms_num, 6=(cx, cy, w, h, θ, score)]
            det_labels:      [post_nms_num]
    '''
    batch_det_bboxes, batch_det_labels = [], []
    for i in range(t_results.shape[0]):
        '''nms'''
        # 输入: [total_box_num, 5], [total_box_num, 16] -> 输出: [nms_num, 6], [nms_num]
        det_bboxes, det_labels = multiclass_nms_rotated(
            multi_bboxes=t_results[i, :, :5],            # [total_box_num, 5]
            multi_scores=t_cls_scores[i],                # [total_box_num, 16]
            score_thr=0.2,                               # 0.05
            nms={'iou_thr': 0.1},                        # {'iou_thr': 0.1}
            max_num=2000,                                # 2000
            score_factors=t_results[i, :, 5]             # [total_box_num]
            )
        # 保证每张图片一定会有一个pgt(否则微调模块不好搞定)
        if det_bboxes.shape[0]==0: 
            max_idx = torch.argmax(t_results[i, :, 5])
            det_bboxes = t_results[i, max_idx,:6].unsqueeze(0)
            det_labels = t_results[i, max_idx,-1].unsqueeze(0).long()

        batch_det_bboxes.append(det_bboxes)
        batch_det_labels.append(det_labels)

    return batch_det_bboxes, batch_det_labels





def batch_grouping_by_gts(dense_bboxes, gt_bboxes, iou_thres=0.1, score_thres=1e-5):
    '''根据pgt或GT对dense的预测结果进行分组聚类(指标:RIoU,是否还应该考虑类别?)
        Args:
            dense_bboxes: [bs, total_anchor_num, 7=(cx, cy, w, h, θ, score, label)]
            gt_bboxes:    List([post_nms_num, 5], ..., [...])
            iou_thres:    group里的框与gt框的最小IoU
            score_thres:  group里的框的置信度最小值

        Returns:
            batch_group_iou:         每个dense bbox 与匹配gt或pgt的IoU            List([keep_num], ,..., [...])     
            batch_group_id_mask:     每个dense bbox 所属的group(与gt的id对应)     List([keep_num], ,..., [...])  
            batch_keep_dense_bboxes: score阈值筛选与grouping IoU筛选后保留的dense bboxes List([keep_num, 7], ,..., [...])  
    '''
    # 遍历batch每张图像的预测结果:
    batch_group_iou, batch_group_id_mask, batch_keep_dense_bboxes = [], [], []
    for i in range(dense_bboxes.shape[0]):
        # 采用联合置信度卡正样本(不满足的样本不参与聚类, 减少计算量)
        pos_mask = dense_bboxes[i, :, 5] > score_thres
        # dense样本和gt计算riou [pre_nms_num, post_nms_num], 得到pre_nms和post_nms的框的两两IoU
        riou = box_iou_rotated(dense_bboxes[i, :, :5][pos_mask], gt_bboxes[i][:, :5])
        # 每个pre_nms属于与最大IoU的post_nms的框的那一组
        group_iou, group_ids_mask = riou.max(dim=1)
        # 再次过滤掉iou小于阈值的框
        fgd_mask = group_iou > iou_thres
        # [pre_nms_num], [pre_nms_num], [pre_nms_num, 6]
        group_iou, group_id_mask, keep_dense_bboxes = group_iou[fgd_mask], group_ids_mask[fgd_mask], dense_bboxes[i, pos_mask, :][fgd_mask]
        batch_group_iou.append(group_iou)
        batch_group_id_mask.append(group_id_mask)
        batch_keep_dense_bboxes.append(keep_dense_bboxes)
    return batch_group_iou, batch_group_id_mask, batch_keep_dense_bboxes