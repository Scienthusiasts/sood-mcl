import torch
import numpy as np
import cv2
import os
from mmrotate.core import rbbox2result, poly2obb_np, obb2poly, poly2obb, build_bbox_coder, obb2xyxy
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from mmcv.ops import box_iou_quadri, box_iou_rotated
from torchvision.transforms.functional import rotate, pad
import torch.nn.functional as F
import random





bbox_coder = build_bbox_coder(dict(type='DistanceAnglePointCoder', angle_version='le90'))
prior_generator = MlvlPointGenerator([8, 16, 32, 64, 128])

def decode_rbbox(bbox_preds, ):
    '''对bbox解码回原图的尺寸
        Args:
            t_bbox_preds: [total_anchor_num, 5=(cx, cy, w, h, θ)]

        Returns:
            decode_bbox_preds: 
    '''
    # 1. 获得grid网格点坐标
    all_level_points = prior_generator.grid_priors(
        [[128, 128], [64,64], [32,32], [16,16], [8,8]],
        dtype=bbox_preds.dtype,
        device=bbox_preds.device
        )
    # [[h1*w1, 2], ..., [h5*w5, 2]] -> [total_anchor_num, 2]
    concat_points = torch.cat(all_level_points, dim=0)
    # 2. 对bbox的乘上对应的尺度
    lvl_range  = [0, 16384, 20480, 21504, 21760, 21824]
    lvl_stride = [8, 16, 32, 64, 128]
    for i in range(5):
        bbox_preds[lvl_range[i]:lvl_range[i+1], :4] *= lvl_stride[i]
    # 3. 对预测的bbox解码得到最终的结果
    decode_bbox_preds = bbox_coder.decode(concat_points, bbox_preds)
    return decode_bbox_preds





def convert_shape_single(logits, dim, bs_dim=True):
    '''将模型输出logit(一种tensor) reshape
    '''
    bs = logits[0].shape[0]   
    # [[bs, dim, h1, w1], ...[bs, dim, h5, w5]] -> [total_grid_num, dim]
    if bs_dim: 
        reshape_logits = [x.permute(0, 2, 3, 1).reshape(bs, -1, dim) for x in logits]
        reshape_logits = torch.cat(reshape_logits, dim=1).view(-1, dim)
    else:
        reshape_logits = [x.reshape(-1, dim) for x in logits]
        reshape_logits = torch.cat(reshape_logits, dim=0).view(-1, dim)
    return reshape_logits


def convert_shape(logits, nc, wo_cls_score=False):
    '''将模型输出logit reshape
    '''
    cls_scores, bbox_preds, angle_preds, centernesses, fpn_feat = logits
    bs = bbox_preds[0].shape[0]
    # wo_cls_score=True时表示cls_scores已经是prototype refine过的
    if wo_cls_score==False:
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses) == len(fpn_feat)
        # [[bs, cat_num, h1, w1], ...[bs, cat_num, h5, w5]] -> [total_grid_num, cat_num]
        cls_scores = convert_shape_single(cls_scores, nc)
    else:
        assert len(bbox_preds) == len(angle_preds) == len(centernesses) == len(fpn_feat)

    # [[bs, 4+1, h1, w1], ...[bs, 4+1, h5, w5]] -> [total_grid_num, 5] box是坐标和角度拼在一起
    bbox_preds = [torch.cat([x, y], dim=1).permute(0, 2, 3, 1).reshape(bs, -1, 5) for x, y in zip(bbox_preds, angle_preds)]
    bbox_preds = torch.cat(bbox_preds, dim=1).view(-1, 5)
    # [[bs, 1, h1, w1], ...[bs, 1, h5, w5]] -> [total_grid_num, 1]
    centernesses = convert_shape_single(centernesses, 1)
    # [[bs, 256, h1, w1], ...[bs, 256, h5, w5]] -> [total_grid_num, 256]
    fpn_feat = convert_shape_single(fpn_feat, 256)
    return [cls_scores, bbox_preds, centernesses, fpn_feat], bs






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
        # 保证每张图片一定会有一个pgt(置信度最大的那个)(否则微调模块不好搞定)
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






def batch_tensor_random_flip(tensor, p=0.5):
    '''对图像进行随机翻转(或以概率p不翻转)'''
    if p==0.0: 
        return tensor, False
    # 生成一个随机数
    rand_num = random.uniform(0,1)
    if rand_num >= p:
        return tensor, False
    else:
        # 在第2维（h维度）上翻转, 相当于垂直翻转 (上下翻转)
        # 如果是水平翻转就第三维度
        vflip_tensor = torch.flip(tensor, dims=[2])
        return vflip_tensor, True



def batch_tensor_random_rotate(tensor, angle_range, rand=True):
    '''对张量图像进行随机角度旋转(使用镜像填充), 并返回旋转mask
        Args:
            tensor: 张量图像, 形状为[bs, 3, h, w]
            angle_range: 随机旋转的角度(角度)范围[min, max]

        Returns:
            rotaug_img: 旋转后的图像
            rand_angle: 旋转的角度
    '''
    # 生成一个随机角度
    if rand:
        rand_angle = random.uniform(angle_range[0], angle_range[1])
    else:
        return tensor, 0.
    # 先进行镜像填充
    padded_tensor, padded_len = pad_tensor(tensor, (2**0.5 - 1) / 2)
    # 再进行旋转操作
    rotaug_img = rotate(padded_tensor, rand_angle, expand=False)
    h, w = rotaug_img.shape[2:]
    # rotaug_img去掉padding的部分, 还原为原始大小(这样就能去除掉padding的黑色填充, 只保留镜像填充)
    rotaug_img = rotaug_img[..., padded_len[0]:h-padded_len[2], padded_len[1]:w-padded_len[3]]
    # ori_mask = F.interpolate(ori_mask.unsqueeze(1), scale_factor=1/16, mode='nearest').squeeze(1)
    return rotaug_img, rand_angle





def pad_tensor(tensor, k, fill='reflect'):
    '''对张量图像进行padding, 
        Args:
            tensor: 张量图像, 形状为[bs, 3, h, w]
            k:      padding的大小为边长的k倍
            fill:   padding部分的填充方式, 默认镜像填充

        Returns:
            padded_tensor: padding后的张量, 形状为[bs, 3, h, w]
            padded_len:    四条边padding的长度
    '''
    h, w = tensor.shape[2:]
    # 计算每个边界的 padding 大小 (k 倍边长)
    padding_top = padding_bottom = round(h * k)
    padding_left = padding_right = round(w * k)
    # 对张量进行 padding, 使用镜像填充
    padded_tensor = pad(tensor, (padding_left, padding_right, padding_top, padding_bottom), padding_mode=fill)
    pad_len = [padding_top, padding_left, padding_bottom, padding_right]
    return padded_tensor, pad_len