import torch
import torch.nn as nn
import numpy as np
import cv2
from mmrotate.core import rbbox2result, poly2obb_np, obb2poly, poly2obb, build_bbox_coder, obb2xyxy
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from mmcv.ops import box_iou_quadri, box_iou_rotated
from torchvision.transforms.functional import rotate, pad
import random
import matplotlib.pyplot as plt




def init_weights(model, init_type, mean=0, std=0.01):
    '''权重初始化方法
    '''
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if init_type=='he':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if init_type=='normal':
                nn.init.normal_(module.weight, mean=mean, std=std)  # 使用高斯随机初始化
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)







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


def convert_shape(logits, nc, wo_cls_score=False, fpn=True):
    '''将模型输出logit reshape
    '''
    if fpn:
        cls_scores, bbox_preds, angle_preds, centernesses, fpn_feat = logits
    else:
        cls_scores, bbox_preds, angle_preds, centernesses = logits
    bs = bbox_preds[0].shape[0]
    # wo_cls_score=True时表示cls_scores已经是prototype refine过的
    if wo_cls_score==False:
        # [[bs, cat_num, h1, w1], ...[bs, cat_num, h5, w5]] -> [total_grid_num, cat_num]
        cls_scores = convert_shape_single(cls_scores, nc)

    # [[bs, 4+1, h1, w1], ...[bs, 4+1, h5, w5]] -> [total_grid_num, 5] box是坐标和角度拼在一起
    bbox_preds = [torch.cat([x, y], dim=1).permute(0, 2, 3, 1).reshape(bs, -1, 5) for x, y in zip(bbox_preds, angle_preds)]
    bbox_preds = torch.cat(bbox_preds, dim=1).view(-1, 5)
    # [[bs, 1, h1, w1], ...[bs, 1, h5, w5]] -> [total_grid_num, 1]
    centernesses = convert_shape_single(centernesses, 1)
    if fpn:
        # [[bs, 256, h1, w1], ...[bs, 256, h5, w5]] -> [total_grid_num, 256]
        fpn_feat = convert_shape_single(fpn_feat, 256)
        return [cls_scores, bbox_preds, centernesses, fpn_feat], bs
    else:
        return [cls_scores, bbox_preds, centernesses], bs






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









def batch_nms(bboxes, cls_scores, centerness, score_thr=0.2):
    '''rotated nms
        Args:
            bboxes:     dense preds, [bs, total_anchor_num, 7=(cx, cy, w, h, θ, score, label)]
            cls_scores: [bs, total_anchor_num, cls_num] (已经过sigmoid)
            centerness: [bs, total_anchor_num] (已经过sigmoid)
            score_thr:  NMS置信度阈值

        Returns:
            batch_det_bboxes: list([post_nms_num, 6=(cx, cy, w, h, θ, score)], ..., [...])
            batch_det_labels: list([post_nms_num], [...])
            batch_det_scores: list([post_nms_num, cls_num], [...])
    '''
    nc = cls_scores[0].shape[-1]
    batch_det_bboxes, batch_det_labels, batch_det_scores = [], [], []
    # 对batch每张图片分别nms
    for i in range(bboxes.shape[0]):
        # 异常处理, 舍弃那些面积等于0的框
        # area = bboxes[i, :, 2] * bboxes[i, :, 3]
        # correct_mask = area>0
        # correct_bboxes = bboxes[i][correct_mask]
        # correct_cls_scores = cls_scores[i][correct_mask]
        # correct_centerness = centerness[i][correct_mask]
        # 异常处理方式2:
        bboxes[i][:, 2:4] = torch.clamp(bboxes[i][:, 2:4], min=2.)
        correct_bboxes = bboxes[i]
        correct_cls_scores = cls_scores[i]
        correct_centerness = centerness[i]

        # multiclass_nms_rotated的输入需要包含背景类预测
        padding = correct_cls_scores.new_zeros(correct_cls_scores.shape[0], 1)
        cls_scores_w_bg = torch.cat([correct_cls_scores, padding], dim=1)
        '''nms'''
        # 输入: [total_box_num, 5], [total_box_num, 16] -> 输出: [nms_num, 6], [nms_num]
        det_bboxes, det_labels, thr_idx, nms_idx = multiclass_nms_rotated(
            multi_bboxes=correct_bboxes[:, :5],            # [total_box_num, 5]
            multi_scores=cls_scores_w_bg,                # [total_box_num, 16]
            score_thr=score_thr,                               # 0.05
            nms={'iou_thr': 0.1},                        # {'iou_thr': 0.1}
            max_num=2000,                                # 2000
            score_factors=correct_centerness,            # centerness[i, :] bboxes[i, :, 5]
            return_inds=True
            )
        # 保证每张图片一定会有一个pgt(置信度最大的那个)(否则微调模块不好搞定)
        if det_bboxes.shape[0]==0: 
            max_idx = torch.argmax(correct_bboxes[:, 5])
            det_bboxes = correct_bboxes[max_idx,:6].unsqueeze(0)
            det_labels = correct_bboxes[max_idx,-1].unsqueeze(0).long()
            det_scores = correct_cls_scores[max_idx, :].unsqueeze(0)
        else:
            # 为什么索引是thr_idx // nc, 因为这个索引基于把所有类别都展平成1维了，需要除以类别数才是正确的索引
            det_scores = correct_cls_scores[thr_idx // nc][nms_idx]

        batch_det_bboxes.append(det_bboxes)
        batch_det_labels.append(det_labels)
        batch_det_scores.append(det_scores)
        # 检查索引是否正确(不对会报越界的cuda error):
        # batch_det_bboxes.append(bboxes[i][thr_idx // nc][nms_idx])
        # batch_det_labels.append(torch.argmax(cls_scores[i][thr_idx // nc][nms_idx],dim=1))

    return batch_det_bboxes, batch_det_labels, batch_det_scores





def batch_grouping_by_nmsboxes(dense_bboxes, nms_bboxes, nms_scores, iou_thres=0.1, score_thres=1e-5, K=8, oversampling_ratio=1.0):
    '''根据pgt或GT对dense的预测结果进行分组聚类(指标:RIoU,是否还应该考虑类别?)
        Args:
            dense_bboxes: [bs, total_anchor_num, 7=(cx, cy, w, h, θ, score, label)]
            nms_bboxes:   list([post_nms_num, 6=(cx, cy, w, h, θ, score)], ..., [...])
            nms_scores:   list([post_nms_num, cls_num], [...]) (已经过sigmoid)
            iou_thres:    group里的框与gt框的最小IoU
            score_thres:  group里的框的置信度最小值

        Returns:
            batch_group_iou:         每个dense bbox 与匹配gt或pgt的IoU    List([keep_num], ,..., [...])     
            batch_keep_dense_bboxes: score阈值筛选与grouping IoU筛选后保留的dense bboxes的初始信息(坐标, 得分, 所属类别)([keep_num, 7], ,..., [...])  
            nms_bboxes:              更新后的nms_bboxes, 去除了那些group数量=0的nms_bbox
            nms_scores:              更新后的nms_scores, 去除了那些group数量=0的nms_scores
    '''
    # 遍历batch每张图像的预测结果:
    batch_groups_iou,  batch_groups_bboxes = [], []
    for i in range(dense_bboxes.shape[0]):
        # 先采用一个很小的联合置信度阈值卡正样本(不满足的样本不参与聚类, 减少计算量)
        pos_mask = dense_bboxes[i, :, 5] > score_thres
        # [FIX] 首先获取筛选后的dense_bboxes
        dense_bboxes_pos = dense_bboxes[i][pos_mask]  # [pos_num, 7]
        
        # 每个dense样本和所有gt计算riou [pre_nms_num, post_nms_num], 得到pre_nms和post_nms的框的两两IoU
        riou = box_iou_rotated(dense_bboxes_pos[:, :5], nms_bboxes[i][:, :5])
        # 每个dense box属于与最大IoU的post_nms的框的那一组
        group_ious, group_ids_mask = riou.max(dim=1)
        # 那些最大IoU小于阈值的dense box也得舍弃
        group_ids_mask[group_ious<iou_thres]=-1

        # 对齐每个group里box的个数
        keep_gt_bboxes_id = []
        groups_iou, groups_bboxes = [], []
        for group_id in range(nms_bboxes[i].shape[0]):
            # 找到属于当前group的所有框
            mask = group_ids_mask == group_id
            # 还需要考虑变成NMSBox的GTBox又没有其他densebox和它一个group的情况:
            if (mask.sum()==0 and nms_bboxes[i][group_id, 5]>=0.99):
                group_iou = torch.tensor([1.0], device=dense_bboxes.device)
                gt_label = torch.argmax(nms_scores[i][group_id])
                gt_box = nms_bboxes[i][group_id]
                group_dense_bboxes = torch.tensor([[gt_box[0], gt_box[1], gt_box[2], gt_box[3], gt_box[4], gt_box[5], gt_label]], device=dense_bboxes.device)
            elif (mask.sum()!=0):
                # [FIX] 使用筛选后的dense_bboxes_pos而不是原始dense_bboxes[i]
                group_iou = group_ious[mask]
                group_dense_bboxes = dense_bboxes_pos[mask]  # 关键修改点
            
            if (mask.sum()!=0 or nms_bboxes[i][group_id, 5]>=0.99):
                # 选择iou前topk(k=8)的那些框
                k = min(K, group_iou.shape[0])
                _, idx = torch.topk(group_iou, k)
                # 如果group里样本数量不足k个, 则在group里随机采样进行padding
                if(k<K):
                    sampled_idx = torch.randint(low=0, high=idx.shape[0], size=(K-k,))
                    padded_idx = idx[sampled_idx]
                    idx = torch.cat([idx, padded_idx])

                groups_iou.append(group_iou[idx])
                # TODO:对额外随机采样的Box进行加噪(因为和其他box是重复的)?
                groups_bboxes.append(group_dense_bboxes[idx])
                keep_gt_bboxes_id.append(group_id)
        
        nms_bboxes[i] = nms_bboxes[i][keep_gt_bboxes_id]
        nms_scores[i] = nms_scores[i][keep_gt_bboxes_id]
        batch_groups_iou.append(torch.stack(groups_iou))
        batch_groups_bboxes.append(torch.stack(groups_bboxes))
    return batch_groups_iou, batch_groups_bboxes, nms_bboxes, nms_scores









def rearrange_order(bs, flatten_tensor):
    '''调整flatten_tensor的拼接顺序 (/data/yht/code/sood-mcl/semi_mmrotate/models/rotated_dt_baseline_ss_gi_head.py会用到)
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









def gen_bboxes_sine_embedings_cxcywha(bboxes, embed_dims, temperature=20):
    """将旋转框坐标(4点)生成位置编码
        Args:
            bboxes:     形状为 [bs, N, 8] 的张量, 8代表 (x0,y0,x1,y1,x2,y2,x3,y3), 且是归一化坐标
            embed_dims: 位置编码的维度(最终是8*embed_dims, 因为是8个坐标的编码拼在一起)
        Returns:
            pe: 位置编码
    """
    # 位置编码的公式: sin(2π·cx / temperature^(2i/embed_dims)) 其他分量(cy, w, h, angle)
    scale = 2 * torch.pi
    dim_t = torch.arange(embed_dims, dtype=torch.float32, device=bboxes.device)
    dim_t = temperature ** (2 * (dim_t // 2) / embed_dims)
    
    embed = bboxes * scale
    # pos.shape = [bs, group_num, 8, embed_dims]
    pos = embed.unsqueeze(-1) / dim_t

    # 和原始的正余弦编码在位置交错编码不一样，这里先正弦编码再余弦编码
    sin_pos = pos[:, :, :, 0::2].sin()  
    cos_pos = pos[:, :, :, 1::2].cos()  
    # [bs, group_num, 8, embed_dims]
    pe = torch.stack((sin_pos, cos_pos), dim=4).flatten(3, 4).flatten(2, 3)
    
    return pe



def normalize_polybboxes(bboxes, img_w, img_h):
    """将旋转矩形框的坐标从像素值归一化到[0,1]范围
        Args:
            bboxes: 形状为 [bs, N, 8] 的张量, 8代表 (x0,y0,x1,y1,x2,y2,x3,y3)
            img_w: 原始图像宽度 (像素)
            img_h: 原始图像高度 (像素)
        Returns:
            归一化后的bboxes, 形状仍为 [bs, N, 8]
    """
    # 创建归一化尺度因子 [1, 1, 8]
    scale_factor = bboxes.new_tensor([img_w, img_h, img_w, img_h, img_w, img_h, img_w, img_h]).view(1, 1, 8)
    # 复制bboxes避免原地修改
    normalized_bboxes = bboxes.clone()
    # 归一化 x0,y0,x1,y1,x2,y2,x3,y3
    normalized_bboxes = normalized_bboxes / scale_factor
    return normalized_bboxes





def revert_shape_single(reshape_logits, feat_sizes, bs=1):
    """将 reshape_logits 恢复为原始的多尺度 logits 组织形式
    
    Args:
        reshape_logits (Tensor): [total_grid_num] 或 [total_grid_num, dim]
        feat_sizes (list): 各层特征图的尺寸列表，如 [128, 64, 32, 16, 8]
        bs (int): batch size，默认为1
        
    Returns:
        list: 恢复后的 logits 列表，每个元素形状为 [bs, dim, h, w]
    """
    # 确保输入是二维的 [total_grid_num, dim]
    if reshape_logits.ndim == 1:
        reshape_logits = reshape_logits.unsqueeze(-1)  # [total_grid_num, 1]
    
    total_grid_num, dim = reshape_logits.shape
    
    # 计算各层网格数并验证
    grid_sizes = [s*s for s in feat_sizes]
    assert sum(grid_sizes) * bs == total_grid_num, (
        f"特征图尺寸总和{sum(grid_sizes)}*bs={bs}≠输入tensor的网格数{total_grid_num}")
    
    # 先按bs分割，再按特征图层级分割
    split_by_bs = torch.split(reshape_logits, sum(grid_sizes), dim=0)
    restored_logits = []
    
    for b in range(bs):
        split_logits = torch.split(split_by_bs[b], grid_sizes, dim=0)
        for i, (logit, size) in enumerate(zip(split_logits, feat_sizes)):
            if b == 0:  # 第一次处理时初始化
                restored_logits.append(torch.empty(bs, dim, size, size, 
                                                device=reshape_logits.device))
            restored_logits[i][b] = logit.view(size, size, dim).permute(2, 0, 1)
    
    return restored_logits