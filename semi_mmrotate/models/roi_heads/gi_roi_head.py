# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmrotate.models import build_loss
from mmrotate.core import rbbox2roi, build_bbox_coder
from mmrotate.models.builder import ROTATED_HEADS
from mmrotate.models.roi_heads.rotate_standard_roi_head import RotatedStandardRoIHead
from mmrotate.models.builder import (ROTATED_HEADS, build_head, build_roi_extractor,
                       build_shared_head)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmdet.core import multi_apply, unmap
from custom.utils import *
from custom.visualize import *
from custom.loss import QFLv2, HungarianWithIoUMatching
# for debug:
# torch.autograd.set_detect_anomaly(True)


class CustomBatchNorm2d(nn.Module):
    '''当bs=1时, 跳过BN
    '''
    def __init__(self, channels):
        super(CustomBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        if x.size(0) == 1:
            return x
        else:
            return self.bn(x)
        

class ShareHead(nn.Module):
    '''Head部分回归和分类之前的共享特征提取层(目的是将ROI的7x7压缩到1x1)
    '''
    def __init__(self, channel):
        super(ShareHead, self).__init__()
        # DWConv
        self.dw_conv = nn.Sequential(
            # 深度卷积 (分组数groups=in_channels)
            nn.Conv2d(channel, channel, 3, 1, groups=channel, bias=False),
            # 逐点卷积 (1x1卷积调整通道数)
            nn.Conv2d(channel, channel, 1, 1, bias=False)
        )
        self.convBlocks = nn.Sequential(
            self._make_conv_block(channel), 
            self._make_conv_block(channel), 
            self._make_conv_block(channel),
        )
        # 权重初始化
        # init_weights(self.convBlocks, 'he')
        init_weights(self.convBlocks, 'normal', 0, 0.01)


    def _make_conv_block(self, out_channels):
        layers = []
        layers.append(self.dw_conv)
        layers.append(CustomBatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.convBlocks(x)
        return x   










@ROTATED_HEADS.register_module()
class GIRoIHead(BaseModule):

    def __init__(self,
                 bbox_roi_extractor,
                 bbox_coder,
                 nc,
                 assigner='HungarianWithIoUMatching',
                 **wargs
                 ):

        super(GIRoIHead, self).__init__()
        self.mode = 'sup'
        self.eps = 1e-7
        self.nc = nc
        self.hidden_dim = 256
        # 提取roi特征, 即RRoIAlign操作
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        # self.bbox_coder用于对回归结果进行编解码操作
        self.bbox_coder = build_bbox_coder(bbox_coder)
        # 正负样本分配方法
        self.assigner = HungarianWithIoUMatching(nc=nc, iou_weight=5, cls_weight=0.1, l1_weight=5)
        # 损失函数
        self.reg_loss = build_loss(dict(type='RotatedIoULoss', reduction='none', mode='log'))
        self.cls_loss = QFLv2()
        # 将roi的7x7压缩到1x1
        self.roi_conv = ShareHead(channel=self.hidden_dim)
        self.attention = MultiheadAttention(embed_dims=self.hidden_dim, num_heads=8, dropout=0.0, batch_first=True)
        self.attention_norm = build_norm_layer(dict(type='LN'), self.hidden_dim)[1]
        self.ffn = FFN(embed_dims=self.hidden_dim, feedforward_channels=2048)
        self.ffn_norm = build_norm_layer(dict(type='LN'), self.hidden_dim)[1]
        # 分类头
        self.cls_fcs = nn.ModuleList()
        for _ in range(1):
            self.cls_fcs.append(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), self.hidden_dim)[1])
            self.cls_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=False))) # inplace=True
        self.fc_cls = nn.Linear(self.hidden_dim, self.nc)
        # 回归头
        self.reg_fcs = nn.ModuleList()
        for _ in range(1):
            self.reg_fcs.append(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
            self.reg_fcs.append(
                build_norm_layer(dict(type='LN'), self.hidden_dim)[1])
            self.reg_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=False))) # inplace=True
        self.fc_reg = nn.Linear(self.hidden_dim, 5)


    def add_GT2NMS_boxes_single(self, nms_bboxes, nms_labels, nms_scores, gt_bboxes, gt_labels, img_meta, pgt_scores=None):
        """如果存在有GTBox和任何NMSBox都匹配不上, 则把这个GTBox加入到NMSBox中(仅在有监督训练时调用)
            Args:
                nms_bboxes: [post_nms_num, 6=(cx, cy, w, h, θ, score)]
                nms_labels: [post_nms_num]
                nms_scores: [post_nms_num, cls_num] (已经过sigmoid)
                gt_bboxes:  [group_nums, 5=(cx, cy, w, h, θ)]  真实的gt / pgt
                gt_labels:  [group_nums]  真实的gt标签 / pgt标签
                img_meta:   图像信息

            Return:

        """
        # gt_pscore用来表示GT框加入到nms框后的伪得分
        gt_pscore = 0.99
        iou = box_iou_rotated(gt_bboxes, nms_bboxes[:, :5]).cpu().numpy()
        # 先进行匈牙利匹配(不是最终的匹配结果)
        _, _, gt_idx, pred_idx = self.assigner.assign_single(gt_bboxes, nms_bboxes, gt_labels, nms_scores, img_meta, maxiou_reassign=False)
        # 找出那些没有和任何pred box匹配上的GT
        gt_zero_iou_mask = torch.ones(gt_bboxes.shape[0], dtype=torch.bool).to(gt_bboxes.device)
        gt_zero_iou_mask[gt_idx] = False
        # 找出那些和任何NMSBox都没有交集的GTBox
        zero_gt_idx_mask = iou[gt_idx, pred_idx]==0
        if zero_gt_idx_mask.sum()>0:
            zero_gt_idx = gt_idx[zero_gt_idx_mask]
            gt_zero_iou_mask[zero_gt_idx] = True

        gt_zero_iou_num = gt_zero_iou_mask.sum()
        if(gt_zero_iou_num > 0):
            score = torch.ones(gt_zero_iou_num, 1).to(gt_bboxes.device)
            # TODO:对gt_zero_box的cxcywhθ做一个随机高斯扰动, 增加鲁棒性
            gt_zero_box = torch.cat([gt_bboxes[gt_zero_iou_mask], score*gt_pscore], dim=1)
            nms_bboxes = torch.cat([nms_bboxes, gt_zero_box])
            gt_zero_label = gt_labels[gt_zero_iou_mask]
            nms_labels = torch.cat([nms_labels, gt_zero_label])
            # 创建基础One-Hot矩阵(全为eps) [n, cls_num]
            gt_scores = torch.full((gt_zero_iou_num, self.nc), self.eps, dtype=torch.float32, device=gt_bboxes.device)
            # 将正确类别位置设为gt_pscore
            gt_scores.scatter_(1, gt_zero_label.unsqueeze(1), gt_pscore)
            nms_scores = torch.cat([nms_scores, gt_scores])
        return nms_bboxes, nms_labels, nms_scores




    def grouping(self, rbb_preds, cls_score, centerness, img_meta, gt_bboxes=None, gt_labels=None):
        """对一阶段网络的dense预测结果进行分组
            Args:
                rbb_preds:  [bs, total_anchor_num, 7=(cx, cy, w, h, θ, joint_score, label)]    
                cls_score:  [bs, total_anchor_num, cls_num] (已经过sigmoid)
                centerness: [bs, total_anchor_num] (已经过sigmoid)
                img_meta:   图像信息
                gt_bboxes:  list([group_nums, 5=(cx, cy, w, h, θ)], ..., [...]) 真实的gt / pgt
                gt_labels:  list([group_nums], ..., [...]) 真实的gt标签 / pgt标签
            Return:
                batch_group_iou:    list([group_nums, box_per_group], ..., [...]) 与匹配gt的IoU (以batch分组, 里面包含batch_nms_bboxes)
                batch_group_bboxes: list([group_nums, box_per_group, 7], ..., [...]) 筛选后保留的dense bboxes的完整信息(cx, cy, w, h, θ, score, label) (以batch分组)
                batch_nms_bboxes:   list([group_nums, 6=(cx, cy, w, h, θ, score)], ..., [...]) nms后的bbox, 会过滤掉一些 (以batch分组)
                batch_nms_scores:   list([group_nums, cls_num], ..., [...]) nms后的score, 会过滤掉一些 (以batch分组)
        """
        # 1.nms(至少会保留一个置信度最大的box): 
        # list([post_nms_num, 6=(cx, cy, w, h, θ, score)], ..., [...]) list([post_nms_num], [...]) list([post_nms_num, cls_num], [...])
        batch_nms_bboxes, batch_nms_labels, batch_nms_scores = batch_nms(rbb_preds, cls_score, centerness, score_thr=0.15)
        if self.mode in ['train_sup', 'train_unsup']:
            # 如果存在有GTBox和任何NMSBox的IoU都为0, 则把这个GTBox加入到NMSBox中(仅在训练时调用)(后续还得处理变成NMSBox的GTBox又没有其他densebox和它一个group的情况)
            batch_nms_bboxes, batch_nms_labels, batch_nms_scores = multi_apply(self.add_GT2NMS_boxes_single, batch_nms_bboxes, batch_nms_labels, batch_nms_scores, gt_bboxes, gt_labels, img_meta['img_metas'])
        # 2.根据nms的框对网络输出的densebbox进行分组(也会对batch_nms_bboxe,batch_nms_scores进行过滤): 
        # 注:batch_group_iou, batch_group_bboxes都是根据与group中心的IoU从大到小排序的, 第一个就是group中心box, 即对应batch_nms_bboxes里的box
        # TODO:排序是否会误导模型学习偏见?是否需要除了第一个外打乱其他的顺序?
        # list([group_nums, box_per_group], ..., [...]) list([group_nums, box_per_group, 7], ..., [...]) list([group_nums, 7], ..., [...])
        batch_group_iou, batch_group_bboxes, batch_nms_bboxes, batch_nms_scores = batch_grouping_by_nmsboxes(rbb_preds, batch_nms_bboxes, batch_nms_scores, iou_thres=0.1, score_thres=1e-6, K=8)
        # 可视化分组结果(依据nms_boxes分组, 通常注释)
        # vis_grouping_batch(batch_nms_bboxes, batch_group_iou, batch_group_bboxes, img_meta, './vis_sup_gt_grouping')
        return batch_group_iou, batch_group_bboxes, batch_nms_bboxes, batch_nms_scores
    

    def group_interact_forward(self, dense_roi_feats, nms_roi_feats):
        """group交互模块(用自注意力+动态卷积进行交互)
        """
        # TODO: 目前是组内交互, 只是局部交互. 是否需要加入全局交互, 即对nms_roi_feats之间的交互?)
        # 只取每个组的第一个预测作为query, 简化运算(这个预测代表每个group中最好的预测, 这个预测和其他预测交互就行了, 其他无所谓)
        # MHSA + LN:  out = Q = [total_group_nums, 1, 256], k = v = [total_group_nums, nums_per_group, 256]
        represent_roi_feat = dense_roi_feats[:, 0, :].unsqueeze(1)
        represent_interative_roi_feat = self.attention_norm(self.attention(query=represent_roi_feat, key=dense_roi_feats, value=dense_roi_feats))
        # TODO:加一个类似SparseRCNN里的动态卷积(输入是dense_roi_feats和nms_roi_feats)?
        # [total_group_nums, 1, 256]
        return represent_interative_roi_feat


    def head_forward(self, dense_roi_feats, nms_roi_feats):
        """这部分可以堆叠
            Args:
                dense_roi_feats: [total_group_nums, nums_per_group, 256] 从特征图中抠出并经过sharehead的groupboxes的roi特征
                nms_roi_feats:   [total_group_nums, 256] 从特征图中抠出的nmsboxes的roi特征
            Return:
                dense_roi_feats: [total_group_nums, nums_per_group, 256] 交互后的roi特征
                cls_score:       [total_gt_nums, cls_num] 预测头输出分类结果(未解码)
                reg_delta:       [total_gt_nums, 5] 预测头输出回归结果(未解码)
        """
        N = dense_roi_feats.shape[0]

        '''接下来就是nms_roi_feats和dense_roi_feats如何交互了'''
        # [total_group_nums, nums_per_group, 256], [total_group_nums, 256] -> [total_group_nums, 256]
        represent_interative_roi_feat = self.group_interact_forward(dense_roi_feats, nms_roi_feats).squeeze(1)
        # FFN [total_group_nums, 1, 256]
        represent_interative_roi_feat = self.ffn_norm(self.ffn(represent_interative_roi_feat))

        cls_feat = represent_interative_roi_feat
        reg_feat = represent_interative_roi_feat

        '''head部分前向'''
        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)
        cls_score = self.fc_cls(cls_feat).view(N, self.nc)
        reg_delta = self.fc_reg(reg_feat).view(N, 5)

        # dense_roi_feats可以作为下一层的dense_roi_feats输入(如果有多层, cls_score, reg_delta则用于输出或计算损失)
        # 有一些问题:如果要堆叠多层，那group里的每个框都需要参与预测, 因为下一层需要依赖上一层的框解码后重新去backbone里抠出roi特征
        return dense_roi_feats, cls_score, reg_delta
    

    def forward(self, fpn_feat, rbb_preds, cls_score, centerness, img_meta, gt_bboxes=None, gt_labels=None):
        """前向
            Args:
                fpn_feat       list([bs, 256, h1, w1], ..., [bs, 256, h5, w5]) 多尺度backbone特征 (以尺度分组而不是batch)
                rbb_preds:     [bs, total_anchor_num, 7=(cx, cy, w, h, θ, joint_score, label)]    
                cls_score:     [bs, total_anchor_num, cls_num] (已经过sigmoid)
                centerness:    [bs, total_anchor_num] (已经过sigmoid)
                img_meta:      batch图像和图像信息
                gt_bboxes:     list([group_nums, 5=(cx, cy, w, h, θ)], ..., [...]) 真实的gt / pgt
                gt_labels:     list([group_nums], ..., [...]) 真实的gt标签 / pgt标签

            Return:
                batch_nms_bboxes: list([group_nums, 6=(cx, cy, w, h, θ, score)], ..., [...]) nms后的bbox, 会过滤掉一些 (以batch分组)
                batch_nms_scores:   list([group_nums, cls_num], ..., [...]) nms后的score, 会过滤掉一些 (以batch分组)
                cls_score:        [total_gt_nums, cls_num] 预测头输出分类结果(未解码)
                reg_delta:        [total_gt_nums, 5] 预测头输出回归结果(未解码)

        """
        '''对一阶段网络的dense预测结果进行分组'''
        # list([group_nums, box_per_group], ..., [...]) list([group_nums, box_per_group, 7], ..., [...]) list([group_nums, 7], ..., [...]) list([group_nums, cls_num], ..., [...])
        batch_group_iou, batch_group_bboxes, batch_nms_bboxes, batch_nms_scores = self.grouping(rbb_preds, cls_score, centerness, img_meta, gt_bboxes, gt_labels)
        '''roialign+sharehead'''
        # 这一步只是进行格式转换同时合并list为tensor而已: list([group_nums * nums_per_group, 5], ,..., [...]) -> [group_nums * nums_per_group, 6=(batch_ind, cx, cy, w, h, a)]
        dense_rois = rbbox2roi([boxes.reshape(-1, 7)[:, :5] for boxes in batch_group_bboxes])
        # rroialign操作: [total_group_nums * nums_per_group, 256, 256, 7, 7] (total_group_nums已经是把batch拼在一起了)
        nms_rois = rbbox2roi([boxes[:, :5] for boxes in batch_nms_bboxes])
        # 从特征图中抠出roi [total_group_nums * nums_per_group, 256, 7, 7]  total_group_nums=每个batch的group拼在一起
        dense_roi_feats = self.bbox_roi_extractor(fpn_feat, dense_rois)
        nms_roi_feats = self.bbox_roi_extractor(fpn_feat, nms_rois)
        # 将roi_feat处理成1维 [total_group_nums, nums_per_group, 256]
        dense_roi_feats = self.roi_conv(dense_roi_feats).reshape(-1, 8, self.hidden_dim)
        nms_roi_feats = self.roi_conv(nms_roi_feats).reshape(-1, self.hidden_dim)

        '''bboxhead(包含特征交互和分类回归)'''
        # 输出的dense_roi_feats已经经过交互 (cls_score, reg_delta则是未解码的原始特征, 不是最终预测结果)
        # TODO: 是否需要堆叠预测头?
        # [total_group_nums, nums_per_group, 256] [total_group_nums, 16], [total_group_nums, 5]
        dense_roi_feats, cls_score, reg_delta = self.head_forward(dense_roi_feats, nms_roi_feats)

        return batch_nms_bboxes, batch_nms_scores, cls_score, reg_delta


    def loss(self, fpn_feat, dense_rbb_preds, dense_cls_score, dense_centerness, gt_bboxes, gt_labels, img_meta, train_mode):
        """前向+损失(训练时调用)
            Args:
                fpn_feat           list([bs, 256, h1, w1], ..., [bs, 256, h5, w5]) 多尺度backbone特征 (以尺度分组而不是batch)
                dense_rbb_preds:   [bs, total_anchor_num, 7=(cx, cy, w, h, θ, joint_score, label)]    
                dense_cls_score:   [bs, total_anchor_num, cls_num] (已经过sigmoid)
                dense_centerness:  [bs, total_anchor_num] (已经过sigmoid)
                gt_bboxes:         list([group_nums, 5=(cx, cy, w, h, θ)], ..., [...]) 真实的gt
                gt_labels:         list([group_nums], ..., [...]) 真实的gt标签
                img_meta:          batch图像信息和标注
                train_mode:        当前的训练模式(训练有监督数据=sup, 训练伪标签=unsup)

            Return:
                losses: 字典组织形式的损失
        """
        self.mode = train_mode
        '''前向(cls_score, reg_delta 未解码)'''
        # list([group_nums, 7], ..., [...]), list([group_nums, cls_num], ..., [...]) [total_group_nums, 16], [total_group_nums, 5]
        batch_nms_bboxes, batch_nms_scores, cls_score, reg_delta = self.forward(fpn_feat, dense_rbb_preds, dense_cls_score, dense_centerness, img_meta, gt_bboxes, gt_labels)
        # TODO:直接用一阶段的分配结果(sup分支)
        # 匈牙利匹配分配正负样本目前发现以下几个情况:
        # 差距过大(IoU=0)的框被匹配在一起(已解决, 把GT加入nms_box), 和GT差距很小的框由于类别不正确因此未匹配上(已解决, 用maxIoU重匹配)
        batch_idx, match_pred_gt_bboxes, match_gt_logits = self.assigner.assign(gt_bboxes, batch_nms_bboxes, gt_labels, batch_nms_scores, img_meta)

        '''回归损失'''
        valid_gt_mask = match_pred_gt_bboxes[1][:, 2]!=0
        fg_num = valid_gt_mask.sum()
        bboxes_preds = self.bbox_coder.decode(match_pred_gt_bboxes[0][valid_gt_mask], reg_delta[valid_gt_mask])
        bbox_loss = self.reg_loss(bboxes_preds, match_pred_gt_bboxes[1][valid_gt_mask]).mean()
        with torch.no_grad():
            # 计算refine_box和gt的iou与nms_box与box的IoU，并计算差值, 能大概反馈改善程度
            refine_iou = box_iou_rotated(bboxes_preds, match_pred_gt_bboxes[1][valid_gt_mask], aligned=True)
            nms_iou = box_iou_rotated(match_pred_gt_bboxes[0][valid_gt_mask], match_pred_gt_bboxes[1][valid_gt_mask], aligned=True)
            diff_iou = (refine_iou-nms_iou).mean()

        # 可视nms_bboxes和gi_head输出的bboxes对比(默认注释)
        # if self.mode=='train_unsup':
        #     all_bboxes_preds = self.bbox_coder.decode(match_pred_gt_bboxes[0], reg_delta)
        #     vis_gi_head_bboxes_batch(img_meta, len(batch_nms_bboxes), batch_idx, match_pred_gt_bboxes[0], all_bboxes_preds, './vis_gi_bboxes_unsup')

        #     vis_cls_score = cls_score.sigmoid()
        #     cat_nms_scores = torch.cat(batch_nms_scores, dim=0)
        #     vis_HM_scores(vis_cls_score.unsqueeze(0), match_gt_logits.unsqueeze(0), img_meta, './vis_unsup_score')

        '''分类损失'''
        cls_score = cls_score.sigmoid()
        # cls_loss = self.cls_loss(cls_score, match_gt_logits, use_weight=True, beta=2.0, reduction='none').sum() / fg_num
        cls_loss = self.cls_loss(cls_score, match_gt_logits, use_weight=True, beta=2.0, reduction='none').mean()
        # 总损失
        losses = {
            'gi_cls_loss':cls_loss, 
            'gi_reg_loss':bbox_loss,
            'iou_improve':diff_iou
            }
        return losses


    def infer(self, fpn_feat, dense_rbb_preds, dense_cls_score, dense_centerness, img_meta):
        """前向+损失(训练时调用)
            Args:
                fpn_feat           list([bs, 256, h1, w1], ..., [bs, 256, h5, w5]) 多尺度backbone特征 (以尺度分组而不是batch)
                dense_rbb_preds:   [bs, total_anchor_num, 7=(cx, cy, w, h, θ, joint_score, label)]    
                dense_cls_score:   [bs, total_anchor_num, cls_num] (已经过sigmoid)
                dense_centerness:  [bs, total_anchor_num] (已经过sigmoid)
                img_meta:          batch图像信息和标注
                train_mode:        当前的训练模式(训练有监督数据=sup, 训练伪标签=unsup)

            Return:
                batch_gi_box: list([nms_box_num, 7=(cx, cy, w, h, θ, cls_score, cls_label)], ..., [...]) gi_head 预测结果
        """
        self.mode = 'infer'
        '''前向(cls_score, reg_delta 未解码)'''
        # list([group_nums, 7], ..., [...]), list([group_nums, cls_num], ..., [...]) [total_group_nums, 16], [total_group_nums, 5]
        batch_nms_bboxes, batch_nms_scores, cls_score, reg_delta = self.forward(fpn_feat, dense_rbb_preds, dense_cls_score, dense_centerness, img_meta)
        nms_scores = torch.cat(batch_nms_scores)
        '''decode'''
        nms_bboxes = torch.cat(batch_nms_bboxes)
        gi_bboxes = self.bbox_coder.decode(nms_bboxes, reg_delta)
        # TODO:这里感觉预测的cls_score很有问题(负样本的噪声很大)，因此还是用nms的score
        # cls_score = cls_score.sigmoid()
        cls_score = nms_scores
        
        last_batch_num = 0
        batch_gi_box = []
        for batch in range(len(batch_nms_bboxes)):
            cur_batch_num = batch_nms_bboxes[batch].shape[0]
            single_nms_boxes = nms_bboxes[last_batch_num:last_batch_num+cur_batch_num]
            single_gi_bboxes = gi_bboxes[last_batch_num:last_batch_num+cur_batch_num]
            cls_scores = cls_score[last_batch_num:last_batch_num+cur_batch_num]
            single_cls_scores, single_cls_preds = torch.max(cls_scores, dim=1)
            pos_mask = single_cls_scores>0.2
            if pos_mask.sum()==0:
                pos_mask = single_cls_scores>0
            last_batch_num += cur_batch_num
            # [nms_box_num, 7=(cx, cy, w, h, θ, cls_score, cls_label)]
            batch_gi_box.append(torch.cat([single_gi_bboxes[pos_mask], single_cls_scores[pos_mask].unsqueeze(1), single_cls_preds[pos_mask].unsqueeze(1)], dim=1))

            # 可视化
            # vis_nms_scores = torch.cat(batch_nms_scores, dim=0)
            # vis_HM_scores(cls_scores.unsqueeze(0), vis_nms_scores.unsqueeze(0), img_meta, './vis_unsup_infer_score')
            # 可视nms_bboxes和gi_head输出的bboxes对比(默认注释)
            # vis_gi_head_bboxes_single(img_meta['img'][batch], img_meta['img_metas'][batch]['ori_filename'], single_nms_boxes, single_gi_bboxes, './vis_gi_bboxes_infer')
        return batch_gi_box










