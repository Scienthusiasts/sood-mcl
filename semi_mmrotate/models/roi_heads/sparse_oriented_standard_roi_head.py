# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrotate.core import rbbox2roi
from mmrotate.models.builder import ROTATED_HEADS, build_shared_head
from mmrotate.models.roi_heads.rotate_standard_roi_head import RotatedStandardRoIHead

from mmcv.ops import box_iou_quadri, box_iou_rotated

import matplotlib.pyplot as plt
import numpy as np
from custom.visualize import *
from custom.features_queue import DistributedClassFeatureQueue



@ROTATED_HEADS.register_module()
class SparseOrientedStandardRoIHead(RotatedStandardRoIHead):
    """Oriented RCNN roi head including one bbox head."""


    def __init__(self,
                 nc,
                 loss_weighting=False,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 version='oc'):

        super(SparseOrientedStandardRoIHead, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.version = version

        if shared_head is not None:
            shared_head.pretrained = pretrained
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        self.init_assigner_sampler()

        self.with_bbox = True if bbox_head is not None else False
        self.with_shared_head = True if shared_head is not None else False

        self.nc = nc
        # 是否开启负样本加权
        self.loss_weighting = loss_weighting
        # 对比学习用到的特征队列:
        self.feats_queue = DistributedClassFeatureQueue(self.nc, 255, keep_iters=50)


    def forward_dummy(self, x, proposals):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        outs = ()
        rois = rbbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def forward_train(self,
                      img, 
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_scores, 
                      gt_labels,
                      mining_gt_bboxes, 
                      mining_flag,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            img: (Tensor): shape=[bs, 3, 1024, 1024] for DOTA datasets.
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_scores (list[Tensor]): Ground truth bboxes scores(confidents as gt)
                shape (num_gts,  ).
            gt_labels (list[Tensor]): class indices corresponding to each box
            mining_gt_bboxes (list[Tensor]): minging pseudo Ground truth bboxes for each image with
                shape (num_gts, 6) in [cx, cy, w, h, a, score] format.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task. Always
                set to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox:

            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                # 正负样本分配
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                # 正负样本平衡采样
                # {'neg_bboxes', 'neg_inds', 'num_gts', 'pos_assigned_gt_inds', 'pos_bboxes'(pos_gt_bboxes), 'pos_inds', 'pos_is_gt'}
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                        (0, gt_bboxes[0].size(-1))).zero_()
                else:
                    sampling_result.pos_gt_bboxes = \
                        gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            # bbox_results.keys() = ['cls_score'(bs*512, nc+1), 'bbox_pred'(bs*512, 5), 'bbox_feats'(bs*512, 256, 7, 7), 'loss_bbox']
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            
            bs = x[0].shape[0]
            # 提取具体到每个样本的损失(reduction=none)
            cls_loss = bbox_results['loss_bbox']['loss_cls']
            reg_loss = bbox_results['loss_bbox']['loss_bbox']
            '''负样本加权(核心)'''
            if self.loss_weighting and mining_flag:
                pos_loss_weight, all_loss_weight = self.neg_weighting(img, img_metas, gt_bboxes, gt_scores, bs, mining_gt_bboxes, sampling_results)
                # bbox_results['loss_bbox']['loss_cls'] = (cls_loss * loss_weight).sum() / loss_weight.sum()
                # 理论上，对于潜在正样本, loss_weight和cls_loss应该呈现反比关系?
                bbox_results['loss_bbox']['loss_cls'] = (cls_loss * all_loss_weight).sum() / (512. * bs)
                bbox_results['loss_bbox']['loss_bbox'] = (reg_loss.sum(dim=-1) * pos_loss_weight).sum() / (512. * bs)
            else:
                bbox_results['loss_bbox']['loss_cls'] = cls_loss.sum() / (512. * bs)
                bbox_results['loss_bbox']['loss_bbox'] = reg_loss.sum() / (512. * bs)
            losses.update(bbox_results['loss_bbox'])





        return losses



    def neg_weighting(self, img, img_metas, gt_bboxes, gt_scores, bs, mining_gt_bboxes, sampling_results):
        """计算负样本损失加权系数
            Args:
                img:              图像, shape=[bs, 3, 1024, 1024] (只可视化用)
                img_metas:        图像信息 list([...], ..., [...]) (只可视化用)
                gt_bboxes:        sgt 坐标, list([sgt_num_per_img, 5], ..., [...]) 
                gt_scores:        sgt 得分, list([sgt_num_per_img   ], ..., [...]) 
                bs:               batch-size
                mining_gt_bboxes: teacher分支通过固定阈值挖掘出的潜在样本 list([bboxes_num_per_img, 6], ..., [...])
                sampling_results: roi_head随机采样的正负样本 
                                  包含字段: {'neg_bboxes', 'neg_inds', 'num_gts', 'pos_assigned_gt_inds', 'pos_bboxes'(pos_gt_bboxes), 'pos_inds', 'pos_is_gt'}

            Returns:
                all_loss_weight:  对类别损失加权的系数, 形状=[bs*512], 值范围在(0,1)之间

        """
        alpha, beta = 1., 5.
        device = sampling_results[0].pos_bboxes.device

        '''负样本加权'''
        all_loss_weight, pos_loss_weight = [], []
        for i in range(bs):
            # 没有mining_gt则负样本权重全为1
            if len(mining_gt_bboxes[i]) == 0: 
                neg_weight = torch.ones(sampling_results[i].neg_bboxes.shape[0], device=device)
            else:
                # 得到负样本与最匹配的挖掘样本的iou和对应的索引
                neg_mining_iou = box_iou_rotated(sampling_results[i].neg_bboxes, mining_gt_bboxes[i][:, :-1])
                max_iou, max_iou_inds = torch.max(neg_mining_iou, dim=-1)
                # 得到负样本与最匹配的挖掘样本对应的置信度
                max_iou_score = mining_gt_bboxes[i][:, -1][max_iou_inds]
                # max_iou_weight可以代表这些负样本实际属于正样本的概率
                max_iou_weight = max_iou * max_iou_score.pow(alpha)
                # 生成正负样本损失加权参数(正样本的权重为1, 负样本的权重为(1 - max_iou_weight)^beta
                # neg_weight越靠近1则是负样本的概率越大, 越接近0则是潜在正样本的概率越大
                neg_weight = (1 - max_iou_weight).pow(beta)
                # 可视化, 默认注释
                # vis_sgt_mininggt_neg_per_img(
                #     img[i], img_metas[i], gt_bboxes[i], 
                #     mining_gt_bboxes[i][:, :-1], sampling_results[i].neg_bboxes, sampling_results[i].pos_bboxes, 
                #     neg_weight, save_dir='vis_neg_weighting')


            '''正样本加权'''
            if len(gt_scores[i]) == 0: 
                # 没有pos_mining_gt则正样本权重全为1
                pos_weight = torch.ones(sampling_results[i].pos_bboxes.shape[0], device=device)
            else:
                # 得到正样本与最匹配的gt的iou和对应的索引
                pos_mining_iou = box_iou_rotated(sampling_results[i].pos_bboxes, gt_bboxes[i])
                max_iou, max_iou_inds = torch.max(pos_mining_iou, dim=-1)
                # 得到正样本与最匹配的gt对应的置信度
                max_iou_score = gt_scores[i][max_iou_inds]
                # max_iou_weight可以代表这些样本实际属于正样本的概率(和负样本处理不同, 这里不乘max_iou)
                pos_weight = max_iou_score.pow(beta)

            all_loss_weight.append(torch.cat([pos_weight, neg_weight], dim=0))
            pos_loss_weight.append(pos_weight)

        all_loss_weight = torch.cat(all_loss_weight)
        pos_loss_weight = torch.cat(pos_loss_weight)
        return pos_loss_weight, all_loss_weight
            













    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training.

        Args:
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[Tensor]): list of sampling results.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains \
                the boxes of the corresponding image in a batch, each \
                tensor has the shape (num_boxes, 5) and last dimension \
                5 represent (cx, cy, w, h, a, score). Each Tensor \
                in the second list is the labels with shape (num_boxes, ). \
                The length of both lists should be equal to batch_size.
        """

        rois = rbbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels












# ========================================================可视化函数========================================================
def vis_sgt_mininggt_neg_per_img(img, img_meta, gt_bboxes, mining_gt_bboxes, neg_bboxes, pos_bboxes, neg_weight, save_dir):
    '''图像处理'''
    # 原图预处理
    std = np.array([58.395, 57.12, 57.375]) / 255.
    mean = np.array([123.675, 116.28, 103.53]) / 255.
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * std + mean, 0, 1)
    img = (img * 255).astype(np.uint8)

    '''box绘制'''
    # 舍弃掉那些负样本权重过大的样本(完全的负样本)
    pos_mask = neg_weight<0.1
    neg_bboxes = neg_bboxes[pos_mask]
    neg_weight = neg_weight[pos_mask]

    # 5参转8参
    poly_gt_bboxes = obb2poly(gt_bboxes).cpu().numpy().astype(np.int32)
    poly_mining_gt_bboxes = obb2poly(mining_gt_bboxes).cpu().numpy().astype(np.int32)
    poly_neg_bboxes = obb2poly(neg_bboxes).cpu().numpy().astype(np.int32)
    poly_pos_bboxes = obb2poly(pos_bboxes).cpu().numpy().astype(np.int32)
    # 可视化strong gts + weak gts
    sgt_img = OpenCVDrawBox(img.copy(), poly_gt_bboxes, (0,255,0), 2)
    mining_gt_img = OpenCVDrawBox(img.copy(), poly_mining_gt_bboxes, (255,255,0), 2)
    neg_img = OpenCVDrawBox(img.copy(), poly_neg_bboxes, (255,0,0), 2)
    pos_img = OpenCVDrawBox(img.copy(), poly_pos_bboxes, (0,255,0), 2)
    # 可视化每个负样本的损失加权值
    neg_bboxes_centers = neg_bboxes[:, :2].cpu().numpy()  # [N, 2], 格式为(x, y)
    for center, score in zip(neg_bboxes_centers, neg_weight.cpu().numpy()):
        x, y = int(center[0]), int(center[1])
        # 绘制得分（保留2位小数）
        text = f"{score:.2f}"
        # 设置文本样式
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_color = (255, 255, 255)  # 白色文字
        bg_color = (0, 0, 0)          # 黑色背景
        # 获取文本大小
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        # 绘制背景矩形（提高可读性）
        cv2.rectangle(neg_img, 
                        (x - text_size[0]//2 - 2, y - text_size[1]//2 - 2),
                        (x + text_size[0]//2 + 2, y + text_size[1]//2 + 2),
                        bg_color, -1)  # -1表示填充
        # 绘制文本
        cv2.putText(neg_img, text, (x - text_size[0]//2, y + text_size[1]//2),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)

    '''绘制+保存'''
    # 获取图像名
    img_name = img_meta['ori_filename']
    # 创建一行两列的画布
    plt.figure(figsize=(12, 12))
    # 绘制 sgt_img
    plt.subplot(2, 2, 1)
    plt.imshow(sgt_img)
    plt.title(f'sgt_img')
    plt.axis('off')
    # 绘制 mining_gt_img
    plt.subplot(2, 2, 2)
    plt.imshow(mining_gt_img)
    plt.title(f'mining_gt_img')
    plt.axis('off')
    # 绘制 pos_img
    plt.subplot(2, 2, 3)
    plt.imshow(pos_img)
    plt.title(f'pos_img')
    plt.axis('off')
    # 绘制 neg_img
    plt.subplot(2, 2, 4)
    plt.imshow(neg_img)
    plt.title(f'neg_img')
    plt.axis('off')

    # 调整布局并保存
    plt.tight_layout()
    if not os.path.isdir(save_dir):os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, img_name), dpi=200)
    plt.close()