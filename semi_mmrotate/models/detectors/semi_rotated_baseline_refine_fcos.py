#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/29 23:53
# @Author : WeiHua
import torch
from mmrotate.models import RotatedFCOS, ROTATED_DETECTORS, RotatedSingleStageDetector
from mmrotate.core import rbbox2result
import mmcv
import numpy as np
# yan
from mmrotate.models.builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck

@ROTATED_DETECTORS.register_module()
class SemiRotatedBLRefineFCOS(RotatedFCOS):
    """Implementation of Rotated `FCOS.`__

    __ https://arxiv.org/abs/1904.01355
    """
    # added by yan
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 # 去噪微调模块
                 roi_head, 
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 get_feature_map=False
                 ):
        super(RotatedFCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                          test_cfg, pretrained, init_cfg)
        self.get_feature_map = get_feature_map

        '''去噪微调模块就是roi head'''
        # reference: /data/yht/code/sood-mcl/mmrotate-0.3.4/mmrotate/models/detectors/two_stage.py
        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # for param in self.roi_head.parameters():
        #     param.requires_grad = False
            
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      get_data=False,
                      get_pred=False,
                      return_fpn_feat=False,
                      fpn_feat_grad=False
                      ):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
            get_data (Bool): If return logit only.
            get_pred (Bool): If return prediction result

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        # torch.Size([2, 256, 128, 128])
        # torch.Size([2, 256, 64, 64])
        # torch.Size([2, 256, 32, 32])
        # torch.Size([2, 256, 16, 16])
        # torch.Size([2, 256, 8, 8])
        x = self.extract_feat(img)
        if not get_pred:
            logits = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, get_data=get_data)
            # NOTE: yan, 将fpn的特征也传回, 目的是为了更新prototype
            if return_fpn_feat:
                if fpn_feat_grad==False:
                    fpn_feat = [lvl_x for lvl_x in x]
                    # fpn_feat = [torch.ones_like(lvl_x.detach())*0.01 for lvl_x in x]
                else:
                    fpn_feat = [lvl_x for lvl_x in x]
                return logits, fpn_feat
            else:
                return logits
        
        with torch.no_grad():
            self.eval()
            bbox_results = self.simple_test(img, img_metas, rescale=True)
            self.train()

        logits = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, get_data=get_data)
        # NOTE: yan, 将fpn的特征也传回, 目的是为了更新prototype
        if return_fpn_feat:
            if fpn_feat_grad==False:
                fpn_feat = [lvl_x.detach() for lvl_x in x]
                # fpn_feat = [torch.ones_like(lvl_x.detach())*0.01 for lvl_x in x]
            else:
                fpn_feat = [lvl_x for lvl_x in x]
            return logits, fpn_feat, bbox_results
        else:
            return logits, bbox_results


    # NOTE: added by yan, 重写simple_test, 使其支持额外返回特征图
    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # 当get_feature_map为True, 额外返回 cls_scores, centernesses
        if self.get_feature_map:
            bbox_list, dense_bboxes_list, dense_scores_list, dense_cnt_list, cls_scores, centernesses = self.bbox_head.get_bboxes(
                *outs, img_metas, rescale=rescale, get_feature_map=True)
            # mlvl_bboxes, mlvl_scores, mlvl_centerness, bbox_list = bbox_lists
        else:
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_metas, rescale=rescale)

        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        # 当get_feature_map为True, 额外返回 cls_scores, centernesses
        if self.get_feature_map:
            return bbox_results, dense_bboxes_list[0], dense_scores_list[0], dense_cnt_list[0], cls_scores, centernesses
        else:
            return bbox_results
