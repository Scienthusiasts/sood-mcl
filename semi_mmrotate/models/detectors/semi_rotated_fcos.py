#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/29 23:53
# @Author : WeiHua
import torch
from mmrotate.models import RotatedFCOS, ROTATED_DETECTORS, RotatedSingleStageDetector
from mmrotate.core import rbbox2result
import mmcv
import numpy as np


@ROTATED_DETECTORS.register_module()
class SemiRotatedFCOS(RotatedFCOS):
    """Implementation of Rotated `FCOS.`__

    __ https://arxiv.org/abs/1904.01355
    """

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      get_data=False,
                      get_pred=False,
                      return_fpn_feat=False
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
                x = [lvl_x.detach() for lvl_x in x]
                return logits, x
            else:
                return logits
        
        with torch.no_grad():
            self.eval()
            bbox_results = self.simple_test(img, img_metas, rescale=True)
            self.train()

        logits = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, get_data=get_data)
        # NOTE: yan, 将fpn的特征也传回, 目的是为了更新prototype
        if return_fpn_feat:
            x = [lvl_x.detach() for lvl_x in x]
            return logits, x.detach(), bbox_results
        else:
            return logits, bbox_results
