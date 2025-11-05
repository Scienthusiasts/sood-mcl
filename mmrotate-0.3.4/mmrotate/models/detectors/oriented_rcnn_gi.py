# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import ROTATED_DETECTORS, build_head

from .two_stage import RotatedTwoStageDetector


@ROTATED_DETECTORS.register_module()
class OrientedRCNNGI(RotatedTwoStageDetector):
    """Implementation of `Oriented R-CNN for Object Detection.`__

    __ https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf  # noqa: E501, E261.
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 # 微调模块:
                 gi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(OrientedRCNNGI, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        '''微调模块gi head'''
        # reference: /data/yht/code/sood-mcl/mmrotate-0.3.4/mmrotate/models/detectors/two_stage.py
        if gi_head is not None:
            # update train and test cfg here for now
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            gi_head.update(train_cfg=rcnn_train_cfg)
            gi_head.update(test_cfg=test_cfg.rcnn)
            gi_head.pretrained = pretrained
            self.gi_head = build_head(gi_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg



    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmrotate/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 6).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs



    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)


        '''训练gi-head分支'''
        # 每张图保留1000个框
        fpn_feat, bboxes, nc_scores = self.simple_test_bboxes(img, img_metas, rcnn_configs=None)
        bboxes = torch.cat([bbox.unsqueeze(0) for bbox in bboxes], dim=0)
        nc_scores = torch.cat([nc_score.unsqueeze(0) for nc_score in nc_scores], dim=0)
        scores, s_labels = torch.max(nc_scores[..., :-1], dim=-1, keepdim=True)
        rbb_preds = torch.cat([bboxes, scores, s_labels], dim=-1)

        # NOTE:Ablation: 断开refine-head与主体检测器的梯度
        fpn_feat = [fpn_feat.detach() for fpn_feat in fpn_feat]
        # 2.送入gi head进行微调(pos_thres=1.0说明gt只有稀疏标注的那部分, 不含挖掘的部分)
        format_data = {'img_metas':img_metas, 'img':img}
        cnt_score = torch.ones_like(scores.squeeze(-1))
        gi_losses = self.gi_head.loss(
            fpn_feat, 
            rbb_preds, nc_scores[..., :-1], cnt_score,
            gt_bboxes, gt_labels,
            format_data, 
            train_mode='train_sup',
            pos_thres=0.1
        )
        # 3.组织gihead微调模块的损失
        for key, val in gi_losses.items():
            losses[key] = val


        return losses




    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)



    def simple_test_bboxes(self, img, img_metas, rcnn_configs=None):
        x = self.extract_feat(img)
        with torch.no_grad():
            self.eval()
            assert self.with_bbox, 'Bbox head must be implemented.'
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            det_bboxes, det_labels = self.roi_head.simple_test_bboxes(x, img_metas, proposal_list, rcnn_configs, rescale=False)
            self.train()

        return x, det_bboxes, det_labels