import torch
from mmrotate.models import OrientedRCNN, ROTATED_DETECTORS, RotatedTwoStageDetector
from mmrotate.models.builder import build_loss, build_head


@ROTATED_DETECTORS.register_module()
class SparseRotatedOrientedRCNNGI(OrientedRCNN):

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
        super(OrientedRCNN, self).__init__(
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



    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      neg_weight_thr=1.0,
                      soft_labels=None,
                      gt_bboxes_ignore=None,
                      get_boxes=False,
                      rcnn_configs=None,
                      loss_configs=None):
        
        super(RotatedTwoStageDetector, self).forward_train(img, img_metas)

        '''处理sgt和mining_gt'''
        mining_gt_bboxes = [[] for _ in range(img.shape[0])]
        rpn_sgt_bboxes = gt_bboxes
        roi_sgt_bboxes = gt_bboxes
        roi_sgt_scores = [[] for _ in range(img.shape[0])]
        roi_sgt_labels = gt_labels
        mining_flag=False
        if gt_bboxes[0].shape[-1] == 6:
            mining_flag=True
            # 把gt的置信度那维暂时先去掉, 后续有用到负样本加权再加上传入rpn_head和roi_head
            rpn_sgt_masks = [gt_bbox[:, -1]==1.0 for gt_bbox in gt_bboxes]
            # roi_sgt_masks阈值越大, 说明参与负损失加权的样本越多，参与正损失加权的样本越少
            roi_sgt_masks = [gt_bbox[:, -1]>=neg_weight_thr for gt_bbox in gt_bboxes]
            rpn_sgt_bboxes = [gt_bbox[:, :-1][rpn_sgt_mask] for gt_bbox, rpn_sgt_mask in zip(gt_bboxes, rpn_sgt_masks)]
            roi_sgt_bboxes = [gt_bbox[:, :-1][roi_sgt_mask] for gt_bbox, roi_sgt_mask in zip(gt_bboxes, roi_sgt_masks)]
            roi_sgt_scores = [gt_bbox[:,  -1][roi_sgt_mask] for gt_bbox, roi_sgt_mask in zip(gt_bboxes, roi_sgt_masks)]
            roi_sgt_labels = [gt_label[roi_sgt_mask] for gt_label, roi_sgt_mask in zip(gt_labels, roi_sgt_masks)]
            # 注意~sgt_mask可能全为False
            mining_gt_bboxes = [gt_bbox[~roi_sgt_mask] for gt_bbox, roi_sgt_mask in zip(gt_bboxes, roi_sgt_masks)]


        if not get_boxes:
            x = self.extract_feat(img)
            losses = dict()

            '''RPN 前向+损失'''
            # TODO: 对伪标签做负样本加权
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            # proposal_list[i].shape = [2000, 6]
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                rpn_sgt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

            # print("="*40+"rpn forward and loss complete!"+"="*40)


            '''roi head 前向+损失 (roihead包含负样本损失加权策略)'''
            if loss_configs != None and loss_configs['type'] == 'FocalLoss':  # for unbiased teacher
                roi_cls_loss_orign = self.roi_head.bbox_head.loss_cls
                self.roi_head.bbox_head.loss_cls = build_loss(loss_configs)
            elif loss_configs == None:
                roi_cls_loss_orign = self.roi_head.bbox_head.loss_cls
            
            if soft_labels != None:
                roi_losses = self.roi_head.forward_train(img, x, img_metas, proposal_list,
                                                        roi_sgt_bboxes, roi_sgt_scores, roi_sgt_labels, mining_gt_bboxes, mining_flag,
                                                        gt_bboxes_ignore)
                
            else:
                roi_losses = self.roi_head.forward_train(img, x, img_metas, proposal_list,
                                                        roi_sgt_bboxes, roi_sgt_scores, roi_sgt_labels, mining_gt_bboxes, mining_flag,
                                                        gt_bboxes_ignore)
            self.roi_head.bbox_head.loss_cls = roi_cls_loss_orign
            losses.update(roi_losses)
            # print("="*40+"roihead forward and loss complete!"+"="*40)
            
            loss_dict = {}
            for loss_name, loss_value in losses.items():
                if isinstance(loss_value, torch.Tensor):
                    loss_dict[loss_name] = loss_value.mean()
                elif isinstance(loss_value, list):
                    loss_dict[loss_name] = sum(_loss.mean() for _loss in loss_value)
                else:
                    raise TypeError(
                        f'{loss_name} is not a tensor or list of tensors')

            return loss_dict, x
        





        # 测试时才用到(get_boxes=True), 训练时用不到:
        with torch.no_grad():
            self.eval()
            assert self.with_bbox, 'Bbox head must be implemented.'
            x = self.extract_feat(img)
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            det_bboxes, det_labels = self.roi_head.simple_test_bboxes(
                                            x, img_metas, proposal_list, rcnn_configs, rescale=False)
            self.train()
        return det_bboxes, det_labels, x
    



    def simple_test_bboxes(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      rcnn_configs=None):
        x = self.extract_feat(img)
        with torch.no_grad():
            self.eval()
            assert self.with_bbox, 'Bbox head must be implemented.'
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            det_bboxes, det_labels = self.roi_head.simple_test_bboxes(x, img_metas, proposal_list, rcnn_configs, rescale=False)
            self.train()

        return x, det_bboxes, det_labels
    

    def simple_test_bboxes_with_grad(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      rcnn_configs=None):
        x = self.extract_feat(img)
        assert self.with_bbox, 'Bbox head must be implemented.'
        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        det_bboxes, det_labels = self.roi_head.simple_test_bboxes(x, img_metas, proposal_list, rcnn_configs, rescale=False)
        return x, det_bboxes, det_labels