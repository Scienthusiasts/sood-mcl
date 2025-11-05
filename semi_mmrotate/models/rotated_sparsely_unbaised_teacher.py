import torch
import numpy as np
from .rotated_semi_detector import RotatedSemiDetector
from mmrotate.models.builder import ROTATED_DETECTORS, build_roi_extractor
from mmrotate.models import build_detector
from mmrotate.core import rbbox2roi

from custom.visualize import *
from custom.fn_mining import FNMining
from custom.features_queue import DistributedClassFeatureQueue
from custom.loss import proto_contrastive_loss







@ROTATED_DETECTORS.register_module()
class RotatedSparselyUnbaisedTeacher(RotatedSemiDetector):
    def __init__(self, nc, bbox_roi_extractor, fn_mining_score_thr, model: dict, semi_loss=None, train_cfg=None, test_cfg=None, symmetry_aware=False):
        super(RotatedSparselyUnbaisedTeacher, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            semi_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            # ugly manner to get start iteration, to fit resume mode
            self.iter_count = train_cfg.get("iter_count", 0)
            # Prepare semi-training config
            # step to start training student (not include EMA update)
            self.burn_in_steps = train_cfg.get("burn_in_steps", 5000)
            # prepare super & un-super weight
            self.sup_weight = train_cfg.get("sup_weight", 1.0)
            self.unsup_weight = train_cfg.get("unsup_weight", 1.0)
            self.weight_suppress = train_cfg.get("weight_suppress", "linear")
            self.rcnn_configs = train_cfg.get("rcnn_configs")
            self.loss_configs = train_cfg.get("loss_configs")
        self.symmetry_aware = symmetry_aware
        self.nc=nc
        self.fn_mining_score_thr = fn_mining_score_thr
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        # 对比学习用到的特征队列:
        self.feats_queue = DistributedClassFeatureQueue(self.nc, 256, keep_iters=50)
        # self.roi_proj = ShareFCHead(channel=256)





    def forward_train(self, imgs, img_metas, **kwargs):
        super(RotatedSparselyUnbaisedTeacher, self).forward_train(imgs, img_metas, **kwargs)
        losses = dict()

        '''数据读取'''
        format_data = self.wrap_datas(imgs, img_metas, **kwargs)
        aug_orders = ['unsup_strong', 'unsup_weak']




        '''burn-in结束, 开启正样本挖掘'''
        if self.iter_count > self.burn_in_steps:
            # burn_in之后的慢启动, 慢慢增加无监督分支损失的权重(只有使用t-s自蒸馏分支才用到)
            unsup_weight = self.ema_weight()

            '''得到teacher预测结果+正样本挖掘'''
            with torch.no_grad():
                t_bboxes, t_labels, t_fpn_feat = self.teacher.forward_train(get_boxes=True, rcnn_configs=self.rcnn_configs, **format_data[aug_orders[1]])
                # 可视化sgt+minging_gt(默认注释)
                # vis_sparse_data(format_data, t_bboxes, t_labels, save_dir='./vis_sgt_mininggt_iter38400')

            bs = len(t_bboxes)
            '''teacher正样本挖掘(sparse-level)'''
            t_scores = [t_bbox[:, -1] for t_bbox in t_bboxes]
            # 挖掘出的正样本会放入format_data中当做gt
            # t_bboxes = list([box_num, 6], ..., [...]), t_scores = list([box_num], ..., [...]), t_labels = list([box_num], ..., [...])
            format_data = FNMining.fp_mining(bs, t_bboxes, t_scores, t_labels, format_data, aug_orders, score_thr=self.fn_mining_score_thr)
            # 可视化sgt+minging_gt(默认注释)
            # vis_sparse_data(format_data, [[],[]], [[],[]], save_dir='./vis_minging_gt_my2')



        '''稀疏监督分支(before burn-in) / student稀疏监督训练(sparse-level) (after burn-in)'''
        # student部分前向+计算损失
        # NOTE:这里会和稀疏GT(挖掘出的正样本)也计算损失, 返回sparse_losses
        # switch_strong_weak = self.iter_count % 2
        sparse_losses, s_fpn_feat = self.student.forward_train(**format_data[aug_orders[0]])

        # 组织稀疏监督损失
        for key, val in sparse_losses.items():
            if key[:4] == 'loss':
                if isinstance(val, list):
                    losses[f"{key}_unsup"] = [self.sup_weight * x for x in val]
                else:
                    losses[f"{key}_unsup"] = self.sup_weight * val
            else:
                losses[key] = val


        '''对比学习'''
        # cont_roifeats_loss = self.cont_learning(self, format_data, aug_orders, s_fpn_feat, t_fpn_feat)
        # losses[f"cont_gtroi_loss"] = cont_roifeats_loss * 0.1


  
        self.iter_count += 1
        return losses







    def cont_learning(self, format_data, aug_orders, s_fpn_feat, t_fpn_feat):
        '''fpn-roi特征的对比学习策略'''
        beta=5.0
        # 取出gt
        gt_bboxes = format_data[aug_orders[0]]['gt_bboxes']
        gt_labels = format_data[aug_orders[0]]['gt_labels']


        # sgt_masks只取sgt
        sgt_masks = [torch.ones_like(gt_label, dtype=torch.bool) for gt_label in gt_labels]
        if gt_bboxes[0].shape[-1]==6:
            sgt_masks = [gt_bbox[:,  -1]==1.0 for gt_bbox in gt_bboxes]
            gt_scores = [gt_bbox[:,  -1] for gt_bbox in gt_bboxes]
            gt_bboxes = [gt_bbox[:, :-1] for gt_bbox in gt_bboxes]
        else:
            gt_scores = [torch.ones_like(gt_label) for gt_label in gt_labels]

        # list -> tensor
        batch_gt_labels = torch.cat(gt_labels, dim=0)
        batch_gt_bboxes = torch.cat(gt_bboxes, dim=0)
        batch_gt_scores = torch.cat(gt_scores, dim=0)
        batch_sgt_masks = torch.cat(sgt_masks, dim=0)

        
        # 1.根据gt_box，通过roialign从fpn中抠出特征(burn-in之前取student, 之后取teacher)
        # 这一步只是合并list为tensor, 并加上batch_id: list([gt_num_per_img, 5], ,..., [...]) -> [batch_gt_num, 6=(batch_ind, cx, cy, w, h, a)]
        batch_gt_rois = rbbox2roi(gt_bboxes)
        # 从特征图中抠出roi并进行avg_pool [batch_gt_num, 256, 7, 7]
        fpn_feat = t_fpn_feat if self.iter_count > self.burn_in_steps else s_fpn_feat
        batch_gt_roi_feats = self.bbox_roi_extractor(fpn_feat, batch_gt_rois)
        # [batch_gt_num, 256, 7, 7] -> [batch_gt_num, 256]
        batch_gt_roi_feats = batch_gt_roi_feats.reshape(-1, 256, 7*7).mean(dim=-1)

        # 2.更新队列
        self.feats_queue.push(batch_gt_roi_feats[batch_sgt_masks], batch_gt_labels[batch_sgt_masks])
        self.feats_queue.pop()
        # 3.取出队列中所有样本
        queue_gt_feats, queue_gt_labels = [], []
        for i in range(self.nc):
            feats = self.feats_queue.get_feats_by_class(i)
            feats_len = feats.shape[0]
            if feats_len > 0:
                queue_gt_feats.append(feats.mean(dim=0, keepdim=True))
                queue_gt_labels.append(torch.full((1, ), i, dtype=torch.long, device=feats.device))
        # 把所有类别拼在一起
        queue_gt_feats = torch.cat(queue_gt_feats, dim=0)
        queue_gt_labels = torch.cat(queue_gt_labels, dim=0)

        # 找到B中每个元素在A中的位置(因为可能存在队列中类别也不全的情况)
        matches = (batch_gt_labels.unsqueeze(1) == queue_gt_labels.unsqueeze(0))
        cont_gt_labels = matches.int().argmax(dim=1)

        # 4.计算对比损失(根据置信度加权)
        cont_roifeats_loss = proto_contrastive_loss(queue_gt_feats, batch_gt_roi_feats, cont_gt_labels)
        factor = batch_gt_scores.pow(beta)
        cont_roifeats_loss = (cont_roifeats_loss * factor).sum() / factor.sum() 

        return cont_roifeats_loss











    def wrap_datas(self, imgs, img_metas, **kwargs):
        """数据读取并组织数据格式
        """
        gt_bboxes = kwargs.get('gt_bboxes')
        gt_labels = kwargs.get('gt_labels')
        # preprocess
        format_data = dict()
        for idx, img_meta in enumerate(img_metas):
            tag = img_meta['tag']
            if tag not in format_data.keys():
                format_data[tag] = dict()
                format_data[tag]['img'] = [imgs[idx]]
                # 'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'tag', 'batch_input_shape'
                format_data[tag]['img_metas'] = [img_metas[idx]]
                format_data[tag]['gt_bboxes'] = [gt_bboxes[idx]]
                format_data[tag]['gt_labels'] = [gt_labels[idx]]
            else:
                format_data[tag]['img'].append(imgs[idx])
                format_data[tag]['img_metas'].append(img_metas[idx])
                format_data[tag]['gt_bboxes'].append(gt_bboxes[idx])
                format_data[tag]['gt_labels'].append(gt_labels[idx])
        for key in format_data.keys():
            format_data[key]['img'] = torch.stack(format_data[key]['img'], dim=0)
        
        return format_data
    


    def ema_weight(self, ):
        """burn_in之后的慢启动, 慢慢增加无监督分支损失的权重
        """
        unsup_weight = self.unsup_weight
        if self.weight_suppress == 'exp':
            target = self.burn_in_steps + 2000
            if self.iter_count <= target:
                scale = np.exp((self.iter_count - target) / 1000)
                unsup_weight *= scale
        elif self.weight_suppress == 'step':
            target = self.burn_in_steps * 2
            if self.iter_count <= target:
                unsup_weight *= 0.25
        elif self.weight_suppress == 'linear':
            target = self.burn_in_steps * 2
            if self.iter_count <= target:
                unsup_weight *= (self.iter_count - self.burn_in_steps) / self.burn_in_steps
        return unsup_weight
    





    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    







# ========================================================================相关可视化函数===================================================================================
def vis_sparse_data(format_data, batch_t_bboxes, batch_t_labels, save_dir='./vis_strong_weak_img'):
    if not os.path.exists(save_dir):os.makedirs(save_dir)
    # 强增强的图像 [bs, 3, 1024, 1024]
    batch_strong_img = format_data['unsup_strong']['img']
    batch_strong_img_meta = format_data['unsup_strong']['img_metas']
    batch_strong_gt_bboxes = format_data['unsup_strong']['gt_bboxes']
    batch_strong_gt_labels = format_data['unsup_strong']['gt_labels']
    # 弱增强的图像 [bs, 3, 1024, 1024]
    batch_weak_img = format_data['unsup_weak']['img']
    batch_weak_img_meta = format_data['unsup_weak']['img_metas']
    batch_weak_gt_bboxes = format_data['unsup_weak']['gt_bboxes']
    batch_weak_gt_labels = format_data['unsup_weak']['gt_labels']

    # 遍历 batch 中的每一对图像
    for i, (strong_img, strong_img_meta, strong_gt_bboxes, strong_gt_labels, 
             weak_img, weak_img_meta, weak_gt_bboxes, weak_gt_labels,
             t_bboxes, t_labels) in enumerate(zip(
        batch_strong_img, batch_strong_img_meta, batch_strong_gt_bboxes, batch_strong_gt_labels,
        batch_weak_img, batch_weak_img_meta, batch_weak_gt_bboxes, batch_weak_gt_labels,
        batch_t_bboxes, batch_t_labels)):
        '''图像处理'''
        # 原图预处理
        std = np.array([58.395, 57.12, 57.375]) / 255.
        mean = np.array([123.675, 116.28, 103.53]) / 255.
        # 处理 strong_img
        strong_img = strong_img.permute(1, 2, 0).cpu().numpy()
        strong_img = np.clip(strong_img * std + mean, 0, 1)
        strong_img = (strong_img * 255).astype(np.uint8) 
        strong_img = np.ascontiguousarray(strong_img) # 确保图像数据是连续内存布局
        # 处理 weak_img
        weak_img = weak_img.permute(1, 2, 0).cpu().numpy()
        weak_img = np.clip(weak_img * std + mean, 0, 1)
        weak_img = (weak_img * 255).astype(np.uint8)
        weak_img = np.ascontiguousarray(weak_img) # 确保图像数据是连续内存布局
        weak_img2 = weak_img.copy()

        '''box绘制'''
        if strong_gt_labels.shape[0]>0:
            # 5参转8参
            poly_strong_gts = obb2poly(strong_gt_bboxes).cpu().numpy().astype(np.int32)
            poly_weak_gts = obb2poly(weak_gt_bboxes).cpu().numpy().astype(np.int32)
            # 可视化strong gts + weak gts
            strong_img = OpenCVDrawBox(strong_img, poly_strong_gts, (0,255,0), 2)
            weak_img = OpenCVDrawBox(weak_img, poly_weak_gts, (0,255,0), 2)
        
        if len(t_bboxes)>0:
            # 5参转8参
            poly_t_gts = obb2poly(t_bboxes[:, :5]).cpu().numpy().astype(np.int32)
            # 可视化teacher pred
            weak_img2 = OpenCVDrawBox(weak_img2, poly_t_gts, (255,0,0), 2)
            # 在每个预测框中心绘制类别和置信度得分(只绘制fn)
            # 获取旋转框的中心点坐标
            centers = t_bboxes[:, :2].cpu().numpy()  # [N, 2], 格式为(x, y)
            # 遍历每个框和得分
            for center, score, label in zip(centers, t_bboxes[:, -1].cpu().numpy(), t_labels.cpu().numpy()):
                x, y = int(center[0]), int(center[1])
                # 绘制得分（保留2位小数）
                text = f"{label} | {score:.2f}"
                # 设置文本样式
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_color = (255, 255, 255)  # 白色文字
                bg_color = (0, 0, 0)          # 黑色背景
                # 获取文本大小
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                # 绘制背景矩形（提高可读性）
                cv2.rectangle(weak_img2, 
                             (x - text_size[0]//2 - 2, y - text_size[1]//2 - 2),
                             (x + text_size[0]//2 + 2, y + text_size[1]//2 + 2),
                             bg_color, -1)  # -1表示填充
                # 绘制文本
                cv2.putText(weak_img2, text, (x - text_size[0]//2, y + text_size[1]//2),
                           font, font_scale, text_color, thickness, cv2.LINE_AA)

        '''绘制+保存'''
        # 获取图像名
        img_name = strong_img_meta['ori_filename']
        # 创建一行两列的画布
        plt.figure(figsize=(12, 6))
        # 绘制 strong_img
        plt.subplot(1, 3, 1)
        plt.imshow(strong_img)
        plt.title(f'Strong Aug Image')
        plt.axis('off')
        # 绘制 weak_img
        plt.subplot(1, 3, 2)
        plt.imshow(weak_img)
        plt.title(f'Weak Aug Image (sgt)')
        plt.axis('off')
        # 绘制 weak_img2
        plt.subplot(1, 3, 3)
        plt.imshow(weak_img2)
        plt.title(f'Weak Aug Image (t_pred)')
        plt.axis('off')

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, img_name), bbox_inches='tight', dpi=150)
        plt.close()

