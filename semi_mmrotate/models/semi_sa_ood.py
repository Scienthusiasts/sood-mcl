import copy
from typing import Sequence

import numpy as np

import torch
import torch.nn as nn

from mmdet.core import reduce_mean

from mmrotate.models import ROTATED_DETECTORS, RotatedBaseDetector, build_detector
from mmrotate.core.bbox.iou_calculators import rbbox_overlaps

import cv2
import matplotlib.pyplot as plt
from custom.visualize import *

def rename_loss_dict(prefix: str, losses: dict) -> dict:
    """Rename the key names in loss dict by adding a prefix.

    Args:
        prefix (str): The prefix for loss components.
        losses (dict):  A dictionary of loss components.

    Returns:
            dict: A dictionary of loss components with prefix.
    """
    return {prefix + k: v for k, v in losses.items()}

def reweight_loss_dict(losses: dict, weight: float) -> dict:
    """Reweight losses in the dict by weight.

    Args:
        losses (dict):  A dictionary of loss components.
        weight (float): Weight for loss components.

    Returns:
            dict: A dictionary of weighted loss components.
    """
    for name, loss in losses.items():
        if 'loss' in name:
            if isinstance(loss, Sequence):
                losses[name] = [item * weight for item in loss]
            else:
                losses[name] = loss * weight
    return losses


@ROTATED_DETECTORS.register_module()
class SemiSAOOD(RotatedBaseDetector):
    def __init__(self, detector: dict, train_cfg=None, test_cfg=None):
        super(SemiSAOOD, self).__init__()
        self.teacher = build_detector(copy.deepcopy(detector))
        self.student = build_detector(copy.deepcopy(detector))
        self.submodules = ['teacher', 'student']
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if train_cfg is not None:
            self.freeze(self.teacher)
        self.inference_on = self.test_cfg.get("inference_on", "teacher")

        self.burn_in_mode=False


    def forward_train(self, imgs, img_metas, **kwargs):
        super(SemiSAOOD, self).forward_train(imgs, img_metas, **kwargs)

        format_data = self.collect_data(imgs, img_metas, **kwargs)
        device = imgs[0].device
    
        losses = dict()

        sup_losses = rename_loss_dict('sup_', self.student.forward_train(**format_data['sup_weak']))
        losses.update(reweight_loss_dict(sup_losses, 1.0))

        # generate pseudo labels using teacher on unsup_weak
        if not self.burn_in_mode:
            losses = dict()
            (teacher_gt_bboxes, teacher_gt_labels) = self.generate_pseudo_labels_by_teacher(format_data)
            format_data['unsup_weak']['gt_bboxes'] = teacher_gt_bboxes
            format_data['unsup_weak']['gt_labels'] = teacher_gt_labels

            num_imgs = len(teacher_gt_labels)
            num_teacher_ann_gt = torch.sum(torch.tensor([len(gt_labels) for gt_labels in teacher_gt_labels], 
                                                      device=device))
            num_teacher_ann_gt = reduce_mean(num_teacher_ann_gt / num_imgs)
            num_sparse_ann_gt = torch.sum(torch.tensor([len(gt_labels) for gt_labels in format_data['unsup_strong']['gt_labels']], 
                                                         device=device)) 
            num_sparse_ann_gt = reduce_mean(num_sparse_ann_gt / num_imgs)


            # merge the pseudo labels with the strongly augmented data
            merged_format_data = self.merge_pseudo_labels(format_data)
            num_merged_ann_gt = torch.sum(torch.tensor([len(gt_labels) for gt_labels in merged_format_data['unsup_strong']['gt_labels']],
                                                       device=device))
            num_merged_ann_gt = reduce_mean(num_merged_ann_gt / num_imgs)

            # 可视化sgt+minging_gt(默认注释)
            # vis_sparse_data(format_data, [[],[]], [[],[]], save_dir='./vis_minging_gt')

            unsup_losses = rename_loss_dict('unsup_', self.student.forward_train(**merged_format_data['unsup_strong']))
            losses.update(reweight_loss_dict(unsup_losses, 0.1))
            losses.update({'num_teacher_ann_gt': num_teacher_ann_gt, 
                           'num_sparse_ann_gt': num_sparse_ann_gt,
                           'num_merged_ann_gt': num_merged_ann_gt})
        # else:
        #     batch_size = len(format_data['unsup_weak']['img'])
        #     teacher_gt_bboxes = [torch.empty((0, 5), dtype=torch.float32, device=device) for _ in range(batch_size)]
        #     teacher_gt_labels = [torch.empty((0,), dtype=torch.long, device=device) for _ in range(batch_size)]

        return losses
    
    def collect_data(self, imgs, img_metas, **kwargs):
        """
        Collects and organizes input data into a structured dictionary grouped by augmentation tags.

        Args:
            imgs (Tensor or list[Tensor]): Input images.
            img_metas (list[dict]): List of image metadata dictionaries, each containing a 'tag' key.
            **kwargs: Additional keyword arguments, must include 'gt_bboxes' and 'gt_labels'.

        Returns:
            dict: A dictionary where each key is a tag and the value is another dict containing:
                - 'img': Stacked images for the tag.
                - 'img_metas': List of image metadata for the tag.
                - 'gt_bboxes': List of ground truth bounding boxes for the tag.
                - 'gt_labels': List of ground truth labels for the tag.
        """
        assert 'gt_bboxes' in kwargs.keys() and 'gt_labels' in kwargs.keys()
        gt_bboxes = kwargs.get('gt_bboxes')
        gt_labels = kwargs.get('gt_labels')

        # preprocess
        format_data = dict()
        for idx, img_meta in enumerate(img_metas):
            tag = img_meta['tag']
            if tag not in format_data.keys():
                format_data[tag] = dict()
                format_data[tag]['img'] = [imgs[idx]]
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
        

    def filter_detections(self, results, score_thr=0.85, min_w=1e-2, min_h=1e-2):
        """
        Filters detection results for a single image by score and minimum box size.

        Args:
            results: list of np.ndarray, each of shape (num_dets, 6) (cx, cy, w, h, angle, score) for a single class
            score_thr: float, minimum score to keep
            min_w: float, minimum width
            min_h: float, minimum height

        Returns:
            filtered_results: list of np.ndarray, filtered detections for the single image,
                with the same shape as input (class label column, if present, is preserved)
        """
        filtered_results = []
        for dets_per_class in results:
            scores = dets_per_class[:, 5]
            ws = dets_per_class[:, 2]
            hs = dets_per_class[:, 3]
            keep = (scores >= score_thr) & (ws >= min_w) & (hs >= min_h)
            filtered_results.append(dets_per_class[keep])
        return filtered_results

    def convert_detections_to_gt(self, results_per_img, device):
        """
        Convert detection results for a single image to gt_bboxes and gt_labels tensors.

        Args:
            results_per_img: list of np.ndarray, one per class, shape (num_dets, 6)
            device: torch device

        Returns:
            gt_bboxes: torch.Tensor, shape (num_valid_dets, 5)
            gt_labels: torch.Tensor, shape (num_valid_dets,)
        """
        all_dets = []
        for class_idx, dets in enumerate(results_per_img):
            if dets.shape[0] == 0:
                continue
            class_col = np.full((dets.shape[0], 1), class_idx)
            dets_with_label = np.concatenate([dets, class_col], axis=1)  # shape (num_dets, 7)
            all_dets.append(dets_with_label)
        if len(all_dets) == 0:
            gt_bboxes = torch.zeros((0, 5), dtype=torch.float32, device=device)
            gt_labels = torch.zeros((0,), dtype=torch.long, device=device)
        else:
            all_dets = np.vstack(all_dets)  # shape (total_dets, 7)
            gt_bboxes = torch.from_numpy(all_dets[:, :5]).float().to(device)
            gt_labels = torch.from_numpy(all_dets[:, 6]).long().to(device)
        return gt_bboxes, gt_labels

    def generate_pseudo_labels_by_teacher(self, format_data):
        """
        Generate pseudo-labels using the teacher model on the unsup_weak branch of format_data.
        Processes each image in the batch independently, handling per-class outputs.
        Returns:
            teacher_gt_bboxes: list of torch.Tensor, each of shape (num_dets, 5)
            teacher_gt_labels: list of torch.Tensor, each of shape (num_dets,)
        """
        assert 'unsup_weak' in format_data.keys(), "format_data must contain 'unsup_weak' key"
        device = format_data['unsup_weak']['img'].device

        score_thr = self.train_cfg.get("pseudo_score_thr", 0.85)
        min_w, min_h = self.train_cfg.get("pseudo_size_thr", (1.e-2, 1.e-2))

        results = self.teacher.simple_test(
            format_data['unsup_weak']['img'],
            format_data['unsup_weak']['img_metas'],
        )

        # Filter detections for each image in the batch using score and size thresholds
        # results_per_img: list of np.ndarray, one per class, shape (num_dets, 6)
        filtered_results = []
        for results_per_img in results:
            filtered_results_per_img = self.filter_detections(results_per_img, score_thr, min_w, min_h)
            filtered_results.append(filtered_results_per_img)

        # Convert filtered detections to gt_bboxes and gt_labels tensors for each image
        teacher_gt_bboxes = []
        teacher_gt_labels = []
        for filtered_results_per_img in filtered_results:
            gt_bboxes, gt_labels = self.convert_detections_to_gt(filtered_results_per_img, device)
            teacher_gt_bboxes.append(gt_bboxes)
            teacher_gt_labels.append(gt_labels)

        return teacher_gt_bboxes, teacher_gt_labels

    def merge_pseudo_labels(self, format_data, iou_thr=0.1):
        """
        Merge pseudo-labels from unsup_weak into unsup_strong, removing any pseudo-label
        that has IoU > iou_thr with any box in unsup_strong for each image.
        Modifies format_data['unsup_strong']['gt_bboxes'] and ['gt_labels'] in place.

        Args:
            format_data: dict with 'unsup_weak' and 'unsup_strong' keys, each containing
                         'gt_bboxes' and 'gt_labels' (lists of tensors per image)
            iou_thr: float, IoU threshold for removal
        """
        iou_thr = self.train_cfg.get("merge_iou_thr", 0.1)

        weak_bboxes_list = format_data['unsup_weak']['gt_bboxes']
        weak_labels_list = format_data['unsup_weak']['gt_labels']
        strong_bboxes_list = format_data['unsup_strong']['gt_bboxes']
        strong_labels_list = format_data['unsup_strong']['gt_labels']

        merged_bboxes_list = []
        merged_labels_list = []

        for weak_bboxes, weak_labels, strong_bboxes, strong_labels in zip(
            weak_bboxes_list, weak_labels_list, strong_bboxes_list, strong_labels_list
        ):
            if strong_bboxes.numel() == 0 or weak_bboxes.numel() == 0:
                # If either is empty, just concatenate
                merged_bboxes = torch.cat([strong_bboxes, weak_bboxes], dim=0)
                merged_labels = torch.cat([strong_labels, weak_labels], dim=0)
            else:
                # Compute IoU between each weak and each strong box
                ious = rbbox_overlaps(weak_bboxes, strong_bboxes)  # (num_weak, num_strong)
                max_ious, _ = ious.max(dim=1)
                keep = max_ious <= iou_thr
                # Keep only weak boxes with IoU <= threshold
                filtered_weak_bboxes = weak_bboxes[keep]
                filtered_weak_labels = weak_labels[keep]
                merged_bboxes = torch.cat([strong_bboxes, filtered_weak_bboxes], dim=0)
                merged_labels = torch.cat([strong_labels, filtered_weak_labels], dim=0)
            merged_bboxes_list.append(merged_bboxes)
            merged_labels_list.append(merged_labels)

        format_data['unsup_strong']['gt_bboxes'] = merged_bboxes_list
        format_data['unsup_strong']['gt_labels'] = merged_labels_list

        return format_data

    def freeze(self, model: RotatedBaseDetector):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

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

    def extract_feat(self, img):
        pass

    def simple_test(self, img, img_metas, **kwargs):
        model = getattr(self, self.inference_on)
        return model.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        pass




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

