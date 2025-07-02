import torch
import torch.nn as nn
import numpy as np
import cv2
from mmrotate.core import rbbox2result, poly2obb_np, obb2poly, poly2obb, build_bbox_coder, obb2xyxy
from mmcv.ops import box_iou_quadri, box_iou_rotated
from custom.visualize import *



class FNMining(nn.Module):
    '''稀疏监督正样本挖掘
    '''
    def __init__(self):
        super(FNMining, self).__init__()




    @staticmethod
    def fp_mining(bs, batch_t_nms_bboxes, batch_t_nms_scores, format_data, aug_orders):
        """teacher正样本挖掘(sparse-level)
        """
        # 1.挖掘正样本
        batch_t_fn_bboxes, batch_t_fn_score = FNMining.batch_fn_mining_strategy(batch_t_nms_bboxes, batch_t_nms_scores, format_data)
        # 2.将挖掘出的正样本作为gt加入format_data(student的输入)中
        for i in range(bs):
            # 没有挖掘出正样本则跳过
            if(batch_t_fn_bboxes[i].shape[0]==0): continue
            # 获取预测类别和对应的置信度
            t_fn_score, t_fn_label = torch.max(batch_t_fn_score[i], dim=-1)
            # 把每个gt的置信度也拼到GT bbox里去(原本gt的置信度为1, 挖掘出的正样本的置信度为其原本的置信度)
            gt_nums = format_data[aug_orders[0]]['gt_bboxes'][i].shape[0]
            scores = torch.ones(gt_nums, device=format_data[aug_orders[0]]['gt_bboxes'][i].device)
            scores = torch.cat([scores, t_fn_score]).unsqueeze(1)
            # 把挖掘出的fp加入GT bbox里去
            t_fn_bboxes = batch_t_fn_bboxes[i][:, :5]
            format_data[aug_orders[0]]['gt_bboxes'][i] = torch.cat([format_data[aug_orders[0]]['gt_bboxes'][i], t_fn_bboxes], dim=0)
            format_data[aug_orders[0]]['gt_labels'][i] = torch.cat([format_data[aug_orders[0]]['gt_labels'][i], t_fn_label], dim=0)
            # 把每个gt的置信度也拼到bbox里去 [sparse_gt+fp, 5] -> [sparse_gt+fp, 6]
            format_data[aug_orders[0]]['gt_bboxes'][i] = torch.cat([format_data[aug_orders[0]]['gt_bboxes'][i], scores], dim=-1)
        # 可视化稀疏标签+挖掘出的正样本(默认注释)
        # vis_batch_gts(format_data, mode='unsup_strong', save_dir='./vis_t_sparse_label_w_mining')

        return format_data



    @staticmethod
    def batch_fn_mining_strategy(batch_nms_bboxes, batch_nms_scores, format_data, iou_thres=0.1):
        """batch正样本挖掘具体策略
        """
        mode = 'unsup_weak'
        batch_gt_bboxes = format_data[mode]['gt_bboxes']
        batch_gt_labels = format_data[mode]['gt_labels']
        batch_fn_bboxes, batch_fn_score, batch_fn_iou = [], [], []
        batch_tn_bboxes, batch_tn_score = [], []
        # 每张图片分别挖掘
        for nms_bboxes, nms_scores, gt_bboxes, gt_labels in zip(batch_nms_bboxes, batch_nms_scores, batch_gt_bboxes, batch_gt_labels):
            max_nms_scores, max_nms_labels = torch.max(nms_scores, dim=-1)
            # 1.潜在正样本的类别必须在稀疏gt中出现
            cat_mask = torch.isin(max_nms_labels, gt_labels)
            # 2.潜在正样本和稀疏GT的IoU不能太大(太大说明和gt冗余)
            riou = box_iou_rotated(nms_bboxes[:, :-1], gt_bboxes)
            riou, _ = torch.max(riou, dim=-1)
            iou_mask = riou <= iou_thres
            # 3.潜在正样本的置信度必须在所有nms box平均置信度之上
            # mean_nms_score = max_nms_scores[iou_mask].mean() if iou_mask.sum() > 0 else max_nms_scores.mean()
            # score_mask = max_nms_scores >= mean_nms_score
            score_mask = max_nms_scores >= 0.0
            # print((max_nms_scores >= 0.6).sum())
            # 满足1.2.3则成为潜在正样本
            pos_mask = cat_mask & score_mask & iou_mask

            batch_fn_bboxes.append(nms_bboxes[pos_mask])
            batch_fn_score.append(nms_scores[pos_mask])
            batch_fn_iou.append(riou[pos_mask])
            batch_tn_bboxes.append(nms_bboxes[~pos_mask])
            batch_tn_score.append(nms_scores[~pos_mask])

        # 可视化解码后的预测框(默认注释)
        vis_batch_preds(format_data, batch_fn_bboxes, batch_fn_score, batch_fn_iou, batch_tn_bboxes, batch_tn_score, save_dir='./vis_t_nms_preds')

        return batch_fn_bboxes, batch_fn_score




    @staticmethod
    def gen_fn_target_feat_single(gt_bboxes, points):
        """得到fcos正负样本分配后的gt特征图(仅对挖掘出的正样本), 在densehead部分使用(改写get_targets)
        """
        num_points = points.size(0)
        num_gts = gt_bboxes.size(0)
        
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]

        gaussian_center = offset_x.pow(2) / (w / 2).pow(2) + offset_y.pow(2) / (h / 2).pow(2)
        return gaussian_center





    @staticmethod
    def get_sample_weight(gaussian_center, scores, pos_thres, beta=1.0):
        """对挖掘出的正样本生成weight_mask在计算损失时加权(单张图像), 在densehead部分使用
        """
        pos_mask = scores >= pos_thres
        neg_mask = ~pos_mask
        # 计算挖掘样本中的正样本的权重mask
        if pos_mask.sum() > 0:
            pos_weight = FNMining.get_sample_weight_by_mode(gaussian_center[:, pos_mask], scores[pos_mask], 'pos', beta=5.0)
        else:
            pos_weight = torch.ones_like(gaussian_center[:, 0])
        # 计算挖掘样本中的负样本的权重mask
        if neg_mask.sum() > 0:
            neg_weight = FNMining.get_sample_weight_by_mode(gaussian_center[:, neg_mask], scores[neg_mask], 'neg', beta=5.0)
        else:
            neg_weight = torch.ones_like(gaussian_center[:, 0])

        all_weight = neg_weight
        pos_region = pos_weight<1.0
        all_weight[pos_region] = pos_weight[pos_region]

        return pos_weight, all_weight


    @staticmethod
    def get_sample_weight_by_mode(gaussian_center, scores, mode, beta=1.0):
        """对挖掘出的正样本生成weight_mask在计算损失时加权(单张图像), 在densehead部分使用
        """
        if mode=='neg':
            w = (1 - scores).pow(beta)
        else:
            w = scores.pow(beta)
        weight = (1 - gaussian_center)
        weight[weight>=0] = 1.0
        weight = weight * w
        weight[weight<0]=1.0
        # 如果一个区域是被多个目标共用，则取最小的那个权重
        weight, _ = torch.min(weight, dim=-1)
        weight = weight

        return weight








# ===============================================================相关可视化函数=======================================================================



def vis_sparse_data(format_data, save_dir='./vis_strong_weak_img'):
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
             weak_img, weak_img_meta, weak_gt_bboxes, weak_gt_labels) in enumerate(zip(
        batch_strong_img, batch_strong_img_meta, batch_strong_gt_bboxes, batch_strong_gt_labels,
        batch_weak_img, batch_weak_img_meta, batch_weak_gt_bboxes, batch_weak_gt_labels)):
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

        '''box绘制'''
        if strong_gt_labels.shape[0]>0:
            # 5参转8参
            poly_strong_gts = obb2poly(strong_gt_bboxes).cpu().numpy().astype(np.int32)
            poly_weak_gts = obb2poly(weak_gt_bboxes).cpu().numpy().astype(np.int32)
            # 可视化strong gts + weak gts
            strong_img = OpenCVDrawBox(strong_img, poly_strong_gts, (0,255,0), 2)
            weak_img = OpenCVDrawBox(weak_img, poly_weak_gts, (0,255,0), 2)

        '''绘制+保存'''
        # 获取图像名
        img_name = strong_img_meta['ori_filename']
        # 创建一行两列的画布
        plt.figure(figsize=(12, 6))
        # 绘制 strong_img
        plt.subplot(1, 2, 1)
        plt.imshow(strong_img)
        plt.title(f'Strong Augmented Image')
        plt.axis('off')
        # 绘制 weak_img
        plt.subplot(1, 2, 2)
        plt.imshow(weak_img)
        plt.title(f'Weak Augmented Image')
        plt.axis('off')
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, img_name), bbox_inches='tight', dpi=150)
        plt.close()





def vis_batch_preds(format_data, batch_fn_bboxes, batch_fn_score, batch_fn_iou, batch_tn_bboxes, batch_tn_score, save_dir):
    '''可视化解码后的预测框(post nms) + 稀疏gt
    '''
    mode = 'unsup_weak'
    # 弱增强的图像 [bs, 3, 1024, 1024]
    batch_img = format_data[mode]['img']
    batch_img_meta = format_data[mode]['img_metas']
    batch_gt_bboxes = format_data[mode]['gt_bboxes']
    batch_gt_labels = format_data[mode]['gt_labels']
    # 可视化每张图像预测结果
    for img, img_meta, gt_bboxes, fn_bboxes, fn_scores, fn_iou, tn_bboxes, tn_scores in \
        zip(batch_img, batch_img_meta, batch_gt_bboxes, batch_fn_bboxes, batch_fn_score, batch_fn_iou, batch_tn_bboxes, batch_tn_score):
        # 原图预处理
        std = np.array([58.395, 57.12 , 57.375]) / 255.
        mean = np.array([123.675, 116.28 , 103.53]) / 255.
        img = img.permute(1,2,0).cpu().numpy()
        img = np.clip(img * std + mean, 0, 1)
        img = (img * 255.).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # bbox5参转8参
        if fn_bboxes.shape[0]>0:
            poly_boxes_fn = obb2poly(fn_bboxes)
        poly_boxes_tn = obb2poly(tn_bboxes)
        poly_boxes_gt = obb2poly(gt_bboxes)

        # 取置信度最大的那个类别的置信度作为该box的置信度
        fn_scores, fn_labels = torch.max(fn_scores, dim=-1)
        # opencv绘制框
        if fn_bboxes.shape[0]>0:
            img = OpenCVDrawBox(img, poly_boxes_fn.cpu().numpy(), (0,255,255), 1)
        img = OpenCVDrawBox(img, poly_boxes_gt.cpu().numpy(), (0,255,0), 2)
        img = OpenCVDrawBox(img, poly_boxes_tn.cpu().numpy(), (0,0,255), 1)

        # 在每个预测框中心绘制类别和置信度得分(只绘制fn)
        if fn_bboxes.shape[0]>0:
            # 获取旋转框的中心点坐标
            centers = fn_bboxes[:, :2].cpu().numpy()  # [N, 2], 格式为(x, y)
            # 遍历每个框和得分
            for center, score, label, iou in zip(centers, fn_scores.cpu().numpy(), fn_labels.cpu().numpy(), fn_iou.cpu().numpy()):
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
                cv2.rectangle(img, 
                             (x - text_size[0]//2 - 2, y - text_size[1]//2 - 2),
                             (x + text_size[0]//2 + 2, y + text_size[1]//2 + 2),
                             bg_color, -1)  # -1表示填充
                # 绘制文本
                cv2.putText(img, text, (x - text_size[0]//2, y + text_size[1]//2),
                           font, font_scale, text_color, thickness, cv2.LINE_AA)
                
        if not os.path.exists(save_dir):os.makedirs(save_dir)
        img_name = img_meta['ori_filename']
        img_save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(img_save_path, img)





def vis_batch_gts(format_data, mode, save_dir):
    '''可视化稀疏gt + 挖掘的正样本
    '''
    mode = 'unsup_strong'
    # 弱增强的图像 [bs, 3, 1024, 1024]
    batch_img = format_data[mode]['img']
    batch_img_meta = format_data[mode]['img_metas']
    batch_gt_bboxes = format_data[mode]['gt_bboxes']
    batch_gt_labels = format_data[mode]['gt_labels']
    # 可视化每张图像预测结果
    for img, img_meta, gt_bboxes in zip(batch_img, batch_img_meta, batch_gt_bboxes):
        # 原图预处理
        std = np.array([58.395, 57.12 , 57.375]) / 255.
        mean = np.array([123.675, 116.28 , 103.53]) / 255.
        img = img.permute(1,2,0).cpu().numpy()
        img = np.clip(img * std + mean, 0, 1)
        img = (img * 255.).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # bbox5参转8参
        poly_boxes_gt = obb2poly(gt_bboxes)

        # opencv绘制框
        img = OpenCVDrawBox(img, poly_boxes_gt.cpu().numpy(), (0,255,0), 2)

        if not os.path.exists(save_dir):os.makedirs(save_dir)
        img_name = img_meta['ori_filename']
        img_save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(img_save_path, img)