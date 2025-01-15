import torch
import numpy as np
import cv2
import os
from mmrotate.core import rbbox2result, poly2obb_np, obb2poly, poly2obb, build_bbox_coder, obb2xyxy
from .utils import *
import copy









def OpenCVDrawBox(image, poly_boxes, color, thickness=1):
    '''画框
        Args:
            :param image:         原始图像(Image格式)
            :param boxes:         网络预测的box坐标
            :param save_res_path: 可视化结果保存路径

        Returns:
            None
    '''
    H, W = image.shape[:2]
    for i, poly_box in enumerate(poly_boxes):
        box = np.array([round(b) for b in poly_box]).reshape((-1, 1, 2))
        # obj的框
        cv2.drawContours(image, [box], 0, color=color, thickness=thickness)
    # 保存
    return image







def vis_sup_bboxes_batch(model, format_data, bs, nc, cls_scores, sup_fpn_feat, rbb_preds, root_dir):
    '''有监督分支可视化微调模块的推理结果(dense, refined, gt)
        Args:
            model:        检测器实例
            format_data:  传来的format_data['sup']
            bs:           batch size
            nc:           数据集类别数
            cls_scores:   [bs, total_anchor_num, nc]
            sup_fpn_feat: 多尺度特征图
            rbb_preds:    一阶段网络预测的dense结果    
            root_dir:     可视化结果保存路径
        Returns:
            None
    '''
    proposal_list = []
    for i in range(bs):
        pos_mask = rbb_preds[i, :, 5] > 1e-1
        proposal_list.append(rbb_preds[i, :, :5][pos_mask])

    # 只推理(refine-head)
    with torch.no_grad():
        refine_box_list = []
        # 还需先将rbbox转成水平外接矩
        # hproposal_list = [obb2xyxy(p, 'le90') for p in proposal_list]
        p = copy.deepcopy(proposal_list)
        refine_bbox = model.roi_head.simple_test(sup_fpn_feat, p, format_data['img_metas'], rescale=True)
        # 调整输出格式, 把所有类别下的预测结果拼在一起
        for i in range(bs):
            res = np.concatenate(refine_bbox[i], axis=0)
            refine_box_list.append(res)

    # 图像和图像名
    img_names = [img_meta['ori_filename'] for img_meta in format_data['img_metas']]
    images = format_data['img']
    '''nms + 分组(目前只回传nms的结果)'''
    # NOTE:注意这里传参共享内存, 所以得.clone()
    batch_nms_bboxes, batch_nms_labels = batch_nms(rbb_preds.clone(), cls_scores.reshape(bs, -1, nc))
    # 每张图片分别可视化
    for batch in range(bs):
        img_name = img_names[batch]
        # 原图预处理
        std = np.array([58.395, 57.12 , 57.375]) / 255.
        mean = np.array([123.675, 116.28 , 103.53]) / 255.
        img = images[batch].permute(1,2,0).cpu().numpy()
        img = np.clip(img * std + mean, 0, 1)
        img = (img * 255.).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 5参转8参
        poly_rbb = obb2poly(proposal_list[batch])
        poly_refine_rbb = obb2poly(torch.tensor(refine_box_list[batch]))
        poly_nms_rbb = obb2poly(batch_nms_bboxes[batch])
        # poly_gt = obb2poly(format_data['gt_bboxes'][batch])
        # 可视化proposals框
        # img = OpenCVDrawBox(img, poly_rbb.cpu().numpy(), (0,0,255), 1)
        # 可视化refine proposals框
        img = OpenCVDrawBox(img, poly_refine_rbb.numpy(), (0,255,0), 2)
        # 可视化nms保留框
        img = OpenCVDrawBox(img, poly_nms_rbb.cpu().numpy(), (0,0,255), 2)
        # 可视化GT框
        # img = OpenCVDrawBox(img, poly_gt.cpu().numpy(), (0,255,0), 2)
        # 保存结果
        if not os.path.exists(root_dir):os.makedirs(root_dir)
        img_save_path = f"{root_dir}/{img_name}"
        cv2.imwrite(img_save_path, img)








def vis_unsup_bboxes_batch(format_data, bs, nms_bboxes_list, proposal_list, refined_proposal_list, root_dir):
    '''无监督分支可视化微调模块的推理结果(dense, refined, pgt)
        Args:
            format_data:           传来的format_data['sup']
            bs:                    batch size
            nms_bboxes_list:       nms后保留的结果
            proposal_list:         一阶段网络预测的dense结果    
            refined_proposal_list: 微调网络预测的结果    
            root_dir:              可视化结果保存路径
        Returns:
            None
    '''
    # 图像和图像名
    img_names = [img_meta['ori_filename'] for img_meta in format_data['img_metas']]
    images = format_data['img']
    # 每张图片分别可视化
    for batch in range(bs):
        img_name = img_names[batch]
        # 原图预处理
        std = np.array([58.395, 57.12 , 57.375]) / 255.
        mean = np.array([123.675, 116.28 , 103.53]) / 255.
        img = images[batch].permute(1,2,0).cpu().numpy()
        img = np.clip(img * std + mean, 0, 1)
        img = (img * 255.).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 5参转8参
        poly_rbb = obb2poly(proposal_list[batch])
        poly_refine_rbb = obb2poly(refined_proposal_list[batch].clone().detach())
        poly_nms_rbb = obb2poly(nms_bboxes_list[batch])
        # 可视化nms保留框
        img = OpenCVDrawBox(img, poly_nms_rbb.cpu().numpy(), (0,0,255), 2)
        # 可视化proposals框
        # img = OpenCVDrawBox(img, poly_rbb.cpu().numpy(), (0,0,255), 1)
        # 可视化refine proposals框
        img = OpenCVDrawBox(img, poly_refine_rbb.cpu().numpy(), (0,255,0), 2)
        # 保存结果
        if not os.path.exists(root_dir):os.makedirs(root_dir)
        img_save_path = f"{root_dir}/{img_name}"
        cv2.imwrite(img_save_path, img)









def vis_grouping_batch(batch_gt_bboxes, batch_group_iou, batch_group_id_mask, batch_dense_bboxes, format_data, root_dir):
    '''可视化分组结果(依据gt_bboxes分组)
        Args: 
            batch_gt_bboxes:     gt或pgt  List([keep_num], ,..., [...])  
            batch_group_iou:     每个dense bbox 与匹配gt或pgt的IoU            List([keep_num], ,..., [...])  
            batch_group_id_mask: 每个dense bbox 所属的group(与gt的id对应)     List([keep_num], ,..., [...])  
            batch_dense_bboxes:  score阈值筛选与grouping IoU筛选后保留的dense bboxes List([keep_num, 7], ,..., [...])  
            format_data:         图像信息
            root_dir:            可视化结果保存路径

        Returns:
            None
    '''
    # 图像和图像名(默认batch=1)
    img_names = [img_meta['ori_filename'] for img_meta in format_data['img_metas']]
    images = format_data['img']

    for img, img_name, gt_bboxes, group_id_mask, dense_bboxes in zip(images, img_names, batch_gt_bboxes, batch_group_id_mask, batch_dense_bboxes):
        # 原图预处理
        std = np.array([58.395, 57.12 , 57.375]) / 255.
        mean = np.array([123.675, 116.28 , 103.53]) / 255.
        img = img.permute(1,2,0).cpu().numpy()
        img = np.clip(img * std + mean, 0, 1)
        img = (img * 255.).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 5参转8参
        poly_dense_bboxes = obb2poly(dense_bboxes[:, :5])
        # 可视化每个group里的框
        for group_id in range(gt_bboxes.shape[0]):
            # 找到属于当前group的所有框
            mask = group_id_mask == group_id
            # 随机颜色
            color = np.random.randint(0, 256, size=3)
            color = tuple([int(x) for x in color])
            # 可视化一个group里的框
            img = OpenCVDrawBox(img, poly_dense_bboxes[mask].cpu().numpy(), color, 1)

        if not os.path.exists(root_dir):os.makedirs(root_dir)
        img_save_path = os.path.join(root_dir, img_name)
        cv2.imwrite(img_save_path, img)