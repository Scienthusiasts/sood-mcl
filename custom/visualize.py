import torch
import numpy as np
import cv2
import os
from mmrotate.core import rbbox2result, poly2obb_np, obb2poly, poly2obb, build_bbox_coder, obb2xyxy
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from .utils import *
import copy
from mmcv.ops import diff_iou_rotated_2d








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






def vis_rgb_tensor(tensor, img_name, save_dir):
    '''仅可视化转化为张量的图像
        Args:
            tensor: 张量, 形状为[3,h,w]
            img_name: 图像名
        Returns:
            None
    '''
    with torch.no_grad():
        # 原图预处理
        std = np.array([58.395, 57.12 , 57.375]) / 255.
        mean = np.array([123.675, 116.28 , 103.53]) / 255.
        img = tensor.permute(1,2,0).cpu().numpy()
        img = np.clip(img * std + mean, 0, 1)
        img = (img * 255.).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 保存结果
        if not os.path.exists(save_dir):os.makedirs(save_dir)
        img_save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(img_save_path, img)


def vis_mask(mask, img_name, save_dir):
    '''仅可视化mask(单通道)
        Args:
            tensor: 张量, 形状为[3,h,w]
            img_name: 图像名
        Returns:
            None
    '''
    with torch.no_grad():
        plt.figure(figsize = (10, 10))
        # 原图预处理
        plt.imshow(mask.cpu().numpy())
        # 保存结果
        if not os.path.exists(save_dir):os.makedirs(save_dir)
        img_save_path = os.path.join(save_dir, img_name)
        plt.axis('off')
        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
        plt.savefig(img_save_path, dpi=100)




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
        # 可视化nms保留框
        img = OpenCVDrawBox(img, poly_nms_rbb.cpu().numpy(), (0,0,255), 2)
        # 可视化refine proposals框
        img = OpenCVDrawBox(img, poly_refine_rbb.numpy(), (0,255,0), 2)
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





def vis_rboxes_on_img(img, rboxes, img_name, save_dir):
    '''解码后的旋转框在原图上可视化(single image)
       (只可视化box本身, 不可视化score和类别) 
        Args: 
            img:      图像tensor
            rboxes:   [n, 5=(cx, cy, w, h, θ)]
            img_name: 图像名
            save_dir: 保存目录

        Returns:
            None
    '''
    # 原图预处理
    std = np.array([58.395, 57.12 , 57.375]) / 255.
    mean = np.array([123.675, 116.28 , 103.53]) / 255.
    img = img.permute(1,2,0).cpu().numpy()
    img = np.clip(img * std + mean, 0, 1)
    img = (img * 255.).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 5参转8参
    poly_boxes = obb2poly(rboxes)
    # opencv绘制框
    img = OpenCVDrawBox(img, poly_boxes.cpu().numpy(), (0,255,0), 1)

    if not os.path.exists(save_dir):os.makedirs(save_dir)
    img_save_path = os.path.join(save_dir, img_name)
    cv2.imwrite(img_save_path, img)





def vis_rotate_feat(ori_img, rot_img, img_name, o_cls_scores, o_centernesses, r_cls_scores, r_centernesses, o_weight_mask, mask):
    '''可视化旋转一致性自监督分支的特征图
    '''
    # 首先把长条状的特征reshape回二维图片的形状
    sizes = [128, 64 ,32, 16, 8]
    e = 1e-20
    # [total_anchor_num, ] -> [[h1*w1, ], [h5*w5, ]]
    o_cls_scores_list = torch.split(o_cls_scores, [size * size for size in sizes], dim=0)
    o_centernesses_list = torch.split(o_centernesses, [size * size for size in sizes], dim=0)
    r_cls_scores_list = torch.split(r_cls_scores, [size * size for size in sizes], dim=0)
    r_centernesses_list = torch.split(r_centernesses, [size * size for size in sizes], dim=0)
    o_weight_mask_list = torch.split(o_weight_mask, [size * size for size in sizes], dim=0)
    mask_list = torch.split(mask, [size * size for size in sizes], dim=0)
    # [[h1*w1, ], [h5*w5, ]] -> [[h1, w1, ], [h5, w5, ]]
    o_cls_scores_list = [x.reshape(size, size, -1) for size, x in zip(sizes, o_cls_scores_list)]
    o_centernesses_list = [x.reshape(size, size) for size, x in zip(sizes, o_centernesses_list)]
    r_cls_scores_list = [x.reshape(size, size, -1).sigmoid() for size, x in zip(sizes, r_cls_scores_list)]
    r_centernesses_list = [x.reshape(size, size).sigmoid() for size, x in zip(sizes, r_centernesses_list)]
    o_weight_mask_list = [x.reshape(size, size).sigmoid() for size, x in zip(sizes, o_weight_mask_list)]
    mask_list = [x.reshape(size, size)for size, x in zip(sizes, mask_list)]
    # max cls score
    o_max_clsscore_list = [torch.max(x, 2)[0]  for x in o_cls_scores_list]
    r_max_clsscore_list = [torch.max(x, 2)[0]  for x in r_cls_scores_list]
    # 联合置信度
    o_joint_score_list = [x * y for x, y in zip(o_max_clsscore_list, o_centernesses_list)]
    r_joint_score_list = [x * y for x, y in zip(r_max_clsscore_list, r_centernesses_list)]

    # 可视化联合置信度
    vis_lvlfeat(ori_img, o_joint_score_list, img_name, './ori_lvl_joint_score', True, mask_list)
    vis_lvlfeat(rot_img, r_joint_score_list, img_name, './rot_lvl_joint_score', True, mask_list)

    # 可视化centerness
    # vis_lvlfeat(ori_img, o_centernesses_list, img_name, './ori_lvl_cnt_score', True, mask_list)
    # vis_lvlfeat(rot_img, r_centernesses_list, img_name, './rot_lvl_cnt_score', True, mask_list)

    # 可视化max cls score
    # vis_lvlfeat(ori_img, o_joint_score_list, img_name, './ori_lvl_cls_score', True, mask_list)
    # vis_lvlfeat(rot_img, r_joint_score_list, img_name, './rot_lvl_cls_score', True, mask_list)

    # 可视化 o_weight_mask
    vis_lvlfeat(ori_img, o_weight_mask_list, img_name, './ori_lvl_weight_mask', True, mask_list)





def vis_lvlfeat(img, feat, img_name, save_dir, use_mask=False, mask_list=None, use_batch_dim=False):
    '''可视化多尺度特征图(bs=1)
        Args: 
            img:           图像tensor  [1, 3, h, w]
            feat:          网络的多尺度中间特征 [[h1, w1], ..., [h5, w5]] 
            img_name:      图像名 
            save_dir:      图像保存目录
            use_mask:      是否对特征乘上一个mask
            mask_list:     use_mask=True时有效
            use_batch_dim: =True时 feat的形状为 [[h1, w1, 1], ..., [h5, w5, 1]]

        Returns:
            None
    '''
    # 对原图预处理
    std = np.array([58.395, 57.12 , 57.375]) / 255.
    mean = np.array([123.675, 116.28 , 103.53]) / 255.
    img = img.squeeze(0).permute(1,2,0).cpu().numpy()
    img = np.clip(img * std + mean, 0, 1)

    '''可视化'''
    plt.figure(figsize = (12, 2))
    for lvl, lvl_feat in enumerate(feat):
        if use_batch_dim: lvl_feat = lvl_feat.squeeze(-1)
        if use_mask: lvl_feat = mask_list[lvl] * lvl_feat.detach()
        # 可视化特征图
        plt.subplot(1,6,lvl+2)
        plt.imshow(lvl_feat.detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        plt.axis('off')
    # 可视化原图
    plt.subplot(1,6,1)
    plt.imshow(img, vmin=0, vmax=1)
    plt.axis('off')
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    # 保存
    if not os.path.exists(save_dir):os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, img_name), dpi=200) 
    plt.close()


def vis_lvlriou(img, rbbox1, rbbox2, img_name, save_dir):
    '''可视化dense bbox的两两riou(所有尺度)
        Args: 
            img:           图像tensor  [1, 3, h, w]
            rbbox1:        box1 [total_anchor_num, 5]
            rbbox2:        box2 [total_anchor_num, 5]
            img_name:      图像名 
            save_dir:      图像保存目录

        Returns:
            None
    '''
    riou = diff_iou_rotated_2d(rbbox1.unsqueeze(0), rbbox2.unsqueeze(0)).squeeze(0)
    sizes = [128, 64 ,32, 16, 8]
    # [total_anchor_num, ] -> [[h1*w1, ], [h5*w5, ]]
    riou_list = torch.split(riou, [size * size for size in sizes], dim=0)
    # [[h1*w1, ], [h5*w5, ]] -> [[h1, w1, ], [h5, w5, ]]
    riou_list = [x.reshape(size, size) for size, x in zip(sizes, riou_list)]

    # 对原图预处理
    std = np.array([58.395, 57.12 , 57.375]) / 255.
    mean = np.array([123.675, 116.28 , 103.53]) / 255.
    img = img.squeeze(0).permute(1,2,0).cpu().numpy()
    img = np.clip(img * std + mean, 0, 1)

    '''可视化'''
    plt.figure(figsize = (12, 2))
    for lvl, lvl_riou in enumerate(riou_list):
        # 可视化特征图
        plt.subplot(1,6,lvl+2)
        plt.imshow(lvl_riou.detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        plt.axis('off')
    # 可视化原图
    plt.subplot(1,6,1)
    plt.imshow(img, vmin=0, vmax=1)
    plt.axis('off')
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    # 保存
    if not os.path.exists(save_dir):os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, img_name), dpi=200) 
    plt.close()