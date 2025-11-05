import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rotate, pad
from scipy.optimize import linear_sum_assignment

import random
import numpy as np
import math

from mmdet.core import multi_apply, unmap
from mmrotate.core import  obb2poly
from mmrotate.models import build_loss
from mmcv.ops import box_iou_quadri, box_iou_rotated

from custom.utils import *
from custom.visualize import *
from custom.loss import QFLv2








def pad_tensor(tensor, k, fill='reflect'):
    '''对张量图像进行padding, 
        Args:
            tensor: 张量图像, 形状为[bs, 3, h, w]
            k:      padding的大小为边长的k倍
            fill:   padding部分的填充方式, 默认镜像填充

        Returns:
            padded_tensor: padding后的张量, 形状为[bs, 3, h, w]
            padded_len:    四条边padding的长度
    '''
    h, w = tensor.shape[2:]
    # 计算每个边界的 padding 大小 (k 倍边长)
    padding_top = padding_bottom = round(h * k)
    padding_left = padding_right = round(w * k)
    # 对张量进行 padding, 使用镜像填充
    padded_tensor = pad(tensor, (padding_left, padding_right, padding_top, padding_bottom), padding_mode=fill)
    pad_len = [padding_top, padding_left, padding_bottom, padding_right]
    return padded_tensor, pad_len



def batch_tensor_random_flip(tensor, p=0.5):
    '''对图像进行随机翻转(或以概率p不翻转)'''
    if p==0.0: 
        return tensor, False
    # 生成一个随机数
    rand_num = random.uniform(0,1)
    if rand_num >= p:
        return tensor, False
    else:
        # 在第2维（h维度）上翻转, 相当于垂直翻转 (上下翻转)
        # 如果是水平翻转就第三维度
        vflip_tensor = torch.flip(tensor, dims=[2])
        return vflip_tensor, True



def gen_rot_mask(n, theta):
    """生成mask, mask里每个位置的值(true/false)代表了原始画布中这个位置在旋转后是否超出边界
    
    Args:
        n (int): Size of the square canvas (n x n).
        theta (float): Rotation angle in degrees, counter-clockwise.
    
    Returns:
        np.ndarray: A boolean mask of shape (n, n) where True indicates the pixel is out of bounds after rotation.
    """
    theta = math.radians(theta)
    cos_theta = math.cos(theta)  
    sin_theta = math.sin(theta)
    
    center = (n - 1) / 2.0
    # Generate coordinate grids
    i, j = np.indices((n, n))
    # Convert to centered coordinates
    x = j - center
    y = center - i  # Because in image coordinates, y increases downward
    # Apply inverse rotation
    x_rot = x * cos_theta - y * sin_theta
    y_rot = x * sin_theta + y * cos_theta
    # Convert back to original coordinate system
    j_rot = x_rot + center
    i_rot = center - y_rot
    # Check if the rotated coordinates are within the canvas bounds
    mask = (0 <= i_rot) & (i_rot < n) & (0 <= j_rot) & (j_rot < n)
    
    return mask


def batch_tensor_random_rotate(tensor, angle_range, rand=True):
    '''对张量图像进行随机角度旋转(使用镜像填充), 并返回旋转mask
        Args:
            tensor: 张量图像, 形状为[bs, 3, h, w]
            angle_range: 随机旋转的角度(角度)范围[min, max]

        Returns:
            rotaug_img: 旋转后的图像
            rand_angle: 旋转的角度
            ori_mask:   原始图像中那些位置在旋转后的图像中超出边界
    '''
    h, w = tensor.shape[2:]
    # 生成一个随机角度
    if rand:
        rand_angle = random.uniform(angle_range[0], angle_range[1])
    else:
        return tensor, 0.
    # 先进行镜像填充
    padded_tensor, padded_len = pad_tensor(tensor, (2**0.5 - 1) / 2)
    # 再进行旋转操作(这里是逆时针旋转)
    rotaug_img = rotate(padded_tensor, rand_angle, expand=False)
    padded_h, padded_w = rotaug_img.shape[2:]
    # rotaug_img去掉padding的部分, 还原为原始大小(这样就能去除掉padding的黑色填充, 只保留镜像填充)
    rotaug_img = rotaug_img[..., padded_len[0]:padded_h-padded_len[2], padded_len[1]:padded_w-padded_len[3]]
    ori_mask = gen_rot_mask(h, rand_angle)
    return rotaug_img, rand_angle, ori_mask








class HungarianMatching():
    """匈牙利匹配(一对一匹配)
    """
    def __init__(self, iou_weight=1.0, l1_weight=0.0):
        # 各项代价的权重
        self.iou_weight = iou_weight




    def assign_single(self, gt_boxes, pred_boxes, img_meta):
        """匈牙利匹配(一张图像)
            Args:
                gt_boxes:    [n, 5] (cx, cy, w, h, θ) (已解码)
                pred_boxes:  [m, 5] (已解码)
                img_meta:    图像信息(只可视化调试时会用到)
            
            Return:
                gt_idx:               索引对, 和pred_idx一一对应
                pred_idx:             索引对, 和gt_idx一一对应
        """
        n, m, = gt_boxes.shape[0], pred_boxes.shape[0]
        '''计算代价(去除了分类代价)'''
        with torch.no_grad():
            # RIoU代价
            iou_cost = torch.clamp(1.0 - box_iou_rotated(gt_boxes, pred_boxes[:, :5]), 0.0, 1.0)
            # 综合代价
            total_cost = self.iou_weight * iou_cost

        '''匈牙利匹配'''
        gt_idx, pred_idx = linear_sum_assignment(total_cost.cpu().numpy())
        iou = box_iou_rotated(gt_boxes[gt_idx], pred_boxes[pred_idx], aligned=True)
        iou_mask = iou > 0.7
        if iou_mask.sum()==0:
            topk_values, topk_indices = torch.topk(iou, 1, dim=-1)
            iou_mask = torch.zeros_like(iou_mask, dtype=torch.bool)
            iou_mask[topk_indices] = True

        iou_mask = iou_mask.cpu().numpy()

        return gt_idx[iou_mask], pred_idx[iou_mask]





    def assign(self, batch_gt_boxes, batch_pred_boxes, batch_img_meta):
        """匈牙利匹配(整个batch)
            Args:
                batch_gt_boxes:    list([group_nums, 5=(cx, cy, w, h, θ)], ..., [...]) 真实的gt
                batch_pred_boxes:  list([group_nums, 7=(cx, cy, w, h, θ, score, label)], ..., [...]) 预测的bbox
                batch_img_meta:    batch图像信息
            
            Return:
                batch_gt_idx:   索引对, 和batch_pred_idx一一对应
                batch_pred_idx: 索引对, 和batch_gt_idx一一对应

        """
        batch_gt_idx, batch_pred_idx = multi_apply(self.assign_single, batch_gt_boxes, batch_pred_boxes, batch_img_meta['img_metas'])

        return batch_gt_idx, batch_pred_idx









class SSBranchSparse(nn.Module):
    '''旋转一致性约束分支(针对sparse框使用)
    '''
    def __init__(self, nc, rand_angle_range, flip_p):
        super(SSBranchSparse, self).__init__()
        self.e = 1e-10
        self.nc = nc
        self.rand_angle_range = rand_angle_range
        self.flip_p = flip_p
        self.aug_mode = ['unsup_strong', 'unsup_weak']
        self.assigner = HungarianMatching()
        self.riou_loss = build_loss(dict(type='RotatedIoULoss', reduction='none'))
        self.cls_loss = QFLv2()


    def gen_aug_data(self, format_data):
        """生成旋转视图,并插入会强增强图像中(注意, 旋转图像是弱增强图像, 原图像是强增强图像)
        """
        # 获取图像及基本信息 注意是aug_orders[0]才是无监督student分支的原始图像不要搞错了
        rot_img = format_data[self.aug_mode[1]]['img'].clone()
        # 旋转增强操作(核心部分): rot_img.shape=[bs, 3, h, w]
        rot_img, rand_angle, ori_mask = batch_tensor_random_rotate(rot_img, self.rand_angle_range, rand=True)
        # 以概率p进行翻转增强操作:
        rot_img, isflip = batch_tensor_random_flip(rot_img, p=self.flip_p)
        # 将原始图像和旋转后的图像沿batch维度拼接(注意原始图像在前面, 旋转图像在后面)后, 注意是覆盖原format_data里的强增强图片
        ori_rot_img = torch.cat((format_data[self.aug_mode[0]]['img'], rot_img), dim=0)
        format_data[self.aug_mode[0]]['img'] = ori_rot_img
        # TODO: 目前是img_metas直接copy, 后续是否也把gt旋转一下:
        ori_rot_img_metas = format_data[self.aug_mode[0]]['img_metas'] * 2
        format_data[self.aug_mode[0]]['img_metas'] = ori_rot_img_metas
        # # 可视化
        # bs = format_data[self.aug_mode[0]]['img'].shape[0] // 2
        # self.vis_batch_rot_img(
        #     format_data[self.aug_mode[0]]['img'][:bs], format_data[self.aug_mode[0]]['img'][bs:], 
        #     [[],[]], [[],[]], [[],[]], [[],[]], 
        #     format_data[self.aug_mode[0]]['img_metas'], ori_mask, rand_angle, isflip, './vis_rot_img'
        # )
        
        return format_data, rand_angle, isflip, ori_mask





    def loss(self, ori_sbboxes, rot_sbboxes, ori_nc_scores, rot_nc_scores, batch_ori_idx, batch_rot_idx, reg_loss_w, cls_loss_w):
        '''计算旋转一致性损失
        '''
        ori_sbboxes = torch.cat([ori_sbboxes[i][batch_ori_idx[i]] for i in range(len(ori_sbboxes))])
        rot_sbboxes = torch.cat([rot_sbboxes[i][batch_rot_idx[i]] for i in range(len(rot_sbboxes))])
        ori_nc_scores = torch.cat([ori_nc_scores[i][batch_ori_idx[i]] for i in range(len(ori_nc_scores))])
        rot_nc_scores = torch.cat([rot_nc_scores[i][batch_rot_idx[i]] for i in range(len(rot_nc_scores))])
        # 得到分类置信度
        ori_scores, ori_preds = torch.max(ori_nc_scores, dim=-1)
        rot_scores, rot_preds = torch.max(rot_nc_scores, dim=-1)
        scores_factor = (ori_scores + rot_scores) * 0.5


        '''回归损失'''
        reg_loss = self.riou_loss(ori_sbboxes, rot_sbboxes)
        # reg_loss = (reg_loss * scores_factor).sum() / scores_factor.sum()
        reg_loss = (reg_loss * scores_factor).sum() / reg_loss.shape[0]

        '''分类损失'''
        cls_loss_l = self.cls_loss(ori_nc_scores, rot_nc_scores)
        cls_loss_r = self.cls_loss(rot_nc_scores, ori_nc_scores)
        cls_loss = 0.5 * (cls_loss_l + cls_loss_r)

        return reg_loss*reg_loss_w, cls_loss*cls_loss_w 



    def inv_rotate_bboxes(self, n, bboxes, angle):
        """将矩形框逆angle旋转(angle是逆时针旋转, 逆就是顺时针)
        
        Args:
            tensor: torch.Tensor, 形状为[bs, 5], 每行代表(cx, cy, w, h, theta)
            n: int, 正方形画布的尺寸
            angle: float, 顺时针旋转角度(角度制)
        
        Returns:
            torch.Tensor: 旋转后的矩形框参数, 形状[bs, 5]
        """
        # 将角度转换为弧度(顺时针旋转所以取负)
        angle_rad = -math.radians(angle)
        # 计算旋转中心(画布中心)
        center = n / 2.0
        # 分离各个参数
        cx = bboxes[:, 0]
        cy = bboxes[:, 1]
        w = bboxes[:, 2]
        h = bboxes[:, 3]
        theta = bboxes[:, 4]
        
        # 1. 将坐标转换为以中心为原点
        x = cx - center
        y = center - cy  # y轴向下为正
        # 2. 应用旋转矩阵(顺时针)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        # 3. 转换回画布坐标系
        new_cx = x_rot + center
        new_cy = center - y_rot  # 转换回y轴向下
        # 4. 更新角度(顺时针旋转angle)
        new_theta = theta - angle_rad  # 减去是因为顺时针旋转
        # 5. 处理角度超出[-π/2, π/2]范围的情况
        new_theta = (new_theta + math.pi/2) % math.pi - math.pi/2
        # 6. 处理宽高交换(当旋转90度时)
        swap_mask = (abs(new_theta) > math.pi/4)
        new_w = torch.where(swap_mask, h, w)
        new_h = torch.where(swap_mask, w, h)
        # 7. 调整角度(如果宽高交换了，角度需要调整π/2)
        sign = torch.sign(new_theta)
        adjusted_theta = torch.where(swap_mask, new_theta - sign * (math.pi/2), new_theta)
        # 8. 确保宽高非负
        new_w = torch.abs(new_w)
        new_h = torch.abs(new_h)
        # 组合结果
        rotated_tensor = torch.stack([new_cx, new_cy, new_w, new_h, adjusted_theta], dim=1)
        return rotated_tensor












    # ======================================================可视化======================================================

    def vis_batch_rot_img(self, batch_ori_img, batch_rot_img, batch_ori_preds, batch_rot_preds, batch_ori_nc_scores, batch_rot_nc_scores, batch_ori_idx, batch_rot_idx, img_metas, ori_mask, rand_angle, isflip, save_dir):
        '''可视化图像(旋转前+旋转后)
            Args:
                ori_img

            Returns:
                None
        '''
        if not os.path.exists(save_dir):os.makedirs(save_dir)

        with torch.no_grad():
            # batch处理
            for ori_img, rot_img, ori_preds, rot_preds, ori_nc_scores, rot_nc_scores, ori_idxs, rot_idxs, img_meta in \
                zip(batch_ori_img, batch_rot_img, batch_ori_preds, batch_rot_preds, batch_ori_nc_scores, batch_rot_nc_scores, batch_ori_idx, batch_rot_idx, img_metas):
                '''原图预处理'''
                std = np.array([58.395, 57.12, 57.375]) / 255.
                mean = np.array([123.675, 116.28, 103.53]) / 255.
                # 处理 ori_img
                ori_img = ori_img.permute(1, 2, 0).cpu().numpy()
                ori_img = np.clip(ori_img * std + mean, 0, 1)
                ori_img = (ori_img * 255).astype(np.uint8) 
                ori_img = np.ascontiguousarray(ori_img) 
                # 处理 rot_img
                rot_img = rot_img.permute(1, 2, 0).cpu().numpy()
                rot_img = np.clip(rot_img * std + mean, 0, 1)
                rot_img = (rot_img * 255).astype(np.uint8)
                rot_img = np.ascontiguousarray(rot_img) 

                '''box绘制'''
                # 需要过滤掉背景类
                # ori_scores, ori_labels = torch.max(ori_nc_scores[:, :-1], dim=-1)
                # rot_scores, rot_labels = torch.max(rot_nc_scores[:, :-1], dim=-1)
                # ori_vis_mask = ori_scores >= 0.01
                # rot_vis_mask = rot_scores >= 0.01

                ori_match_mask = np.zeros(ori_preds.shape[0], dtype=np.bool_)
                rot_match_mask = np.zeros(rot_preds.shape[0], dtype=np.bool_)
                ori_match_mask[ori_idxs] = True
                rot_match_mask[rot_idxs] = True

                # 5参转8参
                poly_ori_preds = obb2poly(ori_preds).cpu().numpy().astype(np.int32)
                poly_rot_preds = obb2poly(rot_preds).cpu().numpy().astype(np.int32)

                # 绘制匹配上的哪些框
                for ori_idx, rot_idx in zip(ori_idxs, rot_idxs):
                    poly_ori_pred = poly_ori_preds[ori_idx]
                    poly_rot_pred = poly_rot_preds[rot_idx]
                    # 随机颜色
                    color = np.random.randint(0, 256, size=3)
                    gt_color = tuple([int(x) for x in color])
                    pred_color = tuple([int(min(x+35, 255)) for x in color])
                    # 绘制匹配上的那些框
                    ori_img = OpenCVDrawBox(ori_img, [poly_ori_pred], pred_color, 1)
                    ori_img = OpenCVDrawBox(ori_img, [poly_rot_pred], gt_color, 1)
                    # 绘制匹配线段
                    cx0, cy0 = round(poly_ori_pred[[0,2,4,6]].mean()), round(poly_ori_pred[[1,3,5,7]].mean())
                    cx1, cy1 = round(poly_rot_pred[[0,2,4,6]].mean()), round(poly_rot_pred[[1,3,5,7]].mean())
                    cv2.line(ori_img, (cx0, cy0), (cx1, cy1), pred_color, thickness=2)

                # 绘制匹配不上的那些框
                # ori_img = OpenCVDrawBox(ori_img, poly_ori_preds[ori_match_mask], (0,255,0), 2)
                # rot_img = OpenCVDrawBox(rot_img, poly_rot_preds[rot_match_mask], (0,255,0), 2)
                # ori_img = OpenCVDrawBox(ori_img, poly_ori_preds[~ori_match_mask], (255,255,255), 1)
                # ori_img = OpenCVDrawBox(ori_img, poly_rot_preds[~rot_match_mask], (255,255,255), 1)



                '''绘制+保存'''
                # 获取图像名
                img_name = img_meta['ori_filename']
                # 创建一行两列的画布
                plt.figure(figsize=(12, 12))
                # 绘制 strong_img
                plt.subplot(2, 2, 1)
                plt.imshow(ori_img)
                plt.title(f'raw image')
                plt.axis('off')
                # 绘制 weak_img
                plt.subplot(2, 2, 2)
                plt.imshow(rot_img)
                plt.title(f'rot image, θ={rand_angle:.2f}, flip={isflip}')
                plt.axis('off')
                # 绘制 ori_mask
                plt.subplot(2, 2, 3)
                plt.imshow(ori_mask)
                plt.title(f'ori_mask')
                plt.axis('off')

                # 调整布局并保存
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, img_name), bbox_inches='tight', dpi=150)
                plt.close()