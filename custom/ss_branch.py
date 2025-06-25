import torch
import torch.nn as nn
import numpy as np
import cv2
from mmrotate.core import rbbox2result, poly2obb_np, obb2poly, poly2obb, build_bbox_coder, obb2xyxy
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from mmcv.ops import box_iou_quadri, box_iou_rotated
from torchvision.transforms.functional import rotate, pad
import random
from mmrotate.models import ROTATED_LOSSES, build_loss
from torchvision.transforms import InterpolationMode
import torch.distributed as dist

from custom.utils import *
from custom.loss import QFLv2, BCELoss, JSDivLoss
from custom.visualize import *

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



def batch_tensor_random_rotate(tensor, angle_range, rand=True):
    '''对张量图像进行随机角度旋转(使用镜像填充), 并返回旋转mask
        Args:
            tensor: 张量图像, 形状为[bs, 3, h, w]
            angle_range: 随机旋转的角度(角度)范围[min, max]

        Returns:
            rotaug_img: 旋转后的图像
            rand_angle: 旋转的角度
    '''
    # 生成一个随机角度
    if rand:
        rand_angle = random.uniform(angle_range[0], angle_range[1])
    else:
        return tensor, 0.
    # 先进行镜像填充
    padded_tensor, padded_len = pad_tensor(tensor, (2**0.5 - 1) / 2)
    # 再进行旋转操作
    rotaug_img = rotate(padded_tensor, rand_angle, expand=False)
    h, w = rotaug_img.shape[2:]
    # rotaug_img去掉padding的部分, 还原为原始大小(这样就能去除掉padding的黑色填充, 只保留镜像填充)
    rotaug_img = rotaug_img[..., padded_len[0]:h-padded_len[2], padded_len[1]:w-padded_len[3]]
    # ori_mask = F.interpolate(ori_mask.unsqueeze(1), scale_factor=1/16, mode='nearest').squeeze(1)
    return rotaug_img, rand_angle






class SSBranch(nn.Module):
    '''Head部分回归和分类之前的共享特征提取层(目的是将ROI的7x7压缩到1x1)
    '''
    def __init__(self, nc, rand_angle_range, flip_p, score_interpolate_mode, box_interpolate_mode, score_loss_w, box_loss_w):
        super(SSBranch, self).__init__()
        self.e = 1e-10
        self.nc = nc
        self.rand_angle_range = rand_angle_range
        self.flip_p = flip_p
        self.JSDLoss = JSDivLoss()
        # 对特征翻转时用的插值方式
        self.score_interpolate_mode = {'nearest':InterpolationMode.NEAREST, 'bilinear':InterpolationMode.BILINEAR}[score_interpolate_mode]
        self.box_interpolate_mode =   {'nearest':InterpolationMode.NEAREST, 'bilinear':InterpolationMode.BILINEAR}[box_interpolate_mode]
        # 损失函数权重
        self.score_loss_w = score_loss_w
        self.box_loss_w = box_loss_w

    def gen_aug_data(self, format_data, aug_orders):
        """
        """
        # 获取图像及基本信息 注意是aug_orders[0]才是无监督student分支的原始图像不要搞错了
        rot_img = format_data[aug_orders[0]]['img'].clone()
        img_name = format_data[aug_orders[0]]['img_metas'][0]['filename'].split('/')[-1]
        # 旋转增强操作(核心部分):
        rot_img, rand_angle = batch_tensor_random_rotate(rot_img, self.rand_angle_range, rand=True)
        # 以概率p进行翻转增强操作:
        rot_img, isflip = batch_tensor_random_flip(rot_img, p=self.flip_p)
        # 将原始图像和旋转后的图像沿batch维度拼接(注意原始图像在前面, 旋转图像在后面)后, 覆盖原aug_orders的无监督图片
        ori_rot_img = torch.cat((format_data[aug_orders[0]]['img'], rot_img), dim=0)
        format_data[aug_orders[0]]['img'] = ori_rot_img
        # 可视化
        # vis_rgb_tensor(ori_rot_img[1, ...], f"{rand_angle}_flip-{isflip}_{img_name}", './vis_rot_img')
        # vis_rgb_tensor(ori_rot_img[0, ...], f"{rand_angle}_flip-{isflip}_{img_name}", './vis_ori_img')

        return format_data, rand_angle, isflip



    def forward(self, format_data, aug_orders, reshape_s_ori_logits, reshape_s_rot_logits, weight_mask, rand_angle, isflip):
        """
        """
        # 注意cls_scores和centernesses都是未经过sigmoid()的logits. _bbox_preds=[total_anchor_num, 5=(cx, cy, w, h, θ)]
        o_cls_scores, o_bbox_preds, o_centernesses, _ = reshape_s_ori_logits
        r_cls_scores, r_bbox_preds, r_centernesses, _ = reshape_s_rot_logits

        # 对原始图像上的预测结果旋转到与旋转结果对齐
        # 0.首先把长条状的特征再还原为二维图像的形状
        H, W = format_data[aug_orders[0]]['img'][0].shape[1:]
        # 这里默认H=W
        sizes = [H//8, H//16, H//32, H//64, H//128]
        # [total_anchor_num, ] -> [[h1*w1, ], [h5*w5, ]]
        o_cls_scores_list = torch.split(o_cls_scores, [size * size for size in sizes], dim=0)
        o_centernesses_list = torch.split(o_centernesses, [size * size for size in sizes], dim=0)
        o_weight_mask_list = torch.split(weight_mask, [size * size for size in sizes], dim=0)
        # [[h1*w1, ], [h5*w5, ]] -> [[h1, w1, ], [h5, w5, ]]
        o_cls_scores_list = [x.reshape(size, size, -1).sigmoid()+self.e for size, x in zip(sizes, o_cls_scores_list)]
        o_centernesses_list = [x.reshape(size, size, -1).sigmoid()+self.e for size, x in zip(sizes, o_centernesses_list)]
        o_weight_mask_list = [x.reshape(size, size, -1) for size, x in zip(sizes, o_weight_mask_list)]
        # 1.对原始图像的特征图逐尺度执行旋转操作(用最近邻差值，保证旋转边界清晰)
        # NOTE: 必须fill一个很小的数, 填充0.会报错:    
        # "/home/yht/.conda/envs/sood-mcl/lib/python3.9/site-packages/mmdet/models/losses/gfocal_loss.py", line 88, in quality_focal_loss_with_prob
        # pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
        # RuntimeError: numel: integer multiplication overflow
        o_cls_scores_list = [rotate(x.permute(2, 0, 1), rand_angle, expand=False, interpolation=self.score_interpolate_mode, fill=[self.e,]).permute(1, 2, 0) for x in o_cls_scores_list]
        o_centernesses_list = [rotate(x.permute(2, 0, 1), rand_angle, expand=False, interpolation=self.score_interpolate_mode, fill=[self.e,]).permute(1, 2, 0) for x in o_centernesses_list]
        o_weight_mask_list = [rotate(x.permute(2, 0, 1), rand_angle, expand=False, interpolation=self.score_interpolate_mode, fill=[self.e,]).permute(1, 2, 0) for x in o_weight_mask_list]
        # 2.如果执行了翻转操作, 则翻转回来
        if isflip:
            o_cls_scores_list = [torch.flip(x, dims=[0]) for x in o_cls_scores_list]
            o_centernesses_list = [torch.flip(x, dims=[0]) for x in o_centernesses_list]
            o_weight_mask_list = [torch.flip(x, dims=[0]) for x in o_weight_mask_list]
        # 3.对旋转后的特征再拉直为一维特征
        o_cls_scores = convert_shape_single(o_cls_scores_list, self.nc, bs_dim=False)
        o_centernesses = convert_shape_single(o_centernesses_list, 1, bs_dim=False)
        o_weight_mask = convert_shape_single(o_weight_mask_list, 1, bs_dim=False).reshape(-1)

        # 把bbox回归值进行旋转:
        # 对bbox_preds解码
        r_decode_bboxes = decode_rbbox(r_bbox_preds)
        o_decode_bboxes = decode_rbbox(o_bbox_preds)
        # 对o_bbox_preds旋转
        # 5参转8参
        o_poly_bboxes = obb2poly(o_decode_bboxes, version='le90').reshape(-1, 4, 2)
        # 定义旋转矩阵
        R = torch.tensor(cv2.getRotationMatrix2D((W / 2, H / 2) , rand_angle, 1), device=o_poly_bboxes.device, dtype=torch.float32)
        one_tensor = torch.ones((o_poly_bboxes.shape[0], 4, 1), device=o_decode_bboxes.device)
        o_poly_bboxes = torch.cat((o_poly_bboxes, one_tensor), dim=-1)
        o_decode_bboxes = torch.einsum("ij, klj -> kli", R, o_poly_bboxes).reshape(-1, 8)
        o_decode_bboxes = poly2obb(o_decode_bboxes, version='le90')
        # 如果执行了翻转操作, 则先对坐标翻转(cy, θ)
        if isflip:
            o_decode_bboxes[:, 1] = H - o_decode_bboxes[:, 1]
            o_decode_bboxes[:, 4] *= -1.
        # 刚刚虽然把bbox旋转到正确的位置上了,还需要把每个grid转到正确的位置上
        o_decode_bboxes_list = torch.split(o_decode_bboxes, [size * size for size in sizes], dim=0)
        # 先旋转
        o_decode_bboxes_list = [x.reshape(size, size, -1) for size, x in zip(sizes, o_decode_bboxes_list)]
        o_decode_bboxes_list = [rotate(x.permute(2, 0, 1), rand_angle, expand=False, interpolation=self.box_interpolate_mode, fill=[self.e,]).permute(1, 2, 0) for x in o_decode_bboxes_list]
        # 生成旋转mask, 用于屏蔽原始特征和旋转特征不一致的部分(旋转产生的padding部分)
        # 有时候h和w会存在=0的情况, 一并做个排除(但是这里只排除了ori预测分支的情况, 没排除rot分支的情况)
        o_h_mask = [(x[..., 2]>self.e) for x in o_decode_bboxes_list]
        o_w_mask = [(x[..., 3]>self.e) for x in o_decode_bboxes_list]
        rot_mask_list = [(w * h).unsqueeze(0) for w, h in zip(o_h_mask, o_w_mask)]
        # 再翻转(注意旋转mask无需翻转)
        if isflip:
            o_decode_bboxes_list = [torch.flip(x, dims=[0]) for x in o_decode_bboxes_list]
        # 拉直
        o_decode_bboxes = convert_shape_single(o_decode_bboxes_list, 5, bs_dim=False)
        rot_mask = convert_shape_single(rot_mask_list, 1, bs_dim=False).reshape(-1)
        # o_pos_mask = (o_weight_mask > 0.15) * rot_mask
        # r_max_clsscore = torch.max(r_cls_scores, dim=1)[0]
        # r_pos_mask = (r_max_clsscore > 0.15) * rot_mask
        o_weight_mask = o_weight_mask * (rot_mask + self.e)
        local_rot_mask_sum = rot_mask.sum()
        rot_mask_sum = local_rot_mask_sum.clone() 
        # 进行多卡之间通信, 此时rot_mask_sum数值为所有gpu上local_rot_mask_sum的值之和
        # 当local_rot_mask_sum * dist.get_world_size() == rot_mask_sum时没问题, 否则说明其中一个gpu上预测出了w或h=0的情况
        dist.all_reduce(rot_mask_sum, op=dist.ReduceOp.SUM)
        print(f"local_mask_sum:{local_rot_mask_sum * dist.get_world_size()}, total_mask_sum:{rot_mask_sum}, {local_rot_mask_sum * dist.get_world_size()==rot_mask_sum}")
        if local_rot_mask_sum * dist.get_world_size() == rot_mask_sum:
            # 可视化
            # 可视化bbox
            # vis_rboxes_on_img(format_data[aug_orders[0]]['img'][1], o_decode_bboxes[o_pos_mask].clone().detach(), f"{rand_angle}_flip-{isflip}_{img_name}", './vis_unsup_o_decode_bbox')
            # vis_rboxes_on_img(format_data[aug_orders[0]]['img'][1], r_decode_bboxes[o_pos_mask].clone().detach(), f"{rand_angle}_flip-{isflip}_{img_name}", './vis_unsup_r_decode_bbox')
            # 可视化特征图
            # vis_rotate_feat(format_data[aug_orders[0]]['img'][1], format_data[aug_orders[0]]['img'][1], f"{rand_angle}_flip-{isflip}_{img_name}", o_cls_scores, o_centernesses, r_cls_scores, r_centernesses, o_weight_mask, rot_mask)

            
            '''自监督分类损失'''
            ss_loss_joint_score = self.cls_ssloss(o_centernesses, r_centernesses, o_cls_scores, r_cls_scores, rot_mask) * self.score_loss_w
            '''自监督回归框(角度+尺度)一致损失'''
            ss_loss_box = self.box_ssloss(rand_angle, o_decode_bboxes, r_decode_bboxes, o_weight_mask, rot_mask) * self.box_loss_w
        else:
            # 否则全返回0损失值
            ss_loss_joint_score = torch.tensor(0.0, device=rot_mask.device)
            ss_loss_box = torch.tensor(0.0, device=rot_mask.device)
    
        return ss_loss_joint_score, ss_loss_box


    def cls_ssloss(self, o_centernesses, r_centernesses, o_cls_scores, r_cls_scores, rot_mask):
        '''自监督类别一致损失'''
        # print(f"o_cls_min:{o_cls_scores[rot_mask].min()}, o_cls_max:{o_cls_scores[rot_mask].max()}")
        # print(f"r_cls_min:{r_cls_scores[rot_mask].min()}, r_cls_max:{r_cls_scores[rot_mask].max()}")
        # print(r_cls_scores.sigmoid()[rot_mask].min(), r_cls_scores.sigmoid()[rot_mask].max())
        # # ss_loss_cls = self.QFLv2(o_cls_scores[rot_mask], r_cls_scores.sigmoid()[rot_mask], weight=torch.ones_like(o_cls_scores[rot_mask], device=o_cls_scores.device, dtype=torch.bool), reduction="none").sum() / o_weight_mask[rot_mask].sum()
        # ss_loss_cls = self.JSDLoss(o_cls_scores[rot_mask], r_cls_scores.sigmoid()[rot_mask], to_distribution=True, dist_dim=0, reduction='sum')

        '''自监督中心度一致损失'''
        # o_centernesses = torch.clamp(o_centernesses, e, 1. - e)
        # r_centernesses = torch.clamp(r_centernesses.sigmoid(), e, 1. - e)
        # print(f"o_cnt_min:{o_centernesses[rot_mask].min()}, o_cnt_max:{o_centernesses[rot_mask].max()}")
        # print(f"r_cnt_min:{r_centernesses[rot_mask].min()}, r_cnt_max:{r_centernesses[rot_mask].max()}")
        # # ss_loss_cnt_all = self.BCE_loss(o_centernesses[rot_mask], r_centernesses[rot_mask], reduction='none')
        # # ss_loss_cnt = (o_weight_mask[rot_mask] * ss_loss_cnt_all.reshape(-1)).sum() / o_weight_mask[rot_mask].sum()
        # ss_loss_cnt_all = self.JSDLoss(o_centernesses[rot_mask], r_centernesses[rot_mask], to_distribution=True, dist_dim=0, reduction='none')
        # ss_loss_cnt = (o_weight_mask[rot_mask] * ss_loss_cnt_all.reshape(-1)).sum() / o_weight_mask[rot_mask].sum()
        
        '''自监督联合置信度一致损失(置信度和类别一起用类别损失优化)'''
        o_centernesses = torch.clamp(o_centernesses, self.e, 1. - self.e)
        r_centernesses = torch.clamp(r_centernesses.sigmoid(), self.e, 1. - self.e)
        o_joint_score = torch.einsum("ij, i -> ij", o_cls_scores, o_centernesses.reshape(-1))
        r_joint_score = torch.einsum("ij, i -> ij", r_cls_scores.sigmoid(), r_centernesses.reshape(-1))
        print(f"o_js_min:{o_joint_score[rot_mask].min()}, o_js_max:{o_joint_score[rot_mask].max()}")
        print(f"r_js_min:{r_joint_score[rot_mask].min()}, r_js_max:{r_joint_score[rot_mask].max()}")
        # ss_loss_joint_score = self.QFLv2(o_joint_score[rot_mask], r_joint_score[rot_mask], weight=torch.ones_like(o_joint_score[rot_mask], device=o_joint_score.device, dtype=torch.bool), reduction="none").sum() / o_weight_mask[rot_mask].sum()
        # ss_loss_joint_score = self.JSDLoss(o_joint_score[rot_mask], r_joint_score[rot_mask], to_distribution=True, dist_dim=1, reduction='mean', loss_weight=1e3)
        # ss_loss_joint_score = self.JSDLoss(o_joint_score[rot_mask], r_joint_score[rot_mask], to_distribution=True, dist_dim=1, reduction='sum', loss_weight=1.) / o_weight_mask[rot_mask].sum()
        ss_loss_joint_score = self.JSDLoss(o_joint_score[rot_mask], r_joint_score[rot_mask], to_distribution=True, dist_dim=0, reduction='sum')

        return ss_loss_joint_score
    

    def box_ssloss(self, rand_angle, o_decode_bboxes, r_decode_bboxes, o_weight_mask, rot_mask):
        '''自监督回归框(角度+尺度)一致损失'''
        # box在计算损失的时候就只取那些旋转前后一致的样本(rot_mask)
        riou_loss = build_loss(dict(type='RotatedIoULoss', reduction='none'))
        # 有时候h和w会存在=0的情况, 用clamp避免
        # 对 o_decode_bboxes 的非原地操作
        o_decode_bboxes = torch.cat([
            o_decode_bboxes[:, :2],
            torch.clamp(o_decode_bboxes[:, 2:4], min=2.),
            o_decode_bboxes[:, 4:]  
        ], dim=1)
        # 对 r_decode_bboxes 的非原地操作
        r_decode_bboxes = torch.cat([
            r_decode_bboxes[:, :2],
            torch.clamp(r_decode_bboxes[:, 2:4], min=2.),
            r_decode_bboxes[:, 4:] 
        ], dim=1)

        o_h_nonzero, o_w_nonzero = o_decode_bboxes[:, 2]!=0., o_decode_bboxes[:, 3]!=0.
        o_nonzero_mask = o_h_nonzero * o_w_nonzero
        r_h_nonzero, r_w_nonzero = r_decode_bboxes[:, 2]!=0., r_decode_bboxes[:, 3]!=0.
        r_nonzero_mask = r_h_nonzero * r_w_nonzero
        nonzero_mask = o_nonzero_mask * r_nonzero_mask
        final_mask = rot_mask * nonzero_mask
        print(f"mask-diff:{final_mask.sum() - rot_mask.sum()}, rand_angle:{rand_angle}")
        if final_mask.sum() - rot_mask.sum() < 0:
            o_decode_bboxes[:, 2:4] += 1
            r_decode_bboxes[:, 2:4] += 1

        ss_loss_box_all = riou_loss(o_decode_bboxes[rot_mask], r_decode_bboxes[rot_mask])
        ss_loss_box = (o_weight_mask[rot_mask] * ss_loss_box_all)
        # 去除掉那些负数的损失
        ss_loss_box = ss_loss_box[ss_loss_box>0]
        print(f"ss_box_loss_min:{ss_loss_box.min()}, ss_box_loss_max:{ss_loss_box.max()}")
        ss_loss_box = ss_loss_box.sum() / o_weight_mask[rot_mask].sum()
        print(f"ss_box_loss:{ss_loss_box}")
        print('='*100)
        # 可视化RIoU
        # vis_lvlriou(format_data[aug_orders[0]]['img'][1], o_decode_bboxes, r_decode_bboxes, img_name, 'vis_ro_branch_riou')

        return ss_loss_box
    
