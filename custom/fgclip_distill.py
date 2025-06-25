import torch
import torch.nn as nn
import numpy as np
import cv2
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from mmcv.ops import box_iou_quadri, box_iou_rotated
from torchvision.transforms.functional import rotate, pad
import torch.nn.functional as F
import random
from mmrotate.models import ROTATED_LOSSES, build_loss
from torchvision.transforms import InterpolationMode
import torch.distributed as dist
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM 
import math
import matplotlib
import matplotlib.pyplot as plt


from custom.utils import *
from custom.loss import KLDivLoss, SmoothL1Loss
from custom.visualize import *





class CustomBatchNorm2d(nn.Module):
    '''当bs=1时, 跳过BN
    '''
    def __init__(self, channels):
        super(CustomBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        if x.size(0) == 1:
            return x
        else:
            return self.bn(x)
        

class ConvDistillProj(nn.Module):
    '''将模型输出的fpn dense feature 维度映射到与clip的维度对齐
    '''
    def __init__(self, in_dim, out_dim):
        super(ConvDistillProj, self).__init__()
        # 两层1x1卷积
        # self.conv_proj = nn.Sequential(
        #     nn.Conv2d(in_dim, in_dim, 1, 1, bias=False),
        #     CustomBatchNorm2d(in_dim),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_dim, out_dim, 1, 1, bias=False)
        # )
        # 一层1x1卷积
        # self.conv_proj = nn.Sequential(
        #     nn.Conv2d(in_dim, out_dim, 1, 1, bias=False)
        # )
        # 一层线性层
        self.linear_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False)
        )
        # 权重初始化
        # init_weights(self.convBlocks, 'he')
        init_weights(self.linear_proj, 'normal', 0, 0.01)

    def forward(self, x):
        return self.linear_proj(x)   














class FGCLIP():
    def __init__(self, ):
        # load model "qihoo360/fg-clip-base"
        model_root = "qihoo360/fg-clip-large"
        self.model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True).cuda()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_root)
        self.image_processor = AutoImageProcessor.from_pretrained(model_root)






class FGCLIPDistillBranch(nn.Module):
    '''使用FGCLIP蒸馏原有模型
    '''
    def __init__(self, ):
        super(FGCLIPDistillBranch, self).__init__()
        self.fgclip = FGCLIP()
        self.proj = ConvDistillProj(in_dim=256*5, out_dim=768)
        self.kldLoss = KLDivLoss()
        self.smoothl1Loss = SmoothL1Loss()
    

    def forward(self, fpn_feat, image, img_metas, mode):
        """
            fpn_feat:
            image:     [bs, 3, 1024, 1024]
            img_metas: [{...}, ..., {...}] {...}.keys() = 'ori_filename',...
        """
        bs = image.shape[0]
        # fg-clip dense feat的尺寸:
        size = (24, 24)
        resize_fpn_feat = []
        for feat in fpn_feat:
            resize_feat = F.interpolate(feat, size=size, mode='bilinear', align_corners=False)
            resize_fpn_feat.append(resize_feat)
        # [bs, 256*5, 24, 24]
        resize_fpn_feat = torch.cat(resize_fpn_feat, dim=1)

        # 使用卷积执行维度变换(特征级蒸馏)
        # [bs, 256*5, 24, 24] ->  [bs, 768, 24, 24]
        # resize_fpn_feat = self.proj(resize_fpn_feat)
        # resize_fpn_feat = resize_fpn_feat.reshape(bs, -1, 768)

        # 使用linear执行维度变换(特征级蒸馏)
        # [bs, 256*5, 24, 24] -> [24*24*bs, 256*5]
        resize_fpn_feat = resize_fpn_feat.reshape(24*24*bs, -1)
        # [24*24*bs, 256*5] -> [24*24*bs, 768]
        resize_fpn_feat = self.proj(resize_fpn_feat)
        # [24*24*bs, 768] -> [bs, 24*24, 768]
        resize_fpn_feat = resize_fpn_feat.reshape(bs, -1, 768)

        # 得到clip dense feat
        with torch.no_grad():
            # [2, 3, 1024, 1024] -> [2, 3, 336, 336]
            input_img = F.interpolate(image, size=(336, 336), mode='bilinear')
            # [bs, 24 * 24, 768]
            clip_dense_feat = self.fgclip.model.get_image_dense_features(input_img)
            # 可视化
            # if mode == 'unsup':
            #     vis_clip_feat_batch(self.fgclip, clip_dense_feat.reshape(bs, 24, 24, -1), image, img_metas, save_dir='./vis_unsup_fgclip_densefeat')

        return resize_fpn_feat, clip_dense_feat


    def loss(self, fpn_feat, image, img_metas, mode='sup'):
        # 形状均为[bs, 24 * 24, 768]
        resize_fpn_feat, clip_dense_feat = self.forward(fpn_feat, image, img_metas, mode)
        # distill_loss = self.kldLoss(resize_fpn_feat, clip_dense_feat, to_distribution=True, dist_dim=2, reduction='mean')
        distill_loss = self.smoothl1Loss(resize_fpn_feat, clip_dense_feat, reduction='mean')
        return distill_loss


