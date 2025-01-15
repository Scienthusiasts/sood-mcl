import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



class FCOSPrototype(nn.Module):
    '''Prototype原型
    '''
    def __init__(self, cat_nums, scale_nums=5, dim=256, decay=0.9996, beta=1000, updates=0, mode='contrast', loss_weight=1.0):
        '''初始化
            # Args:
                - cat_nums: 数据集类别数
                - dim:      prototype的维度
                - decay:    ema的衰减率
                - beta:     ema衰减的超参(单位iter), 越大则decay越慢到达最大值
                - updates:  当前已经ema更新了多少次模型(resume时不为0)
                - mode:     prototype更新方式(contrast, ema, self_contrast, )
            # Returns:
                None
        '''
        super(FCOSPrototype, self).__init__()
        self.mode = mode
        self.scale_nums = scale_nums
        # 类别数
        self.cat_nums = cat_nums + 1
        # self.updates用于记录当前已经ema更新了多少次模型
        # TODO:self.updates还有一个问题就是当resume训练时这部分会直接从0开始
        self.updates = updates 
        # 定义ema衰减函数, 有利于前期训练(初始d=0,之后随着self.updates的增大d慢慢趋近decay常量)
        self.decay = lambda x: decay * (1 - math.exp(-x / beta))  
        # class-wise prototypes (平均初始化)
        self.prototypes = nn.Parameter(torch.ones(size=(self.cat_nums, scale_nums, dim)) * 0.01, requires_grad={'contrast':True,'self_contrast':True,'ema':False}[mode])
        # self.prototypes = nn.Parameter(torch.normal(0, 0.01, size=(self.cat_nums, scale_nums, dim)), requires_grad={'contrast':True,'self_contrast':True,'ema':False}[mode])
        # delta_prototype用于临时存储当前batch下mem_bank的平均特征 (固定初始化)
        self.delta_prototype = torch.ones(size=(self.cat_nums, scale_nums, dim)) * 0.01
        # self.delta_prototype = torch.normal(0, 0.01, size=(self.cat_nums, scale_nums, dim))
        # 用于标记当前batch哪些类别有正样本
        self.mem_bank = np.zeros((self.cat_nums, scale_nums)).astype(np.bool_)
        # mode='contrast'用到
        self.contrast_loss = InfoNCELoss(cat_nums=self.cat_nums, scale_nums=self.scale_nums, t=0.07, reduction='none')
        self.loss_weight = loss_weight


    def forward(self, cls_feats, cls_targets, lvl_idx):
        '''前向过程+损失计算
        '''
        self.update_mem_bank(cls_feats, cls_targets, lvl_idx, device=cls_feats[0].device)
        if self.mode == 'ema':
            prototype_loss = self.ema_update(device=cls_feats[0].device)   
        elif self.mode in ['contrast', 'self_contrast']:
            prototype_loss = self.contrast_learning_update()

        self.updates += 1

        if self.mode in ['ema', 'contrast']:
            return prototype_loss
        elif self.mode == 'self_contrast':
            # prototypes之间进行对比学习
            prototype_contrast_loss = self.prototypes_contrast()
            return prototype_loss, prototype_contrast_loss


    def update_mem_bank(self, feat, cat_gt, scale_idx, device):
        '''更新memory bank(只取GT的正样本区域)
            # Args:
                - feat:   head之前的特征图特征(正样本)
                - cat_gt: 每个特征对应的类别标签(正样本)
                - scale_idx: 每个尺度对应的样本索引
            # Returns:
                None
        '''
        # 第一次则将prototypes转移到对应gpu上
        if self.updates == 0:
            self.prototypes = self.prototypes.to(device)
            self.delta_prototype = self.delta_prototype.to(device)
            
        # 逐类别更新mem_bank
        for i in range(self.cat_nums):
            for lvl, lvl_idx in enumerate(scale_idx):
                lvl_cat_gt = cat_gt[lvl_idx]
                lvl_feat = feat[lvl_idx]
                # 找到对应类别下的正样本索引
                lvl_cat_idx = torch.where(lvl_cat_gt==i)[0]
                # 当前batch有这个类别的正样本才更新
                if len(lvl_cat_idx) > 0:
                    # 这里是截断梯度的, 意味着单纯更新prototype这个操作不会影响到网络其他部分的更新
                    self.mem_bank[i, lvl] = True
                    self.delta_prototype[i, lvl] = lvl_feat[lvl_cat_idx].mean(dim=0).detach()


    def ema_update(self, device):
        '''法1: ema更新prototype
            # Args:
                - device: 特征在哪个gpu
            # Returns:
                None
        '''
        d = self.decay(self.updates)
        # 逐类别ema更新prototypes
        for i in range(self.cat_nums):
            for lvl in range(self.scale_nums):
                # 当前batch有这个类别的正样本才更新
                if self.mem_bank[i, lvl] != False:
                    '''EMA更新核心代码:'''
                    self.prototypes[i, lvl] *= d
                    self.prototypes[i, lvl] += (1. - d) * self.delta_prototype[i, lvl]

        # 返回一个无意义的loss占位
        loss = torch.tensor(0).to(device)
        # 清空
        self.mem_bank *= False
        return loss


    def contrast_learning_update(self):
        '''法2: 对比学习更新prototype
            # Args:
                - device: 特征在哪个gpu
            # Returns:
                - loss: 对比损失
        '''
        # NOTE:这里是不是可以对self.delta_prototype加一个均值为0, 标准差为0.01的高斯噪声
        loss = self.contrast_loss(self.prototypes, self.delta_prototype)
        # 只对那些当前batch有正样本的prototype更新
        loss = loss[self.mem_bank.reshape(-1)].mean()
        # 清空
        self.mem_bank *= False
        return loss


    def prototypes_contrast(self, ):
        '''prototypes之间进行对比学习
        '''
        loss = self.contrast_loss(self.prototypes, self.prototypes)
        return loss.mean()





    def vis_heatmap(self, batch_fpn_feat, batch_imgs, batch_img_names, batch_pos_mask, scale_idx, batch_centerness, batch_cls_score):
        '''可视化prototype与特征图的相似度heatmap (only for experimental validation)
            Args:
                - batch_fpn_feat:  shape = [bs, total_anchor_num, 256]
                - batch_imgs:      shape = [bs, 3, 1024, 1024]
                - batch_img_names: [name_1, ..., name_bs]
            Return:
        '''
        if not os.path.exists('./vis_prototype'):os.makedirs('./vis_prototype')
        sizes = [128, 64 ,32, 16, 8]

        '''逐尺度的进行prototype和特征图交互, 得到相似度heatmap'''
        cat_nums = len(self.prototypes)
        batch_catwise_map = torch.zeros((batch_fpn_feat.shape[0], cat_nums), device=batch_fpn_feat.device)
        # 归一化
        batch_fpn_feat = batch_fpn_feat / batch_fpn_feat.norm(dim=-1, keepdim=True)
        prototypes = self.prototypes / self.prototypes.norm(dim=-1, keepdim=True)
        for i in range(cat_nums):
            for lvl, lvl_idx in enumerate(scale_idx):
                catwise_map = torch.einsum('ij, j -> i', batch_fpn_feat[lvl_idx], self.prototypes[i, lvl])
                batch_catwise_map[lvl_idx, i] = catwise_map

        batch_cls_score_max, batch_cls_score_idx = batch_cls_score.max(dim=1)
        batch_joint_score = batch_cls_score_max * batch_centerness
        # 把batch维度拆开
        batch_catwise_map = batch_catwise_map.reshape(batch_imgs.shape[0], -1, cat_nums)
        batch_pos_mask = batch_pos_mask.reshape(batch_imgs.shape[0], -1)
        batch_joint_score = batch_joint_score.reshape(batch_imgs.shape[0], -1)
        batch_cls_idx = batch_cls_score_idx.reshape(batch_imgs.shape[0], -1)

        '''遍历batch里每一张图像'''
        for catwise_map, img, img_name, pos_mask, joint_score, cls_idx in zip(batch_catwise_map, batch_imgs, batch_img_names, batch_pos_mask, batch_joint_score, batch_cls_idx):

            # 取最大置信度类别的置信度[total_anchor_num, cat_num] -> [[h1*w1, cat_num], ..., [h5*w5, cat_num]]
            active_score_map, active_score_idx = catwise_map.max(dim=1)
            idx = 0
            cls_mask = (cls_idx!=idx) | (active_score_idx!=idx)
            bgd_mask = active_score_idx==16
            active_score_map[bgd_mask | cls_mask] = -1

            '''不同尺度拆分开'''
            active_score_map = torch.split(active_score_map, [size * size for size in sizes], dim=0)
            # [total_anchor_num] -> [[h1*w1], [h5*w5]]
            pos_mask = torch.split(pos_mask, [size * size for size in sizes], dim=0)
            joint_score = torch.split(joint_score, [size * size for size in sizes], dim=0)

            # 对原图预处理
            std = np.array([58.395, 57.12 , 57.375]) / 255.
            mean = np.array([123.675, 116.28 , 103.53]) / 255.
            img = img.permute(1,2,0).cpu().numpy()
            img = np.clip(img * std + mean, 0, 1)

            '''可视化'''
            plt.figure(figsize = (10, 8))
            for lvl, lvl_active_score_map in enumerate(active_score_map):
                lvl_active_score_map = lvl_active_score_map.reshape(sizes[lvl], sizes[lvl])
                lvl_pos_mask = pos_mask[lvl].reshape(sizes[lvl], sizes[lvl])
                lvl_joint_score = joint_score[lvl].reshape(sizes[lvl], sizes[lvl])
                # 可视化特征图
                plt.subplot(4,5,lvl+1)
                plt.imshow(lvl_active_score_map.detach().cpu().numpy(), cmap='jet', vmin=-1, vmax=1)
                plt.axis('off')
                # 可视化联合置信度
                plt.subplot(4,5,lvl+6)
                plt.imshow(lvl_joint_score.detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
                plt.axis('off')
                # 可视化GT
                plt.subplot(4,5,lvl+11)
                plt.imshow(lvl_pos_mask.cpu().numpy(), cmap='jet', vmin=0, vmax=1)
                plt.axis('off')
            # 可视化原图
            plt.subplot(4,5,16)
            plt.imshow(img, vmin=0, vmax=1)
            plt.axis('off')
            plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
            plt.savefig(f"./vis_prototype/{img_name}", dpi=200)  


            











class InfoNCELoss(nn.Module):
    '''InfoNCELoss
    '''
    def __init__(self, cat_nums, scale_nums, t=0.07, reduction='mean'):
        super(InfoNCELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction=reduction)
        # 温度超参数
        self.t = t
        self.scale_nums = scale_nums
        self.cat_nums = cat_nums
    

    def forward(self, vec1, vec2):
        # 先计算两个向量相似度
        sim_logits = self.COSSim(vec1, vec2) / self.t
        sim_logits = sim_logits.reshape(-1, self.cat_nums)
        # 生成对比标签(对角线为1其余为0)
        labels = torch.cat([torch.arange(vec1.shape[0]).to(vec1.device) for _ in range(self.scale_nums)], dim=0)
        # 相似度结果作为logits计算交叉熵损失
        loss = self.loss(sim_logits, labels)
        return loss


    # 计算余弦相似度
    @staticmethod
    def COSSim(vec1, vec2):
        '''计算余弦相似度
        '''
        # 特征向量归一化(欧氏距离=1)
        vec1 = vec1 / vec1.norm(dim=-1, keepdim=True)
        vec2 = vec2 / vec2.norm(dim=-1, keepdim=True)
        # 计算余弦相似度
        logits = torch.bmm(vec1.permute(1,0,2), vec2.permute(1,2,0)).permute(1,0,2)
        return logits
