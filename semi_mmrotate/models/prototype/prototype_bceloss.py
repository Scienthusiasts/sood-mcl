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
    def __init__(self, cat_nums, dim=256, decay=0.9996, beta=1000, updates=0, mode='contrast', loss_weight=1.0):
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
        # 类别数
        self.cat_nums = cat_nums
        # self.updates用于记录当前已经ema更新了多少次模型
        # TODO:self.updates还有一个问题就是当resume训练时这部分会直接从0开始
        self.updates = updates 
        # 定义ema衰减函数, 有利于前期训练(初始d=0,之后随着self.updates的增大d慢慢趋近decay常量)
        self.decay = lambda x: decay * (1 - math.exp(-x / beta))  
        # 用于存储每个batch提取的所有class-wise正样本vectors (初始化全为None)
        self.mem_bank = {i:None for i in range(cat_nums)}
        # class-wise prototypes (平均初始化)
        self.prototypes = nn.Parameter(torch.ones(size=(cat_nums, dim)) * 0.01, requires_grad={'contrast':True, 'ema':False}[mode])
        # self.prototypes = nn.Parameter(torch.normal(0, 0.01, size=(cat_nums, dim)), requires_grad={'contrast':True, 'ema':False}[mode])
        # delta_prototype用于临时存储当前batch下mem_bank的平均特征 (固定初始化)
        self.delta_prototype = torch.ones(size=(cat_nums, dim)) * 0.01
        # self.delta_prototype = torch.normal(0, 0.01, size=(cat_nums, dim))
        # mode='contrast'用到
        self.contrast_loss = InfoNCELoss(t=0.07, reduction='none')
        self.loss_weight = loss_weight


    def forward(self, cls_feats, cls_targets):
        '''前向过程+损失计算
        '''
        self.update_mem_bank(cls_feats, cls_targets)
        if self.mode == 'ema':
            prototype_loss = self.ema_update(device=cls_feats[0].device)   
        elif self.mode in ['contrast', 'self_contrast']:
            prototype_loss = self.contrast_learning_update(device=cls_feats[0].device)

        self.updates += 1

        if self.mode in ['ema', 'contrast']:
            return prototype_loss
        elif self.mode == 'self_contrast':
            # prototypes之间进行对比学习
            prototype_contrast_loss = self.prototypes_contrast()
            return prototype_loss, prototype_contrast_loss


    def update_mem_bank(self, feat, cat_gt):
        '''更新memory bank(只取GT的正样本区域)
            # Args:
                - feat:   head之前的特征图特征(正样本)
                - cat_gt: 每个特征对应的类别标签(正样本)
            # Returns:
                None
        '''
        # 逐类别更新mem_bank
        for i in range(self.cat_nums):
            # 找到对应类别下的正样本索引
            cat_idx = torch.where(cat_gt==i)[0]
            # 当前batch有这个类别的正样本才更新
            if len(cat_idx) > 0:
                 # 这里是截断梯度的, 意味着单纯更新prototype这个操作不会影响到网络其他部分的更新
                 self.mem_bank[i] = feat[cat_idx].detach()


    def ema_update(self, device):
        '''法1: ema更新prototype
            # Args:
                - device: 特征在哪个gpu
            # Returns:
                None
        '''
        # 第一次则将prototypes转移到对应gpu上
        if self.updates == 0:
            self.prototypes = self.prototypes.to(device)
            self.delta_prototype = self.delta_prototype.to(device)
        d = self.decay(self.updates)
        # 逐类别ema更新prototypes
        for i in range(self.cat_nums):
            # 当前batch有这个类别的正样本才更新
            if self.mem_bank[i] != None:
                # [[256], ..., [256]] -> mean -> [256]
                self.delta_prototype[i] = self.mem_bank[i].mean(dim=0)
                '''EMA更新核心代码:'''
                self.prototypes[i] *= d
                self.prototypes[i] += (1. - d) * self.delta_prototype[i]
                # 清空
                self.mem_bank[i] = None

        # 返回一个无意义的loss占位
        loss = torch.tensor(0).to(device)
        return loss


    def contrast_learning_update(self, device):
        '''法2: 对比学习更新prototype
            # Args:
                - device: 特征在哪个gpu
            # Returns:
                - loss: 对比损失
        '''
        # 第一次则将prototypes转移到对应gpu上
        if self.updates == 0:
            self.prototypes = self.prototypes.to(device)
            self.delta_prototype = self.delta_prototype.to(device)
        # not_None_idx用于记录哪些类别在当前batch有正样本
        not_None_idx = []
        # 更新delta_prototype
        for i in range(self.cat_nums):
            # 当前batch有这个类别的正样本才更新delta_prototype
            if self.mem_bank[i] != None:
                # [[256], ..., [256]] -> mean -> [256]
                self.delta_prototype[i] = self.mem_bank[i].mean(dim=0)
                not_None_idx.append(i)
                # 清空
                self.mem_bank[i] = None

        # 计算对比学习损失(not_None_idx: 只对当前batch包含的正样本的类别计算)
        # NOTE:这里是不是可以对self.delta_prototype加一个均值为0, 标准差为0.01的高斯噪声
        loss = self.contrast_loss(self.prototypes, self.delta_prototype)
        # 对损失加权, 保证正负样本的权重均衡
        loss_mask = torch.eye(loss.shape[0]).type(torch.bool)
        loss[loss_mask] *= (1 / (2 * len(not_None_idx)))
        loss[~loss_mask] *= (1 / (2 * len(not_None_idx) * (len(loss_mask) - 1)))
        # 将loss[not_None_idx, :] 改成 loss[:, not_None_idx] (保证prototype每个batch都充分学习)
        loss = loss[:, not_None_idx].sum()
        # loss = loss[not_None_idx, :].sum()
        return loss * self.loss_weight


    def prototypes_contrast(self, ):
        '''prototypes之间进行对比学习
        '''
        loss = self.contrast_loss(self.prototypes, self.prototypes).mean()
        return loss





    def vis_heatmap(self, batch_fpn_feat, batch_imgs, batch_img_names, batch_pos_mask, batch_centerness, batch_cls_score):
        '''可视化prototype与特征图的相似度heatmap (only for experimental validation)
            Args:
                - batch_fpn_feat:  shape = [bs, total_anchor_num, 256]
                - batch_imgs:      shape = [bs, 3, 1024, 1024]
                - batch_img_names: [name_1, ..., name_bs]
            Return:
        '''
        if not os.path.exists('./vis_prototype'):os.makedirs('./vis_prototype')
        sizes = [128, 64 ,32, 16, 8]

        batch_cls_score_max = batch_cls_score.max(dim=1)[0]
        batch_joint_score = batch_cls_score_max * batch_centerness
        # 把batch维度拆开
        batch_fpn_feat = batch_fpn_feat.reshape(batch_imgs.shape[0], -1, 256)
        batch_pos_mask = batch_pos_mask.reshape(batch_imgs.shape[0], -1)
        batch_joint_score = batch_joint_score.reshape(batch_imgs.shape[0], -1)

        '''遍历batch里每一张图像'''
        for fpn_feat, img, img_name, pos_mask, joint_score in zip(batch_fpn_feat, batch_imgs, batch_img_names, batch_pos_mask, batch_joint_score):
            # prototype和特征图交互, 得到相似度heatmap
            active_map = []
            for cat_wise_prototype in self.prototypes:
                cat_wise_active_map = torch.einsum('ij, j -> i', fpn_feat, cat_wise_prototype)
                cat_wise_active_map = cat_wise_active_map / (fpn_feat.norm(dim=1) * cat_wise_prototype.norm(dim=0))
                active_map.append(cat_wise_active_map.unsqueeze(1))
            # 将不同类别的拼在一起
            active_map = torch.cat(active_map, dim=1)
            # 取最大置信度类别的置信度[total_anchor_num, cat_num] -> [[h1*w1, cat_num], ..., [h5*w5, cat_num]]
            active_score_map, active_score_idx = active_map.max(dim=1)
            active_score_map[active_score_idx!=7] = -1

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
    def __init__(self, t=0.07, reduction='mean'):
        super(InfoNCELoss, self).__init__()
        # 温度超参数
        self.t = t
    

    def forward(self, vec1, vec2):
        # 先计算两个向量相似度
        # TODO: 相似度度量试试换成其他的? 比如KLD或JSD
        sim_logits = self.COSSim(vec1, vec2) / self.t
        # 生成对比标签(对角线为1其余为0)
        labels = torch.eye(vec1.shape[0]).to(vec1.device)
        # 相似度结果作为logits计算交叉熵损失(BCE Loss)
        loss = F.binary_cross_entropy(sim_logits.sigmoid(), labels, reduction="none")
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
        logits = vec1 @ vec2.t()
        return logits


    # 计算KLD
    @staticmethod
    def KLD(vec1, vec2):
        # 将s和t第一维度转化为频率(求和=1)
        vec1 = vec1 / vec1.sum(dim=-1, keepdim=True)
        vec2 = vec2 / vec2.sum(dim=-1, keepdim=True)
        # 计算 KL 散度, 添加一个小的 epsilon 防止 log(0)
        eps = 1e-10
        kl_div = vec1 * torch.log((vec1 + eps) / (vec2 + eps))
        return kl_div.sum()


    # 计算JSD
    @staticmethod
    def JSD(vec1, vec2):
        # 将s和t第一维度转化为频率(求和=1)
        vec1 = vec1 / vec1.sum(dim=-1, keepdim=True)
        vec2 = vec2 / vec2.sum(dim=-1, keepdim=True)
        
        # 计算平均分布M
        M = (vec1 + vec2) / 2
        
        # 计算两个方向的KL散度
        kl_div1 = KLD(vec1, M)
        kl_div2 = KLD(vec2, M)
        
        # 计算JS散度
        js_div = (kl_div1 + kl_div2) / 2
        return js_div.sum()