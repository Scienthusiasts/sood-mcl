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
    def __init__(self, cat_nums, dim=256, decay=0.9996, beta=1000, updates=11110, mode='contrast', loss_weight=1.0):
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
        self.to_device = False
        # 类别数
        self.cat_nums = cat_nums + 1
        # self.updates用于记录当前已经ema更新了多少次模型
        # TODO:self.updates还有一个问题就是当resume训练时这部分会直接从0开始
        self.updates = updates 
        # 定义ema衰减函数, 有利于前期训练(初始d=0,之后随着self.updates的增大d慢慢趋近decay常量)
        self.decay = lambda x: decay * (1 - math.exp(-x / beta))  
        # 用于存储每个batch提取的所有class-wise正样本vectors (初始化全为None)
        self.mem_bank = {i:None for i in range(self.cat_nums)}
        # class-wise prototypes (固定初始化)
        self.prototypes = nn.Parameter(torch.ones(size=(self.cat_nums, dim)) * 0.01, requires_grad={'contrast':True, 'ema':False}[mode])
        # delta_prototype用于临时存储当前batch下的平均特征 (固定初始化)
        self.delta_prototype = torch.ones(size=(self.cat_nums, dim)) * 0.01
        # 随机初始化
        # self.prototypes = nn.Parameter(torch.normal(0, 0.01, size=(self.cat_nums, dim)), requires_grad={'contrast':True, 'ema':False}[mode])
        # self.delta_prototype = torch.normal(0, 0.01, size=(self.cat_nums, dim))
        # mode='contrast'用到
        self.contrast_loss = InfoNCELoss(t=0.07, reduction='none')
        self.loss_weight = loss_weight



    def forward(self, fpn_feat, cat_gt, cat_score_pred, cnt_score_pred, branch):
        '''前向过程+损失计算
            # Args:
                - fpn_feat:       head之前的特征图特征(正样本)
                - cat_gt:         每个特征对应的类别标签(正样本)
                - cat_score_pred: 模型head输出的分类置信度 
                - cnt_score_pred: 模型head输出的centerness置信度
                - branch:         'sup' or 'unsup' 
            # Returns:
                None
        '''
        # 1.更新memory bank
        if branch == 'sup':
            self.update_mem_bank_sup(fpn_feat.clone().detach(), cat_gt)
        if branch == 'unsup':
            # pass
            self.update_mem_bank_unsup(fpn_feat.clone().detach(), cat_score_pred.clone().detach())
        # 2.更新prototype
        if self.mode == 'ema':
            prototype_loss = self.ema_update(device=fpn_feat[0].device)   
        elif self.mode in ['contrast', 'self_contrast']:
            prototype_loss = self.contrast_learning_update(device=fpn_feat[0].device)
        # 3.与fpn特征图交互, 微调置信度 [bs*total_anchor_num, cat_num]
        if branch == 'unsup':
            refine_joint_score, refine_cls_score, _ = self.refine_score(fpn_feat.clone().detach(), cat_score_pred, cnt_score_pred)

        # 更新 update iter:
        if branch == 'sup': self.updates += 1

        # 回传损失
        if self.mode in ['ema', 'contrast']:
            if branch == 'sup': return prototype_loss
            elif branch == 'unsup': return prototype_loss, refine_joint_score, refine_cls_score
        elif self.mode == 'self_contrast':
            # prototypes之间进行额外的对比学习
            prototype_contrast_loss = self.prototypes_contrast()
            if branch == 'sup': return prototype_loss, prototype_contrast_loss
            elif branch == 'unsup': return prototype_loss, prototype_contrast_loss, refine_joint_score, refine_cls_score



    def update_mem_bank_sup(self, feat, cat_gt):
        '''有监督分支更新memory bank(只取GT的正样本区域)
            # Args:
                - feat:   head之前的特征图特征(正样本)    [bs * total_anchor_num, 256]
                - cat_gt: 每个特征对应的类别标签(正样本)  [bs * total_anchor_num, ]
            # Returns:
                None
        '''
        '''逐类别更新mem_bank'''
        for i in range(self.cat_nums):
            # 找到对应类别下的正样本索引
            cat_idx = torch.where(cat_gt==i)[0]
            # 当前batch有这个类别的正样本才更新
            if len(cat_idx) > 0:
                # 这里是截断梯度的, 意味着单纯更新prototype这个操作不会影响到网络其他部分的更新
                pos_feats = feat[cat_idx]
                # pos_feats里的每个特征归一化后再取平均
                pos_feats = pos_feats / pos_feats.norm(dim=-1, keepdim=True)
                self.mem_bank[i] = pos_feats.mean(dim=0)
                # mem_bank归一化(取平均后不一定归一化, 再归一化一次)
                self.mem_bank[i] = self.mem_bank[i] / self.mem_bank[i].norm()

                 

    def update_mem_bank_unsup(self, feat, cat_score_pred):
        '''无监督分支更新memory bank(只取GT的正样本区域)
            # Args:
                - feat:           head之前的特征图特征(所有样本)    [bs * total_anchor_num, 256]
                - cat_score_pred: 每个特征对应的类别标签(所有样本)  [bs * total_anchor_num, cat_num]
            # Returns:
                None
        '''
        cat_score_max_score, cat_score_max_label = cat_score_pred.max(dim=1)

        '''topk正负样本筛选(class-agnorstic)'''
        # 正样本数量基于top0.03
        topk_num = int(cat_score_max_score.size(0) * 0.03)
        # 总样本数(正样本+负样本)
        total_sample_nums = topk_num * 2
        # 从大到小排序(取正样本)
        pos_sorted_vals, pos_sorted_inds = torch.topk(cat_score_max_score, cat_score_max_score.size(0))      
        # 从小到大排序(取负样本)
        neg_sorted_vals, neg_sorted_inds = torch.topk(cat_score_max_score, cat_score_max_score.size(0), largest=False)  
        # 创建mask, 指定哪些样本为正样本, 哪些样本为负样本
        certain_mask = torch.zeros_like(cat_score_max_score).type(torch.bool)
        # 前topk个元素为正样本 / 后topk个元素为负样本
        certain_mask[pos_sorted_inds[:topk_num]] = True
        certain_mask[neg_sorted_inds[:topk_num]] = True
        # 设置背景类别
        cat_score_max_label[neg_sorted_inds[:topk_num]] = self.cat_nums - 1
        # 明确的正/负样本参与prototype的更新, 模糊的样本则抛弃
        cat_score_max_score = cat_score_max_score[certain_mask]
        cat_score_max_label = cat_score_max_label[certain_mask]

        '''逐类别更新mem_bank'''
        for i in range(self.cat_nums):
            # 找到对应类别下的正样本索引
            cat_idx = torch.where(cat_score_max_label==i)[0]
            # NOTE:ema_weight这一步貌似很重要, 怀疑是prototype加入unsup分支后效果不好的主要原因
            # 不像有监督部分的更新，无监督的更新是没有GT的，也就是说图像上没有某个类别，这个类别或多或少都会存在极少的样本(误检,极大可能为背景)参与prototype更新
            # 并且样本的数量越少，对prototype的影响也就越大(因为计算余弦相似度是不考虑模长的, 样本越少, 就越不能代表这个类别的一般性特征, 进而加深prototype的bias)
            # 因此还需要有一个方法(权重),保证那些极少样本的类别对prototype的更新的贡献较少.
            ema_weight = len(cat_idx) / total_sample_nums
            # 当前batch有这个类别的正样本(或样本的置信度足够大)才更新
            if ema_weight > 0 and cat_score_max_score[cat_idx].max() > 0.1:
                # 这里是截断梯度的, 意味着单纯更新prototype这个操作不会影响到网络其他部分的更新
                pos_feats = feat[cat_idx]
                # pos_feats里的每个特征归一化后再取平均
                pos_feats = pos_feats / pos_feats.norm(dim=-1, keepdim=True)
                self.mem_bank[i] = pos_feats.mean(dim=0)
                # mem_bank归一化(取平均后不一定归一化, 再归一化一次)
                self.mem_bank[i] = self.mem_bank[i] / self.mem_bank[i].norm() * ema_weight






    def ema_update(self, device):
        '''法1: ema更新prototype
            # Args:
                - device: 特征在哪个gpu
            # Returns:
                None
        '''
        # 第一次则将prototypes转移到对应gpu上
        if self.to_device == False:
            self.prototypes = self.prototypes.to(device)
            self.delta_prototype = self.delta_prototype.to(device)
            self.to_device = True
        d = self.decay(self.updates)
        # 逐类别ema更新prototypes
        for i in range(self.cat_nums):
            # 当前batch有这个类别的正样本才更新
            if self.mem_bank[i] != None:
                self.delta_prototype[i] = self.mem_bank[i]
                '''EMA更新核心代码:'''
                self.prototypes[i] = d * self.prototypes[i] + (1. - d) * self.delta_prototype[i]
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
        if self.to_device == False:
            self.prototypes = self.prototypes.to(device)
            self.delta_prototype = self.delta_prototype.to(device)
            self.to_device = True
        # not_None_idx用于记录哪些类别在当前batch有正样本
        not_None_idx = []
        # 更新delta_prototype
        for i in range(self.cat_nums):
            # 当前batch有这个类别的正样本才更新delta_prototype
            if self.mem_bank[i] != None:
                # [[256], ..., [256]] -> mean -> [256]
                self.delta_prototype[i] = self.mem_bank[i]
                not_None_idx.append(i)
                # 清空
                self.mem_bank[i] = None

        # 计算对比学习损失(not_None_idx: 只对当前batch包含的正样本的类别计算)
        # NOTE:这里是不是可以对self.delta_prototype加一个均值为0, 标准差为0.01的高斯噪声
        loss = self.contrast_loss(self.prototypes, self.delta_prototype)[not_None_idx].mean()
        return loss


    def prototypes_contrast(self, ):
        '''prototypes之间进行对比学习
        '''
        loss = self.contrast_loss(self.prototypes, self.prototypes).mean()
        return loss




    def refine_score(self, batch_fpn_feat, batch_cls_score, batch_cnt_score):
        '''使用prototype来微调网络输出的分类置信度
            Args:
                - batch_fpn_feat:  shape = [bs*total_anchor_num, 256]
                - batch_cls_score: shape = [bs*total_anchor_num, cat_num]
                - batch_cnt_score: shape = [bs*total_anchor_num, 1]
            Return:
                - refine_joint_score: shape = [bs*total_anchor_num, cat_num]
                - refine_cls_score:   shape = [bs*total_anchor_num, cat_num]
        '''
        prototypes = self.prototypes.detach()
        '''批量计算余弦相似度 [bs*total_anchor_num, cat_num+1]'''
        catwise_sim_score = torch.einsum('ij, kj -> ik', 
        batch_fpn_feat / batch_fpn_feat.norm(dim=-1, keepdim=True), 
        prototypes / prototypes.norm(dim=-1, keepdim=True))
        # (-1, 1) -> (0, 1)
        catwise_sim_score = (catwise_sim_score + 1) * 0.5
        catwise_max_score, catwise_max_idx = catwise_sim_score.max(dim=1)
        # 得到背景类mask
        bgd_mask = catwise_max_idx == (self.cat_nums - 1)

        '''关键步骤:使用prototype来微调网络输出的score'''
        refine_cls_score = batch_cls_score.clone()
        # 逐类别进行微调
        for i in range(self.cat_nums - 1):
            cat_mask = catwise_max_idx == i
            # 当前类别是置信度最大类别, 则增强score激活值
            refine_cls_score[:, i][cat_mask] += catwise_sim_score[:, i][cat_mask] * batch_cls_score[:, i][cat_mask]
            # 当前类别不是置信度最大类别, 则抑制score激活值
            refine_cls_score[:, i][~cat_mask] *= catwise_sim_score[:, i][~cat_mask]
        # 背景类logits 则调整为全0
        refine_cls_score[bgd_mask] = 0. 
        # 得到refine的置信度 [bs*total_anchor_num, cat_num]
        refine_joint_score = torch.clamp(refine_cls_score * batch_cnt_score, 0, 1)
        refine_cls_score = torch.clamp(refine_cls_score, 0, 1)
        # 联合置信度基于refine的分类置信度 * 原始的centerness
        # 分类置信度基于refine的分类置信度 
        # TODO:中心置信度怎么refine???
        return refine_joint_score, refine_cls_score, catwise_sim_score



    def vis_heatmap(self, batch_fpn_feat, batch_imgs, batch_img_names, batch_pos_mask, batch_centerness, batch_cls_score):
        '''可视化prototype与特征图的相似度heatmap (only for experimental validation)
            Args:
                - batch_fpn_feat:   shape = [bs, total_anchor_num, 256]
                - batch_imgs:       shape = [bs, 3, 1024, 1024]
                - batch_img_names:  [name_1, ..., name_bs]
                - batch_pos_mask:
                - batch_centerness: 
                - batch_cls_score:
            Return:
        '''
        if not os.path.exists('./vis_prototype'):os.makedirs('./vis_prototype')
        sizes = [128, 64 ,32, 16, 8]

        batch_cls_score_max, batch_cls_score_idx = batch_cls_score.max(dim=1)
        batch_joint_score = batch_cls_score_max * batch_centerness

        '''prototype和特征图交互, 得到相似度heatmap'''
        refine_batch_joint_score, _, batch_catwise_map = self.refine_score(batch_fpn_feat, batch_cls_score, batch_centerness.unsqueeze(-1))

        # 把batch维度拆开
        batch_pos_mask = batch_pos_mask.reshape(batch_imgs.shape[0], -1)
        batch_joint_score = batch_joint_score.reshape(batch_imgs.shape[0], -1)
        batch_cls_idx = batch_cls_score_idx.reshape(batch_imgs.shape[0], -1)
        batch_catwise_map = batch_catwise_map.reshape(batch_imgs.shape[0], -1, self.cat_nums)
        refine_batch_joint_score = refine_batch_joint_score.reshape(batch_imgs.shape[0], -1, self.cat_nums - 1)
        '''遍历batch里每一张图像'''
        for img, img_name, pos_mask, joint_score, cls_idx, catwise_map, refine_joint_score in zip(batch_imgs, batch_img_names, batch_pos_mask, batch_joint_score, batch_cls_idx, batch_catwise_map, refine_batch_joint_score):

            # 取最大置信度类别的置信度 
            refine_joint_score = refine_joint_score.max(dim=1)[0]
            active_score_map, active_score_idx = catwise_map.max(dim=1)
            idx = 0
            bgd_mask = active_score_idx==self.cat_nums-1
            cls_mask = (cls_idx!=idx) | (active_score_idx!=idx)
            active_score_map[bgd_mask] = 0.

            '''不同尺度拆分开'''
            active_score_map = torch.split(active_score_map, [size * size for size in sizes], dim=0)
            # [total_anchor_num] -> [[h1*w1], [h5*w5]]
            pos_mask = torch.split(pos_mask, [size * size for size in sizes], dim=0)
            joint_score = torch.split(joint_score, [size * size for size in sizes], dim=0)
            refine_joint_score = torch.split(refine_joint_score, [size * size for size in sizes], dim=0)

            # 对原图预处理
            std = np.array([58.395, 57.12 , 57.375]) / 255.
            mean = np.array([123.675, 116.28 , 103.53]) / 255.
            img = img.permute(1,2,0).cpu().numpy()
            img = np.clip(img * std + mean, 0, 1)

            '''可视化'''
            plt.figure(figsize = (10, 10))
            for lvl, lvl_active_score_map in enumerate(active_score_map):
                lvl_active_score_map = lvl_active_score_map.reshape(sizes[lvl], sizes[lvl])
                lvl_pos_mask = pos_mask[lvl].reshape(sizes[lvl], sizes[lvl])
                lvl_joint_score = joint_score[lvl].reshape(sizes[lvl], sizes[lvl])
                lvl_refine_joint_score = refine_joint_score[lvl].reshape(sizes[lvl], sizes[lvl])
                # 可视化特征图
                plt.subplot(5,5,lvl+1)
                plt.imshow(lvl_active_score_map.detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
                plt.axis('off')
                # 可视化联合置信度
                plt.subplot(5,5,lvl+6)
                plt.imshow(lvl_joint_score.detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
                plt.axis('off')
                # 可视化refine联合置信度
                plt.subplot(5,5,lvl+11)
                plt.imshow(lvl_refine_joint_score.detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
                plt.axis('off')
                # 可视化GT
                plt.subplot(5,5,lvl+16)
                plt.imshow(lvl_pos_mask.cpu().numpy(), cmap='jet', vmin=0, vmax=1)
                plt.axis('off')
            # 可视化原图
            plt.subplot(5,5,21)
            plt.imshow(img, vmin=0, vmax=1)
            plt.axis('off')
            plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
            plt.savefig(f"./vis_prototype/{img_name}", dpi=200)  


            






    def vis_heatmap_unsup(self, batch_fpn_feat, batch_imgs, batch_img_names, batch_centerness, batch_cls_score):
        '''可视化prototype与特征图的相似度heatmap (only for experimental validation)
            Args:
                - batch_fpn_feat:   shape = [bs, total_anchor_num, 256]
                - batch_imgs:       shape = [bs, 3, 1024, 1024]
                - batch_img_names:  [name_1, ..., name_bs]
                - batch_pos_mask:
                - batch_centerness: 
                - batch_cls_score:
            Return:
        '''
        if not os.path.exists('./vis_prototype'):os.makedirs('./vis_prototype')
        sizes = [128, 64 ,32, 16, 8]

        batch_cls_score_max, batch_cls_score_idx = batch_cls_score.max(dim=1)
        batch_joint_score = batch_cls_score_max * batch_centerness.reshape(-1)

        '''prototype和特征图交互, 得到相似度heatmap'''
        refine_batch_joint_score, _, batch_catwise_map = self.refine_score(batch_fpn_feat, batch_cls_score, batch_centerness)

        # 把batch维度拆开
        batch_joint_score = batch_joint_score.reshape(batch_imgs.shape[0], -1)
        batch_cls_idx = batch_cls_score_idx.reshape(batch_imgs.shape[0], -1)
        batch_catwise_map = batch_catwise_map.reshape(batch_imgs.shape[0], -1, self.cat_nums)
        refine_batch_joint_score = refine_batch_joint_score.reshape(batch_imgs.shape[0], -1, self.cat_nums - 1)
        '''遍历batch里每一张图像'''
        for img, img_name, joint_score, cls_idx, catwise_map, refine_joint_score in zip(batch_imgs, batch_img_names, batch_joint_score, batch_cls_idx, batch_catwise_map, refine_batch_joint_score):

            # 取最大置信度类别的置信度 
            refine_joint_score = refine_joint_score.max(dim=1)[0]
            active_score_map, active_score_idx = catwise_map.max(dim=1)
            idx = 6
            bgd_mask = active_score_idx==self.cat_nums-1
            cls_mask = (cls_idx!=idx) | (active_score_idx!=idx)
            active_score_map[bgd_mask] = 0.

            '''不同尺度拆分开'''
            active_score_map = torch.split(active_score_map, [size * size for size in sizes], dim=0)
            # [total_anchor_num] -> [[h1*w1], [h5*w5]]
            joint_score = torch.split(joint_score, [size * size for size in sizes], dim=0)
            refine_joint_score = torch.split(refine_joint_score, [size * size for size in sizes], dim=0)

            # 对原图预处理
            std = np.array([58.395, 57.12 , 57.375]) / 255.
            mean = np.array([123.675, 116.28 , 103.53]) / 255.
            img = img.permute(1,2,0).cpu().numpy()
            img = np.clip(img * std + mean, 0, 1)

            '''可视化'''
            plt.figure(figsize = (10, 8))
            for lvl, lvl_active_score_map in enumerate(active_score_map):
                lvl_active_score_map = lvl_active_score_map.reshape(sizes[lvl], sizes[lvl])
                lvl_joint_score = joint_score[lvl].reshape(sizes[lvl], sizes[lvl])
                lvl_refine_joint_score = refine_joint_score[lvl].reshape(sizes[lvl], sizes[lvl])
                # 可视化prototype特征图
                plt.subplot(4,5,lvl+1)
                plt.imshow(lvl_active_score_map.detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
                plt.axis('off')
                # 可视化联合置信度
                plt.subplot(4,5,lvl+6)
                plt.imshow(lvl_joint_score.detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
                plt.axis('off')
                # 可视化refine联合置信度
                plt.subplot(4,5,lvl+11)
                plt.imshow(lvl_refine_joint_score.detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
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
        self.loss = nn.CrossEntropyLoss(reduction=reduction)
        # 温度超参数
        self.t = t
    

    def forward(self, vec1, vec2):
        # 先计算两个向量相似度
        sim_logits = self.COSSim(vec1, vec2) / self.t
        # 生成对比标签(对角线为1其余为0)
        labels = torch.arange(vec1.shape[0]).to(vec1.device)
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