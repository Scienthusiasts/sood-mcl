import math
import torch
import torch.nn as nn





class FCOSPrototype(nn.Module):
    '''Prototype原型
    '''
    def __init__(self, cat_nums, dim=256, decay=0.9996, beta=1000, updates=0, mode='contrast'):
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
        # 用于存储每个batch提取的所有class-wise正样本vectors(初始化全为None)
        self.mem_bank = {i:None for i in range(cat_nums)}
        # class-wise prototypes (随机初始化, 目前不计算梯度)
        self.prototypes = nn.Parameter(torch.normal(0, 0.01, size=(cat_nums, dim)), requires_grad={'contrast':True, 'ema':False}[mode])
        # delta_prototype用于临时存储当前batch下mem_bank的平均特征(随机初始化)
        self.delta_prototype = torch.normal(0, 0.01, size=(cat_nums, dim))
        # mode='contrast'用到
        self.contrast_loss = InfoNCELoss(t=0.07, reduction='none')


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
        '''更新memory bank
            # Args:
                - feat:   head之前的特征图特征
                - cat_gt: 每个特征对应的类别标签
            # Returns:
                None
        '''
        # 逐类别更新mem_bank
        for i in range(self.cat_nums):
            if i == self.cat_nums: i = -1
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
        loss = self.contrast_loss(self.prototypes, self.delta_prototype)[not_None_idx].mean()
        return loss


    def prototypes_contrast(self, ):
        '''prototypes之间进行对比学习
        '''
        loss = self.contrast_loss(self.prototypes, self.prototypes).mean()
        return loss








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


    # 计算图像和文本的余弦相似度
    @staticmethod
    def COSSim(vec1, vec2):
        '''计算图像和文本的余弦相似度
        '''
        # 特征向量归一化
        vec1 = vec1 / vec1.norm(dim=-1, keepdim=True)
        vec2 = vec2 / vec2.norm(dim=-1, keepdim=True)
        # 计算余弦相似度
        logits = vec1 @ vec2.t()
        return logits