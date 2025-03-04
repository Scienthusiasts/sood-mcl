import torch
import torch.nn as nn
import torch.nn.functional as F







class QFLv2(nn.Module):
    '''kind of soft二分类交叉熵损失
    '''
    def __init__(self, ):
        super(QFLv2, self).__init__()

    def forward(self, pred_sigmoid, teacher_sigmoid, weight, beta=2.0, reduction='mean'):
        # all goes to 0
        pt = pred_sigmoid
        zerolabel = pt.new_zeros(pt.shape)
        # 一开始假设所有样本都是负样本, 因此实际上有对负样本计算损失, 对应的标签是全0
        loss = F.binary_cross_entropy(pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
        # positive goes to bbox quality

        # 这句话有时候会报错, 不知道为啥(内容如下):
        # ... ...
        # ../aten/src/ATen/native/cuda/Loss.cu:92: operator(): block: [79,0,0], thread: [118,0,0] Assertion `input_val >= zero && input_val <= one` failed.
        # RuntimeError: numel: integer multiplication overflow 
        try:
            pt = teacher_sigmoid[weight] - pred_sigmoid[weight]
        except:
            print(weight.shape, teacher_sigmoid.shape)
            print(torch.isnan(weight).any(), torch.isnan(teacher_sigmoid).any(), torch.isnan(pred_sigmoid).any())
            pt = teacher_sigmoid[weight] - pred_sigmoid[weight]
            
        # 在所有样本都是负样本的基础上更新那些正样本对应位置为正样本损失(当teacher_sigmoid足够低时,也相当于计算负样本损失)
        loss[weight] = F.binary_cross_entropy(pred_sigmoid[weight], teacher_sigmoid[weight], reduction='none') * pt.pow(beta)
        # loss = F.binary_cross_entropy(pred_sigmoid, teacher_sigmoid, reduction='none') * pt.pow(beta)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        return loss






class JSDivLoss(nn.Module):
    '''Jensen-Shannon散度 损失
    '''
    def __init__(self, ):
        super(JSDivLoss, self).__init__()
        self.KLDivLoss = nn.KLDivLoss(reduction='none')
        self.e = 1e-8

    def forward(self, s, t, to_distuibution = True, dist_dim=0, reduction='mean', loss_weight=1.):
        '''
            s:               输入分布 1, 形状为 [n,m] (其中一个维度表示分布的维度, 另一个维度表示有几个分布)
            t:               输入分布 2, 形状为 [n,m] (其中一个维度表示分布的维度, 另一个维度表示有几个分布)
            to_distribution: 是否将输入归一化为概率分布
            dist_dim:        表示哪一个维度表示分布的维度(默认0)
        '''
        # 将s和t第一维度转化为频率(和=1)
        if to_distuibution:
            # 在归一化后加 self.e，然后再重新归一化, 避免直接加 self.e 可能会导致分布的变化
            # 将 s 和 t 归一化为概率分布(.sum一般不可能是0, 所以没加e)
            s = s / s.sum(dim=dist_dim, keepdim=True)
            t = t / t.sum(dim=dist_dim, keepdim=True)
            # 加入小常数，避免零值
            s = s + self.e
            t = t + self.e
            # 重新归一化
            s = s / s.sum(dim=dist_dim, keepdim=True)
            t = t / t.sum(dim=dist_dim, keepdim=True)
        # 计算平均分布的对数
        log_mean = ((s + t) * 0.5 + self.e).log()

        # 调整输入形状，确保 nn.KLDivLoss 对分布维度计算 KLD(如果分布维度是 1, 则无需转置)
        if dist_dim == 0:
            # 如果分布维度是0, 把分布维度调整到1
            log_mean = log_mean.transpose(0, 1)
            s = s.transpose(0, 1)
            t = t.transpose(0, 1)

        # 注意: nn.KLDivLoss 默认认为分布的维度是最后一个维度
        jsd_loss = (self.KLDivLoss(log_mean, s) + self.KLDivLoss(log_mean, t)) * 0.5
        if reduction=='mean':
            return jsd_loss.mean() * loss_weight
        if reduction=='sum':
            return jsd_loss.sum() * loss_weight
        if reduction=='none':
            return jsd_loss.mean(dim=1)





    


class BCELoss(nn.Module):
    '''BCELoss
    '''
    def __init__(self, ):
        super(BCELoss, self).__init__()

    def forward(self, tensor1, tensor2, reduction='none'):
        return F.binary_cross_entropy(tensor1, tensor2, reduction=reduction)
    

# class SSBranchLoss(nn.Module):
#     '''自监督分支损失
#     '''
#     def __init__(self, ):
#         super(SSBranchLoss, self).__init__()

#     def forward(self, tensor1, tensor2, reduction='none'):

