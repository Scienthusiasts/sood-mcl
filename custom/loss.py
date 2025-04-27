import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from mmcv.ops import box_iou_quadri, box_iou_rotated
from mmdet.core import multi_apply, unmap
from custom.visualize import *



class QFLv2(nn.Module):
    '''kind of soft二分类交叉熵损失
    '''
    def __init__(self, ):
        super(QFLv2, self).__init__()

    def forward(self, pred_logits, gt_logits, weight=None, use_weight=True, beta=2.0, reduction='mean'):
        """QFLV2分类损失(sigmoid多类别二分类损失, 和softmax的多类别损失有区别)
            和QFL的区别在于对于正负样本采用不同的策略(负样本的gtlogits就强制为0)
            Args:
                pred_logits: [n, cls_num] (已sigmoid)
                gt_logits:   [n, cls_num] (已sigmoid)
                weight:      正样本mask(也可以不需要, 所有样本按照正样本方式计算, QFLv2退化为QFL)
                use_weight:  是否需要正样本mask
                beta:        超参数
                reduction:   损失组织形式
            
            Return:
                loss
        """

        if use_weight:
            pt = pred_logits
            # all goes to 0
            zerolabel = pt.new_zeros(pt.shape)
            # 一开始假设所有样本都是负样本, 因此实际上有对负样本计算损失, 对应的标签是全0
            loss = F.binary_cross_entropy(pred_logits, zerolabel, reduction='none') * pt.pow(beta)
            # positive goes to bbox quality

            # 这句话有时候会报错, 不知道为啥(内容如下):
            # ... ...
            # ../aten/src/ATen/native/cuda/Loss.cu:92: operator(): block: [79,0,0], thread: [118,0,0] Assertion `input_val >= zero && input_val <= one` failed.
            # RuntimeError: numel: integer multiplication overflow 
            try:
                pt = gt_logits[weight] - pred_logits[weight]
            except:
                print(weight.shape, gt_logits.shape)
                print(torch.isnan(weight).any(), torch.isnan(gt_logits).any(), torch.isnan(pred_logits).any())
                pt = gt_logits[weight] - pred_logits[weight]
                
            # 在所有样本都是负样本的基础上更新那些正样本对应位置为正样本损失(当gt_logits足够低时,也相当于计算负样本损失)
            loss[weight] = F.binary_cross_entropy(pred_logits[weight], gt_logits[weight], reduction='none') * pt.pow(beta)
            # loss = F.binary_cross_entropy(pred_sigmoid, gt_logits, reduction='none') * pt.pow(beta)
        else:
            pt = gt_logits - pred_logits
            loss = F.binary_cross_entropy(pred_logits, gt_logits, reduction='none') * pt.pow(beta)

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
    






class HungarianWithIoUMatching():
    """匈牙利匹配(一对一匹配)
    """
    def __init__(self, nc, iou_weight=5.0, cls_weight=2.0, l1_weight=5.0):
        self.eps = 1e-7
        # 类别数
        self.nc = nc
        # 各项代价的权重
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.l1_weight = l1_weight
        # 分类代价
        self.cls_cost = QFLv2()


    def max_iou_assign_single(self, gt_boxes, pred_boxes, pred_idx, gt_idx, iou_thr=0.2):
        """用于解决和GT差距很小的框由于类别不正确因此未匹配上(训练时的类别应该是和GT的类别而不是负样本)
           因此对那些负样本重新分配GT, 这次只考虑IoU
           (为什么不考虑一开始就直接用MaxIoUAssign: 多个GT框聚集时,MAXIoUAssigner可能将多个pred分配给同一个GT框,导致其他GT框匹配不足)
            Args:
                gt_boxes:    [n, 5] (cx, cy, w, h, θ) (已解码)
                pred_boxes:  [m, 7] (已解码)
                pred_idx:    正样本pred索引, 和gt_idx是一对
                gt_idx:      正样本gt索引, 和pred_idx是一对
            
            Return:
                pred_idx:    更新后的正样本pred索引, 和gt_idx是一对
                gt_idx:      更新后的正样本gt索引, 和pred_idx是一对
        """
        neg_box_mask = torch.ones(pred_boxes.shape[0], dtype=torch.bool).to(pred_boxes.device)
        neg_box_mask[pred_idx] = False

        if(neg_box_mask.sum()>=0):
            max_iou, max_iou_idx = box_iou_rotated(pred_boxes[:, :5], gt_boxes).max(1)
            # 找出那些和某个GT的IoU大于阈值的所有预测样本
            neg2pos_mask = max_iou > iou_thr
            # 同时考虑1.原来是负样本 and 2.和某个GT的IoU大于阈值 -> 把那些和某个GT的IoU大于阈值的负样本改为正样本
            neg2pos_mask *= neg_box_mask
            # 找出那些样本的索引，添加到正样本上去
            if(neg2pos_mask.sum()>0):
                neg2pos_pred_idx = torch.where(neg2pos_mask)[0]
                neg2pos_gt_idx = max_iou_idx[neg2pos_mask]
                pred_idx = np.concatenate([pred_idx, neg2pos_pred_idx.cpu().numpy()])
                gt_idx = np.concatenate([gt_idx, neg2pos_gt_idx.cpu().numpy()])
        return pred_idx, gt_idx



    def assign_single(self, gt_boxes, pred_boxes, gt_labels, pred_logits, img_meta, maxiou_reassign=True):
        """匈牙利匹配(一张图像)
            Args:
                gt_boxes:    [n, 5] (cx, cy, w, h, θ) (已解码)
                pred_boxes:  [m, 7] (已解码)
                gt_labels:   [n] 整数类别标签
                pred_logits: [m, num_classes] 预测类别logits
                img_meta:    图像信息(只可视化调试时会用到)
            
            Return:
                match_pred_gt_bboxes: 匹配的GT和预测索引(未匹配的部分用全0替代) [2, m, 5] 
                match_gt_logits:      未匹配的预测框索引(未匹配的部分用全0替代) [m, 5] 
                gt_idx:               索引对, 和pred_idx是一对
                pred_idx:             索引对, 和gt_idx是一对
        """
        device = gt_boxes.device
        n, m, = gt_boxes.shape[0], pred_boxes.shape[0]

        '''将gt_label也转化为logits, 这样方便计算QFLv2类别损失:'''
        # 创建基础One-Hot矩阵(全为eps) [n, cls_num]
        gt_logits = torch.full((n, self.nc), self.eps, dtype=torch.float32, device=device)
        # 将正确类别位置设为1-eps
        gt_logits.scatter_(1, gt_labels.unsqueeze(1), 1. - self.eps)
        '''因为gtbox数量和predbox数量一般是不同的, 且为了计算代价, 需要两两进行配对(同时也能对齐):'''
        # [m, cls_num] -> [n, m, cls_num] -> [n*m, cls_num]
        expanded_pred_logits = pred_logits.unsqueeze(0).expand(n, -1, -1).reshape(-1, self.nc)
        # [n, cls_num] -> [n, m, cls_num] -> [n*m, cls_num]
        expanded_gt_logits = gt_logits.unsqueeze(1).expand(-1, m, -1).reshape(-1, self.nc)
        '''计算代价'''
        with torch.no_grad():
            # 分类代价
            cls_cost = self.cls_cost(expanded_pred_logits, expanded_gt_logits, use_weight=False, reduction="none").mean(1).reshape(n, m)
            # RIoU代价
            iou_cost = 1.0 - box_iou_rotated(gt_boxes, pred_boxes[:, :5])  
            # L1位置代价(可以考虑不重叠的情况)
            img_h, img_w, _ = img_meta['img_shape']
            factor = gt_boxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            # [n, m, 4]
            gt_xywh = (gt_boxes[:, :4] / factor).unsqueeze(1).expand(-1, m, -1)
            # [n, m, 4]
            pred_xywh = (pred_boxes[:, :4] / factor).unsqueeze(0).expand(n, -1, -1)
            l1_cost = F.l1_loss(gt_xywh, pred_xywh, reduction='none').mean(dim=-1) 
            # 综合代价
            total_cost = self.cls_weight * cls_cost + self.iou_weight * iou_cost + self.l1_weight * l1_cost
            # 可视化代价矩阵(默认注释)
            # vis_HM_cost_matrix(cls_cost, './vis_HM_cls_cost', img_meta['ori_filename'])
            # vis_HM_cost_matrix(iou_cost, './vis_HM_iou_cost', img_meta['ori_filename'])
            # vis_HM_cost_matrix(l1_cost, './vis_HM_box_cost', img_meta['ori_filename'])
            # vis_HM_cost_matrix(total_cost, './vis_HM_total_cost', img_meta['ori_filename'], range_limit=False)

        '''匈牙利匹配'''
        # 匈牙利匹配:
        gt_idx, pred_idx = linear_sum_assignment(total_cost.cpu().numpy())

        '''负样本maxIoU重匹配'''
        if maxiou_reassign:
            # 对那些负样本重新分配GT, 这次只考虑IoU:
            pred_idx, gt_idx = self.max_iou_assign_single(gt_boxes, pred_boxes, pred_idx, gt_idx)

        # 根据配对索引生成配对的结果 [2, m, 5=(cx, cy, w, h, θ)] (未匹配上的结果则用全0代替)
        match_pred_gt_bboxes = torch.zeros((2, m, 5), device=gt_boxes.device)
        match_pred_gt_bboxes[0, ...] = pred_boxes[:, :5]
        match_pred_gt_bboxes[1, pred_idx, :] = gt_boxes[gt_idx, :5]
        # 根据配对索引生成配对的结果 [m, 5=(cx, cy, w, h, θ)]
        match_gt_logits = torch.zeros((m, self.nc), device=gt_boxes.device)
        match_gt_logits[pred_idx] = gt_logits[gt_idx]

        return match_pred_gt_bboxes, match_gt_logits, gt_idx, pred_idx



    def assign(self, batch_gt_boxes, batch_pred_boxes, batch_gt_labels, batch_pred_logits, batch_img_meta):
        """匈牙利匹配(整个batch)
            Args:
                batch_gt_boxes:    list([group_nums, 5=(cx, cy, w, h, θ)], ..., [...]) 真实的gt
                batch_pred_boxes:  list([group_nums, 7=(cx, cy, w, h, θ, score, label)], ..., [...]) 预测的bbox
                batch_gt_labels:   list([group_nums], ..., [...]) 真实的gt的标签
                batch_pred_logits: list([group_nums, cls_num], ..., [...]) 预测的logits(all classes)
                batch_img_meta:    batch图像信息
            
            Return:
                match_pred_gt_bboxes: [2, total_box_num, 5] 所有pred样本的分配情况(bbox)
                match_gt_logits:      [total_box_num, cls_num] 所有pred样本的分配情况(cls score)
                batch_idx:            [total_box_num] batch索引

        """
        batch_match_pred_gt_bboxes, batch_match_gt_logits, _, _ = multi_apply(self.assign_single, batch_gt_boxes, batch_pred_boxes, batch_gt_labels, batch_pred_logits, batch_img_meta['img_metas'])
        # 可视化匹配box(一般情况下注释)
        # vis_HM_boxes(batch_match_pred_gt_bboxes, batch_match_gt_logits, batch_img_meta, './vis_HM_result')
        # 可视化匹配score(一般情况下注释)
        # vis_HM_scores(batch_pred_logits, batch_match_gt_logits, batch_img_meta, './vis_HM_score')

        # 按batch进行拼接: list([2, box_num_1, 5], ...., [2, box_num_bs, 5]) -> [2, total_box_num, 5]
        match_pred_gt_bboxes = torch.cat(batch_match_pred_gt_bboxes, dim=1)
        # 按batch进行拼接: list([box_num_1, cls_num], ...., [box_num_bs, cls_num]) -> [total_box_num, cls_num]
        match_gt_logits = torch.cat(batch_match_gt_logits, dim=0)
        # 生成batch索引
        batch_idx = []
        for i in range(len(batch_match_gt_logits)):
            idx = torch.ones(batch_match_gt_logits[i].shape[0]).to(match_gt_logits.device)
            batch_idx.append(idx * i)
        batch_idx = torch.cat(batch_idx) 

        return batch_idx, match_pred_gt_bboxes, match_gt_logits






