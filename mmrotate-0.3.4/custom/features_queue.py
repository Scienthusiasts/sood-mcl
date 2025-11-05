import torch
from collections import deque
import torch.distributed as dist

# 请用python实现一个队列数据结构类：
# 队列里通过参数实现c个子队列代表有c个类别，每个类别有一个子队列
# 同时也需要再维护c个iter队列用于记录每个特征被push进入队列时的iter数(假设此时子队列中特征的维度为(n, dim), 则iter队列的维度为n)
# 对于push操作要实现的功能:
# push操作输入是一个tensor(维度为(bs, dim), bs每次可能不一样，但dim是固定的), 和tensor对应所属的类别tensor(维度为bs)
# 每次执行push操作时, 遍历所有类别, 将属于当前类别的tensor push到对应类别的子队列中, 同时记录push时对应的iter数
# 注意push进入的tensor包含梯度, 需要先进行detach然后clone
# 每次迭代时，将属于当前类别的特征tensor(维度为(bs, dim), bs每次可能不一样，但dim是固定的)push进入这个类别下的队列(需要和队列中已经有的的特征concat在一起, concat在队列末尾)，
# 对于pop操作要实现的功能:
# pop时会遍历所有类别，pop出那些iter数最小的特征, 更新特征队列和iter队列
# get_feats_by_class操作:
# 给定一个类别id，返回对应类别的队列中的所有元素, 维度为(n, dim), n为当前队列中的元素数量
# 需要注意的地方:
# 无需对队列的长度上限做限制


import torch
import torch.distributed as dist
from collections import deque

class DistributedClassFeatureQueue:
    
    def __init__(self, num_classes, dim, keep_iters=5, device='cuda'):
        """多类别队列
        
        参数:
            num_classes (int): 类别数量
            dim (int): 特征的维度
            keep_iters (int): 特征在队列中保留的迭代次数
            device (str): 存储设备
        """
        self.num_classes = num_classes
        self.dim = dim
        self.keep_iters = keep_iters
        self.current_iter = 0
        self.device = device
        
        # 每个rank都有自己的队列
        self.feats_queues = [deque() for _ in range(num_classes)]
        self.iter_queues = [deque() for _ in range(num_classes)]


    def push(self, features, class_ids):
        """将特征推入对应类别的队列中
        
        参数:
            features (torch.Tensor): 特征张量，形状为(bs, dim)
            class_ids (torch.Tensor): 类别ID张量，形状为(bs,)
        """
        features = features.to(self.device)
        class_ids = class_ids.to(self.device)
        
        # 只处理当前GPU的数据
        counts = torch.bincount(class_ids, minlength=self.num_classes)
        
        for class_id in range(self.num_classes):
            if counts[class_id] == 0:
                continue
                
            mask = (class_ids == class_id)
            self.feats_queues[class_id].append(features[mask].detach().clone())
            iters = torch.full((mask.sum().item(),), self.current_iter, 
                             dtype=torch.long, device=self.device)
            self.iter_queues[class_id].append(iters)
        
        self.current_iter += 1


    def pop(self):
        """移除所有队列中保存时间超过keep_iters的特征
        """
        threshold = self.current_iter - self.keep_iters
        
        if threshold <= 0:
            return
        
        for class_id in range(self.num_classes):
            if not self.iter_queues[class_id]: 
                continue
                
            remaining_feats = []
            remaining_iters = []
            for feat_block, iter_block in zip(self.feats_queues[class_id], self.iter_queues[class_id]):
                mask = (iter_block > threshold)
                
                if mask.any():
                    remaining_feats.append(feat_block[mask])
                    remaining_iters.append(iter_block[mask])
            
            self.feats_queues[class_id] = deque(remaining_feats)
            self.iter_queues[class_id] = deque(remaining_iters)


    def get_feats_by_class(self, class_id):
        """
        获取指定类别的所有特征和IOU值（从所有GPU收集）
        
        参数:
            class_id (int): 类别ID
            
        返回:
            - features (torch.Tensor): 该类别的所有特征，形状为(n, dim)
        """
        assert 0 <= class_id < self.num_classes, "类别ID超出范围"
        
        # 收集当前rank的特征数据
        if not self.feats_queues[class_id]:
            local_feats = torch.empty(0, self.dim, device=self.device)
        else:
            local_feats = torch.cat(list(self.feats_queues[class_id]))
        
        # 获取所有rank的数据量
        local_size = torch.tensor([local_feats.shape[0]], device=self.device)
        sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, local_size)
        
        # 准备接收缓冲区
        max_size = max(s.item() for s in sizes)
        if max_size == 0:
            return torch.empty(0, self.dim, device=self.device)
            
        # 填充本地数据到统一尺寸
        if local_feats.shape[0] < max_size:
            feat_padding = torch.zeros(max_size - local_feats.shape[0], self.dim,
                                    device=self.device, dtype=local_feats.dtype)
            local_feats = torch.cat([local_feats, feat_padding])
            
        
        # 收集所有特征数据
        gathered_feats = [torch.zeros_like(local_feats) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_feats, local_feats)
        
        
        # 拼接有效数据
        valid_feats = []
        for feats, size in zip(gathered_feats, sizes):
            if size > 0:
                valid_feats.append(feats[:size])
        
        if not valid_feats:
            return torch.empty(0, self.dim, device=self.device)
        
        return torch.cat(valid_feats)
    

    def __len__(self):
        """返回所有类别队列中元素的总数（汇总所有GPU的数据）
        以列表形式返回每个类别的总数，如[10, 5, 8,...]表示类别0有10个，类别1有5个，等等
        
        返回:
            list[int]: 每个类别的特征总数
        """
        # 计算本地每个类别的数量
        local_counts = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        for class_id in range(self.num_classes):
            local_counts[class_id] = sum(feat_block.shape[0] for feat_block in self.feats_queues[class_id])
        
        # 汇总所有GPU的数据 - 这里直接使用local_counts作为接收缓冲区
        dist.all_reduce(local_counts, op=dist.ReduceOp.SUM, async_op=False)
        
        # 转换为Python列表返回
        return local_counts


    def class_len(self, class_id):
        """返回指定类别队列中的元素数量（汇总所有GPU的数据）
        
        参数:
            class_id (int): 要查询的类别ID
            
        返回:
            int: 该类别在所有GPU上的特征总数
        """
        assert 0 <= class_id < self.num_classes, "类别ID超出范围"
        
        # 计算本地该类的数量
        local_count = torch.tensor(
            sum(feat_block.shape[0] for feat_block in self.feats_queues[class_id]),
            dtype=torch.long,
            device=self.device
        )
        # 汇总所有GPU的数据 - 直接使用local_count作为接收缓冲区
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM, async_op=False)
        
        return local_count.item()
