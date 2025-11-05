import torch
import numpy as np
from .rotated_semi_detector import RotatedSemiDetector
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models import build_detector
from custom.visualize import *
from custom.fn_mining import FNMining
from custom.ss_branch_sparse import SSBranchSparse


@ROTATED_DETECTORS.register_module()
class RotatedSemiUnbaisedTeacherGI(RotatedSemiDetector):
    def __init__(self, fn_mining_score_thr, gi_head_pos_thres, model: dict, use_refine_head, use_ss_branch, ss_branch, train_cfg=None, test_cfg=None, symmetry_aware=False):
        super(RotatedSemiUnbaisedTeacherGI, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            semi_loss=None,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            # ugly manner to get start iteration, to fit resume mode
            self.iter_count = train_cfg.get("iter_count", 0)
            # Prepare semi-training config
            # step to start training student (not include EMA update)
            self.burn_in_steps = train_cfg.get("burn_in_steps", 5000)
            # prepare super & un-super weight
            self.sup_weight = train_cfg.get("sup_weight", 1.0)
            self.unsup_weight = train_cfg.get("unsup_weight", 1.0)
            self.weight_suppress = train_cfg.get("weight_suppress", "linear")
            self.rcnn_configs = train_cfg.get("rcnn_configs")
            self.loss_configs = train_cfg.get("loss_configs")
        self.symmetry_aware = symmetry_aware
        # 是否开启gi-head
        self.use_refine_head = use_refine_head
        self.fn_mining_score_thr = fn_mining_score_thr
        self.gi_head_pos_thres = gi_head_pos_thres
        # 是否开启ss_bransh
        self.use_ss_branch=use_ss_branch
        if self.use_ss_branch:
            self.SSBranch = SSBranchSparse(**ss_branch)








    def forward_train(self, imgs, img_metas, **kwargs):
        super(RotatedSemiUnbaisedTeacherGI, self).forward_train(imgs, img_metas, **kwargs)
        losses = dict()

        '''数据读取'''
        sup_format_data, format_data = self.wrap_datas(imgs, img_metas, **kwargs)
        aug_orders = ['unsup_strong', 'unsup_weak']
        # vis_sparse_data(format_data, [[],[]], [[],[]], save_dir='./vis_sgt')
        # vis_sup_data(sup_format_data, save_dir='./vis_sup_img')



        '''全监督分支'''
        # student部分前向+计算损失
        # NOTE:这里会和稀疏GT(挖掘出的正样本)也计算损失, 返回sparse_losses
        sup_losses, _ = self.student.forward_train(**sup_format_data['sup_weak'])

        # 组织全监督损失
        for key, val in sup_losses.items():
            if key[:4] == 'loss':
                if isinstance(val, list):
                    losses[f"{key}_sup"] = [self.sup_weight * x for x in val]
                else:
                    losses[f"{key}_sup"] = self.sup_weight * val
            else:
                losses[key] = val
                

        '''burn-in结束, 开启正样本挖掘'''
        if self.iter_count > self.burn_in_steps:
            # burn_in之后的慢启动, 慢慢增加无监督分支损失的权重(只有使用t-s自蒸馏分支才用到)
            unsup_weight = self.ema_weight()

            '''得到teacher预测结果+正样本挖掘'''
            with torch.no_grad():
                if self.use_refine_head:
                    '''用gihead挖掘出正样本'''
                    # rcnn_configs=None, 则会返回preNMS的dense结果(2000个框)
                    t_fpn_feat, t_bboxes, t_nc_scores = self.teacher.simple_test_bboxes(rcnn_configs=None, **format_data[aug_orders[1]])
                    t_bboxes = torch.cat([t_bbox.unsqueeze(0) for t_bbox in t_bboxes], dim=0)
                    t_nc_scores = torch.cat([t_nc_score.unsqueeze(0) for t_nc_score in t_nc_scores], dim=0)
                    t_scores, t_labels = torch.max(t_nc_scores[..., :-1], dim=-1, keepdim=True)
                    t_rbb_preds = torch.cat([t_bboxes, t_scores, t_labels], dim=-1)

                    '''预测结果经过teacher gi_head的微调'''
                    t_cnt_score = torch.ones_like(t_scores.squeeze(-1))
                    batch_preds = self.teacher.gi_head.infer(
                        t_fpn_feat, 
                        t_rbb_preds, t_nc_scores[..., :-1], t_cnt_score,
                        format_data[aug_orders[0]],
                        )
                    batch_t_nms_bboxes, batch_t_nms_scores, batch_t_nms_labels = [], [], []
                    for preds in batch_preds:
                        # preds = [nms_boxes_num, 7=(cx, cy, w, h, θ, score, label)]
                        batch_t_nms_bboxes.append(preds[:, :6])
                        batch_t_nms_scores.append(preds[:, 5])
                        batch_t_nms_labels.append(preds[:, 6].to(torch.long))
                else:
                    '''用原始的teacher预测的bboxes挖掘出正样本'''
                    batch_t_nms_bboxes, batch_t_nms_labels, t_fpn_feat = self.teacher.forward_train(get_boxes=True, rcnn_configs=self.rcnn_configs, **format_data[aug_orders[1]])
                    batch_t_nms_scores = [t_bbox[:, -1] for t_bbox in batch_t_nms_bboxes]

                bs = len(batch_t_nms_bboxes)
                # 可视化sgt+minging_gt(默认注释)
                # vis_sparse_data(format_data, batch_t_nms_bboxes, batch_t_nms_labels, save_dir='./vis_minging_gt', vis_text=False, thickness=1)

            '''teacher正样本挖掘(sparse-level)'''
            # 挖掘出的正样本会放入format_data中当做gt
            #  list([box_num, 6], ..., [...]), list([box_num], ..., [...]) list([box_num], ..., [...])
            format_data = FNMining.fp_mining(bs, batch_t_nms_bboxes, batch_t_nms_scores, batch_t_nms_labels, format_data, aug_orders, score_thr=self.fn_mining_score_thr, mode='unsup')
            # 可视化sgt+minging_gt(默认注释)
            # vis_sparse_data(format_data, [[],[]], [[],[]], save_dir='./vis_minging_gt')



            '''稀疏监督分支(before burn-in) / student稀疏监督训练(sparse-level) (after burn-in)'''
            # student部分前向+计算损失
            # NOTE:这里会和稀疏GT(挖掘出的正样本)也计算损失, 返回sparse_losses
            sparse_losses, s_fpn_feat = self.student.forward_train(**format_data[aug_orders[0]], neg_weight_thr=0.5)



            # 组织无监督损失
            for key, val in sparse_losses.items():
                if key[:4] == 'loss':
                    if isinstance(val, list):
                        losses[f"{key}_unsup"] = [self.sup_weight * x for x in val]
                    else:
                        losses[f"{key}_unsup"] = self.sup_weight * val
                else:
                    losses[key] = val



    

        '''训练gi-head分支'''
        if self.use_refine_head:
            # 已经截断梯度:
            # 每张图保留1000个框
            s_fpn_feat, s_bboxes, s_nc_scores = self.student.simple_test_bboxes(rcnn_configs=None, **format_data[aug_orders[0]])
            s_bboxes = torch.cat([s_bbox.unsqueeze(0) for s_bbox in s_bboxes], dim=0)
            s_nc_scores = torch.cat([s_nc_score.unsqueeze(0) for s_nc_score in s_nc_scores], dim=0)
            s_scores, s_labels = torch.max(s_nc_scores[..., :-1], dim=-1, keepdim=True)
            s_rbb_preds = torch.cat([s_bboxes, s_scores, s_labels], dim=-1)

            # NOTE:Ablation: 断开refine-head与主体检测器的梯度
            # s_fpn_feat = [fpn_feat.detach() for fpn_feat in s_fpn_feat]

            # 2.送入gi head进行微调(pos_thres=1.0说明gt只有稀疏标注的那部分, 不含挖掘的部分)
            s_cnt_score = torch.ones_like(s_scores.squeeze(-1))
            s_gi_losses = self.student.gi_head.loss(
                s_fpn_feat, 
                s_rbb_preds, s_nc_scores[..., :-1], s_cnt_score,
                format_data[aug_orders[0]]['gt_bboxes'], format_data[aug_orders[0]]['gt_labels'], 
                format_data[aug_orders[0]], 
                train_mode='train_sup',
                pos_thres=self.gi_head_pos_thres
            )
            # 3.组织gihead微调模块的损失
            for key, val in s_gi_losses.items():
                if key[-4:] == 'loss':
                    losses[key] = val
                else:
                    losses[key] = val


        # TODO: 0728旋转一致性约束
        if self.use_ss_branch and self.iter_count > self.burn_in_steps:
            '''生成旋转视角图像'''
            # format_data参数共享内存, 不返回也会同步修改
            format_data, rand_angle, isflip, ori_mask = self.SSBranch.gen_aug_data(format_data)
            '''得到不同视角推理结果(目前采用post nms的结果)'''
            s_fpn_feat, s_bboxes, s_nc_scores = self.student.simple_test_bboxes_with_grad(rcnn_configs=None, **format_data[aug_orders[0]])
            bs = format_data[aug_orders[0]]['img'].shape[0] // 2

            # 先用阈值卡一波，不然匈牙利匹配速度很慢
            score_masks = []
            for s_nc_score in s_nc_scores:
                s_score, _ = torch.max(s_nc_score[:, :-1], dim=-1)
                score_mask = s_score > 0.01
                if score_mask.sum()==0:
                    topk_values, topk_indices = torch.topk(s_score, 10)
                    score_mask = torch.zeros_like(s_score, dtype=torch.bool)
                    score_mask[topk_indices] = True
                score_masks.append(score_mask)



            # 去除掉最后一维score
            ori_sbboxes = [s_bboxes[i][score_masks[i]] for i in [0, 1]]
            rot_sbboxes = [s_bboxes[i][score_masks[i]] for i in [2, 3]]
            ori_nc_scores = [s_nc_scores[i][score_masks[i]] for i in [0, 1]]
            rot_nc_scores = [s_nc_scores[i][score_masks[i]] for i in [2, 3]]
            # 将rot_sbboxes逆旋转回ori_sbboxes坐标系
            rot_sbboxes = [self.SSBranch.inv_rotate_bboxes(1024, rot_sbbox, rand_angle) for rot_sbbox in rot_sbboxes]
            # 二分图匹配将ori与rot框进行一一配对
            batch_ori_idx, batch_rot_idx = self.SSBranch.assigner.assign(ori_sbboxes, rot_sbboxes, format_data[aug_orders[0]])
            # 可视化
            # self.SSBranch.vis_batch_rot_img(
            #     format_data[aug_orders[0]]['img'][:bs], format_data[aug_orders[0]]['img'][:bs], 
            #     ori_sbboxes, rot_sbboxes, ori_nc_scores, rot_nc_scores,
            #     batch_ori_idx, batch_rot_idx,
            #     format_data[aug_orders[0]]['img_metas'], ori_mask, rand_angle, isflip, './vis_ori_rot_img'
            # )

            reg_loss = self.SSBranch.loss(ori_sbboxes, rot_sbboxes, ori_nc_scores, rot_nc_scores, batch_ori_idx, batch_rot_idx)
            losses['ss_reg_loss'] = reg_loss * 0.1

    
        self.iter_count += 1
        return losses







    def wrap_datas(self, imgs, img_metas, **kwargs):
        """数据读取并组织数据格式
        """
        gt_bboxes = kwargs.get('gt_bboxes')
        gt_labels = kwargs.get('gt_labels')
        # preprocess
        sup_format_data, unsup_format_data = dict(), dict()
        for idx, img_meta in enumerate(img_metas):
            tag = img_meta['tag']
            if tag!='sup_weak':
                if tag not in unsup_format_data.keys():
                    unsup_format_data[tag] = dict()
                    unsup_format_data[tag]['img'] = [imgs[idx]]
                    # 'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 
                    # 'flip', 'flip_direction', 'img_norm_cfg', 'tag', 'batch_input_shape'
                    unsup_format_data[tag]['img_metas'] = [img_metas[idx]]
                    unsup_format_data[tag]['gt_bboxes'] = [gt_bboxes[idx]]
                    unsup_format_data[tag]['gt_labels'] = [gt_labels[idx]]
                else:
                    unsup_format_data[tag]['img'].append(imgs[idx])
                    unsup_format_data[tag]['img_metas'].append(img_metas[idx])
                    unsup_format_data[tag]['gt_bboxes'].append(gt_bboxes[idx])
                    unsup_format_data[tag]['gt_labels'].append(gt_labels[idx])
            if tag=='sup_weak':
                if tag not in sup_format_data.keys():
                    sup_format_data[tag] = dict()
                    sup_format_data[tag]['img'] = [imgs[idx]]
                    sup_format_data[tag]['img_metas'] = [img_metas[idx]]
                    sup_format_data[tag]['gt_bboxes'] = [gt_bboxes[idx]]
                    sup_format_data[tag]['gt_labels'] = [gt_labels[idx]]
                else:
                    sup_format_data[tag]['img'].append(imgs[idx])
                    sup_format_data[tag]['img_metas'].append(img_metas[idx])
                    sup_format_data[tag]['gt_bboxes'].append(gt_bboxes[idx])
                    sup_format_data[tag]['gt_labels'].append(gt_labels[idx])

        for key in sup_format_data.keys():
            sup_format_data[key]['img'] = torch.stack(sup_format_data[key]['img'], dim=0)
        for key in unsup_format_data.keys():
            unsup_format_data[key]['img'] = torch.stack(unsup_format_data[key]['img'], dim=0)

        return sup_format_data, unsup_format_data
    


    def ema_weight(self, ):
        """burn_in之后的慢启动, 慢慢增加无监督分支损失的权重
        """
        unsup_weight = self.unsup_weight
        if self.weight_suppress == 'exp':
            target = self.burn_in_steps + 2000
            if self.iter_count <= target:
                scale = np.exp((self.iter_count - target) / 1000)
                unsup_weight *= scale
        elif self.weight_suppress == 'step':
            target = self.burn_in_steps * 2
            if self.iter_count <= target:
                unsup_weight *= 0.25
        elif self.weight_suppress == 'linear':
            target = self.burn_in_steps * 2
            if self.iter_count <= target:
                unsup_weight *= (self.iter_count - self.burn_in_steps) / self.burn_in_steps
        return unsup_weight
    





    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    












# ========================================================================相关可视化函数===================================================================================
def vis_sparse_data(format_data, batch_t_bboxes, batch_t_labels, save_dir='./vis_strong_weak_img', vis_text=True, thickness=2):
    if not os.path.exists(save_dir):os.makedirs(save_dir)
    # 强增强的图像 [bs, 3, 1024, 1024]
    batch_strong_img = format_data['unsup_strong']['img']
    batch_strong_img_meta = format_data['unsup_strong']['img_metas']
    batch_strong_gt_bboxes = format_data['unsup_strong']['gt_bboxes']
    batch_strong_gt_labels = format_data['unsup_strong']['gt_labels']
    # 弱增强的图像 [bs, 3, 1024, 1024]
    batch_weak_img = format_data['unsup_weak']['img']
    batch_weak_img_meta = format_data['unsup_weak']['img_metas']
    batch_weak_gt_bboxes = format_data['unsup_weak']['gt_bboxes']
    batch_weak_gt_labels = format_data['unsup_weak']['gt_labels']
    # 遍历 batch 中的每一对图像
    for i, (strong_img, strong_img_meta, strong_gt_bboxes, strong_gt_labels, 
             weak_img, weak_img_meta, weak_gt_bboxes, weak_gt_labels,
             t_bboxes, t_labels) in enumerate(zip(
        batch_strong_img, batch_strong_img_meta, batch_strong_gt_bboxes, batch_strong_gt_labels,
        batch_weak_img, batch_weak_img_meta, batch_weak_gt_bboxes, batch_weak_gt_labels,
        batch_t_bboxes, batch_t_labels)):
        '''图像处理'''
        # 原图预处理
        std = np.array([58.395, 57.12, 57.375]) / 255.
        mean = np.array([123.675, 116.28, 103.53]) / 255.
        # 处理 strong_img
        strong_img = strong_img.permute(1, 2, 0).cpu().numpy()
        strong_img = np.clip(strong_img * std + mean, 0, 1)
        strong_img = (strong_img * 255).astype(np.uint8) 
        strong_img = np.ascontiguousarray(strong_img) # 确保图像数据是连续内存布局
        # 处理 weak_img
        weak_img = weak_img.permute(1, 2, 0).cpu().numpy()
        weak_img = np.clip(weak_img * std + mean, 0, 1)
        weak_img = (weak_img * 255).astype(np.uint8)
        weak_img = np.ascontiguousarray(weak_img) # 确保图像数据是连续内存布局

        '''box绘制'''
        if strong_gt_labels.shape[0]>0:
            # 5参转8参
            poly_strong_gts = obb2poly(strong_gt_bboxes).cpu().numpy().astype(np.int32)
            poly_weak_gts = obb2poly(weak_gt_bboxes).cpu().numpy().astype(np.int32)
            # 可视化strong gts + weak gts
            strong_img = OpenCVDrawBox(strong_img, poly_strong_gts, (0,255,0), 2)
            weak_img = OpenCVDrawBox(weak_img, poly_weak_gts, (0,255,0), 2)
        
        if len(t_bboxes)>0:
            # 5参转8参
            poly_t_gts = obb2poly(t_bboxes[:, :5]).cpu().numpy().astype(np.int32)
            # 可视化teacher pred
            strong_img = OpenCVDrawBox(strong_img, poly_t_gts, (255,0,0), thickness)
            if vis_text:
                # 在每个预测框中心绘制类别和置信度得分(只绘制fn)
                # 获取旋转框的中心点坐标
                centers = t_bboxes[:, :2].cpu().numpy()  # [N, 2], 格式为(x, y)
                # 遍历每个框和得分
                for center, score, label in zip(centers, t_bboxes[:, -1].cpu().numpy(), t_labels.cpu().numpy()):
                    x, y = int(center[0]), int(center[1])
                    # 绘制得分（保留2位小数）
                    text = f"{label} | {score:.2f}"
                    # 设置文本样式
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    text_color = (255, 255, 255)  # 白色文字
                    bg_color = (0, 0, 0)          # 黑色背景
                    # 获取文本大小
                    text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
                    # 绘制背景矩形（提高可读性）
                    cv2.rectangle(strong_img, 
                                (x - text_size[0]//2 - 2, y - text_size[1]//2 - 2),
                                (x + text_size[0]//2 + 2, y + text_size[1]//2 + 2),
                                bg_color, -1)  # -1表示填充
                    # 绘制文本
                    cv2.putText(strong_img, text, (x - text_size[0]//2, y + text_size[1]//2),
                            font, font_scale, text_color, 1, cv2.LINE_AA)

        '''绘制+保存'''
        # 获取图像名
        img_name = strong_img_meta['ori_filename']
        # 创建一行两列的画布
        plt.figure(figsize=(12, 6))
        # 绘制 strong_img
        plt.subplot(1, 2, 1)
        plt.imshow(strong_img)
        plt.title(f'Strong Augmented Image')
        plt.axis('off')
        # 绘制 weak_img
        plt.subplot(1, 2, 2)
        plt.imshow(weak_img)
        plt.title(f'Weak Augmented Image')
        plt.axis('off')
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, img_name), bbox_inches='tight', dpi=150)
        plt.close()





def vis_sup_data(format_data, save_dir='./vis_sup_img'):
    if not os.path.exists(save_dir):os.makedirs(save_dir)
    # 有监督图像 [bs, 3, 1024, 1024]
    batch_img = format_data['sup_weak']['img']
    batch_img_meta = format_data['sup_weak']['img_metas']
    batch_gt_bboxes = format_data['sup_weak']['gt_bboxes']
    batch_gt_labels = format_data['sup_weak']['gt_labels']

    # 遍历 batch 中的每一对图像
    for img, img_meta, gt_bboxes, gt_labels in zip(batch_img, batch_img_meta, batch_gt_bboxes, batch_gt_labels):
        '''图像处理'''
        # 原图预处理
        std = np.array([58.395, 57.12, 57.375]) / 255.
        mean = np.array([123.675, 116.28, 103.53]) / 255.
        # 处理 img
        img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * std + mean, 0, 1)
        img = (img * 255).astype(np.uint8) 
        img = np.ascontiguousarray(img) # 确保图像数据是连续内存布局

        '''box绘制'''
        # 5参转8参
        poly_gts = obb2poly(gt_bboxes).cpu().numpy().astype(np.int32)
        # 可视化strong gts + weak gts
        img = OpenCVDrawBox(img, poly_gts, (0,255,0), 2)
        
        '''绘制+保存'''
        # 获取图像名
        img_name = img_meta['ori_filename']
        # 创建一行两列的画布
        plt.figure(figsize=(6, 6))
        # 绘制 strong_img
        plt.subplot(1, 1, 1)
        plt.imshow(img)
        plt.title(f'Image')
        plt.axis('off')
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, img_name), bbox_inches='tight', dpi=150)
        plt.close()

