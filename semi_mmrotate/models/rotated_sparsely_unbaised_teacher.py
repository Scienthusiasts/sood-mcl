import torch
import numpy as np
from .rotated_semi_detector import RotatedSemiDetector
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models import build_detector


@ROTATED_DETECTORS.register_module()
class RotatedSparselyUnbaisedTeacher(RotatedSemiDetector):
    def __init__(self, model: dict, semi_loss=None, train_cfg=None, test_cfg=None, symmetry_aware=False):
        super(RotatedSparselyUnbaisedTeacher, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            semi_loss,
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









    def forward_train(self, imgs, img_metas, **kwargs):
        super(RotatedSparselyUnbaisedTeacher, self).forward_train(imgs, img_metas, **kwargs)
        losses = dict()

        '''数据读取'''
        format_data = self.wrap_datas(imgs, img_metas, **kwargs)
        aug_orders = ['unsup_strong', 'unsup_weak']
        # 可视化稀疏数据和标签(通常注释)
        # vis_sparse_data(format_data, save_dir='./vis_strong_weak_img')





        if self.iter_count > self.burn_in_steps:

            # burn_in之后的慢启动, 慢慢增加无监督分支损失的权重(只有使用t-s自蒸馏分支才用到)
            unsup_weight = self.ema_weight()

            with torch.no_grad():
                # get teacher data
                pseudo_bboxes, pseudo_labels = self.teacher.forward_train(
                    get_boxes=True, rcnn_configs=self.rcnn_configs, **format_data['unsup_weak'])
                format_data['unsup_strong']['gt_bboxes'] = [pseudo_bboxes_img[:, :5] for pseudo_bboxes_img in pseudo_bboxes]
                format_data['unsup_strong']['gt_labels'] = pseudo_labels

            unsup_losses = self.student.forward_train(**format_data['unsup_strong'], loss_configs=self.loss_configs)

            for key, val in unsup_losses.items():
                if key[:4] == 'loss':
                    if 'bbox' in key:
                        losses[f"{key}_unsup"] = 0 * val
                    else:
                        losses[f"{key}_unsup"] = unsup_weight * val







        '''稀疏监督分支(before burn-in) / student稀疏监督训练(sparse-level) (after burn-in)'''
        # student部分前向+计算损失
        # NOTE:这里会和稀疏GT(挖掘出的正样本)也计算损失, 返回sparse_losses
        sparse_losses = self.student.forward_train(**format_data[aug_orders[0]])

        # 组织稀疏监督损失
        for key, val in sparse_losses.items():
            if key[:4] == 'loss':
                if isinstance(val, list):
                    losses[f"{key}_sup"] = [self.sup_weight * x for x in val]
                else:
                    losses[f"{key}_sup"] = self.sup_weight * val
            else:
                losses[key] = val








        self.iter_count += 1
        return losses







    def wrap_datas(self, imgs, img_metas, **kwargs):
        """数据读取并组织数据格式
        """
        gt_bboxes = kwargs.get('gt_bboxes')
        gt_labels = kwargs.get('gt_labels')
        # preprocess
        format_data = dict()
        for idx, img_meta in enumerate(img_metas):
            tag = img_meta['tag']
            if tag not in format_data.keys():
                format_data[tag] = dict()
                format_data[tag]['img'] = [imgs[idx]]
                # 'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'tag', 'batch_input_shape'
                format_data[tag]['img_metas'] = [img_metas[idx]]
                format_data[tag]['gt_bboxes'] = [gt_bboxes[idx]]
                format_data[tag]['gt_labels'] = [gt_labels[idx]]
            else:
                format_data[tag]['img'].append(imgs[idx])
                format_data[tag]['img_metas'].append(img_metas[idx])
                format_data[tag]['gt_bboxes'].append(gt_bboxes[idx])
                format_data[tag]['gt_labels'].append(gt_labels[idx])
        for key in format_data.keys():
            format_data[key]['img'] = torch.stack(format_data[key]['img'], dim=0)
        
        return format_data
    


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
