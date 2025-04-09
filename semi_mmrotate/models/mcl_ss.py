import torch
import numpy as np
from .rotated_semi_detector import RotatedSemiDetector
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models import build_detector
# yan
from custom.utils import *
from custom.visualize import *
from torchvision.transforms import InterpolationMode
from custom.loss import QFLv2, BCELoss, JSDivLoss
from mmrotate.models import ROTATED_LOSSES, build_loss
import torch.distributed as dist

@ROTATED_DETECTORS.register_module()
class MCLTeacherSS(RotatedSemiDetector):
    def __init__(self, model: dict, semi_loss, nc, train_cfg=None, test_cfg=None, pretrained=None):
        super(MCLTeacherSS, self).__init__(
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
            self.logit_specific_weights = train_cfg.get("logit_specific_weights")
        self.nc = nc
        self.QFLv2 = QFLv2()
        self.BCE_loss = BCELoss()
        self.JSDLoss = JSDivLoss()

    def forward_train(self, imgs, img_metas, **kwargs):
        super(MCLTeacherSS, self).forward_train(imgs, img_metas, **kwargs)
        gt_bboxes = kwargs.get('gt_bboxes')
        gt_labels = kwargs.get('gt_labels')
        # preprocess
        format_data = dict()
        for idx, img_meta in enumerate(img_metas):
            tag = img_meta['tag']
            if tag in ['sup_strong', 'sup_weak']:
                tag = 'sup'
            if tag not in format_data.keys():
                format_data[tag] = dict()
                format_data[tag]['img'] = [imgs[idx]]
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
            # print(f"{key}: {format_data[key]['img'].shape}")
        losses = dict()
        # supervised forward
        sup_losses = self.student.forward_train(**format_data['sup'])
        for key, val in sup_losses.items():
            if key[:4] == 'loss':
                if isinstance(val, list):
                    losses[f"{key}_sup"] = [self.sup_weight * x for x in val]
                else:
                    losses[f"{key}_sup"] = self.sup_weight * val
            else:
                losses[key] = val
        if self.iter_count > self.burn_in_steps:
            # Train Logic
            # unsupervised forward
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



            # get student data
            # NOTE: yan, 这里可以调整教师和学生增强的顺序
            aug_orders = ['unsup_strong', 'unsup_weak']     # 1.正常
            # aug_orders = ['unsup_weak', 'unsup_strong']   # 2.调换
            # aug_orders = ['unsup_strong', 'unsup_strong'] # 3.一致(强)
            # aug_orders = ['unsup_weak', 'unsup_weak']     # 4.一致(弱)


            '''对无监督分支的图像进行旋转增强处理(旋转一致性自监督学习)'''
            # 获取图像及基本信息 注意是aug_orders[0]才是无监督student分支的原始图像不要搞错了
            rot_img = format_data[aug_orders[0]]['img'].clone()
            img_name = format_data[aug_orders[0]]['img_metas'][0]['filename'].split('/')[-1]
            # 旋转增强操作(核心部分):
            rot_img, rand_angle = batch_tensor_random_rotate(rot_img, [45, 135], rand=True)
            # 以概率p进行翻转增强操作:
            rot_img, isflip = batch_tensor_random_flip(rot_img, p=0.0)
            # 将原始图像和旋转后的图像沿batch维度拼接(注意原始图像在前面, 旋转图像在后面)
            ori_rot_img = torch.cat((format_data[aug_orders[0]]['img'], rot_img), dim=0)
            format_data[aug_orders[0]]['img'] = ori_rot_img
            # 可视化
            # vis_rgb_tensor(ori_rot_img[1, ...], f"{rand_angle}_flip-{isflip}_{img_name}", './vis_rot_img')
            # vis_rgb_tensor(ori_rot_img[0, ...], f"{rand_angle}_flip-{isflip}_{img_name}", './vis_ori_img')





            with torch.no_grad():
                # get teacher data
                teacher_logits = self.teacher.forward_train(get_data=True, **format_data[aug_orders[1]])

            student_logits = self.student.forward_train(get_data=True, **format_data[aug_orders[0]])






            '''格式调整(旋转一致性自监督学习)'''
            # 将原始图像的推理结果与旋转图像的推理结果分离开
            s_ori_logits,  s_rot_logits = [], []
            for pred in student_logits:
                ori_logits, rot_logits = [], [] 
                ori_logits = [x[0].unsqueeze(0) for x in pred]
                rot_logits = [x[1].unsqueeze(0) for x in pred]
                s_ori_logits.append(ori_logits)
                s_rot_logits.append(rot_logits)
            # 对输出的特征进行reshape
            # [total_grid_num, cat_num], [total_grid_num, 4+1], [total_grid_num, 1]
            reshape_s_ori_logits, bs = convert_shape(s_ori_logits, self.nc, fpn=False)
            reshape_s_rot_logits, bs = convert_shape(s_rot_logits, self.nc, fpn=False)
            reshape_t_logits, bs = convert_shape(teacher_logits, self.nc, fpn=False)

            '''旋转一致性自监督学习获取soft权重矩阵用于加权ss损失'''
            # 注意cls_scores和centernesses都是未经过sigmoid()的logits
            t_cls_scores, _, t_centernesses = reshape_t_logits
            # 首先获取联合置信度(ss分支会用到)
            # t_scores, t_pred提取最大的类别置信度和对应的类别索引 [bs * h * w, cat_num] -> [bs * h * w], [bs * h * w]
            t_scores, _ = torch.max(t_cls_scores.sigmoid(), 1)
            t_joint_scores = t_centernesses.sigmoid().reshape(-1) * t_scores
            weight_mask = 1 / (1 + torch.exp(-10 * t_joint_scores)).pow(10) - 1/1024. 

            '''去除旋转一致性自监督学习的格式调整'''
            # s_ori_logits = student_logits
            # reshape_s_ori_logits, bs = convert_shape(s_ori_logits, self.nc)
            # reshape_t_logits, bs = convert_shape(teacher_logits, self.nc)






            unsup_losses = self.semi_loss(teacher_logits, s_ori_logits, img_metas=format_data['unsup_weak'], alone_angle=True)

            for key, val in self.logit_specific_weights.items():
                if key in unsup_losses.keys():
                    unsup_losses[key] *= val
            for key, val in unsup_losses.items():
                if key[:4] == 'loss':
                    losses[f"{key}_unsup"] = unsup_weight * val
                else:
                    losses[key] = val





            '''旋转一致性自监督分支(旋转一致性自监督学习)'''
            # 注意cls_scores和centernesses都是未经过sigmoid()的logits. _bbox_preds=[total_anchor_num, 5=(cx, cy, w, h, θ)]
            o_cls_scores, o_bbox_preds, o_centernesses = reshape_s_ori_logits
            r_cls_scores, r_bbox_preds, r_centernesses = reshape_s_rot_logits

            # 对原始图像上的预测结果旋转到与旋转结果对齐
            # 0.首先把长条状的特征再还原为二维图像的形状
            H, W = format_data[aug_orders[0]]['img'][0].shape[1:]
            # 这里默认H=W
            sizes = [H//8, H//16, H//32, H//64, H//128]
            e = 1e-10
            # [total_anchor_num, ] -> [[h1*w1, ], [h5*w5, ]]
            o_cls_scores_list = torch.split(o_cls_scores, [size * size for size in sizes], dim=0)
            o_centernesses_list = torch.split(o_centernesses, [size * size for size in sizes], dim=0)
            o_weight_mask_list = torch.split(weight_mask, [size * size for size in sizes], dim=0)
            # [[h1*w1, ], [h5*w5, ]] -> [[h1, w1, ], [h5, w5, ]]
            o_cls_scores_list = [x.reshape(size, size, -1).sigmoid()+e for size, x in zip(sizes, o_cls_scores_list)]
            o_centernesses_list = [x.reshape(size, size, -1).sigmoid()+e for size, x in zip(sizes, o_centernesses_list)]
            o_weight_mask_list = [x.reshape(size, size, -1) for size, x in zip(sizes, o_weight_mask_list)]
            # 1.对原始图像的特征图逐尺度执行旋转操作(用最近邻差值，保证旋转边界清晰)
            # NOTE: 必须fill一个很小的数, 填充0.会报错:    
            # "/home/yht/.conda/envs/sood-mcl/lib/python3.9/site-packages/mmdet/models/losses/gfocal_loss.py", line 88, in quality_focal_loss_with_prob
            # pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
            # RuntimeError: numel: integer multiplication overflow
            o_cls_scores_list = [rotate(x.permute(2, 0, 1), rand_angle, expand=False, interpolation=InterpolationMode.NEAREST, fill=[e,]).permute(1, 2, 0) for x in o_cls_scores_list]
            o_centernesses_list = [rotate(x.permute(2, 0, 1), rand_angle, expand=False, interpolation=InterpolationMode.NEAREST, fill=[e,]).permute(1, 2, 0) for x in o_centernesses_list]
            o_weight_mask_list = [rotate(x.permute(2, 0, 1), rand_angle, expand=False, interpolation=InterpolationMode.NEAREST, fill=[e,]).permute(1, 2, 0) for x in o_weight_mask_list]
            # 2.如果执行了翻转操作, 则翻转回来
            if isflip:
                o_cls_scores_list = [torch.flip(x, dims=[0]) for x in o_cls_scores_list]
                o_centernesses_list = [torch.flip(x, dims=[0]) for x in o_centernesses_list]
                o_weight_mask_list = [torch.flip(x, dims=[0]) for x in o_weight_mask_list]
            # 3.对旋转后的特征再拉直为一维特征
            o_cls_scores = convert_shape_single(o_cls_scores_list, self.nc, bs_dim=False)
            o_centernesses = convert_shape_single(o_centernesses_list, 1, bs_dim=False)
            o_weight_mask = convert_shape_single(o_weight_mask_list, 1, bs_dim=False).reshape(-1)

            # 把bbox回归值进行旋转:
            # 对bbox_preds解码
            r_decode_bboxes = decode_rbbox(r_bbox_preds)
            o_decode_bboxes = decode_rbbox(o_bbox_preds)
            # 对o_bbox_preds旋转
            # 5参转8参
            o_poly_bboxes = obb2poly(o_decode_bboxes, version='le90').reshape(-1, 4, 2)
            # 定义旋转矩阵
            R = torch.tensor(cv2.getRotationMatrix2D((W / 2, H / 2) , rand_angle, 1), device=o_poly_bboxes.device, dtype=torch.float32)
            one_tensor = torch.ones((o_poly_bboxes.shape[0], 4, 1), device=o_decode_bboxes.device)
            o_poly_bboxes = torch.cat((o_poly_bboxes, one_tensor), dim=-1)
            o_decode_bboxes = torch.einsum("ij, klj -> kli", R, o_poly_bboxes).reshape(-1, 8)
            o_decode_bboxes = poly2obb(o_decode_bboxes, version='le90')
            # 如果执行了翻转操作, 则先对坐标翻转(cy, θ)
            if isflip:
                o_decode_bboxes[:, 1] = H - o_decode_bboxes[:, 1]
                o_decode_bboxes[:, 4] *= -1.
            # 刚刚虽然把bbox旋转到正确的位置上了,还需要把每个grid转到正确的位置上
            o_decode_bboxes_list = torch.split(o_decode_bboxes, [size * size for size in sizes], dim=0)
            # 先旋转
            o_decode_bboxes_list = [x.reshape(size, size, -1) for size, x in zip(sizes, o_decode_bboxes_list)]
            o_decode_bboxes_list = [rotate(x.permute(2, 0, 1), rand_angle, expand=False, interpolation=InterpolationMode.NEAREST, fill=[e,]).permute(1, 2, 0) for x in o_decode_bboxes_list]
            # 生成旋转mask, 用于屏蔽原始特征和旋转特征不一致的部分(旋转产生的padding部分)
            # 有时候h和w会存在=0的情况, 一并做个排除(但是这里只排除了ori预测分支的情况, 没排除rot分支的情况)
            o_h_mask = [(x[..., 2]>e) for x in o_decode_bboxes_list]
            o_w_mask = [(x[..., 3]>e) for x in o_decode_bboxes_list]
            rot_mask_list = [(w * h).unsqueeze(0) for w, h in zip(o_h_mask, o_w_mask)]
            # 再翻转(注意旋转mask无需翻转)
            if isflip:
                o_decode_bboxes_list = [torch.flip(x, dims=[0]) for x in o_decode_bboxes_list]
            # 拉直
            o_decode_bboxes = convert_shape_single(o_decode_bboxes_list, 5, bs_dim=False)
            rot_mask = convert_shape_single(rot_mask_list, 1, bs_dim=False).reshape(-1)
            # o_pos_mask = (o_weight_mask > 0.15) * rot_mask
            # r_max_clsscore = torch.max(r_cls_scores, dim=1)[0]
            # r_pos_mask = (r_max_clsscore > 0.15) * rot_mask
            o_weight_mask = o_weight_mask * (rot_mask + e)
            local_rot_mask_sum = rot_mask.sum()
            rot_mask_sum = local_rot_mask_sum.clone() 
            # 进行多卡之间通信, 此时rot_mask_sum数值为所有gpu上local_rot_mask_sum的值之和
            # 当local_rot_mask_sum * dist.get_world_size() == rot_mask_sum时没问题, 否则说明其中一个gpu上预测出了w或h=0的情况
            dist.all_reduce(rot_mask_sum, op=dist.ReduceOp.SUM)
            print(f"local_mask_sum:{local_rot_mask_sum * dist.get_world_size()}, total_mask_sum:{rot_mask_sum}, {local_rot_mask_sum * dist.get_world_size()==rot_mask_sum}")
            if local_rot_mask_sum * dist.get_world_size() == rot_mask_sum:
                # 可视化
                # 可视化bbox
                # vis_rboxes_on_img(format_data[aug_orders[0]]['img'][1], o_decode_bboxes[o_pos_mask].clone().detach(), f"{rand_angle}_flip-{isflip}_{img_name}", './vis_unsup_o_decode_bbox')
                # vis_rboxes_on_img(format_data[aug_orders[0]]['img'][1], r_decode_bboxes[o_pos_mask].clone().detach(), f"{rand_angle}_flip-{isflip}_{img_name}", './vis_unsup_r_decode_bbox')
                # 可视化特征图
                # vis_rotate_feat(format_data[aug_orders[0]]['img'][1], format_data[aug_orders[0]]['img'][1], f"{rand_angle}_flip-{isflip}_{img_name}", o_cls_scores, o_centernesses, r_cls_scores, r_centernesses, o_weight_mask, rot_mask)

                
                '''自监督类别一致损失'''
                # print(f"o_cls_min:{o_cls_scores[rot_mask].min()}, o_cls_max:{o_cls_scores[rot_mask].max()}")
                # print(f"r_cls_min:{r_cls_scores[rot_mask].min()}, r_cls_max:{r_cls_scores[rot_mask].max()}")
                # print(r_cls_scores.sigmoid()[rot_mask].min(), r_cls_scores.sigmoid()[rot_mask].max())
                # # ss_loss_cls = self.QFLv2(o_cls_scores[rot_mask], r_cls_scores.sigmoid()[rot_mask], weight=torch.ones_like(o_cls_scores[rot_mask], device=o_cls_scores.device, dtype=torch.bool), reduction="none").sum() / o_weight_mask[rot_mask].sum()
                # ss_loss_cls = self.JSDLoss(o_cls_scores[rot_mask], r_cls_scores.sigmoid()[rot_mask], to_distuibution=True, dist_dim=0, reduction='sum')

                '''自监督中心度一致损失'''
                # o_centernesses = torch.clamp(o_centernesses, e, 1. - e)
                # r_centernesses = torch.clamp(r_centernesses.sigmoid(), e, 1. - e)
                # print(f"o_cnt_min:{o_centernesses[rot_mask].min()}, o_cnt_max:{o_centernesses[rot_mask].max()}")
                # print(f"r_cnt_min:{r_centernesses[rot_mask].min()}, r_cnt_max:{r_centernesses[rot_mask].max()}")
                # # ss_loss_cnt_all = self.BCE_loss(o_centernesses[rot_mask], r_centernesses[rot_mask], reduction='none')
                # # ss_loss_cnt = (o_weight_mask[rot_mask] * ss_loss_cnt_all.reshape(-1)).sum() / o_weight_mask[rot_mask].sum()
                # ss_loss_cnt_all = self.JSDLoss(o_centernesses[rot_mask], r_centernesses[rot_mask], to_distuibution=True, dist_dim=0, reduction='none')
                # ss_loss_cnt = (o_weight_mask[rot_mask] * ss_loss_cnt_all.reshape(-1)).sum() / o_weight_mask[rot_mask].sum()
                
                '''自监督联合置信度一致损失(置信度和类别一起用类别损失优化)'''
                o_centernesses = torch.clamp(o_centernesses, e, 1. - e)
                r_centernesses = torch.clamp(r_centernesses.sigmoid(), e, 1. - e)
                o_joint_score = torch.einsum("ij, i -> ij", o_cls_scores, o_centernesses.reshape(-1))
                r_joint_score = torch.einsum("ij, i -> ij", r_cls_scores.sigmoid(), r_centernesses.reshape(-1))
                print(f"o_js_min:{o_joint_score[rot_mask].min()}, o_js_max:{o_joint_score[rot_mask].max()}")
                print(f"r_js_min:{r_joint_score[rot_mask].min()}, r_js_max:{r_joint_score[rot_mask].max()}")
                # ss_loss_joint_score = self.QFLv2(o_joint_score[rot_mask], r_joint_score[rot_mask], weight=torch.ones_like(o_joint_score[rot_mask], device=o_joint_score.device, dtype=torch.bool), reduction="none").sum() / o_weight_mask[rot_mask].sum()
                ss_loss_joint_score = self.JSDLoss(o_joint_score[rot_mask], r_joint_score[rot_mask], to_distuibution=True, dist_dim=0, reduction='sum')
                # ss_loss_joint_score = self.JSDLoss(o_joint_score[rot_mask], r_joint_score[rot_mask], to_distuibution=True, dist_dim=1, reduction='mean', loss_weight=1e3)
                # ss_loss_joint_score = self.JSDLoss(o_joint_score[rot_mask], r_joint_score[rot_mask], to_distuibution=True, dist_dim=1, reduction='sum', loss_weight=1.) / o_weight_mask[rot_mask].sum()
                

                '''自监督回归框(角度+尺度)一致损失'''
                # box在计算损失的时候就只取那些旋转前后一致的样本(rot_mask)
                riou_loss = build_loss(dict(type='RotatedIoULoss', reduction='none'))
                # 有时候h和w会存在=0的情况
                o_h_nonzero, o_w_nonzero = o_decode_bboxes[:, 2]!=0., o_decode_bboxes[:, 3]!=0.
                o_nonzero_mask = o_h_nonzero * o_w_nonzero
                r_h_nonzero, r_w_nonzero = r_decode_bboxes[:, 2]!=0., r_decode_bboxes[:, 3]!=0.
                r_nonzero_mask = r_h_nonzero * r_w_nonzero
                nonzero_mask = o_nonzero_mask * r_nonzero_mask
                final_mask = rot_mask * nonzero_mask
                print(f"mask-diff:{final_mask.sum() - rot_mask.sum()}, rand_angle:{rand_angle}")
                if final_mask.sum() - rot_mask.sum() < 0:
                    o_decode_bboxes[:, 2:4] += 1
                    r_decode_bboxes[:, 2:4] += 1
                ss_loss_box_all = riou_loss(o_decode_bboxes[rot_mask], r_decode_bboxes[rot_mask])
                ss_loss_box = (o_weight_mask[rot_mask] * ss_loss_box_all)
                # 去除掉那些负数的损失
                ss_loss_box = ss_loss_box[ss_loss_box>0]
                print(f"ss_box_loss_min:{ss_loss_box.min()}, ss_box_loss_max:{ss_loss_box.max()}")
                ss_loss_box = ss_loss_box.sum() / o_weight_mask[rot_mask].sum()
                print(f"ss_box_loss:{ss_loss_box}")
                print('='*100)
                # 可视化RIoU
                # vis_lvlriou(format_data[aug_orders[0]]['img'][1], o_decode_bboxes, r_decode_bboxes, img_name, 'vis_ro_branch_riou')


                # 组织自监督一致性损失
                ss_weight = 0.1
                # losses['ss_loss_cls'] = ss_loss_cls * ss_weight
                # losses['ss_loss_cnt'] = ss_loss_cnt * ss_weight
                losses['ss_loss_joint_score'] = ss_loss_joint_score * ss_weight
                losses['ss_loss_box'] = ss_loss_box * ss_weight








































        self.iter_count += 1

        return losses