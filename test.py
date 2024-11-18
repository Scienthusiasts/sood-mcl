# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, replace_ImageToTensor

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, setup_multi_processes
# yan:
from mmcv.image import tensor2imgs
from mmdet.core import encode_mask_results
from semi_mmrotate.models.rotated_dt_baseline import RotatedDTBaseline
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F




def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    # NOTE: added by yan
    parser.add_argument(
        '--get-feature-map',
        type=bool,
        default=False,
        help='whether to return cls_score and centerness feature map.')
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if 'samples_per_gpu' in cfg.data.test:
            warnings.warn('`samples_per_gpu` in `test` field of '
                          'data will be deprecated, you should'
                          ' move it to `test_dataloader` field')
            test_dataloader_default_args['samples_per_gpu'] = \
                cfg.data.test.pop('samples_per_gpu')
        if test_dataloader_default_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
            if 'samples_per_gpu' in ds_cfg:
                warnings.warn('`samples_per_gpu` in `test` field of '
                              'data will be deprecated, you should'
                              ' move it to `test_dataloader` field')
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        test_dataloader_default_args['samples_per_gpu'] = samples_per_gpu
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    # NOTE: added by yan
    cfg.model['model']['get_feature_map'] = args.get_feature_map
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        # NOTE: yan, add args.get_feature_map
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, args.get_feature_map,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals', 'type'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)









# NOTE: added by yan, 重写single_gpu_test
def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    get_feature_map=False, 
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # NOTE: added by yan
            if get_feature_map:
                result, dense_bboxes, dense_scores, dense_cnt, cls_scores, centernesses = model(return_loss=False, rescale=True, **data)
                # 处理并可视化热力图
                # vis_feature_map(cls_scores, centernesses, data['img'][0].data[0], data['img_metas'][0].data[0][0]['ori_filename'])
                # 处理并可视化dense boxes
                vis_dense_bboxes(dense_bboxes, dense_scores, dense_cnt, data['img'][0].data[0], data['img_metas'][0].data[0][0]['ori_filename'])
            else:
                result = model(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results




# NOTE: added by yan, 用于处理并可视化热力图
def vis_feature_map(cls_scores, centernesses, raw_img, img_name):
    cls_score_list, cnt_score_list, joint_score_list = [], [], []
    # 最大的那层特征的尺寸
    lvl0_h, lvl0_w = cls_scores[0].shape[2:]

    # 遍历每一层
    for cls_score, cnt_score in zip(cls_scores, centernesses):
        h, w = cls_score.shape[2:]
        cls_score, _ = cls_score.max(dim=1, keepdim=True)
        cls_score = cls_score.sigmoid()
        cnt_score = cnt_score.sigmoid()
        joint_score = cls_score * cnt_score
        # 将所有尺度特征图都插值下采样到和最大的那层特征图尺寸一致
        if h != lvl0_h:
            cls_score = F.interpolate(cls_score, size=(lvl0_h, lvl0_w), mode='bicubic')
            cnt_score = F.interpolate(cnt_score, size=(lvl0_h, lvl0_w), mode='bicubic')
            joint_score = F.interpolate(joint_score, size=(lvl0_h, lvl0_w), mode='bicubic')

        cls_score_list.append(cls_score.reshape(lvl0_h, lvl0_w).cpu().numpy())
        cnt_score_list.append(cnt_score.reshape(lvl0_h, lvl0_w).cpu().numpy())
        joint_score_list.append(joint_score.reshape(lvl0_h, lvl0_w).cpu().numpy())
    
    '''可视化'''
    plt.figure(figsize = (10, 6))
    i = 1
    rol, col = 3, 5
    # 可视化类别最大置信度热力图
    for map in cls_score_list:
        plt.subplot(rol, col, i)
        plt.imshow(map, cmap='jet', vmin=0, vmax=1)
        plt.axis('off')
        i += 1
    # 可视化联合置信度激活热力图
    for map in joint_score_list:
        plt.subplot(rol, col, i)
        plt.imshow(map, cmap='jet', vmin=0, vmax=1)
        plt.axis('off')
        i += 1
    # 可视化原图
    std = np.array([58.395, 57.12 , 57.375]) / 255.
    mean = np.array([123.675, 116.28 , 103.53]) / 255.
    raw_img = raw_img.squeeze(0).permute(1,2,0).cpu().numpy()
    raw_img = np.clip(raw_img * std + mean, 0, 1)
    plt.subplot(rol, col, i)
    plt.imshow(raw_img)
    plt.axis('off')

    if not os.path.isdir('./vis_same-size/fig'):os.makedirs('./vis_same-size/fig')
    if not os.path.isdir('./vis_same-size/npfile/cls_score'):os.makedirs('./vis_same-size/npfile/cls_score')
    if not os.path.isdir('./vis_same-size/npfile/joint_score'):os.makedirs('./vis_same-size/npfile/joint_score')
    # 保存可视化结果
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    plt.savefig(f'./vis_same-size/fig/{img_name}', dpi=200)  
    # 保存二进制np文件
    # for i in range(5):
    #     np.save(f'./vis_same-size/npfile/cls_score/{img_name}_lvl{i}.npy', cls_score_list[i])
    #     np.save(f'./vis_same-size/npfile/joint_score/{img_name}_lvl{i}.npy', joint_score_list[i])
    
    plt.clf()





def vis_dense_bboxes(dense_bboxes, dense_scores, dense_cnt, raw_img, img_name):
    # 对原图预处理:
    std = np.array([58.395, 57.12 , 57.375]) / 255.
    mean = np.array([123.675, 116.28 , 103.53]) / 255.
    raw_img = raw_img.squeeze(0).permute(1,2,0).cpu().numpy()
    raw_img = np.clip(raw_img * std + mean, 0, 1)

    dense_score_max, _ = dense_scores.max(dim=1)
    joint_score = dense_score_max * dense_cnt

    dense_bboxes = dense_bboxes.cpu().numpy()
    dense_scores = dense_scores.cpu().numpy()
    dense_cnt = dense_cnt.cpu().numpy()
    joint_score = joint_score.cpu().numpy()

    # 只可视化伪框的点坐标
    dense_y, dense_x = dense_bboxes[:, 0].astype(np.int32), dense_bboxes[:, 1].astype(np.int32)
    dense_x = np.clip(dense_x, 0, 1024)
    dense_y = np.clip(dense_y, 0, 1024)
    # 创建空白画布
    fig = np.zeros((128, 128))
    for x, y, score in zip(dense_x, dense_y, joint_score):
        if 0<x<1024 and 0<y<1024:
            fig[x//8, y//8] += score

    '''可视化'''
    plt.figure(figsize = (10, 5))
    # 可视化原图
    plt.subplot(1,2,1)
    plt.imshow(raw_img)
    plt.axis('off')
    # 可视化伪框中心点
    plt.subplot(1,2,2)
    plt.imshow(fig)
    plt.axis('off')
    if not os.path.isdir('./vis_same-size/dense_centers'):os.makedirs('./vis_same-size/dense_centers')
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    plt.savefig(f'./vis_same-size/dense_centers/{img_name}', dpi=200)  







if __name__ == '__main__':
    main()
