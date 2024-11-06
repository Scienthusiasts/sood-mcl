#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import json
import os
import shutil
import tqdm


def split_img_vis_list(list_file, img_dir, ann_dir, out_dir):
    with open(list_file, 'r', encoding='utf-8') as f:
        file_list = json.load(f)
    all_img_files = dict()
    all_ann_files = dict()
    for file_ in glob.glob(os.path.join(img_dir, '*.png')):
        all_img_files[file_.split('/')[-1]] = file_
    for file_ in glob.glob(os.path.join(ann_dir, '*.txt')):
        all_ann_files[file_.split('/')[-1]] = file_
    print(f"Total images: {len(all_img_files)}")
    print(f"Total labels: {len(all_ann_files)}")
    labeled_img_out_dir = out_dir['labeled_img']
    unlabeled_img_out_dir = out_dir['unlabeled_img']
    labeled_ann_out_dir = out_dir['labeled_ann']
    unlabeled_ann_out_dir = out_dir['unlabeled_ann']

    if not os.path.isdir(labeled_img_out_dir):os.makedirs(labeled_img_out_dir)
    if not os.path.isdir(unlabeled_img_out_dir):os.makedirs(unlabeled_img_out_dir)
    if not os.path.isdir(labeled_ann_out_dir):os.makedirs(labeled_ann_out_dir)
    if not os.path.isdir(unlabeled_ann_out_dir):os.makedirs(unlabeled_ann_out_dir)

    labeled_img_num = 0
    labeled_ann_num = 0

    for file_name, file_path in all_img_files.items():
        # NOTE:modified by yan, which allow split after crop
        if file_name.split('__')[0]+'.png' in file_list:
            shutil.copyfile(file_path, os.path.join(labeled_img_out_dir, file_name))
            labeled_img_num += 1
            print(f'copy {file_name} to labeled data')
        else:
            shutil.copyfile(file_path, os.path.join(unlabeled_img_out_dir, file_name))
            print(f'copy {file_name} to unlabeled data')

    for file_name, file_path in all_ann_files.items():
        # NOTE:modified by yan, which allow split after crop
        if file_name.split('__')[0]+'.png' in file_list:
            shutil.copyfile(file_path, os.path.join(labeled_ann_out_dir, file_name))
            labeled_ann_num += 1
            print(f'copy {file_name} to labeled data')
        else:
            with open(os.path.join(unlabeled_ann_out_dir, file_name), "w") as _: pass
            print(f'create empty {file_name} to unlabeled data') 


    assert labeled_img_num == len(file_list) and labeled_ann_num == len(file_list)
    print(f"Finish saving {labeled_img_num} labeled image and annfile.")


if __name__ == '__main__':
    # example
    per = 10
    version = 1.0
    list_file = f'/data/yht/code/sood-mcl/data_lists/{per}p_list.json'
    img_dir =   f'/data/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/train/images'
    ann_dir =   f'/data/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/train/{version}/annfiles'
    # split结果保存的路径：
    labled_image_dir =   f'/data/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/train_{per}per/{version}/labeled/images'
    unlabled_image_dir = f'/data/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/train_{per}per/{version}/unlabeled/images'
    labled_ann_dir =     f'/data/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/train_{per}per/{version}/labeled/annfiles'
    unlabled_ann_dir =   f'/data/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/train_{per}per/{version}/unlabeled/empty_annfiles'
    out_dir = dict(
        labeled_img=labled_image_dir,
        unlabeled_img=unlabled_image_dir,
        labeled_ann=labled_ann_dir,
        unlabeled_ann=unlabled_ann_dir,
    )
    # For spliting the DOTA-v1.5's train set (半监督)
    split_img_vis_list(list_file, img_dir, ann_dir, out_dir)