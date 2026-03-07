import os
from torch.hub import load_state_dict_from_url
from random import shuffle
import numpy as np
import random
from functools import partial
import math

def download_weights(backbone, model_dir="./pretrained_model"):
    download_urls = {
        'vgg16'         : 'https://download.pytorch.org/models/vgg16-397923af.pth',
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)

def load_dataset(dataset_path, train_ratio):
    types       = 0
    train_path  = dataset_path
    lines       = []
    labels      = []
    
    for character in os.listdir(train_path):
        #-------------------------------------------------------------#
        #   对每张图片进行遍历
        #-------------------------------------------------------------#
        character_path = os.path.join(train_path, character)
        for image in os.listdir(character_path):
            lines.append(os.path.join(character_path, image))
            labels.append(types)
        types += 1

    #-------------------------------------------------------------#
    #   将获得的所有图像进行打乱。
    #-------------------------------------------------------------#
    random.seed(1)
    shuffle_index = np.arange(len(lines), dtype=np.int32)
    shuffle(shuffle_index)
    random.seed(None)
    lines    = np.array(lines,dtype=object)
    labels   = np.array(labels)
    lines    = lines[shuffle_index]
    labels   = labels[shuffle_index]
    
    #-------------------------------------------------------------#
    #   将训练集和验证集进行划分
    #-------------------------------------------------------------#
    num_train           = int(len(lines)*train_ratio)

    val_lines      = lines[num_train:]
    val_labels     = labels[num_train:]

    train_lines    = lines[:num_train]
    train_labels   = labels[:num_train]
    return train_lines, train_labels, val_lines, val_labels

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func