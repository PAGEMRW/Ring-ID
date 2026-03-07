import yaml
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
with open('config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
os.environ["CUDA_VISIBLE_DEVICES"] = cfg['train']['gpuid']
import torch
import torch.optim as optim
import numpy as np
from nets.siamese import Siamese
import torch.nn as nn
from utils.callbacks import LossHistory
import torch.backends.cudnn as cudnn
from utils.train_utils import download_weights, load_dataset, show_config,set_optimizer_lr, get_lr_scheduler
from utils.dataloader import SiameseDataset, dataset_collate
from torch.utils.data import DataLoader
from utils.utils_fit import fit_one_epoch



if __name__ == "__main__":
    Cuda = cfg['train']['cuda']
    pretrained = cfg['train']['pretrained']
    model_path = cfg['train']['model_path']
    Init_Epoch = cfg['train']['Init_Epoch']
    Epoch = cfg['train']['Epoch']
    batch_size = cfg['train']['batch_size']
    Init_lr = cfg['train']['Init_lr']
    Min_lr = cfg['train']['Min_lr']
    dataset_path = cfg['train']['dataset_path']
    method       = 'method'
    flat_shape = cfg['train']['flat_shape']
    
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    optimizer_type = cfg['train']['optimizer_type']
    momentum = cfg['train']['momentum']
    weight_decay = cfg['train']['weight_decay']
    pre_net_path = cfg['train']['pre_net_path']
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = cfg['train']['lr_decay_type']
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    #------------------------------------------------------------------#
    save_period         = cfg['train']['save_period']
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = cfg['train']['save_dir']
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    num_workers         = cfg['train']['num_workers']
    device              = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #------------------------------------------------------------------#
    #   backbone      使用到的模型名称
    backbone            = cfg['train']['backbone']
    lossfunc            = cfg['train']['lossfunc']
    if pretrained and backbone == 'vgg16':
        download_weights(backbone)

    model = Siamese(backbone, method=method, flat_shape = flat_shape, pretrained=pretrained, pre_net_path=pre_net_path)
    if model_path != '' and (pre_net_path == '' or pre_net_path == 'None'):
        print('Load weights {}.'.format(model_path))
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    elif model_path != '' and pre_net_path != '':
        model_dict = model.state_dict()

        # === 1. 加载 pre_net ===
        print("Loading pre_net weights from:", pre_net_path)
        prenet_state = torch.load(pre_net_path, map_location=device)
        prenet_state = prenet_state['model']
        prenet_state = prenet_state.get('model_state_dict', prenet_state)


        prenet_state = {
            f"pre_net.{k}": v
            for k, v in prenet_state.items()
            if f"pre_net.{k}" in model_dict
            and v.shape == model_dict[f"pre_net.{k}"].shape
        }


        pre_net_total = sum(1 for k in model_dict if k.startswith("pre_net."))
        print(f"\n pre_net matched params: {len(prenet_state)} / {pre_net_total}")

        # === 2. 加载 Siamese backbone ===
        print("\nLoading Siamese backbone weights from:", model_path)
        siamese_state = torch.load(model_path, map_location=device)
        siamese_state = siamese_state.get('model_state_dict', siamese_state)
        siamese_state = {
            k: v for k, v in siamese_state.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }


        # === 3. 合并加载 ===
        combined = {**siamese_state, **prenet_state}
        msg = model.load_state_dict(combined, strict=False)

        print("\n Loaded successfully:")
        print("   Missing keys:", msg.missing_keys)
        print("   Unexpected keys:", msg.unexpected_keys)

            
    #   获得损失函数
    if lossfunc == 'BCEWithLogitsLoss':
        loss = torch.nn.BCEWithLogitsLoss()
    #   记录Loss
    loss_history = LossHistory(save_dir, model, backbone)
    model = model.to(device)
    cudnn.benchmark = True
    #   训练集和验证集的比例。
    train_ratio = cfg['train']['train_ratio']
    train_lines, train_labels, val_lines, val_labels = load_dataset(dataset_path, train_ratio)
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    show_config(
            model_path = model_path, \
            Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
    #---------------------------------------------------------#
    #   总训练世代指的是遍历全部数据的总次数
    #   总训练步长指的是梯度下降的总次数 
    #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
    #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
    #----------------------------------------------------------#
    wanted_step = 3e4 if optimizer_type == "sgd" else 1e4
    total_step  = num_train // batch_size * Epoch
    if total_step <= wanted_step:
        wanted_epoch = wanted_step // (num_train // batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, batch_size, Epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))
    
    #-------------------------------------------------------------------#
    #   判断当前batch_size，自适应调整学习率
    #-------------------------------------------------------------------#
    nbs             = 64
    lr_limit_max    = 1e-2 if optimizer_type == 'adam' else 1e-1
    lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
    
    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    #---------------------------------------#
    #   根据optimizer_type选择优化器
    #---------------------------------------#
    optimizer = {
        'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
        'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay = weight_decay)
    }[optimizer_type]

    #---------------------------------------#
    #   获得学习率下降的公式
    #---------------------------------------#
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
    
    #---------------------------------------#
    #   判断每一个世代的长度
    #---------------------------------------#
    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size
    
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    train_dataset   = SiameseDataset(train_lines, train_labels, True)
    val_dataset     = SiameseDataset(val_lines, val_labels, False)

    gen             = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                            drop_last=True, collate_fn=dataset_collate)
    gen_val         = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                            drop_last=True, collate_fn=dataset_collate)

    for epoch in range(Init_Epoch, Epoch):
        
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        
        fit_one_epoch(model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, Cuda, save_period, save_dir)

    loss_history.writer.close()

    