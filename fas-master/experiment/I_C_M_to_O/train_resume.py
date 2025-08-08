import sys
sys.path.append('../../')

from utils.utils import save_checkpoint2, AverageMeter, Logger, accuracy, mkdirs, adjust_learning_rate, time_to_str
from utils.evaluate import eval
from utils.utils import init_distributed_mode
from utils.get_loader import build_loader, build_dataset
from models.DGFAS import DG_model, simple_model, Discriminator
from loss.AdLoss import Real_AdLoss1
from utils.dataset import FASDataset

import random
import numpy as np
from config import config
from datetime import datetime
import time
from timeit import default_timer as timer

from torchtoolbox.tools import mixup_data, mixup_criterion

import os
import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as T

import torch.nn.functional as F

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = 'cuda'

def train():
    mkdirs(config.checkpoint_path, config.best_model_path, config.logs)
    # 初始化分布式模式
    init_distributed_mode(config)
    
    # 构建数据加载器
    train_loader, val_loader, num_classes = build_loader(config)
    tgt_valid_dataloader = val_loader
    iter_per_epoch = len(train_loader)

    # 初始化最佳模型指标
    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_ACER = 1.0
    best_model_AUC = 0.0
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]

    # 初始化累计器
    loss_classifier = AverageMeter()
    acc_classifier = AverageMeter()

    # -------------------------- 加载模型参数(.pth.tar) --------------------------
    # 定义模型
    net = DG_model().to(device)
    net2 = simple_model(config.model).to(device)
    ad_net_real = Discriminator().to(device)

    # 模型加载路径（根据实际保存路径修改）
    net_ckpt_path = os.path.join(config.best_model_path, "net_model_best_0.13872_73.pth.tar")  # .pth.tar格式
    # print(net_ckpt_path)
    net2_ckpt_path = os.path.join(config.best_model_path, "net2_model_best_0.13872_73.pth.tar")
    # print(net2_ckpt_path)
    start_iter = 50  # 起始迭代次数（默认从0开始，若加载模型可调整）

    # 尝试加载模型
    if os.path.exists(net_ckpt_path) and os.path.exists(net2_ckpt_path):
        print(f"训练脚本加载模型参数: {net_ckpt_path} 和 {net2_ckpt_path}")
        
        # 加载net模型
        net_ckpt = torch.load(net_ckpt_path, map_location=device)
        # 处理分布式训练的module前缀
        net_state_dict = net_ckpt["state_dict"]
        new_net_state_dict = {k.replace("module.", ""): v for k, v in net_state_dict.items()}
        net.load_state_dict(new_net_state_dict)
        
        # 加载net2模型
        net2_ckpt = torch.load(net2_ckpt_path, map_location=device)
        net2_state_dict = net2_ckpt["state_dict"]
        new_net2_state_dict = {k.replace("module.", ""): v for k, v in net2_state_dict.items()}
        net2.load_state_dict(new_net2_state_dict)
        
        # 恢复最佳指标（若checkpoint中包含）
        if "best_HTER" in net_ckpt:
            best_model_HTER = net_ckpt["best_HTER"]
            best_model_ACC = net_ckpt.get("best_ACC", 0.0)
            best_model_AUC = net_ckpt.get("best_AUC", 0.0)
            print(f"恢复最佳指标: HTER={best_model_HTER:.3f}, ACC={best_model_ACC:.3f}")
        
        # 恢复迭代次数（若需要从断点继续）
        if "iter_num" in net_ckpt:
            start_iter = net_ckpt["iter_num"]
            print(f"从迭代次数 {start_iter} 继续训练")
    else:
        print("训练脚本未输入模型文件，使用随机初始化模型")
    # ------------------------------------------------------------------------------

    # 日志初始化
    log = Logger()
    log.open(config.logs + config.tgt_data + '_log_SSDG_resnet18_withoutloss2.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    print("Norm_flag: ", config.norm_flag)
    log.write('** start training target model! **\n')
    log.write(
        '--------|------------- VALID -------------|--- classifier ---|------ Current Best ------|--------------|\n')
    log.write(
        '  iter  |   loss   top-1   HTER    AUC    |   loss   top-1   |   top-1   HTER    AUC    |    time      |\n')
    log.write(
        '-------------------------------------------------------------------------------------------------------|\n')
    start = timer()
    criterion = nn.CrossEntropyLoss().cuda()
    
    # 优化器和调度器
    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": config.init_lr},
        {"params": filter(lambda p: p.requires_grad, net2.parameters()), "lr": config.init_lr},
        {"params": filter(lambda p: p.requires_grad, ad_net_real.parameters()), "lr": config.init_lr},
    ]
    optimizer = optim.AdamW(optimizer_dict, lr=config.init_lr, betas=(0.9, 0.995), eps=1e-8, weight_decay=5e-2)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=200, eta_min=1e-9)
    
    # 分布式训练适配
    if config.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[config.gpu])

    # 训练循环（从start_iter开始）
    for iter_num in range(start_iter, config.max_iter + 1):  # 从起始迭代次数开始
        # 每个epoch重置累计器
        if iter_num % iter_per_epoch == 0:
            loss_classifier.reset()
            acc_classifier.reset()

        # 训练模式
        net.train()
        net2.train()
        ad_net_real.train()
        optimizer.zero_grad()

        # 获取批次数据
        images, labels = next(iter(train_loader))
        images = images.cuda()
        labels = labels.cuda()

        # 混合增强
        input_data, labels_a, labels_b, lam = mixup_data(images, labels, 0.75)
        
        ######### forward #########
        if torch.isnan(input_data).any():
            print("input_data contains NaN!")
        input_fft = torch.rfft(
                            input_data, 
                            signal_ndim=2, 
                            onesided=False
                        )
        if torch.isnan(input_fft).any():
            print("input_fft contains NaN!")
        input_phase = torch.atan2(input_fft[..., 1], input_fft[..., 0])
        pred2, feature1, ifftfeature = net2(input_phase)
        pred, lowfeature, allfeature, lowfeature1 = net(input_data, ifftfeature)##DG

        ######### backward #########
        loss1 = mixup_criterion(criterion, pred, labels_a, labels_b, lam)
        loss2 = F.mse_loss(feature1, lowfeature)
        loss3 = mixup_criterion(criterion, pred2, labels_a, labels_b, lam)

        current_cls_loss = (loss1 + loss3).item()
        acc = accuracy(pred, labels, topk=(1,))[0].item()
        loss_classifier.update(current_cls_loss, n=input_data.size(0))
        acc_classifier.update(acc, n=input_data.size(0))

        loss = loss1 + config.lambda_app * loss2 + loss3
        if torch.isnan(loss).any():
            print(f"Loss is NaN! loss1={loss1}, loss2={loss2}, loss3={loss3}")
        loss.backward()
        optimizer.step()

        ######### 打印进度 #########
        print('\r', end='', flush=True)
        print(
            '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s'
            % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                loss_classifier.avg, acc_classifier.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min'))
            , end='', flush=True)

        ######### 每个epoch结束处理 #########
        if (iter_num != 0 and (iter_num+1) % iter_per_epoch == 0):
            valid_args = eval(tgt_valid_dataloader, net, net2)
            is_best = valid_args[3] <= best_model_HTER
            best_model_HTER = min(valid_args[3], best_model_HTER)
            threshold = valid_args[5]
            if is_best:
                best_model_ACC = valid_args[6]
                best_model_AUC = valid_args[4]

            # 保存模型时包含迭代次数和最佳指标（便于后续恢复）
            epoch = config.epoch
            save_list = [
                epoch, valid_args, best_model_HTER, best_model_ACC, best_model_ACER, threshold,
                optimizer.state_dict(),  # 保存优化器状态
                iter_num  # 保存当前迭代次数
            ]
            string1 = 'net'
            string2 = 'net2'
            save_checkpoint2(save_list, is_best, net, config.gpus, config.checkpoint_path, config.best_model_path, string1)
            save_checkpoint2(save_list, is_best, net2, config.gpus, config.checkpoint_path, config.best_model_path, string2)

            print('\r', end='', flush=True)
            log.write(
                '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s'
                % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                loss_classifier.avg, acc_classifier.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min'))
            )
            log.write('\n')
            time.sleep(0.01)


if __name__ == "__main__":
    train()