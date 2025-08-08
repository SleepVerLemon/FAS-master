import sys
sys.path.append('../../')

from utils.utils import save_checkpoint2, AverageMeter, Logger, accuracy, mkdirs, adjust_learning_rate, time_to_str
from utils.evaluate import eval
from utils.utils import init_distributed_mode
from utils.get_loader import build_loader, build_dataset
from models.DGFAS import DG_model, simple_model,  Discriminator
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
    # load data
    # 初始化分布式模式（参考PVT）
    init_distributed_mode(config)
    
    # 构建数据加载器
    train_loader, val_loader, num_classes = build_loader(config)
    

    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_ACER = 1.0
    best_model_AUC = 0.0
    # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:ACER, 5:AUC, 6:threshold
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]
    
    tgt_valid_dataloader = val_loader
    

    loss_classifier = AverageMeter()
    acc_classifier = AverageMeter()

    net = DG_model().to(device)
    net2 = simple_model(config.model).to(device)
    ad_net_real = Discriminator().to(device)
    iter_per_epoch = len(train_loader)

    #net = models.resnet18()

    #修改网络结构，将fc层1000个输出改为2个输出
    # fc_input_feature = net.fc.in_features
    # net.fc = nn.Linear(fc_input_feature, 2)


    log = Logger()
    log.open(config.logs + config.tgt_data + '_log_SSDG_resnet18_withoutloss2.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    print("Norm_flag: ", config.norm_flag)
    log.write('** start training target model! **\n')
    log.write(
        '--------|------------- VALID -------------|--- classifier ---|------ Current Best ------|--------------|\n')
    log.write(
        '  iter  |   loss   top-1   HTER    AUC    |   loss   top-1   |   top-1   HTER    AUC    |    time      |\n')
    log.write(
        '-------------------------------------------------------------------------------------------------------|\n')
    start = timer()
    criterion = nn.CrossEntropyLoss().cuda()
    
    
    
    
    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": config.init_lr},
        {"params": filter(lambda p: p.requires_grad, net2.parameters()), "lr": config.init_lr},
        {"params": filter(lambda p: p.requires_grad, ad_net_real.parameters()), "lr": config.init_lr},
        
    ]
    # optimizer = optim.SGD(optimizer_dict, lr=config.init_lr, momentum=config.momentum, weight_decay=config.weight_decay)   
    optimizer = optim.AdamW(optimizer_dict, lr=config.init_lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=5e-2)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=200,eta_min=1e-9)  #初始学习率不能太大，否则loss越来越大直到无穷
    # 分布式训练适配
    if config.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[config.gpu])

    
    # 训练循环（调整数据迭代方式）
    for iter_num in range(config.max_iter + 1):

        if iter_num % iter_per_epoch == 0:
            loss_classifier.reset()  # 重置损失累计器
            acc_classifier.reset()   # 重置准确率累计器

        # 训练模式
        net.train()
        net2.train()
        ad_net_real.train()
        optimizer.zero_grad()


        # 获取批次数据（直接从统一的train_loader迭代）
        images, labels = next(iter(train_loader))
        images = images.cuda()
        labels = labels.cuda()

        # 混合增强（保持原有逻辑）
        input_data, labels_a, labels_b, lam = mixup_data(images, labels, 0.75)
        
        ######### forward #########
        # print(torch.__version__)  # 确保版本≥1.7.0
        # print(hasattr(torch.fft, 'fftn'))
        # input_phase = torch.angle(torch.fft.fft2(input_data, dim=(2, 3)))
        # 检测输入是否有NaN
        if torch.isnan(input_data).any():
            print("input_data contains NaN!")
        input_fft = torch.rfft(
                            input_data, 
                            signal_ndim=2,  # 对最后2维（h, w）进行变换
                            onesided=False  # 返回完整的双边频谱
                            )  # 输出shape: (batch, channel, h, w, 2)，最后一维为[实部, 虚部]
        if torch.isnan(input_fft).any():
            print("input_fft contains NaN!")
        # print(input_fft.shape)
        # 计算相位角（angle = arctan2(虚部, 实部)）
        input_phase = torch.atan2(input_fft[..., 1], input_fft[..., 0])
        pred2, feature1, ifftfeature = net2(input_phase)
        pred, lowfeature, allfeature, lowfeature1 = net(input_data, ifftfeature)##DG



        ######### backward #########
        loss1 = mixup_criterion(criterion, pred, labels_a, labels_b, lam)
        loss2 = F.mse_loss(feature1, lowfeature)
        loss3 = mixup_criterion(criterion, pred2, labels_a, labels_b, lam)

        current_cls_loss = (loss1 + loss3).item()
        acc = accuracy(pred, labels, topk=(1,))[0].item()
        loss_classifier.update(current_cls_loss, n=input_data.size(0))  # n是当前批次样本数
        acc_classifier.update(acc, n=input_data.size(0))  # 同样传入n

        loss = loss1 + config.lambda_app * loss2 + loss3  # 可根据需要保留鉴别器损失
        # 检测损失值
        if torch.isnan(loss).any():
            print(f"Loss is NaN! loss1={loss1}, loss2={loss2}, loss3={loss3}")
        loss.backward()
        optimizer.step()
       

        

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

        if (iter_num != 0 and (iter_num+1) % iter_per_epoch == 0):
            # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:AUC, 5:threshold, 6:ACC_threshold
            valid_args = eval(tgt_valid_dataloader, net, net2)
            # judge model according to HTER
            is_best = valid_args[3] <= best_model_HTER
            best_model_HTER = min(valid_args[3], best_model_HTER)
            threshold = valid_args[5]
            if (valid_args[3] <= best_model_HTER):
                best_model_ACC = valid_args[6]
                best_model_AUC = valid_args[4]

            epoch = config.epoch
            save_list = [epoch, valid_args, best_model_HTER, best_model_ACC, best_model_ACER, threshold]
            string1 = 'net'
            string2 = 'net2'
            save_checkpoint2(save_list, is_best, net, config.gpus, config.checkpoint_path, config.best_model_path,string1)
            save_checkpoint2(save_list, is_best, net2, config.gpus, config.checkpoint_path, config.best_model_path, string2)
            print('\r', end='', flush=True)
            param_lr_tmp = [group['lr'] for group in optimizer.param_groups]
            log.write(
                '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s'
                % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                loss_classifier.avg, acc_classifier.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min')#,
                # param_lr_tmp[0],param_lr_tmp[1]
                ))
            log.write('\n')
            time.sleep(0.01)




if __name__ == "__main__":
    train()
