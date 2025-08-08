import sys
sys.path.append('../../')

from utils.utils import save_checkpoint2, AverageMeter, Logger, accuracy, mkdirs, adjust_learning_rate, time_to_str
from utils.evaluate import eval
from utils.get_loader import get_dataset,get_dataset2
from models.DGFAS import DG_model, simple_model,  Discriminator
from loss.AdLoss import Real_AdLoss1

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
    src1_train_dataloader_fake, src1_train_dataloader_real, \
    src2_train_dataloader_fake, src2_train_dataloader_real, \
    src3_train_dataloader_fake, src3_train_dataloader_real, \
    tgt_valid_dataloader = get_dataset2(config.src1_data, config.src1_train_num_frames, 
                                       config.src2_data, config.src2_train_num_frames, 
                                       config.src3_data, config.src3_train_num_frames,
                                       config.tgt_data, config.tgt_test_num_frames, config.batch_size)
    # src1_train_dataloader_fake, src1_train_dataloader_real, \
    # tgt_valid_dataloader = get_dataset(config.src1_data, config.src1_train_num_frames, 
    #                                    config.tgt_data, config.tgt_test_num_frames, config.batch_size)
    

    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_ACER = 1.0
    best_model_AUC = 0.0
    # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:ACER, 5:AUC, 6:threshold
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]

    loss_classifier = AverageMeter()


    net = DG_model().to(device)
    net2 = simple_model(config.model).to(device)
    ad_net_real = Discriminator().to(device)
    #net = models.resnet18()

    #修改网络结构，将fc层1000个输出改为2个输出
    # fc_input_feature = net.fc.in_features
    # net.fc = nn.Linear(fc_input_feature, 2)


    log = Logger()
    log.open(config.logs + config.tgt_data + '_log_train_DBDG_resnet18.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    
    log.write('** start training target model! **\n')
    log.write(
        '--------|------------- VALID -------------|classifier|------ Current Best ------|--------------|\n')
    log.write(
        '  iter  |   loss   top-1   HTER    AUC    |   loss   |   top-1   HTER    AUC    |    time      |\n')
    log.write(
        '-----------------------------------------------------------------------------------------------|\n')
    start = timer()
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": config.init_lr},
        {"params": filter(lambda p: p.requires_grad, ad_net_real.parameters()), "lr": config.init_lr},
        {"params": filter(lambda p: p.requires_grad, net2.parameters()), "lr": config.init_lr},
    ]
    #optimizer = optim.SGD(optimizer_dict, lr=config.init_lr, momentum=config.momentum, weight_decay=config.weight_decay)   
    optimizer = optim.AdamW(optimizer_dict, lr=config.init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-2)
    # cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=100,eta_min=1e-9)  #初始学习率不能太大，否则loss越来越大直到无穷
    # cosine_schedule = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
    init_param_lr = []
    for param_group in optimizer.param_groups:
        init_param_lr.append(param_group["lr"])

    iter_per_epoch = 10

    src1_train_iter_real = iter(src1_train_dataloader_real)
    src1_iter_per_epoch_real = len(src1_train_iter_real)
    src2_train_iter_real = iter(src2_train_dataloader_real)
    src2_iter_per_epoch_real = len(src2_train_iter_real)
    src3_train_iter_real = iter(src3_train_dataloader_real)
    src3_iter_per_epoch_real = len(src3_train_iter_real)
    src1_train_iter_fake = iter(src1_train_dataloader_fake)
    src1_iter_per_epoch_fake = len(src1_train_iter_fake)
    src2_train_iter_fake = iter(src2_train_dataloader_fake)
    src2_iter_per_epoch_fake = len(src2_train_iter_fake)
    src3_train_iter_fake = iter(src3_train_dataloader_fake)
    src3_iter_per_epoch_fake = len(src3_train_iter_fake)

    max_iter = config.max_iter
    epoch = 1
    if(len(config.gpus) > 1):
        net = torch.nn.DataParallel(net).cuda()

    for iter_num in range(max_iter+1):
        if (iter_num % src1_iter_per_epoch_real == 0):
            src1_train_iter_real = iter(src1_train_dataloader_real)
        if (iter_num % src2_iter_per_epoch_real == 0):
            src2_train_iter_real = iter(src2_train_dataloader_real)
        if (iter_num % src3_iter_per_epoch_real == 0):
            src3_train_iter_real = iter(src3_train_dataloader_real)
        if (iter_num % src1_iter_per_epoch_fake == 0):
            src1_train_iter_fake = iter(src1_train_dataloader_fake)
        if (iter_num % src2_iter_per_epoch_fake == 0):
            src2_train_iter_fake = iter(src2_train_dataloader_fake)
        if (iter_num % src3_iter_per_epoch_fake == 0):
            src3_train_iter_fake = iter(src3_train_dataloader_fake)
        if (iter_num != 0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1
        param_lr_tmp = []
        for param_group in optimizer.param_groups:
            param_lr_tmp.append(param_group["lr"])

        net.train(True)
        net2.train(True)
        ad_net_real.train(True)
        optimizer.zero_grad()

        adjust_learning_rate(optimizer, epoch, init_param_lr, config.lr_epoch_1, config.lr_epoch_2)
        
        ######### data prepare #########
        src1_img_real, src1_label_real = src1_train_iter_real.next()
        src1_img_real = src1_img_real.cuda()
        src1_label_real = src1_label_real.cuda()
        input1_real_shape = src1_img_real.shape[0]
        # print('ccc',input1_real_shape )  #10

        src2_img_real, src2_label_real = src2_train_iter_real.next()
        src2_img_real = src2_img_real.cuda()
        src2_label_real = src2_label_real.cuda()
        input2_real_shape = src2_img_real.shape[0]

        src3_img_real, src3_label_real = src3_train_iter_real.next()
        src3_img_real = src3_img_real.cuda()
        src3_label_real = src3_label_real.cuda()
        input3_real_shape = src3_img_real.shape[0]

        src1_img_fake, src1_label_fake = src1_train_iter_fake.next()
        src1_img_fake = src1_img_fake.cuda()
        src1_label_fake = src1_label_fake.cuda()
        input1_fake_shape = src1_img_fake.shape[0]

        src2_img_fake, src2_label_fake = src2_train_iter_fake.next()
        src2_img_fake = src2_img_fake.cuda()
        src2_label_fake = src2_label_fake.cuda()
        input2_fake_shape = src2_img_fake.shape[0]

        src3_img_fake, src3_label_fake = src3_train_iter_fake.next()
        src3_img_fake = src3_img_fake.cuda()
        src3_label_fake = src3_label_fake.cuda()
        input3_fake_shape = src3_img_fake.shape[0]

        input_data = torch.cat([src1_img_real, src1_img_fake, src2_img_real, src2_img_fake, src3_img_real, src3_img_fake], dim=0)

        source_label = torch.cat([src1_label_real, src1_label_fake,
                                src2_label_real, src2_label_fake,
                                src3_label_real, src3_label_fake], dim=0)
  

        input_data, labels_a, labels_b, lam = mixup_data(input_data, source_label, 0.8)
        input_data, labels_a, labels_b = input_data.cuda(), labels_a.cuda(), labels_b.cuda()
        ######### forward #########
        input_phase = torch.angle(torch.fft.fftn(input_data, dim=(2, 3)))
       
        pred2, feature1, ifftfeature = net2(input_phase)
        pred, lowfeature, allfeature, lowfeature1 = net(input_data, ifftfeature)##DG
        
        ######### backward #########
        input1_shape = input1_real_shape + input1_fake_shape
        input2_shape = input2_real_shape + input2_fake_shape
   
        feature_real_1 = allfeature.narrow(0, 0, input1_real_shape)
        # print('ccc',feature_real_1.shape)  #[10*128]
        feature_real_2 = allfeature.narrow(0, input1_shape, input2_real_shape)
        feature_real_3 = allfeature.narrow(0, input1_shape+input2_shape, input3_real_shape)
        feature_real = torch.cat([feature_real_1, feature_real_2, feature_real_3], dim=0)
        
        discriminator_out_real = ad_net_real(feature_real)
        real_shape_list = []
        real_shape_list.append(input1_real_shape)
        real_shape_list.append(input2_real_shape)
        real_shape_list.append(input3_real_shape)
        real_adloss = Real_AdLoss1(discriminator_out_real, criterion, real_shape_list)
        
        
        loss1 = mixup_criterion(criterion, pred, labels_a, labels_b, lam)

        loss2 = F.mse_loss(feature1, lowfeature)
        ###消融实验/尝试效果
        # loss2 = torch.mean(F.cosine_similarity(feature1, lowfeature)) 
        pred2 = pred2.narrow(0, 0, input_data.size(0))
        # print('pred2.shape',pred2.shape) #[60,2]
       
        # print('source_label',source_label.shape)
        loss3 = mixup_criterion(criterion, pred2, labels_a, labels_b, lam)
        

        loss = loss1  +   config.lambda_app * loss2 +loss3 + config.lambda_adreal * real_adloss  #
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
 
        loss_classifier.update(loss.item())
        # pred3 = pred +
        # acc = accuracy(pred.narrow(0, 0, input_data.size(0)), source_label, topk=(1,))

        # classifer_top1.update(acc[0])


        print('\r', end='', flush=True)
        print(
            '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f |  %6.3f  %6.3f  %6.3f  | %s'
            % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                loss_classifier.avg, 
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
            
            save_list = [epoch, valid_args, best_model_HTER, best_model_ACC, best_model_ACER, threshold]
            string1 = 'net'
            string2 = 'net2'
            save_checkpoint2(save_list, is_best, net, config.gpus, config.checkpoint_path, config.best_model_path,string1)
            save_checkpoint2(save_list, is_best, net2, config.gpus, config.checkpoint_path, config.best_model_path, string2)
            print('\r', end='', flush=True)
            log.write(
                '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s   %s'
                % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                loss_classifier.avg, 
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min'),
                param_lr_tmp[0]))
            log.write('\n')
            time.sleep(0.01)



if __name__ == '__main__':
    
    train()
