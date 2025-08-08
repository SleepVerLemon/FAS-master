import sys
sys.path.append('../../')
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from config import config
from utils.utils import sample_frames
from utils.dataset import YunpeiDataset2
from utils.utils import AverageMeter, accuracy, draw_roc
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from models.DGFAS import DG_model, simple_model
import shutil 

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

def test(test_dataloader, model1, model2, threshold):
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    model1.eval()
    model2.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    number = 0
    with torch.no_grad():
        for iter, (input, target, videoID) in enumerate(test_dataloader):
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            input_phase = torch.angle(torch.fft.fftn(input, dim=(2, 3)))
            cls_out1, feature1, feature= model2(input_phase)
            cls_out, lowfeature, allfeature,lowfeature1 = model1(input, feature)
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            videoID = videoID.cpu().data.numpy()
            for i in range(len(prob)):
                if (videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                    number += 1
                    if (number % 100 == 0):
                        print('**Testing** ', number, ' photos done!')
    print('**Testing** ', number, ' photos done!')
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
        # compute loss and acc for every video
        avg_single_video_output = sum(output_dict_tmp[key]) / len(output_dict_tmp[key])
        avg_single_video_target = sum(target_dict_tmp[key]) / len(target_dict_tmp[key])
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))
        valid_top1.update(acc_valid[0])

    cur_EER_valid, threshold, FRR_list, FAR_list = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    auc_score = roc_auc_score(label_list, prob_list)
    draw_roc(FRR_list, FAR_list, auc_score)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)
    return [valid_top1.avg, cur_EER_valid, cur_HTER_valid, auc_score, ACC_threshold, threshold]

def test1(test_dataloader, model2, threshold):
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}

    model2.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    number = 0
    with torch.no_grad():
        for iter, (input, target, videoID) in enumerate(test_dataloader):
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            # input_phase = torch.angle(torch.fft.fftn(input, dim=(2, 3)))
            # cls_out1, feature1, feature= model2(input_phase)
            cls_out1= model2(input)
            prob = F.softmax(cls_out1, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            videoID = videoID.cpu().data.numpy()
            for i in range(len(prob)):
                if (videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out1[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out1[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                    number += 1
                    if (number % 100 == 0):
                        print('**Testing** ', number, ' photos done!')
    print('**Testing** ', number, ' photos done!')
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
        # compute loss and acc for every video
        avg_single_video_output = sum(output_dict_tmp[key]) / len(output_dict_tmp[key])
        avg_single_video_target = sum(target_dict_tmp[key]) / len(target_dict_tmp[key])
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))
        valid_top1.update(acc_valid[0])

    cur_EER_valid, threshold, FRR_list, FAR_list = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    auc_score = roc_auc_score(label_list, prob_list)
    draw_roc(FRR_list, FAR_list, auc_score)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)
    return [valid_top1.avg, cur_EER_valid, cur_HTER_valid, auc_score, ACC_threshold, threshold]

import torchvision.models as models
import torch.nn as nn

def main():
    net = DG_model().cuda()
    net2 = simple_model(config.model).cuda()

    # net3 = models.resnet18().cuda()

    # #修改网络结构，将fc层1000个输出改为2个输出
    # fc_input_feature = net3.fc.in_features
    # net3.fc = nn.Linear(fc_input_feature, 2)

    test_data = sample_frames(flag=2, num_frames=config.tgt_test_num_frames, dataset_name=config.tgt_data)
    test_dataloader = DataLoader(YunpeiDataset2(test_data, train=False), batch_size=1, shuffle=False)
    print('\n')
    print("**Testing** Get test files done!")
    # load model
    net1_ = torch.load(config.best_model_path + config.tgt_best_model_name1)  #net
    net2_ = torch.load(config.best_model_path + config.tgt_best_model_name2)
    # net_ = torch.load(config.best_model_path + config.tgt_best_model_name)
  
    # net3.load_state_dict(net_["state_dict"])
    net.load_state_dict(net1_["state_dict"])
    net2.load_state_dict(net2_["state_dict"])
    
    threshold = net1_["threshold"]
    # threshold = net_["threshold"]
    # device = 'cuda'
    # net3.to(device)
    # test model
    test_args = test(test_dataloader, net, net2, threshold)
    # test_args = test1(test_dataloader, net3, threshold)
    # test_args = test1(test_dataloader, net2, threshold)
    print('\n===========Test Info===========\n')
    print(config.tgt_data, 'Test acc: %5.4f' %(test_args[0]))
    print(config.tgt_data, 'Test EER: %5.4f' %(test_args[1]))
    print(config.tgt_data, 'Test HTER: %5.4f' %(test_args[2]))
    print(config.tgt_data, 'Test AUC: %5.4f' % (test_args[3]))
    print(config.tgt_data, 'Test ACC_threshold: %5.4f' % (test_args[4]))
    print('\n===============================\n')

if __name__ == '__main__':
    main()
