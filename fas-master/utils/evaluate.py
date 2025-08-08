from utils.utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate_threshold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


def eval(valid_dataloader, model1, model2):
    """
    基于单张图像评估，无需视频层面聚合
    """
    criterion = nn.CrossEntropyLoss()
    valid_losses = AverageMeter()  # 平均损失
    valid_top1 = AverageMeter()    # 平均准确率
    prob_list = []                  # 所有图像的预测概率（正类）
    label_list = []                 # 所有图像的真实标签
    all_outputs = []                # 所有图像的模型输出（用于计算损失）
    all_targets = []                # 所有图像的真实标签（用于计算损失）

    # 模型设为评估模式
    model1.eval()
    model2.eval()

    with torch.no_grad():  # 关闭梯度计算
        for iter, (input, target) in enumerate(valid_dataloader):
            # 处理输入（傅里叶变换得到相位特征，与之前逻辑一致）
            # 1. 对输入进行2D傅里叶变换（实部+虚部输出）
            input_fft = torch.rfft(
                input, 
                signal_ndim=2,  # 对最后2个维度（H, W）进行变换，等效于dim=(2,3)
                onesided=False  # 输出完整的双边频谱
            )  # shape: (batch, channel, H, W, 2)，最后一维为[实部, 虚部]
            if torch.isnan(input_fft).any():
                print("input_fft contains NaN!")

            # 2. 计算相位角（用atan2替代angle，确保可微分）
            input_phase = torch.atan2(
                input_fft[..., 1],  # 虚部
                input_fft[..., 0]   # 实部
            ) 
            
            # 转移到GPU
            input = Variable(input).cuda()
            input_phase = Variable(input_phase).cuda()
            target = Variable(target.long()).cuda()  # 确保标签为长整型

            # 模型推理
            cls_out1, feature1, feature = model2(input_phase)
            cls_out, lowfeature, allfeature, lowfeature1 = model1(input, feature)

            # 计算softmax概率（正类的概率）
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]  # 取正类（索引1）的概率
            label = target.cpu().data.numpy()  # 真实标签

            # 收集单张图像的结果（无需按视频聚合）
            prob_list.extend(prob)
            label_list.extend(label)
            all_outputs.append(cls_out.cpu())  # 保存模型输出（用于计算损失）
            all_targets.append(target.cpu())   # 保存真实标签（用于计算损失）

        # 计算整体损失和准确率（基于所有图像）
        # 拼接所有批次的输出和标签
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 计算平均损失
        loss = criterion(all_outputs, all_targets)
        valid_losses.update(loss.item(), n=len(all_outputs))  # 累计所有样本

        # 计算准确率
        acc_valid = accuracy(all_outputs, all_targets, topk=(1,))
        valid_top1.update(acc_valid[0].item(), n=len(all_outputs))

        prob_list = np.array(prob_list)
        label_list = np.array(label_list)

        # 计算AUC、EER、HTER等指标（基于所有单张图像的概率和标签）
        auc_score = roc_auc_score(label_list, prob_list)
        cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
        ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
        cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)

    # 返回评估指标（与原格式保持一致，便于后续日志打印）
    return [
        valid_losses.avg,    # 平均损失
        valid_top1.avg,      # 平均准确率（top1）
        cur_EER_valid,       # EER
        cur_HTER_valid,      # HTER
        auc_score,           # AUC
        threshold,           # 最佳阈值
        ACC_threshold * 100  # 阈值对应的准确率（百分比）
    ]