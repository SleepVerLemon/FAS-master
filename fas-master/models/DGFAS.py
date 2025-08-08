import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import sys
import numpy as np
from torch.autograd import Variable
import random
import os
from collections import OrderedDict
import torch.nn.functional as F

# def resnet18(pretrained=False, **kwargs):
# 	"""Constructs a ResNet-18 model.
# 	Args:
# 	   pretrained (bool): If True, returns a model pre-trained on ImageNet
# 	"""
# 	model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
# 	# change your path
# 	model_path = '/root/classification/ckp_org/BetterOulu.pth'
# 	pretrained_weight = torch.load(model_path)
# 	del pretrained_weight['fc.weight']
# 	del pretrained_weight['fc.bias']
# 	model.load_state_dict(pretrained_weight, strict = False)
# 	print("loading model: ", model_path)
# 	return model

def resnet18(model_path, pretrained=True,** kwargs):
    """Constructs a ResNet-18 model.
    Args:
       pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    # 仅当pretrained为True时，才加载预训练权重True
    if pretrained:
        # 预训练权重路径
        
        pretrained_weight = torch.load(model_path)
        # 删除不需要的全连接层参数
        if 'fc.weight' in pretrained_weight:
            del pretrained_weight['fc.weight']
        if 'fc.bias' in pretrained_weight:
            del pretrained_weight['fc.bias']
        # 加载权重（strict=False忽略不匹配的键）
        model.load_state_dict(pretrained_weight, strict=False)
        print(f"已加载预训练模型: {model_path}")
    else:
        # 不加载预训练权重时的提示
        print("未加载预训练权重，使用随机初始化的模型")
    
    return model

class ChannelAttention(nn.Module):
	def __init__(self, in_planes, ratio=16):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False) #512

		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
		max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
		# avg_out = self.fc1(self.avg_pool(x))
		# max_out = self.fc1(self.max_pool(x))
		out = avg_out + max_out
		return self.sigmoid(out)
def l2_norm(input, axis=1):
	norm = torch.norm(input, 2, axis, True)
	output = torch.div(input, norm)
	return output
class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		self.classifier_layer = nn.Linear(512, 2)
		self.classifier_layer.weight.data.normal_(0, 0.01)
		self.classifier_layer.bias.data.fill_(0.0)

	def forward(self, input, norm_flag):
		if(norm_flag):
			self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
			classifier_out = self.classifier_layer(input)
		else:
			classifier_out = self.classifier_layer(input)
		return classifier_out
# class simple_model(nn.Module):
#     def __init__(self):
#         super(simple_model, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=8)  #全局池化层GAP
#         self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=1)  #全局池化层GAP

#         # self.in_features = model_resnet.fc.in_features
#         # in_channel = self.in_features
#         # print("in_channel:", in_channel)  #[512]
#         self.classifier = nn.Linear(128, 2)

#     def forward(self, input, feature):
#         lowfeature = self.conv1(input)
#         #print("lowfeature:", lowfeature.shape)  #[60,64,256,256]
#         lowfeature = self.avgpool(lowfeature)   #[60,64,8,8]

#         allfeature = torch.cat((feature, lowfeature),1)  ##特征融合
#         #print("allfeature size", allfeature.shape) # [60,128,8,8]
#         allfeature = self.avgpool1(allfeature)
#         #feature = self.avgpool1(feature)
#         ###

#         allfeature = allfeature.view(allfeature.size(0), -1)
#         #print("allfeature size", allfeature.shape) # [60,8192]
#         classifier_out = self.classifier(allfeature)
#         return classifier_out, lowfeature

# class DG_model(nn.Module):
#     def __init__(self, model):
#         super(DG_model, self).__init__()
#         model_resnet = resnet18()
#         self.conv1 = model_resnet.conv1
#         self.bn1 = model_resnet.bn1
#         self.relu = model_resnet.relu
#         self.maxpool = model_resnet.maxpool
#         self.layer1 = model_resnet.layer1
#         self.layer2 = model_resnet.layer2
#         self.layer3 = model_resnet.layer3
#         self.layer4 = model_resnet.layer4

#         ###
#         self.inplanes = 512
#         self.ca1 = ChannelAttention(64)
#         ###
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=8)  #全局池化层GAP
#         self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=1)  #全局池化层GAP
#         self.fc1   = nn.Conv2d(self.inplanes, self.inplanes // 8, 1, bias=False)

#         self.in_features = model_resnet.fc.in_features
#         in_channel = self.in_features
#         print("in_channel:", in_channel)  #[512]
#         self.classifier = nn.Linear(64, 2)
#         # self.bottleneck_layer_fc = nn.Linear(512, 512)
#         # self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
#         # self.bottleneck_layer_fc.bias.data.fill_(0.1)
#         # self.bottleneck_layer = nn.Sequential(
#         #     self.bottleneck_layer_fc,
#         #     nn.ReLU(),
#         #     nn.Dropout(0.5)
#         # )

#     def forward(self, input):
#         feature1 = self.conv1(input)
#         feature1 = self.bn1(feature1)
#         feature1 = self.relu(feature1)
#         feature1 = self.maxpool(feature1)
#         feature1 = self.layer1(feature1)

#         feature1 = self.layer2(feature1)
#         feature1 = self.layer3(feature1)
#         feature1 = self.layer4(feature1)
#         #print("feature1 size", feature1.shape)  #[60,512,8,8]
#         feature1 = self.fc1(feature1)
#         #print("feature1 size", feature1.shape)   #[60,64,8,8]
#         ###
#         feature = self.ca1(feature1) * feature1
#         #print("注意力机制后feature size", feature.shape)  #[60,64,8,8]
#         ifftfeature = torch.angle(torch.fft.ifftn(feature, dim=(2, 3)))
#         #print("IFFT后feature size", feature.shape)  #[60,64,8,8]
#         ####
#         feature = self.avgpool1(ifftfeature)
#         #print("IFFT后avgfeature size", feature.shape)  #[60,64,1,1]
#         feature = feature.view(feature.size(0), -1)
#         #print("feature size", feature.shape)  #[60,64]
#         #feature = self.fc1(feature)
#         ###
#         #print("feature size", feature.shape)  #[60,64,1,1]
#         #print("lowfeature size", lowfeature.shape) # [60,64,8,8] 加了avgpool后变相同

#         # feature = self.bottleneck_layer(feature)
#         # if (1):
#         #     feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
#         #     feature = torch.div(feature, feature_norm)
#         #print("feature size", feature.shape)

#         classifier_out = self.classifier(feature)
#         return classifier_out, feature1, ifftfeature

class DG_model(nn.Module):
    def __init__(self):
        super(DG_model, self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        model_resnet = resnet18('/root/FAS_model_xiaopang114/fas-master/experiment/I_C_M_to_O/oulu_checkpoint/resnet18/best_model/net_model_best_0.13872_73.pth.tar')
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1   #64
        self.layer2 = model_resnet.layer2   #128
        self.layer3 = model_resnet.layer3   #256
        self.layer4 = model_resnet.layer4  #512

        self.fc1  = nn.Conv2d(512, 64, 1, bias=False)
        # model = densenet()
        # self.conv1 = model
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=8) #全局池化层GAP
        self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=1) #全局池化层GAP

        # self.in_features = model_resnet.fc.in_features
        # in_channel = self.in_features
        # print("in_channel:", in_channel)  #[512]
        self.classifier = nn.Linear(128, 2)

    def forward(self, input, feature):
        #lowfeature = self.conv1(input)
        feature1 = self.conv1(input)
        feature1 = self.bn1(feature1)
        feature1 = self.relu(feature1)
        feature1 = self.maxpool(feature1)
        feature1 = self.layer1(feature1)   #64
        feature1 = self.layer2(feature1)
        lowfeature = self.layer3(feature1)
        lowfeature = self.layer4(lowfeature)
        lowfeature = self.fc1(lowfeature)
        # lowfeature = self.layer1(feature1) 
        
        #print("lowfeature:", lowfeature.shape) #[60,64,256,256]
        lowfeature = self.avgpool(lowfeature) #[60,64,8,8]
        lowfeature1 = self.avgpool1(lowfeature)
        # print("lowfeature:", lowfeature.shape) #[60,64,8,8]
        # print("feature:", feature.shape) #加了随即裁剪100之后[60,64,4,4]
        allfeature1 = torch.cat((feature, lowfeature),1) ##特征融合  增加列。allfeature前半部分是feature后半部分是low feature
        #print("allfeature size", allfeature.shape) # [60,128,8,8]
        allfeature = self.avgpool1(allfeature1)
        #feature = self.avgpool1(feature)
        ###

        allfeature = allfeature.view(allfeature.size(0), -1)
        ##  L2归一化
        feature_norm = torch.norm(allfeature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
        allfeature = torch.div(allfeature, feature_norm)
        ##
        #print("allfeature size", allfeature.shape) # [60,8192]
        classifier_out = self.classifier(allfeature)
        return classifier_out, lowfeature, allfeature1,lowfeature1,
class cam_model(nn.Module):
    def __init__(self):
        super(DG_model, self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        model_resnet = resnet18()
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1   #64
        self.layer2 = model_resnet.layer2   #128
        self.layer3 = model_resnet.layer3   #256
        self.layer4 = model_resnet.layer4  #512

        self.fc1  = nn.Conv2d(512, 64, 1, bias=False)
        # model = densenet()
        # self.conv1 = model
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=8) #全局池化层GAP
        self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=1) #全局池化层GAP

        # self.in_features = model_resnet.fc.in_features
        # in_channel = self.in_features
        # print("in_channel:", in_channel)  #[512]
        self.classifier = nn.Linear(128, 2)

    def forward(self, input):
        #lowfeature = self.conv1(input)
        feature1 = self.conv1(input)
        feature1 = self.bn1(feature1)
        feature1 = self.relu(feature1)
        feature1 = self.maxpool(feature1)
        feature1 = self.layer1(feature1)   #64
        feature1 = self.layer2(feature1)
        lowfeature = self.layer3(feature1)
        lowfeature = self.layer4(lowfeature)
        lowfeature = self.fc1(lowfeature)
        # lowfeature = self.layer1(feature1) 
        
        #print("lowfeature:", lowfeature.shape) #[60,64,256,256]
        lowfeature = self.avgpool(lowfeature) #[60,64,8,8]
        lowfeature1 = self.avgpool1(lowfeature)
        # print("lowfeature:", lowfeature.shape) #[60,64,8,8]
        # print("feature:", feature.shape) #加了随即裁剪100之后[60,64,4,4]
        # allfeature = torch.cat((feature, lowfeature),1) ##特征融合  增加列。allfeature前半部分是feature后半部分是low feature
        #print("allfeature size", allfeature.shape) # [60,128,8,8]
        allfeature = self.avgpool1(allfeature)
        #feature = self.avgpool1(feature)
        ###

        allfeature = allfeature.view(allfeature.size(0), -1)
        ##  L2归一化
        feature_norm = torch.norm(allfeature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
        allfeature = torch.div(allfeature, feature_norm)
        ##
        #print("allfeature size", allfeature.shape) # [60,8192]
        classifier_out = self.classifier(allfeature)
        return classifier_out, lowfeature, allfeature,lowfeature1

class simple_model(nn.Module):
    def __init__(self, model):
        super(simple_model, self).__init__()
        model_resnet = resnet18('/root/FAS_model_xiaopang114/fas-master/experiment/I_C_M_to_O/oulu_checkpoint/resnet18/best_model/net2_model_best_0.13872_73.pth.tar')
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4

        ###
        self.inplanes = 512
        self.ca1 = ChannelAttention(64)
        ###
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=8) #全局池化层GAP
        self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=1) #全局池化层GAP
        self.fc1  = nn.Conv2d(self.inplanes, self.inplanes // 8, 1, bias=False)

        self.in_features = model_resnet.fc.in_features
        in_channel = self.in_features
        print("in_channel:", in_channel) #[512]
        self.classifier = nn.Linear(64, 2)
        # self.bottleneck_layer_fc = nn.Linear(512, 512)
        # self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        # self.bottleneck_layer_fc.bias.data.fill_(0.1)
        # self.bottleneck_layer = nn.Sequential(
        #     self.bottleneck_layer_fc,
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )

    def forward(self, input):

        feature1 = self.conv1(input)
        feature1 = self.bn1(feature1)
        feature1 = self.relu(feature1)
        feature1 = self.maxpool(feature1)
        feature1 = self.layer1(feature1)

        feature1 = self.layer2(feature1)
        feature1 = self.layer3(feature1)
        feature1 = self.layer4(feature1)
        # print("feature1 size", feature1.shape) #[60,512,8,8]
        feature1 = self.fc1(feature1)
        # print("feature1 size", feature1.shape)  #[60,64,8,8]
        ###
        feature = self.ca1(feature1) * feature1
        
        # print("注意力机制后feature size", feature.shape) #[60,64,8,8]  加了随即裁剪变成4，4
        # ifftfeature = torch.angle(torch.fft.ifftn(feature, dim=(2, 3)))
        feature_fft = torch.rfft(
            feature,  # 输入应为包含实部和虚部的张量，shape为(..., 2)
            signal_ndim=2,  # 对最后2维进行逆变换
            onesided=False  # 与正向变换保持一致
        )

        ifft_result = torch.irfft(
            feature_fft,  # 输入复数频谱（..., 2）
            signal_ndim=2,
            onesided=False
        )

        if torch.isnan(feature_fft).any():
            print("feature_fft contains NaN!")
        if torch.isnan(ifft_result).any():
            print("ifft_result contains NaN!")

        # 计算角度
        ifftfeature = torch.angle(ifft_result)

        real = feature_fft[..., 0]  # 实部：[batch, 64, 8, 8]
        imag = feature_fft[..., 1]  # 虚部：[batch, 64, 8, 8]
        # 用atan2(虚部, 实部)计算角度（等价于angle，但支持反向传播）
        fft_angle = torch.atan2(imag, real)  # 替代原来的torch.angle()

        
        # print("IFFT后feature size", feature.shape) #[60,64,8,8]
        ####
        feature = self.avgpool1(fft_angle)     #为了分类
        #print("IFFT后avgfeature size", feature.shape) #[60,64,1,1]
        feature = feature.view(feature.size(0), -1)
        # print("ori feature size", feature.shape) #[60,64]
        #feature = self.fc1(feature)
        ###
        #print("feature size", feature.shape) #[60,64,1,1]
        #print("lowfeature size", lowfeature.shape) # [60,64,8,8] 加了avgpool后变相同

        # feature = self.bottleneck_layer(feature)
        # if (1):
        #     feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
        #     feature = torch.div(feature, feature_norm)
        #print("feature size", feature.shape)
        ####
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
        feature = torch.div(feature, feature_norm)
        ###
        classifier_out = self.classifier(feature)
        return classifier_out, feature1, fft_angle





class GRL(torch.autograd.Function):
    def __init__(self):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 4000  # be same to the max_iter of config.py

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        return -coeff * gradOutput

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(128, 512)   #128->64
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(512, 3)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL()

    def forward(self, feature):
        adversarial_out = self.ad_net.forward(self.grl_layer.forward(feature))
        return adversarial_out

if __name__ == '__main__': 
    x = Variable(torch.ones(1, 3, 256, 256))
    model1 = DG_model()
    model2 = simple_model()
    y, v = model1.forward(x, True)
    y1, v1 = model2.forward(x, True)