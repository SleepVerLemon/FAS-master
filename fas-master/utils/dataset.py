# fas/utils/dataset.py
import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import math
import random


# 保留原有核心数据增强组件
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


class RandomShufflePatch(object):
    def __init__(self, image_size, ratio=0.5, total_patch_num=9):
        self.ratio = ratio
        self.total_patch_num = total_patch_num
        self.patch_num = int(math.sqrt(self.total_patch_num))
        self.image_size = image_size  # 目标图像尺寸（正方形）
        self.patch_size = image_size // self.patch_num

    def __call__(self, img):
        # 首先将图像转换为指定尺寸的正方形
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # 如果随机数小于比例，则不进行patch shuffle
        if random.random() < self.ratio:
            return img
            
        # 转换为numpy数组进行处理
        img = np.array(img)
        h, w, c = img.shape
        
        # 计算patch尺寸和坐标
        patch_size = self.patch_size
        patches = []
        
        # 提取所有patch
        for i in range(self.patch_num):
            for j in range(self.patch_num):
                # 计算每个patch的坐标
                start_h = i * patch_size
                end_h = start_h + patch_size if i < self.patch_num - 1 else h
                start_w = j * patch_size
                end_w = start_w + patch_size if j < self.patch_num - 1 else w
                
                # 提取并调整patch大小
                img_patch = img[start_h:end_h, start_w:end_w]
                img_patch = cv2.resize(img_patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                patches.append(img_patch)
        
        # 打乱patch顺序
        random.shuffle(patches)
        
        # 重新组合patch
        rows = []
        for i in range(self.patch_num):
            # 每一行的patch
            row_patches = patches[i * self.patch_num : (i + 1) * self.patch_num]
            row = np.concatenate(row_patches, axis=1)
            rows.append(row)
        
        # 组合所有行
        shuffled_img = np.concatenate(rows, axis=0)
        # 确保最终尺寸正确
        shuffled_img = cv2.resize(shuffled_img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        
        return Image.fromarray(shuffled_img)


# 适配ImageNet格式的FAS数据集（核心改造）
class FASDataset(ImageFolder):
    """
    FAS数据集类，适配ImageNet目录结构：
    /path/to/imagenet/
      train/
        class1(real)/
          img1.jpeg
        class2(spoof)/
          img2.jpeg
      val/
        class1(real)/
          img3.jpeg
        class2(spoof)/
          img4.jpeg
    """
    def __init__(self, root, transform=None, is_train=True):
        """参考PVT的INatDataset实现，手动加载样本，避免ImageFolder的属性依赖问题"""
        self.transform = transform
        self.loader = default_loader  # 使用默认图像加载器
        self.data_dir = os.path.join(root, 'train' if is_train else 'val')
        
        # 类别映射：class1->real(1)，class2->spoof(0)
        self.class_to_idx = {'class1': 1, 'class2': 0}
        
        # 手动收集所有样本（参考PVT的self.samples初始化方式）
        self.samples = self._collect_samples()
        
        # 调试信息
        print(f"加载数据集路径: {self.data_dir}")
        print(f"有效样本数: {len(self.samples)}")

    def _collect_samples(self):
        """手动遍历目录收集样本，类似PVT的样本收集逻辑"""
        samples = []
        # 遍历class1和class2文件夹
        for cls_name in self.class_to_idx.keys():
            cls_dir = os.path.join(self.data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"警告：类别文件夹 {cls_dir} 不存在，跳过该类别")
                continue
            # 遍历文件夹中的所有图像文件
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                # 简单过滤非图像文件
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    target = self.class_to_idx[cls_name]
                    samples.append((img_path, target))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        # 加载图像（参考PVT的loader用法）
        sample = self.loader(img_path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


# 从torchvision.folder导入默认加载器（保持与ImageFolder一致的行为）
def default_loader(path):
    from torchvision.datasets.folder import default_loader as torch_default_loader
    return torch_default_loader(path)