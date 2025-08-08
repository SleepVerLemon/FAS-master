import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
import math
import random

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import numpy as np

import cv2

from PIL import Image, ImageFile, ImageEnhance

import albumentations as A  # 引入PVT常用的albumentations增强库
from albumentations.pytorch import ToTensorV2
def crop_face_from_scene(image, face_name_full, scale = 1.5):
    #print(face_name_full)
    f=open(face_name_full,'gbk')
    lines=f.readlines()
    y1,x1,w,h=[float(ele) for ele in lines[:4]]
    f.close()
    y2=y1+w
    x2=x1+h

    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    #w_img,h_img=image.size
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)


    region=image[x1:x2,y1:y2]
    return region
def _is_numpy_image(img):
    return img.ndim in {2, 3}
def resize(img, size, interpolation=cv2.INTER_LINEAR):
    if not _is_numpy_image(img):
        raise TypeError('img should be OpenCV numpy Image. Got {}'.format(type(img)))
    # if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
    #     raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        h, w, _ = img.shape
        #h, w= img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, (ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, (ow, oh), interpolation)
    else:
        return cv2.resize(img, size[::-1], interpolation)
class RandomShufflePatch(object):
    def __init__(self, image_size, ratio=0.5, total_patch_num=9):
        self.ratio = ratio
        self.total_patch_num = total_patch_num
        self.patch_num = int(math.sqrt(self.total_patch_num))
        self.image_size = image_size
        self.patch_size = image_size // self.patch_num
        self.w_list, self.h_list = [], []
        for i in range(self.patch_num):
            if i == self.patch_num - 1:
                self.w_list.append([i * self.patch_size, image_size])
                self.h_list.append([i * self.patch_size, image_size])
            else:
                self.w_list.append([i * self.patch_size, (i + 1) * self.patch_size])
                self.h_list.append([i * self.patch_size, (i + 1) * self.patch_size])
        # print('RandomShufflePatch')

    def __call__(self, img):
        # print('RandomShufflePatch write img')
        img = np.array(img)
        h_list = self.h_list
        w_list = self.w_list
        if random.random() < self.ratio:
            return img

        shape = img.shape
        assert shape[0] == shape[1]
        assert shape[0] == self.image_size

        self.patches = []
        for i in range(self.patch_num):
            for j in range(self.patch_num):
                img_patch = img[h_list[i][0]:h_list[i][1], w_list[j][0]:w_list[j][1]]
                img_patch = resize(img_patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
                self.patches.append(img_patch)

        random.shuffle(self.patches)
        self.patches_2 = []
        for i in range(self.patch_num):
            self.patches_2.append(np.concatenate(self.patches[i * self.patch_num: (i + 1) * self.patch_num], axis=1))

        shuffled_img = np.concatenate(self.patches_2, axis=0)

        shuffled_img = resize(shuffled_img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        shuffled_img = Image.fromarray(shuffled_img)
        return shuffled_img

class YunpeiDataset1(Dataset):
    def __init__(self, data_pd, transforms=None, train=True):
        self.train = train
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        # 单独设置
        # 随机改变图像的亮度
        brightness_change = T.ColorJitter(brightness=0.5)
        # 随机改变图像的色调
        hue_change = T.ColorJitter(hue=0.5)
        # 随机改变图像的对比度
        contrast_change = T.ColorJitter(contrast=0.2)

        # if transforms is None:
        if not train:
            self.transforms = T.Compose([
                # T.CenterCrop(128),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            # self.transforms = T.Compose([
            #     T.RandomHorizontalFlip(),
            #     T.RandomVerticalFlip(),
            #     T.CenterCrop(128),
            #     # brightness_change,
            #     # hue_change,
            #     # contrast_change,
            #     T.ToTensor(),
            #     T.Normalize(mean=[0.485, 0.456, 0.406],
            #                 std=[0.229, 0.224, 0.225])
            # ])
            self.transforms = transforms
        # else:
        #     self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            img = Image.open(img_path)
            # face = crop_face_from_scene(img, img_path, scale=1.5)
            img = self.transforms(img)
            return img, label
        else:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            videoID = self.photo_belong_to_video_ID[item]
            img = Image.open(img_path)
            img = self.transforms(img)
            return img, label, videoID



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
        img[img > 255] = 255                       # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

class YunpeiDataset(Dataset):
    def __init__(self, data_pd, transforms=None, train=True):
        self.train = train
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist() if not train else None
        self.transform = transform
        # 单独设置
        # 随机改变图像的亮度
        brightness_change = T.ColorJitter(brightness=0.1)
        # 随机改变图像的色调
        hue_change = T.ColorJitter(hue=0.1)
        # 随机改变图像的对比度
        contrast_change = T.ColorJitter(contrast=0.1)
        

        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    # T.CenterCrop(100),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    AddGaussianNoise(mean=0, variance=1, amplitude=20),
                    # brightness_change,
                    # hue_change,
                    # contrast_change,
                    RandomShufflePatch(image_size=256),
                    # T.RandomVerticalFlip(),
                    # T.CenterCrop(100),
                    
                    T.ToTensor(),
                    
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    # T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
                ])


    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            img = Image.open(img_path)
            img = self.transforms(img)
            
            # print("xxxxxxx",img.shape)  #[3, 256, 256]
            return img, label
        else:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            videoID = self.photo_belong_to_video_ID[item]
            img = Image.open(img_path)
            img = self.transforms(img)

            return img, label, videoID


class YunpeiDataset1(Dataset):
    def __init__(self, data_pd, transforms=None, train=True):
        self.train = train
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        # 单独设置
        # 随机改变图像的亮度
        brightness_change = T.ColorJitter(brightness=0.1)
        # 随机改变图像的色调
        hue_change = T.ColorJitter(hue=0.1)
        # 随机改变图像的对比度
        contrast_change = T.ColorJitter(contrast=0.1)
        

        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    # T.CenterCrop(100),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    AddGaussianNoise(mean=0, variance=1, amplitude=20),
                    # brightness_change,
                    # hue_change,
                    # contrast_change,
                    # RandomShufflePatch(image_size=256),
                    # T.RandomVerticalFlip(),
                    # T.CenterCrop(100),
                    
                    T.ToTensor(),
                    
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    # T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
                ])


    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            img = Image.open(img_path)
            img = self.transforms(img)
            
            # print("xxxxxxx",img.shape)  #[3, 256, 256]
            return img, label
        else:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            videoID = self.photo_belong_to_video_ID[item]
            img = Image.open(img_path)
            img = self.transforms(img)

            return img, label, videoID
    

class YunpeiDataset2(Dataset):
    def __init__(self, data_pd, transforms=None, train=True):
        self.train = train
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()


        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    # T.CenterCrop(100),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                 
                    T.ToTensor(),
                    
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    # T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
                ])


    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            img = Image.open(img_path)
            img = self.transforms(img)
            
            # print("xxxxxxx",img.shape)  #[3, 256, 256]
            return img, label
        else:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            videoID = self.photo_belong_to_video_ID[item]
            img = Image.open(img_path)
            img = self.transforms(img)

            return img, label, videoID
    


