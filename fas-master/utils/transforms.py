# fas/utils/transforms.py
import torchvision.transforms as T
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .dataset import AddGaussianNoise, RandomShufflePatch


def build_transform(args, is_train=True):
    """构建数据增强管道（兼容PVT风格的配置驱动）"""
    input_size = args.input_size
    if is_train:
        # 训练增强：整合原有FAS增强策略 + PVT常用增强
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        # 插入FAS特有的增强（随机补丁打乱和高斯噪声）
        transform.transforms.insert(0, T.RandomHorizontalFlip(p=0.5))
        transform.transforms.insert(1, RandomShufflePatch(image_size=input_size))
        transform.transforms.insert(2, AddGaussianNoise(mean=0, variance=1, amplitude=20))
        return transform

        # 验证增强：仅 resize 和标准化
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=3),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])