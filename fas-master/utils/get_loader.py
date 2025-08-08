# fas/utils/get_loader.py
import os
import torch
from torch.utils.data import DataLoader
from .dataset import FASDataset
from .transforms import build_transform
from utils.utils import is_dist_avail_and_initialized, get_world_size, get_rank


def build_dataset(args, is_train=True):
    """构建数据集（兼容ImageNet格式）"""
    transform = build_transform(args, is_train)
    dataset = FASDataset(
        root=args.data_path,
        transform=transform,
        is_train=is_train
    )
    return dataset, 2  # FAS任务固定2类（real/spoof）


def build_loader(args):
    """构建数据加载器（支持分布式）"""
    dataset_train, num_classes = build_dataset(args, is_train=True)
    dataset_val, _ = build_dataset(args, is_train=False)

    # 分布式采样器配置
    if is_dist_avail_and_initialized():
        num_tasks = get_world_size()
        global_rank = get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 训练加载器
    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # 验证加载器
    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    return data_loader_train, data_loader_val, num_classes