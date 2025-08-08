import torch
import torch.nn as nn
def Real_AdLoss(discriminator_out, criterion, shape_list):
    # generate ad_label
    ad_label1_index = torch.LongTensor(shape_list[0], 1).fill_(0)
    ad_label1 = ad_label1_index.cuda()
    # ad_label2_index = torch.LongTensor(shape_list[1], 1).fill_(1)
    # ad_label2 = ad_label2_index.cuda()
    # ad_label3_index = torch.LongTensor(shape_list[2], 1).fill_(2)
    # ad_label3 = ad_label3_index.cuda()
    # ad_label = torch.cat([ad_label1, ad_label2, ad_label3], dim=0).view(-1)
    ad_label = torch.cat([ad_label1], dim=0).view(-1)
    real_adloss = criterion(discriminator_out, ad_label)
    return real_adloss
def Real_AdLoss1(discriminator_out, criterion, shape_list):
    # generate ad_label
    ad_label1_index = torch.LongTensor(shape_list[0], 1).fill_(0)
    ad_label1 = ad_label1_index.cuda()
    ad_label2_index = torch.LongTensor(shape_list[1], 1).fill_(1)
    ad_label2 = ad_label2_index.cuda()
    ad_label3_index = torch.LongTensor(shape_list[2], 1).fill_(2)
    ad_label3 = ad_label3_index.cuda()
    ad_label = torch.cat([ad_label1, ad_label2, ad_label3], dim=0).view(-1)
    
    real_adloss = criterion(discriminator_out, ad_label)
    return real_adloss
def Real_AdLoss2(discriminator_out, criterion, shape_list):
    # generate ad_label
    ad_label1_index = torch.LongTensor(shape_list[0], 1).fill_(0)
    ad_label1 = ad_label1_index.cuda()
    ad_label2_index = torch.LongTensor(shape_list[1], 1).fill_(1)
    ad_label2 = ad_label2_index.cuda()
    
    ad_label = torch.cat([ad_label1, ad_label2], dim=0).view(-1)
    
    real_adloss = criterion(discriminator_out, ad_label)
    return real_adloss

def Fake_AdLoss(discriminator_out, criterion, shape_list):
    # generate ad_label
    ad_label1_index = torch.LongTensor(shape_list[0], 1).fill_(0)
    ad_label1 = ad_label1_index.cuda()
    ad_label2_index = torch.LongTensor(shape_list[1], 1).fill_(1)
    ad_label2 = ad_label2_index.cuda()
    ad_label3_index = torch.LongTensor(shape_list[2], 1).fill_(2)
    ad_label3 = ad_label3_index.cuda()
    ad_label = torch.cat([ad_label1, ad_label2, ad_label3], dim=0).view(-1)

    fake_adloss = criterion(discriminator_out, ad_label)
    return fake_adloss

def AdLoss_Limited(discriminator_out, criterion, shape_list):
    # generate ad_label
    ad_label2_index = torch.LongTensor(shape_list[0], 1).fill_(0)
    ad_label2 = ad_label2_index.cuda()
    ad_label3_index = torch.LongTensor(shape_list[1], 1).fill_(1)
    ad_label3 = ad_label3_index.cuda()
    ad_label = torch.cat([ad_label2, ad_label3], dim=0).view(-1)

    real_adloss = criterion(discriminator_out, ad_label)
    return real_adloss


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=2, feat_dim=512, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # x = x.size(-1) 
        # print(x.shape)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
