# coding=gb2312

import torch
import torch.nn.functional as F
from torch import nn
from icecream import ic


class CapsuleLossMarginLoss(nn.Module):
    def __init__(self, reconstruction=True):
        super(CapsuleLossMarginLoss, self).__init__()
        self.reconstruction = reconstruction
        if reconstruction:
            self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, labels, classes, images,  reconstructions):
        
        classes = torch.norm(classes, dim=2, keepdim=False)
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        # ic(left.size())         # (B*10)
        # ic(right.size())        # (B*10)
        # ic(labels.size())       # (B*10)
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        if not self.reconstruction:
            return margin_loss / images.size(0) 
        
        # ic(images.size())       # (64*1*32*32)
        # images = images.view(reconstructions.size()[0], -1)     # (64*1024)
        assert torch.numel(images) == torch.numel(reconstructions)
        # ic(images.size())
        # ic(reconstructions.size())
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        return (margin_loss + 0.0005* reconstruction_loss) / images.size(0)


class CapsuleLossCrossEntropy(nn.Module):
    def __init__(self, rl, reconstruction=True):
        super(CapsuleLossCrossEntropy, self).__init__()
        self.reconstruction = reconstruction
        self.cross_entropy = nn.CrossEntropyLoss()
        if reconstruction:
            # self.reconstruction_loss = nn.MSELoss(size_average=False)
            self.reconstruction_loss = nn.MSELoss(reduction='sum')
            self.rl = rl

    def forward(self, labels, classes, images,  reconstructions):
        # targets = torch.argmax(labels, dim=1)
        targets = labels
        # classes = torch.norm(classes, dim=2, keepdim=False)
        
        if classes.dtype != torch.float32:
            classes = classes.float()  
        targets = targets.float()
            
        print("Classes shape:", classes.shape, "Classes dtype:", classes.dtype)
        print("Targets shape:", targets.shape, "Targets dtype:", targets.dtype)
        

        
        loss = self.cross_entropy(classes,targets)
        if not self.reconstruction:
            return loss 
        
        # ic(images.size())
        # ic(reconstructions.size())
        
        assert torch.numel(images) == torch.numel(reconstructions)
        # images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (loss + self.rl * reconstruction_loss) / images.size(0)
    
def caps_loss(y_true, y_pred, x=None, x_recon=None, lam_recon=0.0005):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
  
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    
    
    # ic(y_true.size())
    # ic(y_pred.size())
    # ic(x.size())
    
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()
    
    # print("x range: ", x.min().item(), " to ", x.max().item())
    # print("x_recon range: ", x_recon.min().item(), " to ", x_recon.max().item())
    
    # L_recon = nn.MSELoss()(x_recon, x)
    
    # print("L_margin: ", L_margin.item(), " L_recon: ", L_recon.item())
    # return L_margin + lam_recon * L_recon
    return L_margin