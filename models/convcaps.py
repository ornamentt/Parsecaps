import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary
from icecream import ic

from Utils.helpers import compute_conv,reshape_tensor
from layers.PrimaryCaps import *
from layers.Convcaps_layer import *
from layers.pruned_layer import PruningLayer
from layers.Block import *
      
  
class UCapsuleNet(nn.Module):
   
  def __init__(self, num_class, input_shape, i_c, similarity_threshold, activation, device='cpu' ):
      super(UCapsuleNet, self).__init__()
      
      self.dim_caps = 16
      self.B = input_shape[0]
      self.i_w = input_shape[-1]
      self.in_channel = i_c
      self.num_classes = num_class
      self.theta = similarity_threshold
      self.device = device
      self.mask = Mask_CID(self.device)

      self.convlayer = ConvLayer(input_channels=self.in_channel,decrease_resolution=True, filters=128,device=self.device)
      h2, w2, f2 = compute_conv(input_shape[-2], input_shape[-1], 3, 1, 'same', 128)
      self.Primary_Cap = PrimaryCap_32( input_shape = [None,f2, w2,w2], n_channels = 8, dim_capsule = self.dim_caps, device=self.device)
      hp,wp,fp = compute_conv(h2, w2, 3, 2, 1, 8)
      # ic(input_shape)
      # ic(fp,hp,wp)
       ## Pruned Capsules 
      self.pruned_caps = PruningLayer(input_shape = [None, fp, 16, hp, wp], similarity_threshold=self.theta, device=self.device)
       
       ## Convolutional Capsules
      self.cell1 = Add_ConvCaps(input_shape = [None, fp, 16, hp, wp], layer_num = 's',dim_capsule = self.dim_caps,device=self.device)
      self.cell2 = Add_ConvCaps(input_shape = [None, 2*fp, 16, hp/2, wp/2], layer_num = 's', dim_capsule = self.dim_caps,device=self.device)
      self.cell3 = Add_ConvCaps(input_shape = [None, 4*fp, 16, hp/4, wp/4], layer_num = 's',dim_capsule = self.dim_caps,device=self.device) 
      self.cell4 = Add_ConvCaps(input_shape = [None, 8*fp, 16, hp/8, wp/8], layer_num = 's',dim_capsule = self.dim_caps,device=self.device)
       
       
       #TODO: UPSAMPLE OR CONVTRANSPOSE
      #  self.shortcut = nn.Sequential(
      #     nn.ConvTranspose2d(in_channels=128, out_channels=fp*16, kernel_size=1, stride=2, padding=1, output_padding=1),
      #     nn.BatchNorm2d(fp*16))
       
      self.shortcut1 = nn.Sequential(
          nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
          nn.Conv2d(16*16, 8*16, kernel_size=1),
          nn.BatchNorm2d(fp*16))                                                   # fp = 8
      self.shortcut2 = nn.Sequential(
          nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
          nn.Conv2d(32*16, 16*16, kernel_size=1),
          nn.BatchNorm2d(fp*32))
      self.shortcut3 = nn.Sequential(
          nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
          nn.Conv2d(64*16, 32*16, kernel_size=1),
          nn.BatchNorm2d(fp*64))
      self.shortcut4 = nn.Sequential(
          nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
          nn.Conv2d(128*16, 64*16, kernel_size=1),
          nn.BatchNorm2d(fp*128))
       
      self.FullyConvCaps = FullyConvCaps(input_shape = [None, 16*fp, 16, hp/16, wp/16],
                                         num_classes=num_class, activation = activation, device=self.device)   
      # if bugs appear, change the in_channels to what tips say
      self.recon = nn.Sequential(
                  nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
                  nn.Sigmoid())

      # self.decoder = Decoder(input_size = input_shape, device = self.device)
      self.ReLU = nn.ReLU()
   
  def forward(self, inputs, onehot_label):
     
    conv = self.convlayer(inputs)                         #([B, 128, 30, 30])
    # ic(conv.shape)
    # primary_cap= self.LeakyReLU(self.Primary_Cap(conv))
    primary_cap= self.ReLU(self.Primary_Cap(conv))
    # pruned_cap = self.pruned_caps(primary_cap)
  
    x1 = self.cell1(primary_cap)
    x2 = self.cell2(x1)
    x3 = self.cell3(x2)
    x4 = self.cell4(x3)
    x_4 = x4.view(x4.size(0), x4.size(1)*x4.size(2), x4.size(3), x4.size(4))                   #([B, 128*16, 8, 8])
    # ic(x_4.shape)
    
    x_3 = self.shortcut4(x_4).view(x3.shape)                                                   #([B, 64, 16, 16, 16])
    # ic(x_3.shape)
    # ic(x3.shape)
    x_3_skip = x_3 + x3
    x_3_skip = x_3_skip.view(x_3_skip.size(0), x_3_skip.size(1)*x_3_skip.size(2), x_3_skip.size(3), x_3_skip.size(4))
    
    x_2 = self.shortcut3(x_3_skip).view(x2.shape)                                              #(B, 32, 16, 32, 32)
    # ic(x_2.shape)
    x_2_skip = x_2 + x2
    x_2_skip = x_2_skip.view(x_2_skip.size(0), x_2_skip.size(1)*x_2_skip.size(2), x_2_skip.size(3), x_2_skip.size(4))
    
    x_1 = self.shortcut2(x_2_skip).view(x1.shape)                                              #(B, 16, 16, 64, 64)
    # ic(x_1.shape)
    x_1_skip = x_1 + x1 
    x = x_1_skip.view(x_1_skip.size(0), -1, self.i_w, self.i_w)                                #(B, 4, 512, 512)
     
    # x = self.shortcut1(x_1_skip).view(primary_cap.shape)                                        #(B, 8, 16, 128, 128)
    # ic(x.shape)
    
    x_fully = self.ReLU(self.FullyConvCaps(x4))
    
    outputs = x_fully.norm(dim=-1).squeeze(-1)
      # ic(outputs.shape)          #([128, 10, 16])
      # ic(onehot_label.shape)     #([128, 10])
    masked, indices, classes = self.mask(outputs, onehot_label)
      # ic(masked.shape)           #([128, 1, 16])
      # ic(indices.shape)          #([128, 1])
      # ic(classes.shape)          #([128, 10, 1])
    x_recon = self.recon(x)                                      #([B, 1, 512, 512])
    # x_recon = None
    # ic(x_recon.shape)
        
    return outputs, indices.squeeze(-1), x_recon, classes.squeeze(-1)

  
  
class ShallowNet(nn.Module):
   
  def __init__(self, num_class, input_shape, layernum, i_c, similarity, activation ,device='cpu'):
       super(ShallowNet, self).__init__()
       dim_caps = 16
       self.layernum = layernum
       self.in_channel = i_c
       self.num_classes = num_class
       self.theta = similarity
       self.activation = activation
       self.device = device
       self.mask = Mask_CID(self.device)
       
      #  self.BN1 = Conv2d_bn(input_shape = input_shape,input_channels=self.in_channel,filters=64,kernel_size=3, strides=1, padding='same',device=self.device)
      #  h1, w1, f1 = compute_conv(input_shape[-2], input_shape[-1], 3, 1, 'same', 64)
      # #  ic(f1,h1,w1)
      #  self.BN2 = Conv2d_bn(input_shape = [None,f1,h1,w1],input_channels=64,filters=64,kernel_size=3, strides=1, padding='same',device=self.device)
      #  h2, w2, f2 = compute_conv(h1, w1, 3, 1, 'same', 64)
      # #  ic(f2,h2,w2)
       
       self.convlayer = ConvLayer_s(input_channels=self.in_channel,decrease_resolution=True, filters=128,device=self.device)
       self.Primary_Cap = PrimaryCap_32( input_shape = [None,128, 30, 30], n_channels = 8, dim_capsule = dim_caps, device=self.device)


       ## Convolutional Capsules
       if self.layernum == 0:
         self.FullyConvCaps = FullyConvCap(input_shape = [None, 8, 16, 15, 15],
                                          num_classes=num_class, activation = activation, device=self.device) 
        
       elif self.layernum == 1:
         self.ConvCaps = ConvCaps(input_shape =[None, 8, 16, 15, 15],n_channels=8, dim_capsule=dim_caps, decrease_resolution = True)
         self.FullyConvCaps = FullyConvCap(input_shape = [None, 8, 16, 15, 15],
                                          num_classes=num_class, activation = activation, device=self.device) 
         
       elif self.layernum == 3:
         self.ConvCaps_3 =  Add_ConvCaps(input_shape = [None, 8, 16, 15, 15], layer_num = 's',dim_capsule = dim_caps, activation= activation ,device=self.device)
         self.FullyConvCaps = FullyConvCap(input_shape = [None, 16, 16, 8, 8],
                                          num_classes=num_class, activation = activation, device=self.device) 
         
       else:
          raise NotImplementedError('Unknown layer_num: {}'.format(self.layernum))
         
      #  self.decoder = Decoder(input_size = input_shape, dim=dim_caps, device = self.device)
      #  self.LeakyReLU = nn.LeakyReLU()
       self.ReLU = nn.ReLU()

   
  def forward(self, inputs, onehot_label=None):
     
      conv = self.convlayer(inputs)
    #   ic(conv.shape)
      # primary_cap= self.LeakyReLU(self.Primary_Cap(conv))
      primary_cap= self.ReLU(self.Primary_Cap(conv))

    #   ic(primary_cap.shape)
      
        ## Convolutional Capsules
      if self.layernum == 0:
          # FullyConvCaps = self.LeakyReLU(self.FullyConvCaps(primary_cap))
          FullyConvCaps = self.ReLU(self.FullyConvCaps(primary_cap))

          # ic(FullyConvCaps.shape)   #([128, 10, 16, 1, 1])
      elif self.layernum == 1:
          # ConvCaps = self.LeakyReLU(self.ConvCaps(primary_cap))
          # FullyConvCaps = self.LeakyReLU(self.FullyConvCaps(ConvCaps))
          ConvCaps = self.ReLU(self.ConvCaps(primary_cap))
          FullyConvCaps = self.ReLU(self.FullyConvCaps(ConvCaps))

      elif self.layernum == 3:
          ConvCaps = self.ConvCaps_3(primary_cap)
        #   ic(ConvCaps.shape)
          # FullyConvCaps = self.LeakyReLU(self.FullyConvCaps(ConvCaps)) 
          FullyConvCaps = self.ReLU(self.FullyConvCaps(ConvCaps))
        #   ic(FullyConvCaps.shape)  
 
        
      outputs = FullyConvCaps.norm(dim=-1).squeeze(-1)
      # ic(outputs.shape)          #([128, 10, 16])
      # ic(onehot_label.shape)     #([128, 10])
      masked, indices, classes = self.mask(outputs, onehot_label)
      # ic(masked.shape)           #([128, 1, 16])
      # ic(indices.shape)          #([128, 1])
      # ic(classes.shape)          #([128, 10, 1])
      # x_recon = self.decoder(masked)
      x_recon = None
        
      return indices.squeeze(-1), x_recon, classes.squeeze(-1), classes.squeeze(-1)