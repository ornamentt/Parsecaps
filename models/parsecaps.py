import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary
from icecream import ic
import math

from Utils.helpers import compute_conv,reshape_tensor
from layers.PrimaryCaps import *
# from layers.Convcaps_layer import *
from layers.pruned_layer import PruningLayer
from layers.Block import *



# class DeepUCaps(nn.Module):
   
#   def __init__(self, num_class, input_shape, similarity_threshold, kwargs_dict, 
#                 dim=[3, 8, 16, 32, 64], act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm, init='trunc_norm', device='cpu'):
#       super(DeepUCaps, self).__init__()
      
#       self.dim_caps = dim     # [3，8，16, 32, 64]
#       self.dim_original = [dim[2], dim[3], dim[4]]  # [16, 32, 64], same as paper cvt
#       self.B = input_shape[0]
#       self.num_classes = num_class
#       self.theta = similarity_threshold
#       self.device = device
#       self.mask = Mask_CID(self.device)
      
#       # get kwargs for each layer
#       kwargs = {}
#       for i in range(len(self.dim_original)):
#         kwargs[i] = {key: value[i] for key, value in kwargs_dict.items()}
      
#       n0,_,_ = compute_conv(input_shape[-1], input_shape[-1], 3, 2, 1, None)
#       n1,_,_ = compute_conv(n0, n0, 3, 2, 1, None)
#       n2,_,_ = compute_conv(n1, n1, 3, 2, 1, None)
#       n3,_,_ = compute_conv(n2, n2, 3, 2, 1, None)
      
#       self.ConvLayer = ConvLayer(input_channels=input_shape[1], filters=self.dim_caps[1], decrease_resolution=True, device=self.device)

#       self.Primary_Cap = PrimaryCaps(in_chans = self.dim_caps[1], dim_capsule = self.dim_caps[2], device=self.device, **kwargs[0])
#       ## Pruned Capsules 
#       # self.pruned_caps = PruningLayer(input_shape = [None, fp, 16, hp, wp], similarity_threshold=self.theta, device=self.device)
       
#       self.downblock1 = Block(in_chans=self.dim_caps[2], init=init, act_layer=act_layer, norm_layer=norm_layer, **kwargs[1])
#       self.downblock2 = Block(in_chans=self.dim_caps[3], init=init, act_layer=act_layer, norm_layer=norm_layer, **kwargs[2])
#       # self.downblock3 = Block(in_chans=self.in_chans[3], init=init, act_layer=act_layer, norm_layer=norm_layer, **kwargs[3])
      
#       self.norm = norm_layer(self.dim_original)
      
      
#       # self.upblock1 = UpSampleBlock(self.dim_caps[2], self.dim_caps[1], scale_factor=2, mode='nearest')
#       # self.upblock2 = UpSampleBlock(self.dim_caps[1], self.dim_caps[0], scale_factor=2, mode='nearest')
#       # self.upblock3 = UpSampleBlock(self.dim_caps[0], self.in_chans[0], scale_factor=2, mode='nearest', layer='recon')
#       # self.upconv = nn.Conv2d(in_channels=self.dim_caps[0], out_channels=self.in_chans[0], kernel_size=1, stride=1, padding=0)
      
#       self.FullyConvCaps = Fully(input_shape = [None,self.dim_caps[4],n3,n3],
#                                          num_classes=num_class, device=self.device)
        
#       # self.recon = nn.Sequential(
#       #             nn.Conv2d(in_channels=, out_channels=1, kernel_size=1, stride=1, padding=0),
#       #             nn.Sigmoid())

#       self.decoder = Decoder(input_size = input_shape, device = self.device)
#       self.act_layer = act_layer()
   
#   def forward(self, inputs, onehot_label):   
#     '''
#     downsample
#     '''   
#     conv = self.ConvLayer(inputs)             #([128, 8, 16, 16])
#     # ic(conv.shape)                
#     primary_cap= self.act_layer(self.Primary_Cap(conv))   #([128, 32, 8, 8])
#     # ic(primary_cap.shape)
#     # pruned_cap = self.pruned_caps(primary_cap)
#     x1 = self.downblock1(primary_cap)         #([128, 32, 4, 4])
#     # ic(x1.shape)
#     x2 = self.downblock2(x1)                  #([128, 64, 2, 2])
#     # ic(x2.shape)
#     # x3 = self.downblock3(x2)
#     # x = x2.reshape(self.B, x2.size(2)*x2.size(3), x2.size(1))
    
#     x_fully = self.act_layer(self.FullyConvCaps(x2))
    
#     # outputs = x_fully.norm(dim=-1).squeeze(-1)
#     outputs = x_fully
#     # ic(outputs.shape)          #([128, 10, 64])
#     # ic(onehot_label.shape)     #([128, 10])
#     masked, indices, classes = self.mask(outputs, onehot_label)
#     # ic(masked.shape)           #([128, 1, 64])
#     # ic(indices.shape)          #([128])
#     # ic(classes.shape)          #([128, 10])
    
#     # '''
#     # upsample
#     # '''
#     # # x_3 = self.upblock1(x3, x2)
#     # # ic(x_3.shape)
#     # masked = masked.permute(0,2,1).unsqueeze(-1)  #([128, 64, 1, 1])
#     # # ic(masked.shape)
#     # x_m = self.upblock0(masked, x2)
#     # x_2 = self.upblock1(x2, x1)               #([128, 32, 8, 8])
#     # # ic(x_2.shape)
#     # x_1 = self.upblock2(x_2, primary_cap)     #([128, 16, 16, 16])
#     # # ic(x_1.shape)
#     # x_recon = self.upblock3(x_1)  
#     x_recon = None     
    
#     return indices, x_recon, classes
#     # return outputs
    

class ParseCaps(nn.Module):
   
  def __init__(self, num_class, input_shape, conceptnum, kwargs_dict, 
                dim=[3, 8, 16, 32, 64], act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, init='trunc_norm', device='cuda'):
      super(ParseCaps, self).__init__()
      
      self.dim_caps = dim     # [3，8，16, 32, 64]
      self.dim_original = [dim[2], dim[3], dim[4]]  # [16, 32, 64]
      self.B = input_shape[0]
      self.num_classes = num_class
      self.device = device
      self.mask = Mask_CID(self.device)
      
      
      # get kwargs for each layer
      kwargs = {}
      for i in range(len(self.dim_original)):
        kwargs[i] = {key: value[i] for key, value in kwargs_dict.items()}
      
      n0,_,_ = compute_conv(input_shape[-1], input_shape[-1], 3, 2, 1, None)
      n1,_,_ = compute_conv(n0, n0, 3, 2, 1, None)
      n2,_,_ = compute_conv(n1, n1, 3, 2, 1, None)
      n3,_,_ = compute_conv(n2, n2, 3, 2, 1, None)
      
      self.ConvLayer = ConvLayer(input_channels=input_shape[1], filters=self.dim_caps[1], decrease_resolution=True, device=self.device)

      self.Primary_Cap = PrimaryCaps(in_chans = self.dim_caps[1], dim_capsule = self.dim_caps[2], device=self.device, **kwargs[0])
      ## Pruned Capsules 
      # self.pruned_caps = PruningLayer(input_shape = [None, fp, 16, hp, wp], similarity_threshold=self.theta, device=self.device)
       
      self.downblock1 = Block(in_chans=self.dim_caps[2], init=init, act_layer=act_layer, norm_layer=norm_layer, **kwargs[1])
      self.downblock2 = Block(in_chans=self.dim_caps[3], init=init, act_layer=act_layer, norm_layer=norm_layer, **kwargs[2])
      
      self.norm = norm_layer(self.dim_original)
  
      
      self.FullyConvCaps = Fully(input_shape = [None,self.dim_caps[4],n3,n3],
                                         num_classes=num_class, device=self.device)
        

      # self.decoder = Decoder(self.dim_caps[4], self.device, num_caps=1, img_size=input_shape[-1], img_channels=input_shape[1])
      self.act_layer = act_layer()
      
      self.concept_layer = nn.Linear(self.dim_caps[4], 10)
      # self.concept_layer = ConceptLayer(n3, self.dim_caps[4], conceptnum, True, self.device)
   
  def forward(self, inputs, onehot_label):   
    '''
    downsample
    '''   
    conv = self.ConvLayer(inputs)            
    # ic(conv.shape)                
    primary_cap= self.act_layer(self.Primary_Cap(conv))   
    # ic(primary_cap.shape)
    # pruned_cap = self.pruned_caps(primary_cap)
    x1 = self.downblock1(primary_cap)         
    # ic(x1.shape)
    x2 = self.downblock2(x1)                  
    # ic(x2.shape)

    
    x_fully = self.act_layer(self.FullyConvCaps(x2))
    
    # outputs = x_fully.norm(dim=-1).squeeze(-1)
    outputs = x_fully
    # ic(outputs.shape)          
    # ic(onehot_label.shape)    
    masked, indices, classes = self.mask(outputs, onehot_label)
    # ic(masked.shape)           
    concept = self.concept_layer(masked)
 
    # x_recon = self.decoder(masked) 
    x_recon = None    
    
    return indices, x_recon, classes, concept
    # return outputs
    
    


    