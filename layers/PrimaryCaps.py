import torch
import torch.nn as nn
from icecream import ic
from einops import rearrange
from einops.layers.torch import Rearrange
import math

from layers.Block import Cell


class PrimaryCaps(nn.Module):
      
      # 对应第一个ConvEmbed，其stride=4，padding=2，kernel_size=7
      
    def __init__(self, in_chans, dim_capsule=16,
                 depth=1,
                 num_heads=16,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 device='cpu',
                 patch_size = 7,
                 patch_stride = 4,
                 patch_padding = 2,
                 **kwargs):
      super(PrimaryCaps, self).__init__()
      '''
      input_shape: [None, channel, w, h]
      n = H * W, d = C (verifiable, not constant)
      output_shape: [None, capsule_number, capsule_dim]
      '''
      self.dim_capsule = dim_capsule
      self.device = device
      self.input_num_features = in_chans
      # ic(input_shape)
      # ic(self.input_num_features)
      self.DW_Conv2D = nn.Sequential(
              nn.Conv2d(in_channels=self.input_num_features, out_channels=self.input_num_features,
                        kernel_size=patch_size, stride=patch_stride, padding=patch_padding, groups=self.input_num_features),
              nn.LeakyReLU(),
              # nn.ReLU(),
              nn.Conv2d(in_channels=self.input_num_features, out_channels=self.dim_capsule,
                        kernel_size=1))
      
      self.pos_drop = nn.Dropout(p=drop_rate)
      dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
      cells = []
      for j in range(depth):
          cells.append(
              Cell(
                  dim_in=self.dim_capsule,
                  dim_out=self.dim_capsule,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[j],
                  act_layer=act_layer,
                  norm_layer=norm_layer,
                  **kwargs
              )
          )
        #将堆叠的cells组成神经网络
      self.cells = nn.ModuleList(cells) 
      
    def forward(self, inputs):   # [B, 3, 224, 224]  
      conv1s = self.DW_Conv2D(inputs)      #([B, 16, 56, 56])
      B, C, H, W = conv1s.size()
      # ic(conv1s.size())
      x = conv1s.view(B, H*W, C)     #([B, 3136, 16])
      # ic(x.size())
      x = self.pos_drop(x)
      for i, cel in enumerate(self.cells):
          x = cel(x)
      outputs = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)      
      return outputs
  
  
# class PrimaryCap_91(nn.Module):
#     def __init__(self,input_shape, n_channels, dim_capsule,device='cpu'):
#       super(PrimaryCap_91, self).__init__()
#       '''
#       input_shape: [None, cahnnel, w, h]
#       output_shape: [None, capsule_number, capsule_dim, h, w]
#       '''
#       self.dim_capsule = dim_capsule
#       self.n_channels = n_channels
#       self.device = device
      
#       self.input_height = input_shape[2]
#       self.input_width = input_shape[3]
#       self.input_num_features = input_shape[1]
      
#       #  9*9 convolution with stride 1 padding 0
      
#       self.DW_Conv2D = nn.Sequential(
#               nn.Conv2d(in_channels=self.input_num_features, out_channels=self.input_num_features, 
#                         kernel_size=9, stride=1, padding=0, groups=self.input_num_features),
#               nn.LeakyReLU(),
#               # nn.ReLU(),
#               nn.Conv2d(in_channels=self.input_num_features, out_channels=self.dim_capsule*self.n_channels, 
#                         kernel_size=1))
      
      
#     def forward(self, inputs):     
      
#       conv1s = self.DW_Conv2D(inputs)
#       outputs = conv1s.view(conv1s.size(0), self.n_channels, self.dim_capsule, conv1s.size(-2), conv1s.size(-1))

#       return outputs

class PrimaryCap_32(nn.Module):
    def __init__(self,input_shape, n_channels, dim_capsule,device='cpu'):
      super(PrimaryCap_32, self).__init__()
      '''
      input_shape: [None, channel, w, h]
      output_shape: [None, capsule_number, capsule_dim, h, w]
      '''
      self.dim_capsule = dim_capsule
      self.n_channels = n_channels
      self.device = device
      
      # self.input_height = input_shape[2]
      # self.input_width = input_shape[3]
      self.input_num_features = input_shape[1]
      
      #  3*3 convolution with stride 2 padding 1
      self.DW_Conv2D = nn.Sequential(
              nn.Conv2d(in_channels=self.input_num_features, out_channels=self.input_num_features,
                        kernel_size=3, stride=2, padding=1, groups=self.input_num_features),
              nn.LeakyReLU(),
              # nn.ReLU(),
              nn.Conv2d(in_channels=self.input_num_features, out_channels=self.dim_capsule*self.n_channels,
                        kernel_size=1))
      
    def forward(self, inputs):     
      
      conv1s = self.DW_Conv2D(inputs)      #([B, 128, 112, 112])
      # ic(conv1s.size())
      outputs = conv1s.view(conv1s.size(0), self.n_channels, self.dim_capsule, conv1s.size(-2), conv1s.size(-1))     #([B, 8, 16, 112, 112])
      # ic(outputs.size())
      return outputs
  
  
# if __name__ == '__main__':
#     x = torch.randn(1, 3, 224, 224)
#     primarycaps = PrimaryCaps(input_shape=x.shape, dim_capsule=16)
#     y = primarycaps(x)
#     ic(y.size())
    