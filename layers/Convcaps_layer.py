# 仅含有前置卷积和fullycaps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from icecream import ic
from layers.Orth_attention_routing import simple_orth_attention, deep_orth_attention

class ConvLayer(nn.Module):
    def __init__(self,input_channels,filters,decrease_resolution=True, layer = 4, device='cpu'):
        super(ConvLayer, self).__init__()
        self.device = device
        self.layer = layer
        
        if decrease_resolution == True:
          self.stride = 2
        else:
          self.stride = 1
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=5, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        # self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        # # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        # self.bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=filters, kernel_size=3, stride=self.stride, padding=1)
        # # self.bn4 = nn.BatchNorm2d(filters, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        # self.bn4 = nn.BatchNorm2d(filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = x.to(self.device)
          # x = F.leaky_relu(self.bn1(self.conv1(x)))
          # x = F.leaky_relu(self.bn2(self.conv2(x)))
          # x = F.leaky_relu(self.bn3(self.conv3(x)))
          # x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))
        # ic(x.size())                      #([32, 128, 30, 30])
        return x

class ConvLayer_s(nn.Module):
    def __init__(self,input_channels,filters,decrease_resolution=True, layer = 4, device='cpu'):
        super(ConvLayer_s, self).__init__()
        self.device = device
        self.layer = layer
        
        if decrease_resolution == True:
          self.stride = 2
        else:
          self.stride = 1
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=1, padding='same')
        # self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # self.conv4 = nn.Conv2d(in_channels=128, out_channels=filters, kernel_size=3, stride=self.stride, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=filters, kernel_size=3, stride=self.stride, padding=1)
        self.bn4 = nn.BatchNorm2d(filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # ic(x.size())                      #([32, 128, 30, 30])
        return x
 
class ConvCaps(nn.Module):
    def __init__(self,input_shape, n_channels, dim_capsule,  device='cpu', decrease_resolution=False, kernel_size = 3):
      super(ConvCaps, self).__init__()
      self.num_capsule = n_channels
      self.dim_capsule = dim_capsule
      self.kernel_size = kernel_size
      self.device = device
      
      if decrease_resolution == True:
        self.stride = 2
      else:
          self.stride = 1
          
      # n_channels：capsule number of next layer
      # input_ch：capsule number of this layer
      
      self.height = input_shape[3]
      self.width = input_shape[4]
      self.input_dim = input_shape[2]
      self.input_ch = input_shape[1]

      Att_W = torch.empty(self.input_ch*self.num_capsule,self.input_ch,self.dim_capsule,1, 1)
      self.Att_W  = nn.init.xavier_uniform_(nn.Parameter(Att_W), gain=nn.init.calculate_gain('relu'))
      
      self.depthwise_conv = nn.Conv2d(self.dim_capsule, self.dim_capsule, kernel_size=self.kernel_size, stride=self.stride, padding=1, groups=self.dim_capsule) 
      self.pointwise_conv = nn.Conv2d(self.dim_capsule, self.dim_capsule * self.num_capsule, kernel_size=1)

    def forward(self, inputs):
        
        input_caps = torch.chunk(inputs, self.input_ch, dim=1)
        # ic(input_caps[0].size())
        input_caps = [torch.squeeze(input_cap, dim=1) for input_cap in input_caps]
      
        all_conv_results = []
        for input_cap in input_caps:
            # Depthwise convolution
            depthwise_result = self.depthwise_conv(input_cap)
            # ic(depthwise_result.size())
            pointwise_result = self.pointwise_conv(depthwise_result)
              # ic(pointwise_result.size())
            reshaped_result = pointwise_result.view(pointwise_result.size(0), self.num_capsule, self.dim_capsule, depthwise_result.size(-2), depthwise_result.size(-1))
            all_conv_results.append(reshaped_result.unsqueeze(2))
        conv1s = torch.cat(all_conv_results, dim=2)

        return conv1s


class Add_ConvCaps(nn.Module):
    '''
    total stride = 2
    '''
    def __init__(self,input_shape,layer_num,dim_capsule,activation='squash',device='cpu'):
      super(Add_ConvCaps, self).__init__()
      self.layer_num = layer_num
      height = input_shape[3]
      width = input_shape[4]
      input_dim = input_shape[2]
      input_ch = input_shape[1]
      self.activation = activation
      self.device = device
      
      # ic(input_shape)
      
      # 3*3, 3*3
      self.ConvCaps11 = ConvCaps(input_shape = input_shape,n_channels=2*input_ch, dim_capsule=dim_capsule, decrease_resolution = True,device=self.device)
      self.ConvCaps12 =   ConvCaps(input_shape = [None, 2*input_ch, input_dim, int(height/2), int(width/2)],n_channels=2*input_ch, dim_capsule=dim_capsule, decrease_resolution = False,device=self.device)
      
      self.ConvCaps21 =   ConvCaps(input_shape = [None, input_ch, input_dim, int(height), int(width)],n_channels=input_ch, dim_capsule=dim_capsule, decrease_resolution = True,device=self.device)
      self.ConvCaps22 =   ConvCaps(input_shape = [None, input_ch, input_dim, int(height/2), int(width/2)],n_channels=input_ch, dim_capsule=dim_capsule, decrease_resolution = False,device=self.device)
      
      
      self.attention_routing0 = deep_orth_attention(input_ch, input_ch, input_dim, 
                              [None, input_ch, input_dim, int(height/2), int(width/2)], device=self.device,activation=self.activation).to(self.device)
      self.attention_routing1 = deep_orth_attention(input_ch, input_ch*2, input_dim, 
                              [None, input_ch, input_dim, int(height/2), int(width/2)], device=self.device,activation=self.activation).to(self.device)
      self.attention_routing2 = deep_orth_attention(input_ch*2, input_ch*2, input_dim, 
                              [None, input_ch, input_dim, int(height/2), int(width/2)], device=self.device,activation=self.activation).to(self.device)
      
      self.attention_routing3 = simple_orth_attention(input_ch, input_ch, input_dim, 
                              [None, input_ch, input_dim, int(height/2), int(width/2)], device=self.device,activation=self.activation).to(self.device)
      self.attention_routing4 = simple_orth_attention(input_ch, input_ch*2, input_dim, 
                              [None, input_ch, input_dim, int(height/2), int(width/2)], device=self.device,activation=self.activation).to(self.device)
      self.attention_routing5 = simple_orth_attention(input_ch*2, input_ch*2, input_dim, 
                              [None, input_ch, input_dim, int(height/2), int(width/2)], device=self.device,activation=self.activation).to(self.device)
      
      self.shortcut1 = nn.Sequential(
          nn.Conv2d(in_channels=input_ch*input_dim, out_channels=input_ch*input_dim*2, kernel_size=1, stride=2, padding=0),
          # nn.BatchNorm2d(input_ch*input_dim*2, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True))
          nn.BatchNorm2d(input_ch*input_dim*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
      
      # self.shortcut2 = nn.Sequential(
      #    nn.PixelShuffle(2),
      #    nn.BatchNorm2d(input_ch*input_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
      
      # self.shortcut2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    
    def forward(self,inputs):
      if self.layer_num== 's':
        '''
         (B,N,D,H,W) -> (B,2*N,D,H/2,W/2)
        '''
        C1 = self.ConvCaps11(inputs)                  #([B, 16, 8, 16, 12, 12])
        C1F = F.leaky_relu(C1)
        # C1F = F.relu(C1)                                                  
        C1A = self.attention_routing1(C1F)            #([B, 16, 16, 12, 12])
                                    
        C2 = self.ConvCaps12(C1A)                     #([B, 16, 16, 12, 12, 12])
        C2F = F.leaky_relu(C2)
        # C2F = F.relu(C2)   
        C2A = self.attention_routing2(C2F)
      
        inputs = inputs.view(inputs.size(0), inputs.size(1)*inputs.size(2), inputs.size(-2), inputs.size(-1))
        INPUTS = self.shortcut1(inputs).view(C2A.shape)
        output = C2A+INPUTS
        return output
        
      elif self.layer_num == 'r':
        ConvCaps1 = F.leaky_relu(self.ConvCaps21(inputs))
        # ConvCaps1 = F.relu(self.ConvCaps21(inputs))

        ConvCaps1 = self.attention_routing0(ConvCaps1)
        ConvCaps2 = self.ConvCaps22(ConvCaps1)
        ConvCaps2 = F.leaky_relu(self.attention_routing0(ConvCaps2)+ConvCaps1)
        # ConvCaps2 = F.relu(self.attention_routing(ConvCaps2)+ConvCaps1)

        ConvCaps3 = self.ConvCaps22(ConvCaps2)
        output = F.leaky_relu(self.attention_routing0(ConvCaps3)+ConvCaps2)
        # output = F.relu(self.attention_routing(ConvCaps3)+ConvCaps2)

        return output
      
      else:
        raise ValueError('layer_num must be s(standard) or r(relu)')

class Mask_CID(nn.Module):
    '''
    Masks out all capsules except the capsules that represent the class.
    '''
    def __init__(self, device):
        super(Mask_CID, self).__init__()
        self.device = device
        
    def forward(self, x, target=None):
        batch_size = x.size()[0]
        classes = torch.norm(x, dim=2)
        if target is None:
            max_len_indices = classes.max(dim=-1)[1]#.squeeze(dim=-1)
        else:
            max_len_indices = target.max(dim=-1)[1]
  
        batch_ind = torch.arange(start=0, end=batch_size).to(self.device) #a tensor containing integer from 0 to batch size.
        m = torch.stack([batch_ind, max_len_indices], dim=-1).to(self.device) #records the label's index for every batch.
        masked = torch.zeros((batch_size, 1) + x.size()[2:]).to(self.device)

        for i in range(batch_size):
            masked[i] = x[m[i][0], m[i][1], :].unsqueeze(0)
        if target is None:
            return masked.squeeze(-1), max_len_indices, classes
      
        masked = masked.squeeze(-1)
        indices = classes.max(dim=1)[1]#.squeeze()
        
        return masked, indices, classes
    
      
      
class FullyConvCaps(nn.Module):
  # for UCapsuleNet
    def __init__(self,input_shape, num_classes, activation, device='cpu'):
      super(FullyConvCaps, self).__init__()
      '''
      input_shape: [None, num_capsule=8, dim_capsule =16, w, h]
      output_shape: [None, num_classes=10, dim_capsule,1,1]
      '''
      self.n_channels = num_classes
      self.activation = activation
      self.device = device
      
      input_shape = [None,int(input_shape[1]),int(input_shape[2]),int(input_shape[3]),int(input_shape[4])]
      self.height = input_shape[3]
      self.width = input_shape[4]
      self.dim_capsule = input_shape[2]
      self.num_capsule = input_shape[1]
      
      # ic(input_shape)    #[None, 8, 16, 8, 8]
      
      # Shallow Capsnet, Mnist
      if self.height == 8:
        self.convs = nn.ModuleList(nn.Conv2d(self.dim_capsule, self.dim_capsule, kernel_size=3, stride=2, padding=0, groups=self.dim_capsule) for i in range(3))
        self.final_conv = nn.Conv2d(self.dim_capsule, self.dim_capsule, kernel_size=3, groups=self.dim_capsule)
      # Deep Capsnet, CIFAR10
      elif self.height == 1:
        # ic(input_shape)    #[None, 64, 16, 1, 1]
        self.depthwise_convs = nn.ModuleList([nn.Conv2d(self.dim_capsule, self.dim_capsule, kernel_size=1, stride=1, padding=0, groups=self.dim_capsule)])
      elif self.height == 7:
        self.convs = nn.ModuleList(nn.Conv2d(self.dim_capsule, self.dim_capsule, kernel_size=3, stride=2, padding=0, groups=self.dim_capsule) for i in range(3))
        self.final_conv = nn.Conv2d(self.dim_capsule, self.dim_capsule, kernel_size=3, groups=self.dim_capsule)

      else:
        raise ValueError(f'cant calculate height {self.height}')
      # Pointwise convolution to adjust the channels using 1x1 kernel
      self.pointwise_conv = nn.Conv2d(self.dim_capsule, self.dim_capsule * self.n_channels, kernel_size=1)
      
      output_shape = [None, num_classes, 1,1,1]
      self.attention_routing = simple_orth_attention(self.num_capsule, self.n_channels, self.dim_capsule, output_shape, self.activation,self.device).to(self.device)
      
      
    def forward(self, inputs):
      # ic(inputs.size())
      inputs = F.dropout(inputs, p=0.3)
      input_caps = torch.chunk(inputs, self.num_capsule, dim=1)
      # ic(input_caps[0].size())  
      input_caps = [torch.squeeze(input_cap, dim=1) for input_cap in input_caps]
      # ic(input_caps[0].size())
      
      all_conv_results = []
      for input_cap in input_caps:
          # Depthwise convolution
          if self.height == 8:
            for conv in self.convs:
              depthwise1 = conv(input_cap)
              # ic(depthwise1.size())
              depthwise_result = self.final_conv(depthwise1)
          elif self.height == 1:
            depthwise_result = input_cap                                    #([32, 16, 1, 1])
          elif self.height == 7:
            for conv in self.convs:
              depthwise1 = conv(input_cap)
              # ic(depthwise1.size())
              depthwise_result = self.final_conv(depthwise1)
          else:
            raise ValueError(f'cant calculate height {self.height}')
          # ic(depthwise_result.size())
          pointwise_result = self.pointwise_conv(depthwise_result)          #([32, 160, 1, 1])
          # ic(pointwise_result.size())
          reshaped_result = pointwise_result.view(pointwise_result.size(0), self.n_channels, self.dim_capsule, 1, 1)
          
          all_conv_results.append(reshaped_result.unsqueeze(2))
      
      conv1s = torch.cat(all_conv_results, dim=2)
      outputs = self.attention_routing(conv1s)
      
      return outputs
    
class FullyConvCap(nn.Module):
  # for ShallowNet
    def __init__(self,input_shape, num_classes, activation, device='cpu'):
      super(FullyConvCap, self).__init__()
      '''
      input_shape: [None, num_capsule=8, dim_capsule =16, w, h]
      output_shape: [None, num_classes=10, dim_capsule,1,1]
      '''
      self.n_channels = num_classes
      self.activation = activation
      self.device = device
      
      input_shape = [None,int(input_shape[1]),int(input_shape[2]),int(input_shape[3]),int(input_shape[4])]
      self.height = input_shape[3]
      self.width = input_shape[4]
      self.dim_capsule = input_shape[2]
      self.num_capsule = input_shape[1]
      
      # ic(input_shape)    #[None, 8, 16, 8, 8]
      
      # Shallow Capsnet, Mnist
      self.convs = nn.ModuleList(nn.Conv2d(self.dim_capsule, self.dim_capsule, kernel_size=3, stride=2, padding=0, groups=self.dim_capsule) for i in range(3))
      self.final_conv = nn.Conv2d(self.dim_capsule, self.dim_capsule, kernel_size=3, groups=self.dim_capsule)
      
      # Pointwise convolution to adjust the channels using 1x1 kernel
      self.pointwise_conv = nn.Conv2d(self.dim_capsule, self.dim_capsule * self.n_channels, kernel_size=1)
      output_shape = [None, num_classes, 1,1,1]
      self.attention_routing = simple_orth_attention(self.num_capsule, self.n_channels, self.dim_capsule, output_shape, self.activation,self.device).to(self.device)
      
      
    def forward(self, inputs):
      # ic(inputs.size())
      inputs = F.dropout(inputs, p=0.3)
      input_caps = torch.chunk(inputs, self.num_capsule, dim=1)
      # ic(input_caps[0].size())  
      input_caps = [torch.squeeze(input_cap, dim=1) for input_cap in input_caps]
      # ic(input_caps[0].size())
      
      all_conv_results = []
      for input_cap in input_caps:
          # Depthwise convolution
          for conv in self.convs:
            depthwise1 = conv(input_cap)
            # ic(depthwise1.size())
            depthwise_result = self.final_conv(depthwise1)
          # ic(depthwise_result.size())
          pointwise_result = self.pointwise_conv(depthwise_result)          #([32, 160, 1, 1])
          # ic(pointwise_result.size())
          reshaped_result = pointwise_result.view(pointwise_result.size(0), self.n_channels, self.dim_capsule, 1, 1)
          
          all_conv_results.append(reshaped_result.unsqueeze(2))
      
      conv1s = torch.cat(all_conv_results, dim=2)
      outputs = self.attention_routing(conv1s)
      
      return outputs
    

