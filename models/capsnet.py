import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.autograd import Variable
from icecream import ic
    
from Utils import helpers
# from layers.Orth_attention_routing import AttentionRouting, LastRouting


# device = torch.device("cpu")


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)

def dynamic_routing(u,device):
    """
    Args:
        x: u_hat,  (B, 10, 32x6x6, 16, 1)      
    Return:
        v: next layer output (B, 10, 16)
    """
    # N = 32*6*6 # previous layer (num_capsules*6*6???)(dim_capsules = 8)
    N = 1568
    N1 = 3 # next layer (num_classes???)
    B = u.shape[0]                                          # (B,10,32*6*6,16,1)
    b = torch.zeros(B,N1,N,1,1).to(device)                  # (B,10,32*6*6,1,1)
    T=3
    
    for _ in range(T):                                      
        # probability of each vector to be distributed is 1
        # (B,10,32*6*6,1, 1)
        c = F.softmax(b, dim=1).to(device)  
 
        # (B,10,16)
        s = torch.sum(u.matmul(c), dim=2).squeeze(-1).to(device)      
     
        # (B,10,16)
        v = squash(s).to(device)

        # (B,10,32*6*6,1,1)
        b = b + v[:,:,None,None,:].matmul(u).to(device)                
    
    return v

class Primary_CapsLayer(nn.Module):
    def __init__(self,device,map=32,in_channels=256,out_channels=8,kernel_size=9,stride=3):
        super(Primary_CapsLayer,self).__init__()
        # self.height = height
        # self.width = width
        self.primary_capsule_layer = \
            nn.ModuleList([nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size, stride=stride) for _ in range(map)])    
        self.device = device    
            
    def forward(self, x):
        """ Produce primary capsules
        Args:
            x: features with (B, 256, 24, 24), (batchsize,channels,W,H)  
        Return:
            vectors (B, 32*6*6, 8)
        """
        # ic(x.size())
        batch_size = x.size()[0]
        capsules = [conv(x) for conv in self.primary_capsule_layer]  # [[B, 8, 6, 6] * 32] 
        dim = capsules[0].shape[1]
        capsules_reshaped = [c.reshape(batch_size,dim,-1) for c in capsules]  # [[B, 8, 36] * 32] 
        s = torch.cat(capsules_reshaped, dim=-1).permute(0, 2, 1).to(self.device)  # (B, 32*6*6, 8)
        return squash(s)
    
class Digit_CapsLayer(nn.Module):
    def __init__(self, nclasses, device, out_channels_dim=16, routing='dynamic'):
        super(Digit_CapsLayer,self).__init__()
        # self.W = nn.Parameter(1e-3 * torch.randn(1,nclasses,32*6*6,out_channels_dim,8))
        self.W = nn.Parameter(1e-3 * torch.randn(1,nclasses,1568,out_channels_dim,8))
        self.device = device
        self.routing = routing
        # self.attention_routing = LastRouting(d=16,num_classes=10,capsule_number_l=32*6*6, activation='squash',device=self.device)
        
        
    def forward(self, x):
        """Predict and routing
        
        Args:
            x: Input vectors, (B, 32*6*6, 8)
            
        Return:
            class capsules, (B, 10, 16)
        """
        # ic(x.shape)
        x = x[:,None,...,None].to(self.device)
        u_hat = self.W.matmul(x)  # (B, 10, 32x6x6, 16, 1)
        # assert u_hat.shape[1:] == (10, 32*6*6, 16, 1)
        
        if self.routing == 'attention':
            class_capsules = self.attention_routing(u_hat)
        elif self.routing == 'dynamic':
            class_capsules = dynamic_routing(u_hat, self.device)
        else:
            raise NotImplementedError('No routing method named {}'.format(self.routing))
        
        return class_capsules

class Mask_CID(nn.Module):
    '''
    Masks out all capsules except the capsules that represent the class.
    '''

    def __init__(self, device):

        super(Mask_CID, self).__init__()
        self.device = device

    def forward(self, x, target=None):

        batch_size = x.size()[0]
        # ic(x.size())

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


# class Decoder(nn.Module):
#     """Decode the input predicted vectors tor origin images
    
#     Usage:
#         decoder = MLPDecoder([512, 1024], 16, (32,32))      16:vector size 1*16, (32,32): MMIST图片size, [512,1024]ￄ1�7?隐藏层参ￄ1�7?
#         reconstructed_x = decoder(selected_capsules)
#     """
#     def __init__(self, hidden, in_channels, out_shape, i_c):
#         super().__init__()
#         self.out_shape = out_shape
#         self.in_channel = i_c
#         h,w = out_shape
#         out_channels = w*h
#         self.mlp = nn.Sequential(*[
#             nn.Linear(_in, _out)                                              
#             for _in,_out in zip([in_channels]+hidden, hidden+[i_c*out_channels])
#         ])
        
#     def forward(self, x):
#         """
#         Args:
#             x: (B,16)
            
#         Return:
#             reconstructed images with (B,1,32,32)
#         """
#         B = x.shape[0]
#         x = self.mlp(x)                      # (B,ic*32*32)
#         x = x.reshape(B, self.in_channel, *self.out_shape)              
#         return x


class CapsNet(nn.Module):
     
    def __init__(self, num_class, img_height, img_width, in_channel, device, routing='dynamic'):
        super(CapsNet,self).__init__()
        self.num_class = num_class
        self.height,self.width = img_height, img_width
        self.in_channel = in_channel
        self.device = device
        self.routing = routing
        self.conv_layer = nn.Sequential(
        nn.Conv2d(self.in_channel,64,3,stride=2,padding=1),                          # 对应Primary_CapsuleLayer的输入通道数为256
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64,128,3,stride=2,padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.Conv2d(128,256,3,stride=2,padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        )
        self.primary_layer = Primary_CapsLayer(self.device).to(self.device)
        self.caps_layer = Digit_CapsLayer(device=self.device,nclasses=self.num_class, out_channels_dim=16, routing=self.routing)
        self.mask = Mask_CID(self.device)
        # self.decoder = Decoder([512,1024], 16, (self.height, self.width),self.in_channel)
        # self.decoder = Decoder(input_size = [None,in_channel,img_height,img_width], num_classes=num_class, device = self.device)
        
        
    # def forward(self, x, onehot_label=None):
    #     """
    #     Args:
    #         x : Input img, (B, i_c, 32, 32)
        
    #     Return:
    #         the class capsules, each capsule is a 16 dimension vector
    #     """
    #     # ic(x.size())
    #     x = self.conv_layer(x)  # (B, 256, 20, 20)                     
    #     x = self.primary_layer(x)  # (B, 32*6*6, 8)
    #     dig_caps = self.caps_layer(x)  # (B, 10, 16)
        
    #     indices = torch.norm(dig_caps, dim=2, keepdim=False)
    #     masked = indices.max(dim=1)[1].squeeze()
        
    #     outputs = dig_caps
    #     # ic(dig_caps.size())
    #     # outputs = dig_caps.norm(dim=-1)
    #     # ic(outputs.size())
    #     x_recon = self.decoder(outputs, onehot_label)
    #     y_pred = torch.norm(outputs, dim=2, keepdim=False).squeeze(-1)
        
    #     classes = torch.norm(outputs, dim=2)
    #     indices = classes.max(dim=1)[1].squeeze()
        
    #     return outputs, indices, x_recon, y_pred 
    
    def forward(self, x, target=None):
        """
        Args:
            x : Input img, (B, i_c, 32, 32)
        
        Return:
            the class capsules, each capsule is a 16 dimension vector
        """
        x = self.conv_layer(x)  # (B, 256, 20, 20)                     
        x = self.primary_layer(x)  # (B, 32*6*6, 8)
        # ic(x.shape)
        dig_caps = self.caps_layer(x)  # (B, 10, 16)
        # ic(dig_caps.shape)
        masked, indices,classes = self.mask(dig_caps, target)
        # ic(masked.shape)
        # ic(indices.shape)
        # decoded = self.decoder(masked)
        
        return indices, None, classes, masked
        # return dig_caps, indices, decoded, classes
        

    
if __name__ == '__main__':
    device = helpers.get_device()
    net = CapsNet(10,32,32,3,device).to(device)
    x = torch.randn(1, 3, 32, 32).to(device)
    # summary(net, (3, 32, 32))
    y = net(x)
    print(y[0].size(), y[1].size(), y[2].size(), y[3].size())