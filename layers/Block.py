import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from einops.layers.torch import Rearrange
import collections.abc as container_abcs
from itertools import repeat
import logging
import math

from layers.Orth_attention_routing import *
from icecream import ic

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse
 
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class ConvLayer(nn.Module):
    def __init__(self,input_channels,filters,decrease_resolution=True, layer = 4, device='cpu'):
        super(ConvLayer, self).__init__()
        self.device = device
        self.layer = layer
        
        if decrease_resolution == True:
          self.stride = 2
        else:
          self.stride = 1
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=1, padding='same')
        # self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=filters, kernel_size=3, stride=self.stride, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=32, out_
        # channels=filters, kernel_size=3, stride=self.stride, padding=1)
        self.bn4 = nn.BatchNorm2d(filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # ic(x.size())                      #([32, 128, 30, 30])
        return x
    
class Mlp(nn.Module):
    '''
    input: [B, N, D], capsules 
    output: [B, N, D], capsules
    structure: fc + Gelu + dropout + fc + dropout
    '''
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        # 下面调用时，out_features是None，由下面这行代码，意为不改变维度
        # 在 Transformer 架构中，我们通常希望保持特征维度的一致性，以便于层与层之间的连接。
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None,patch_size=7,stride=4,padding=2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=patch_size, stride=stride, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class ConvEmbed(nn.Module):
    '''
    输入是2维的图片数据或者2D的capsule reshape成特征图)，输出的是特征图，目的是让特征图的尺寸变小
    图片数据经过一个卷积层，输出一个特征图
    '''

    def __init__(self, patch_size=7, in_chans=3, embed_dim=64, stride=4, padding=2, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        #(s,s),一个patch的大小
        self.patch_size = patch_size
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.proj = DoubleConv(in_channels=in_chans, out_channels=embed_dim,mid_channels=(in_chans+embed_dim)//2, 
                               patch_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        #输入是特征图
        x = self.proj(x)      #(b,c,h,w)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        #输出是特征图
        return x 
    
class Cell(nn.Module):
    '''
    输入是token，输出是token
    输入的token先reshape成特征图，送入self-attention操作得到结果token
    然后将上面的结果送入MLP中进行运算，输出token
    '''
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        
        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )
        # self.attn = AxialAttention(
        #     1, 1, stay=1, num_dimensions=2, heads=num_heads, dim_index=-1, sum_axial_out=True,
        #     qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop,
        #     **kwargs
        # )
        
        # 如果 drop_path 大于 0，那么会使用 DropPath 模块。DropPath(drop_path) 创建了一个 DropPath 实例，其丢弃概率（drop_prob）设置为 drop_path 的值。
        # 如果 drop_path 的值为 0 或者更小，使用 nn.Identity 作为一个占位符。nn.Identity 是一个不进行任何操作的模块，它简单地返回输入数据而不做任何改变。
        
        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x):
        res = x

        x = self.norm1(x)
        attn = self.attn(x).squeeze()
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
class Block(nn.Module):
    '''
    该模块由一个Convolutional Token Embedding 模块+Convolutional Projection 模块 两部分组成
    输入是图片，输出是特征图和cls_token
    图片数据先经过ConvEmbed，得到一个特征图
    然后这个特征图会被reshape成token
    这个token会组合上cls_token，一起送入堆叠Block中，输出token
    最后会将这个token分离出cls_token和图片数据token，然后将图片数据reshape成图片数据的特征图
    这种设计可以根据不同的应用场景进行定制和扩展，体现了深度学习中模块化和可重用性的原则。
    '''
    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=1,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='xavier',
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None
        #ConvEmbed
        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 生成一个 dropout 概率列表，这个列表用于之后的 DropPath。
        
        #堆叠cell
        #cell中d不变
        cells = []
        for j in range(depth):
            cells.append(
                Cell(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
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
        
        #初始化网络权重参数
        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        '''
        用于根据截断正态分布初始化网络中的线性层和标准化层的权重和偏置。
        '''
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        '''
        根据 Xavier初始化方法初始化网络中的线性层和标准化层的权重和偏置。
        '''
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        #首先输入的是图片（2d），经过卷积，将图片尺寸变小了，输出的是特征图（2d）
        x = self.patch_embed(x)
        B, C, H, W = x.size()
        
        # Reshape to capsule
        x = rearrange(x, 'b c h w -> b (h w) c')

        x = self.pos_drop(x)
        #执行堆叠的cell
        for i, cel in enumerate(self.cells):
            x = cel(x)
            
        # 将胶囊映射回特征图的形式
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x
    
class Fully(nn.Module):
  # for DeepUCaps
    def __init__(self,input_shape, num_classes,device='cpu'):
      super(Fully, self).__init__()
      '''
      input_shape: [None, dim_capsule, W, H]
      output_shape: [None, num_classes=10, dim_capsule]
      '''
      self.n_channels = num_classes
      self.device = device
      
      self.dim_capsule = int(input_shape[1])
      self.num_capsule = int(input_shape[2]*input_shape[3]/4)
    #   ic(input_shape)   
      
      self.conv = nn.Conv2d(self.dim_capsule, self.dim_capsule, kernel_size=3, stride=2, padding=1)
      self.linear = nn.Sequential(
        nn.Linear(self.num_capsule, 2*self.num_capsule),
        nn.ReLU(inplace=True),
        nn.Linear(2*self.num_capsule, self.n_channels),
        nn.Sigmoid()
        )
      
    def forward(self, inputs):
      # ic(inputs.size())
      inputs = F.dropout(inputs, p=0.3)
      B = inputs.size()[0]
      x = self.conv(inputs).reshape(B * self.dim_capsule, -1)
      x = x.reshape(B * self.dim_capsule, -1)
      x = F.batch_norm(x, running_mean=None, running_var=None, training=True)
      x = F.relu(x)
      outputs = self.linear(x).reshape(B, self.n_channels, self.dim_capsule)
    #   ic(outputs.size())
      return outputs
  

  
class Mask_CID(nn.Module):
    '''
    Masks out all capsules except the capsules that represent the class.
    '''
    def __init__(self, device):
        super(Mask_CID, self).__init__()
        self.device = device
        
    def forward(self, x, target=None):
        # ic(x.size())
        
        batch_size = x.size()[0]
        classes = torch.norm(x, dim=2)
        # ic(classes.size())
        if target is None:
            max_len_indices = classes.max(dim=-1)[1]#.squeeze(dim=-1)
        else:
            max_len_indices = target.max(dim=-1)[1]
  
        batch_ind = torch.arange(start=0, end=batch_size).to(self.device) #a tensor containing integer from 0 to batch size.
        m = torch.stack([batch_ind, max_len_indices], dim=-1).to(self.device) #records the label's index for every batch.
        masked = torch.zeros((batch_size, 1) + x.size()[2:]).to(self.device)

        for i in range(batch_size):
            masked[i] = x[m[i][0], m[i][1]].unsqueeze(0)
        if target is None:
            return masked.squeeze(-1), max_len_indices, classes
        masked = masked.squeeze(-1)
        # masked = None
        indices = classes.max(dim=-1)[1]#.squeeze()
        # ic(indices.shape)
        
        return masked, indices, classes



class Decoder(nn.Module):
    '''
    Reconstruct back the input image from the prediction capsule using transposed Convolutions.
    '''

    def __init__(self, caps_dimension, device, num_caps=1, img_size=224, img_channels=1):

        super(Decoder, self).__init__()

        self.num_caps = num_caps
        self.img_channels = img_channels
        self.img_size = img_size
        self.caps_dimension = caps_dimension
        self.neurons = self.img_size//4
        self.device = device

        self.fc = nn.Sequential(torch.nn.Linear(self.caps_dimension*self.num_caps, self.neurons*self.neurons*16), nn.ReLU(inplace=True)).to(self.device)

        self.reconst_layers1 = nn.Sequential(nn.BatchNorm2d(num_features=16, momentum=0.8),
                                                nn.ConvTranspose2d(in_channels=16, out_channels=64,
                                                kernel_size=3, stride=1, padding=1))

        self.reconst_layers2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.reconst_layers3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.reconst_layers4 = nn.Sequential(nn.ConvTranspose2d(in_channels=16, out_channels=self.img_channels, kernel_size=3, stride=1, padding=1),
                                                nn.ReLU(inplace=True))


    def forward(self, x):
        '''
        Forward Propagation
        '''

        x = x.type(torch.FloatTensor).to(self.device)

        x = self.fc(x)
        x = x.reshape(-1, 16, self.neurons, self.neurons)
        x = self.reconst_layers1(x)
        x = self.reconst_layers2(x)

        p2d = (1, 0, 1, 0)
        x = F.pad(x, p2d, "constant", 0)
        x = self.reconst_layers3(x)

        x = F.pad(x, p2d, "constant", 0)
        x = self.reconst_layers4(x)

        x = x.view(-1, self.img_channels, self.img_size, self.img_size)
        # ic(x.size())

        return x
    

class ConceptLayer(nn.Module):
    
  def __init__(self, n, d, conceptnum=8, sup=True, device='cuda'):
    super(ConceptLayer, self).__init__()
    
    self.n = n
    self.d = d
    self.conceptnum = conceptnum
    self.sup = sup
    self.device = device
    
    self.concept = nn.Sequential(
        nn.Linear(self.d, 10),
        nn.ReLU(),
        nn.Linear(self.n, conceptnum)
           
    )
    self.globalcaps = nn.Sequential(
      nn.linear(10,1),
      nn.Softmax(dim=-1))
    self.concept_layer = nn.Linear(self.d, 10)
    
  def forward(self, x):
    if self.sup:
      return self.concept_layer(x)
    else:
      return self.globalcaps(self.concept(x))



if __name__ == '__main__':
    
    def test_block():
        # 设定输入参数
        batch_size = 4      # 输入批次大小
        in_channels = 3     # 输入通道数，对于彩色图像通常是3
        img_size = 224      # 假设输入图片的大小是 224x224
        embed_dim = 768     # 嵌入维度
        depth = 12          # Block 的深度
        num_heads = 16      # 注意力头的数量
        mlp_ratio = 4.      # MLP 比率

        # 创建 Block 实例
        block = Block(
            patch_size=16,
            patch_stride=16,
            patch_padding=0,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.1,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
        )

        # 生成一个假的输入数据
        input_tensor = torch.randn(batch_size, in_channels, img_size, img_size)

        # 前向传播
        output_tensor = block(input_tensor)

        # 检查输出的维度是否正确
        assert output_tensor.ndim == 4, "输出特征图的维度应该为4"
        print("输出特征图的维度：", output_tensor.shape)
        print("测试通过！")

    # 运行测试
    test_block()
