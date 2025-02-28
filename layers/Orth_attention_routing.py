
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.activation_functions import squash, capsule_relu, Capsule_ReLU, Squash
# from activation_functions import squash, capsule_relu, Capsule_ReLU
from collections import OrderedDict
from operator import itemgetter
from einops import rearrange
from einops.layers.torch import Rearrange
from icecream import ic
import math
import cfg

def exists(val):
    return val is not None

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))

def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape

def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)
      
    return permutations

class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)

        # attention
        axial = self.fn(axial,**kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


class Attention(nn.Module):
    '''
    主要内容：
    输入是token,将token重新reshape成特征图，然后在特征图的基础进行卷积，
    进行三次卷积得到q,k,v三个特征图，然后将这三个特征图再reshape成token,
    然后用这个q,k这个两个token,进行attention操作，得到打分值score,
    打分值score经过softmax变成概率值，然后与V这个token相乘，得到结果token
    '''
    '''
    input: [B, N, D], token 
    output: [B, N, D], token
    '''
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        
        #卷积实现得到q,k,v
        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        
        #扩充维度
        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
    
    # 卷积操作->BN->然后将卷积图（b,c,h,w）reshape成Token（b,h*w,c)
    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                # ('bn', nn.BatchNorm2d(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x):
        
        #将token转变成特征图
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        #TODO:
        
        h = int(math.sqrt(x.size(1)))
        # ic(h)
        # ic(x.shape)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=h)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        return q, k, v    #输出的是token

    def forward(self, x):
        # ic(x.shape)
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x)        #'b c h w -> b (h w) c'
        
        #先扩宽token的维度，然后再实现mult-head，最后的维度是[b,h,t,d]
        '''
        q,k,v矩阵最初的维度是 [B, N, D]，其中 B 是批次大小（batch size），N 是序列长度（对应图像处理中的 h * w，即图像被划分成的块的总数），D 是特征维度。
        多头注意力要求将 D 维度分割成多个头。假设有 H 个头，每个头的维度是 d，则 D = H * d。
        'b t (h d) -> b h t d' 表示将维度 [B, N, H*d] 重排列为 [B, H, N, d]。
        其中 b 代表批次大小，t 代表序列长度(向量个数)，(h d) 是合并的头维度和每个头的特征维度，重排列后变成 b h t d，即分离出每个头和其对应的特征维度。
        '''
        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        
        #将attention得到概率值与value信息值相乘得到结果，结构是，b,h,t,d
        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')
        
        #扩充维度
        x = self.proj(x)
        x = self.proj_drop(x)

        return x           #b,t,(h,d),输出的是token
   
    
class AxialAttention(nn.Module):
    def __init__(self, dim_in, dim_out, num_dimensions=2, heads=8, dim_index=-1, sum_axial_out=True,
                 qkv_bias=False,drop=0.,attn_drop=0.,**kwargs):
        '''
        dim_in, dim_out:ATTENTTION的输入和输出维度。
        num_dimensions: 在输入张量中，除了批处理维度和特征维度外，还有多少维度需要注意力机制。通常对于图像来说是2（高和宽）。
        heads: 多头注意力中的头数。
        dim_heads: 每个注意力头的维度。如果为 None，则自动计算。
        dim_index: 指示哪个维度是特征维度。
        sum_axial_out: 是否将不同轴的输出求和。
        '''
        
        assert (dim_in % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = 1  #不变的特征维度，类似图片中的c,让n，d原轴向注意力中的w和h
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)
        
        self.attention = Attention(dim_in, dim_out, heads,
                                   qkv_bias,drop,attn_drop,
                                   **kwargs)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, self.attention))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        x = x.unsqueeze(-1)      # [64, 64, 16, 1]
        # ic(x.shape)
        # ic(self.total_dimensions)
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        # ic(self.dim_index)
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            # return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))
            
            iterator = iter(self.axial_attentions)
            attn1 = next(iterator)
            out1 = attn1(x)
            
            num_Caps = x.shape[1] # number of capsules
            batchSize = x.shape[0]
            
            if(num_Caps * batchSize < cfg.ATTENTION_THRESHOLD):
                attn2 = next(iterator)
                out2 = attn2(x)
                return sum(out1, out2)
            
            return out1    

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out
  
        
    
class simple_orth_attention(nn.Module):
    def __init__(self, input_ch, n_channels, dim_capsule, input_shape, activation = 'Squash', device='cpu'):
        super(simple_orth_attention, self).__init__()
        self.i_c = input_ch           # Current capsule number
        self.device = device
        self.n = n_channels
        self.d = dim_capsule
        # #TODO: activation function
        # if activation == 'Squash':
        #     self.activation = Squash()
        # elif activation == 'Capsule_ReLU':
        #     self.activation = Capsule_ReLU(input_shape)
        # else:
        #     assert False, 'Unknown activation function'
        
        i = nn.Parameter(torch.empty(dim_capsule,n_channels,input_ch))
        self.i = nn.init.xavier_uniform_(i, gain=nn.init.calculate_gain('relu')).to(self.device)
        i_caps = nn.Parameter(torch.empty(dim_capsule,n_channels))
        self.i_caps = nn.init.xavier_uniform_(i_caps, gain=nn.init.calculate_gain('relu')).to(self.device)
        
        self.deepwise_conv = nn.Conv2d(self.d, self.d, kernel_size=1, stride=1, padding=0, groups=self.d, bias=False)
        # self.deepwise_conv = nn.Conv2d(self.d, self.d, kernel_size=1, stride=1, padding=0, groups=self.i_c, bias=False)
        
        # ic(self.n)
        # ic(self.i_c)
        
    def alpha_entmax(self, logits, alpha):
        unnormalized_scores = alpha * logits
        attention_weights = F.softmax(unnormalized_scores / np.sqrt(self.d), dim=1)
        return attention_weights   

    def householder_matrix(self, vector):
        I = torch.eye(vector.size(-1), device=self.device).unsqueeze(0)  
        vector = vector.unsqueeze(-1).to(self.device) 
        vector_transpose = vector.transpose(-2, -1)
        H = I - 2 * torch.matmul(vector, vector_transpose) / torch.matmul(vector_transpose, vector)
        return H.squeeze()
        
    def forward(self, x):
        '''
        x: (B, n, i_c, d, 1, 1)
        '''
        # ic(x.shape)
        householder_matrices = [self.householder_matrix(self.i[:, i, :]) for i in range(self.n)]
        Att_W = torch.stack(householder_matrices).unsqueeze(2)
        Att_W = Att_W.view(self.i_c * self.n, self.i_c, self.d, 1, 1).to(self.device) 
        # ic(Att_W.shape)
        Att_inputs = torch.chunk(x, self.n, dim=1)
        # ic(Att_inputs[0].shape)                                              # (B, 1, i_c, d, 1, 1)
        Att_ws = torch.chunk(Att_W, self.n, dim=0)                           # (i_c, i_c, d, 1, 1)
        # ic(Att_ws[0].shape) 
        
        
        Att_inputs = [torch.squeeze(Att_input, dim=1) for Att_input in  Att_inputs]
        outputs = []
        for Att_input,Att_w,i in zip(Att_inputs, Att_ws,range(self.n)):
            x = torch.squeeze(Att_input, dim=1)
            # ic(x.shape)
            attentions = F.conv3d(x, Att_w)
            attentions = self.alpha_entmax(attentions, 1.5)
            final_attentions = torch.mul(x, attentions)
            # final_attentions = torch.sum(final_attentions, dim=1).squeeze()         # (B, d, 1, 1)->(B, d)
            final_attentions = torch.sum(final_attentions, dim=1).permute(0,2,3,1)         # (B, d, W, H)->(B, W, H, d)
            # ic(final_attentions.shape)
            # ic(self.i_caps[:, i].shape)                                            # (d)
            #TODO: NO NEED TO USE HOUSEHOLDER MATRIX
            final_attentions = torch.matmul(final_attentions, self.householder_matrix(self.i_caps[:, i])).permute(0,3,1,2).view(-1, self.d, x.shape[-2], x.shape[-1])
            # final_attentions = torch.matmul(final_attentions, self.householder_matrix(self.i_caps[:, i])).view(-1, self.d, 1, 1)
            conv3 = self.deepwise_conv(final_attentions)
            # ic(conv3.shape)                                                # (B, d, 1, 1)
            outputs.append(conv3.unsqueeze(1))
        
    
        outputs = torch.cat(outputs, dim=1)
        # ic(outputs.shape)                                                   # (B, n, d, 1, 1)
        #TODO: activation function
        # outputs = self.activation(outputs)
        
        return outputs   


class deep_orth_attention(nn.Module):
    def __init__(self, input_ch, n_channels, dim_capsule, input_shape, activation = 'Squash', device='cpu'):
        super(deep_orth_attention, self).__init__()
        self.i_c = input_ch
        self.device = device
        self.n = n_channels
        self.d = dim_capsule
        self.b = input_shape[0]
        
        # ic(self.n)
        #TODO: activation function
        if activation == 'squash':
            self.activation = Squash()
        elif activation == 'capsule_relu':
            self.activation = Capsule_ReLU(n_channels)

        i = nn.Parameter(torch.empty(dim_capsule,n_channels,input_ch))
        self.i = nn.init.xavier_uniform_(i, gain=nn.init.calculate_gain('relu')).to(self.device)
        i_caps = nn.Parameter(torch.empty(dim_capsule,n_channels))
        self.i_caps = nn.init.xavier_uniform_(i_caps, gain=nn.init.calculate_gain('relu')).to(self.device)

    def alpha_entmax(self, logits, alpha):
        '''
        α-Entmax函数
        '''
        unnormalized_scores = alpha * logits
        attention_weights = F.softmax(unnormalized_scores / np.sqrt(self.d), dim=1)
        return attention_weights   

    def householder_matrix(self, vector):
        I = torch.eye(vector.size(-1), device=self.device).unsqueeze(0)  
        vector = vector.unsqueeze(-1).to(self.device) 
        vector_transpose = vector.transpose(-2, -1)
        H = I - 2 * torch.matmul(vector, vector_transpose) / torch.matmul(vector_transpose, vector)
        return H.squeeze()

    def forward(self, x):
        '''
        x: (B, n, i_c, d, W, H)
        '''
        householder_matrices = [self.householder_matrix(self.i[:, i, :]) for i in range(self.n)]
        # ic(householder_matrices[0].shape)
        Att_W = torch.stack(householder_matrices).unsqueeze(2)
        # ic(Att_W.shape)
        Att_W = Att_W.view(self.i_c * self.n, self.i_c, self.d, 1, 1).to(self.device)
        # ic(Att_W.shape) 

        householder_matrices_caps = [self.householder_matrix(self.i_caps[:, i]) for i in range(self.n)]
        CapsAct_W = torch.stack(householder_matrices_caps).unsqueeze(-1).view(self.d * self.n, self.d, 1, 1).to(self.device)

        CapsAct_B = torch.nn.Parameter(torch.zeros(self.d * self.n)).to(self.device)

        CapsAct_ws = torch.chunk(CapsAct_W, self.n, dim=0)
        CapsAct_bs = torch.chunk(CapsAct_B, self.n, dim=0)
        Att_inputs = torch.chunk(x, self.n, dim=1)
        # ic(Att_inputs[0].shape)                                              # (B, 1, i_c, d, W, H)
        Att_ws = torch.chunk(Att_W, self.n, dim=0) 
        # ic(Att_ws[0].shape)


        outputs = []
        for Att_input, Att_w, CapsAct_w, CapsAct_b in zip(Att_inputs, Att_ws, CapsAct_ws, CapsAct_bs):
            x = torch.squeeze(Att_input, dim=1)
            # ic(x.shape)                    # (i_c, d, W, H)
            # ic(Att_w.shape)
            attentions = F.conv3d(x, Att_w)
            attentions = self.alpha_entmax(attentions, 1.5)
            final_attentions = torch.mul(x, attentions)
            final_attentions = torch.sum(final_attentions, dim=1)
            conv3 = F.conv2d(final_attentions, CapsAct_w, bias=CapsAct_b)
            outputs.append(conv3.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)

        #TODO: activation function
        # outputs = self.activation(outputs)

        return outputs 
        

# if __name__ == '__main__':
    
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
#     x = torch.randn(32, 10, 8, 16, 1, 1).to(device)
#     attention_routing = simple_orth_attention(input_ch=8, n_channels=10, dim_capsule=16, device=device).to(device)
#     y = attention_routing(x)
#     print(y.shape)