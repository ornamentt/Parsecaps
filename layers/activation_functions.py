import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
from icecream import ic

class Capsule_ReLU(nn.Module):
  def __init__(self, input_shape, dim=2, eps=0.00001):
    super(Capsule_ReLU, self).__init__()
    
    #TODO:
    # ic([input_shape[1], 1, input_shape[-2],input_shape[-1]])
    
    self.dim = dim
    self.ln = nn.LayerNorm([input_shape[1], 1, input_shape[-2],input_shape[-1]], elementwise_affine=False)
    self.eps = eps

  def _check_input_dim(self, input):
      if (input.dim() != 5):
        raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    self._check_input_dim(input)
    input_magnitude = torch.clamp_min(LA.norm(input, dim=self.dim,keepdim=True), min=self.eps)
    input_magnitude_ln = self.ln(input_magnitude)  # 使用LayerNorm
    input_magnitude_ln = F.relu(input_magnitude_ln)
    # print('Capsule_ReLU')
    return (input_magnitude_ln/input_magnitude)*input



def capsule_relu(input, dim=-1, bar = 1.0, eps=1e-8):
    input_norm = torch.clamp_min_(LA.norm(input=input, dim=dim, keepdim=True), min=eps) 
    input = input / input_norm
    return input * torch.clamp_min_(F.relu(input_norm-bar), min=eps)


class ConvertToCaps(nn.Module):
    def __init__(self):
        super(ConvertToCaps, self).__init__()
    
    # size(input): b,in_c, h,w
    # size(output): b, in_c, h, w, 1
    def forward(self, x):
        return x.unsqueeze(-1)
    
    
def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)


  
class Squash(nn.Module):
    def __init__(self, dim=-1):
        super(Squash, self).__init__()
        self.dim = dim
    
    def forward(self, tensor):
        squared_norm = (tensor ** 2).sum(dim=self.dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        # print('SQUASH')
        return scale * tensor / torch.sqrt(squared_norm)
      
if __name__ == '__main__':
    # test Capsule_ReLU
    x = torch.randn(32, 10, 16, 28, 28)
    print(x.shape)
    x = Capsule_ReLU(x.shape)(x)
    print(x.shape)
   