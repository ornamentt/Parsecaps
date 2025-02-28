import torch
import torch.nn as nn

class ConvertToCaps(nn.Module):
    def __init__(self,dim=16):
        super().__init__()
        self.dim = dim
    
    def forward(self, inputs):
        # channels first
        inputs = torch.unsqueeze(inputs, 2)
        # inputs.shape = (batch, channels, dimensions, height, width)
        batch, channels, dimensions, height, width = inputs.shape
        inputs = inputs.permute(0, 3, 4, 1, 2).contiguous()
        # output_shape = (batch, channels * height * width, dimensions)
        return inputs.view(batch, -1, self.dim)


class CapsToScalars(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        # inputs.shape = (batch, num_capsules, dimensions)
        return torch.norm(inputs, dim=2)
    
if __name__ == '__main__':
    caps = torch.randn(128, 1024, 16)
    scalars = torch.randn(128, 3, 32, 32)
    
    convert = ConvertToCaps()
    scalar = CapsToScalars()
    output1 = convert(scalars)
    print(output1.shape)
    output2 = scalar(caps)
    print(output2.shape)
    print(output2[0, :])