"""
Temporal Implementation of C2SP of 'Convolution with even-sized kernels and symmetricpadding(NeurIPS 2019)'
link : https://arxiv.org/pdf/1903.08385.pdf
Detailed performance testing will be updated soon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#### Need to check performance        
class SymmetricOnePad2d(nn.Module):
    """symmetric 0-pad to splited tensors and concat"""
    
    def __init__(self):
        super(SymmetricOnePad2d, self).__init__()
        self.padding1 = nn.ZeroPad2d((1, 0, 1 ,0))
        self.padding2 = nn.ZeroPad2d((1, 0, 0, 1))
        self.padding3 = nn.ZeroPad2d((0, 1, 1, 0))
        self.padding4 = nn.ZeroPad2d((0, 1, 0, 1))        
    def forward(self, x):
        sub = x.shape[1] // 4
        x1, x2, x3, x4 = x[:,:sub], x[:,sub:2*sub], x[:,2*sub:3*sub], x[:,3*sub:]
        x1, x2, x3, x4 = self.padding1(x1), self.padding2(x2), self.padding3(x3), self.padding4(x4)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x
    
    
class Conv2dEvenKernel2x2(nn.Conv2d):
    """2x2, stride1 Conv2d with symmetric zero pad2d"""
    
    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super(Conv2dEvenKernel2x2, self).__init__(in_channels, out_channels, 2, 1, 0, 1, groups, bias)
        self.symmetric1pad = SymmetricOnePad2d()
        
    def forward(self, x):
        x = self.symmetric1pad(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x
