import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class HyperNetwork(nn.Module):
    
    """
    HyperNetwork gets latent dim vector z and return kernel parameters
    
    kernel_size : size of kernel to be generated
    z_dim : dimensionality for latent vector z
    unit_size : unit size for in/out channels
    
    returns a kernel weights with shape [unit_size, unit_size, kernel_size, kernel_size]
    """
    
    def __init__(self, z_dim=64, hidden_size=64, kernel_size=3, unit_size=16):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.unit_size = unit_size
        self.in_channel = self.unit_size
        self.out_channel = self.unit_size
        
        # fc layers to generate a kernel
        self.fc1 = nn.Linear(self.z_dim, self.in_channel * self.hidden_size)
        self.fc2 = nn.Linear(self.z_dim, self.out_channel * self.kernel_size * self.kernel_size)
        
        
    def forward(self, z):
        
        out = self.fc1(z)
        out = out.view(self.in_channel, self.hidden_size)

        # generate kernel weights for each in_channel dimension
        out = [self.fc2(out[i]) for i in range(self.in_channel)]

        # concatenate and reshape kernel
        out = torch.stack(out, dim=0)
        out = out.view(self.out_channel, self.in_channel, self.kernel_size, self.kernel_size)     
        
        return out
    
    
    
class ResBlock(nn.Module):
    expansion=1
    """
    Residual Block to build ResNet. In forward propagation stage, it gets 2 conv weights.
    Each corresponds to conv1 and conv2 layer and remaining processes are same with plain resblock.
    
    conv1_w, conv2_w are the results from a hypernetwork.
    """
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResBlock, self).__init__()        
        self.downsample = downsample
        self.stride = stride
        
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channel)

        
    def forward(self, x, conv1_w, conv2_w):
        residual = x
        
        out = F.conv2d(x, conv1_w, stride=self.stride, padding=1)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = F.conv2d(out, conv2_w, padding=1)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out


class Embedding(nn.Module):
    """
    Embedding contains multiple z embdeeing parameters to be trained.
    Returns a completed kernel by concatenating the results of hypernetwork.
    
    z_dim : dimensionality for latent vector z
    scale_factor : scale factor for channels. []
    (channel_num = unit_size * scale_factor)
    
    returns a kernel weights with shape [channel_out, channel_in, kernel_size, kernel_size]
    """
    def __init__(self, z_dim=64, scale_factor=[1, 1]):
        super(Embedding, self).__init__()
        self.z_dim = z_dim
        self.z_list = nn.ParameterList()
        self.in_unit = scale_factor[0]
        self.out_unit = scale_factor[1]
        
        # Embedding parameters to be trained 
        for _ in range(self.in_unit * self.out_unit):
            self.z_list.append(Parameter(torch.randn(self.z_dim)))
            
    def forward(self, hyper_net):
        kernel = []
        
        # concat kernel weights for out_channel dimmension
        for i in range(self.out_unit):
            in_weights = []
            # concat kernel weights for in_channel dimension
            for j in range(self.in_unit):
                in_weights.append(hyper_net(self.z_list[i*self.in_unit + j]))
            kernel.append(torch.cat(in_weights, dim=1))
        kernel = torch.cat(kernel, dim=0)
        return kernel
    