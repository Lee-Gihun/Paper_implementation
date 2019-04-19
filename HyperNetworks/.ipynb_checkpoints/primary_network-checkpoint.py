import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from hypernet_modules import *


class PrimaryNetwork(nn.Module):
    
    """
    Primary Network to be trained. Overall structure is same with ResNet.
    Each kernels have degrees of fredom 64*scale_rate. Here, scale_rate means num of embeddings used to make that kernel.
    
    block : block component for ResNet. BasicBlock is implemented in this code and named as ResBlock.
    layers : [layer1_size, layer2_size, layer3_size] ex) [7, 7, 7] for Res44
    num_classes : 10 for cifar-10, 100 for cifar-100
    """
    
    def __init__(self, block, layers, num_classes=10, z_dim=64, unit_size=16):
        super(PrimaryNetwork, self).__init__()
        self.z_dim = z_dim
        self.unit_size = unit_size
        self.inplanes = unit_size
        self.hypernet = HyperNetwork(z_dim=self.z_dim, hidden_size=self.z_dim, kernel_size=3, unit_size=self.unit_size)
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, self.unit_size, layers[0])
        self.layer2 = self._make_layer(block, self.unit_size*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.unit_size*4, layers[2], stride=2)
        
        self.embed1 = self._make_embedding(i=0, layer=layers[0])
        self.embed2 = self._make_embedding(i=1, layer=layers[1])
        self.embed3 = self._make_embedding(i=2, layer=layers[2])
        self.tembed = nn.ModuleList([Embedding(z_dim=self.z_dim, scale_factor=[1, 1]), Embedding(z_dim=self.z_dim, scale_factor=[1, 1])])
        
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.unit_size*4, num_classes)
                
    
    def _make_layer(self, block, channel_num, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.inplanes != channel_num:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, channel_num, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel_num)
            )
            
        layer = nn.ModuleList()
        layer.append(block(self.inplanes, channel_num, stride, downsample))
        self.inplanes = channel_num
        
        for _ in range(1, blocks):
            layer.append(block(self.inplanes, channel_num))
            
        return layer

    
    def _make_embedding(self, i, layer):
        embeddings = nn.ModuleList()
        
        block_embed = nn.ModuleList()
        if i==0:
            block_embed.append(Embedding(z_dim=self.z_dim, scale_factor=[1, 1]))
            block_embed.append(Embedding(z_dim=self.z_dim, scale_factor=[1, 1]))
        else:
            block_embed.append(Embedding(z_dim=self.z_dim, scale_factor=[i, 2**i]))
            block_embed.append(Embedding(z_dim=self.z_dim, scale_factor=[2**i, 2**i]))
        
        embeddings += block_embed
        
        for _ in range(layer-1):
            block_embed = []
            block_embed.append(Embedding(z_dim=self.z_dim, scale_factor=[2**i, 2**i]))
            block_embed.append(Embedding(z_dim=self.z_dim, scale_factor=[2**i, 2**i]))
            embeddings += block_embed
            
        return embeddings


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        for i, block in enumerate(self.layer1):
            conv1_w, conv2_w = self.embed1[2*i](self.hypernet), self.embed1[2*i+1](self.hypernet)
            x = block(x, conv1_w, conv2_w)
            
        for i, block in enumerate(self.layer2):
            conv1_w, conv2_w = self.embed2[2*i](self.hypernet), self.embed2[2*i+1](self.hypernet)
            x = block(x, conv1_w, conv2_w)
            
        for i, block in enumerate(self.layer3):
            conv1_w, conv2_w = self.embed3[2*i](self.hypernet), self.embed3[2*i+1](self.hypernet)
            x = block(x, conv1_w, conv2_w)
            
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    