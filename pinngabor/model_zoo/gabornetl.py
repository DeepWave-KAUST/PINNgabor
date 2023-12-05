import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Basicblock(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(Basicblock, self).__init__()
        self.layer1 = nn.Linear(in_planes,out_planes)

    def forward(self, x):
        out = torch.sin(self.layer1(x))
        return out

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear')!=-1:
        nn.init.xavier_normal_(m.weight,gain=1)
        nn.init.constant_(m.bias, 0)
    if classname.find('Conv')!=-1:
        nn.init.xavier_normal_(m.weight,gain=1)
        nn.init.constant_(m.bias,0)
        
class GaborLayer(nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """

    def __init__(self, in_features, out_features, freq, last_layer_type, alpha=1.0, beta=1.0):
        super().__init__()
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.phi = nn.Parameter(
            torch.distributions.Uniform(-np.pi, np.pi).sample((out_features,))
        )
        self.omega = torch.ones(in_features,out_features) * freq
        self.last_layer_type = last_layer_type
        return

    def forward(self, x, v):
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        if self.last_layer_type == 0:
            return torch.sin(torch.matmul(x * v,self.omega.to(x.device)) + self.phi) * torch.exp(-0.5 * D * self.gamma[None, :]) # sin(freq / v * x + phi) * exp(-0.5 * D * gamma) # navie version
        elif self.last_layer_type == 1:
            return torch.cos(torch.matmul(x * v,self.omega.to(x.device)) + self.phi) * torch.exp(-0.5 * D * self.gamma[None, :]), torch.sin(torch.matmul(x * v,self.omega.to(x.device)) + self.phi) * torch.exp(-0.5 * D * self.gamma[None, :]) # sin(freq / v * x + phi) * exp(-0.5 * D * gamma)
        else:
            print('unknown last layer type')
            
class GaborLayerD(nn.Module):
    """
    Gabor-like filter as used in GaborNet. but the center point D is learnable
    """

    def __init__(self, in_features, out_features, freq, last_layer_type, alpha=1.0, beta=1.0):
        super().__init__()
        # self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.phi = nn.Parameter(
            torch.distributions.Uniform(-np.pi, np.pi).sample((out_features,))
        )
        self.omega = torch.ones(in_features,out_features) * freq
        self.last_layer_type = last_layer_type
        return

    def forward(self, x, D, v):
        if self.last_layer_type == 2:
            return torch.sin(torch.matmul(x * v,self.omega.to(x.device)) + self.phi) * torch.exp(-0.5 * D * self.gamma[None, :]) # sin(freq / v * x + phi) * exp(-0.5 * D * gamma) # navie version
        elif self.last_layer_type == 3:
            return torch.cos(torch.matmul(x * v,self.omega.to(x.device)) + self.phi) * torch.exp(-0.5 * D * self.gamma[None, :]), torch.sin(torch.matmul(x * v,self.omega.to(x.device)) + self.phi) * torch.exp(-0.5 * D * self.gamma[None, :]) # sin(freq / v * x + phi) * exp(-0.5 * D * gamma)
        else:
            print('unknown last layer type')
            
class GaborLayerd(nn.Module):
    """
    Gabor-like filter as used in GaborNet. but the center point d is learnable
    """

    def __init__(self, in_features, out_features, freq, last_layer_type, alpha=1.0, beta=1.0):
        super().__init__()
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.phi = nn.Parameter(
            torch.distributions.Uniform(-np.pi, np.pi).sample((out_features,))
        )
        self.omega = torch.ones(in_features,out_features) * freq
        self.last_layer_type = last_layer_type
        return

    def forward(self, x, d, v):
        D = (
            (x ** 2).sum(-1)[..., None]
            + (d ** 2).sum(-1)[..., None]
            - 2 * torch.einsum("ij,ik->ik", x, d)
        )
        if self.last_layer_type == 4:
            return torch.sin(torch.matmul(x * v,self.omega.to(x.device)) + self.phi) * torch.exp(-0.5 * D * self.gamma[None, :]) # sin(freq / v * x + phi) * exp(-0.5 * D * gamma) # navie version
        elif self.last_layer_type == 5:
            return torch.cos(torch.matmul(x * v,self.omega.to(x.device)) + self.phi) * torch.exp(-0.5 * D * self.gamma[None, :]), torch.sin(torch.matmul(x * v,self.omega.to(x.device)) + self.phi) * torch.exp(-0.5 * D * self.gamma[None, :]) # sin(freq / v * x + phi) * exp(-0.5 * D * gamma)
        else:
            print('unknown last layer type')
    

class GaborNetL(nn.Module):
    '''
    Gabor layer applied to the last layer of the PINN
    1: the version adding one linear layer after the gabor layer
    2: direct sum the real and imaginary part of the gabor layer
    3: add another branch for center ppint of the gabor layer
    '''
    
    def __init__(self, in_channels, out_channels, layers, **kwargs) -> None:
        super(GaborNetL, self).__init__()
        self.layers = layers
        self.in_planes = in_channels
        self.last_layer_type = kwargs['last_layer_type']
        if self.last_layer_type == 0:
            self.layer1 = self._make_layer(Basicblock,self.layers[:len(layers)-1]).apply(weight_init)
            self.filter = GaborLayer(in_channels, layers[-1], **kwargs)
            self.linear = nn.Linear(layers[-2], layers[-1])
            self.out_linear = nn.Linear(layers[-1], out_channels) # navie version
        elif self.last_layer_type == 1:
            self.layer1 = self._make_layer(Basicblock,self.layers[:len(layers)-1]).apply(weight_init)
            self.filter = GaborLayer(in_channels, layers[-1], **kwargs)
            self.linear = nn.Linear(layers[-2], layers[-1])
        elif self.last_layer_type == 2:
            self.layer1 = self._make_layer(Basicblock,self.layers[:len(layers)-1]).apply(weight_init)
            self.filter = GaborLayerD(in_channels, layers[-1], **kwargs)
            self.linear = nn.Linear(layers[-2], layers[-1])
            self.linearD = nn.Linear(in_channels, layers[-1])
            self.out_linear = nn.Linear(layers[-1], out_channels) # navie version
        elif self.last_layer_type == 3:
            self.layer1 = self._make_layer(Basicblock,self.layers[:len(layers)-1]).apply(weight_init)
            self.filter = GaborLayerD(in_channels, layers[-1], **kwargs)
            self.linear = nn.Linear(layers[-2], layers[-1])
            self.linearD = nn.Linear(in_channels, layers[-1])
        elif self.last_layer_type == 4:
            self.layer1 = self._make_layer(Basicblock,self.layers[:len(layers)-1]).apply(weight_init)
            self.filter = GaborLayerd(in_channels, layers[-1], **kwargs)
            self.linear = nn.Linear(layers[-2], layers[-1])
            self.lineard = nn.Linear(in_channels, layers[-1])
            self.out_linear = nn.Linear(layers[-1], out_channels) # navie version
        elif self.last_layer_type == 5:
            self.layer1 = self._make_layer(Basicblock,self.layers[:len(layers)-1]).apply(weight_init)
            self.filter = GaborLayerd(in_channels, layers[-1], **kwargs)
            self.linear = nn.Linear(layers[-2], layers[-1])
            self.lineard = nn.Linear(in_channels, layers[-1])
        else:
            raise ValueError('unknown last layer type')
            
        
    def _make_layer(self, block, layers):
        layers_net = []
        for layer in layers:
            layers_net.append(block(self.in_planes,layer))
            self.in_planes = layer
        return nn.Sequential(*layers_net)
    
    def forward(self, x, v=1.0/1.5):
        out = self.layer1(x)
        if self.last_layer_type==0:
            # the version adding one linear layer after the gabor layer
            out = self.filter(x,v) * self.linear(out)
            out = self.out_linear(out)
        elif self.last_layer_type==1:
            # direct sum the real and imaginary part of the gabor layer
            out_real, out_imag = self.filter(x,v)
            out_real, out_imag = out_real * self.linear(out), out_imag * self.linear(out)
            out = torch.cat([torch.sum(out_real, dim=1, keepdim=True), torch.sum(out_imag, dim=1, keepdim=True)], dim=1)
        elif self.last_layer_type==2:
            out = self.filter(x, self.linearD(x), v) * self.linear(out)
            out = self.out_linear(out)
        elif self.last_layer_type==3:
            out_real, out_imag = self.filter(x, self.linearD(x), v)
            out_real, out_imag = out_real * self.linear(out), out_imag * self.linear(out)
            out = torch.cat([torch.sum(out_real, dim=1, keepdim=True), torch.sum(out_imag, dim=1, keepdim=True)], dim=1)
        elif self.last_layer_type==4:
            out = self.filter(x, self.lineard(x), v) * self.linear(out)
            out = self.out_linear(out)
        elif self.last_layer_type==5:
            out_real, out_imag = self.filter(x, self.lineard(x), v)
            out_real, out_imag = out_real * self.linear(out), out_imag * self.linear(out)
            out = torch.cat([torch.sum(out_real, dim=1, keepdim=True), torch.sum(out_imag, dim=1, keepdim=True)], dim=1)
        return out
    