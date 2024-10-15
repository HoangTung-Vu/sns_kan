import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Wavelet:
    """All Wavelet family
    Returns:
        Use some function below for WavKANLayer
    """
    @staticmethod
    def mexican_hat(x):
        term1 = ((x ** 2) - 1)
        term2 = torch.exp(-0.5 * x ** 2)
        wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
        return wavelet
    
    @staticmethod 
    def morlet(x):
        omega0 = 5.0  # Central frequency
        real = torch.cos(omega0 * x)
        envelope = torch.exp(-0.5 * x ** 2)
        wavelet = envelope * real
        return wavelet
    
    @staticmethod
    def dog(x):
        return -x * torch.exp(-0.5 * x ** 2)
    
    @staticmethod
    def meyer(x):
        v = torch.abs(x)
        pi = math.pi

        def meyer_aux(v):
            return torch.where(v <= 1/2, torch.ones_like(v), torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

        def nu(t):
            return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

        wavelet = torch.sin(pi * v) * meyer_aux(v)
        return wavelet

class WavKANLayer_test(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type=Wavelet.mexican_hat, base_function=nn.SiLU(), device='cpu', lin_enable=False):
        """_summary_

        Args:
            in_features (int): number of input dimension
            out_features (int): number of output dimension
            wavelet_type (Wavelet class function, optional): wavelet type. Defaults to Wavelet.mexican_hat.
            base_function (Base function for linear part, optional): _description_. Defaults to nn.SiLU().
            device (str, optional): device. Defaults to 'cpu'.
            lin_enable (bool): Linear part enable Defaults to False.
        """
        super(WavKANLayer_test, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        self.lin_enable = lin_enable

        # Layer parameters
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.lin_weights = nn.Parameter(torch.Tensor(out_features, in_features)) if lin_enable else None

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        if self.lin_enable:
            nn.init.kaiming_uniform_(self.lin_weights, a=math.sqrt(5))

        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # Efficient broadcasting without unnecessary expansion
        # Perform Wavelet transform
        wav = self.wavelet_type((x.unsqueeze(1) - self.translation) / self.scale)
        wavelet_output = torch.einsum('boc,oc->bo', wav, self.wavelet_weights)

        if self.lin_enable:
            # Compute linear part only if enabled
            bias = F.linear(self.base_function(x), self.lin_weights)
        else:
            bias = 0
        
        return self.bn(wavelet_output + bias)
