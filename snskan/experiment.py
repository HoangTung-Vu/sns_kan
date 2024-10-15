import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WavKANLayer_test(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat', base_function=nn.SiLU(), device='cpu', lin_enable=False):
        """
        Args:
            in_features (int): number of input dimensions.
            out_features (int): number of output dimensions.
            wavelet_type (str, optional): The type of wavelet to use ('mexican_hat', 'morlet', 'dog', or 'meyer').
            base_function (torch.nn.Module, optional): Base function for linear part. Defaults to nn.SiLU().
            device (str, optional): Device ('cpu' or 'cuda'). Defaults to 'cpu'.
            lin_enable (bool, optional): Whether to enable linear part. Defaults to False.
        """
        super(WavKANLayer, self).__init__()
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

    def wavelet_transform(self, x_scaled):
        """Apply the selected wavelet function."""
        if self.wavelet_type == 'mexican_hat':
            term1 = (x_scaled ** 2) - 1
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
            del term1, term2  # Free memory after usage
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            del real, envelope  # Free memory after usage
        elif self.wavelet_type == 'dog':
            wavelet = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
        elif self.wavelet_type == 'meyer':
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1/2, torch.ones_like(v), torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * self.nu(2 * v - 1))))

            wavelet = torch.sin(pi * v) * meyer_aux(v)
            del v  # Free memory after usage
        else:
            raise ValueError(f"Unknown wavelet type: {self.wavelet_type}")

        return wavelet

    @staticmethod
    def nu(t):
        """Auxiliary function for the Meyer wavelet."""
        return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

    def forward(self, x):
        # Efficient broadcasting without unnecessary expansion
        x_scaled = (x.unsqueeze(1) - self.translation) / self.scale
        
        # Perform wavelet transform using the selected wavelet function
        wav = self.wavelet_transform(x_scaled)
        wavelet_output = torch.einsum('boc,oc->bo', wav, self.wavelet_weights)
        
        # Explicitly delete intermediate variables to free memory
        del x_scaled, wav  # Free memory after usage

        if self.lin_enable:
            # Compute linear part only if enabled
            bias = F.linear(self.base_function(x), self.lin_weights)
        else:
            bias = 0

        # Apply batch normalization and return final result
        output = self.bn(wavelet_output + bias)
        
        # Explicitly delete wavelet_output and bias after use
        del wavelet_output, bias
        
        return output
