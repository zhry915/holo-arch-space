"""
Holo-MobilenetV3: Holomorphic MobileNet V3 with Complex BatchNorm and GELU.
Modified for Hugging Face Spaces deployment.
"""
import torch
import torch.nn as nn
import math
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class HoloBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x):
        if not torch.is_complex(x):
            return nn.functional.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps
            )
            
        # Complex forward pass (used in eval mode during HoloPGFEngine spectral extraction)
        mean = self.running_mean.view(1, -1, 1, 1).to(x.dtype)
        var = self.running_var.view(1, -1, 1, 1).to(x.dtype)
        weight = self.weight.view(1, -1, 1, 1).to(x.dtype) if self.affine else 1.0
        bias = self.bias.view(1, -1, 1, 1).to(x.dtype) if self.affine else 0.0
        
        std_inv = torch.rsqrt(var + self.eps)
        return (x - mean) * std_inv * weight + bias

class HoloGELU(nn.Module):
    def forward(self, x):
        if not torch.is_complex(x):
            return nn.functional.gelu(x)
        # Complex GELU approximation (analytical continuation)
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def replace_with_holo_modules(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            new_bn = HoloBatchNorm2d(
                child.num_features, child.eps, child.momentum, 
                child.affine, child.track_running_stats
            )
            if child.affine:
                new_bn.weight.data.copy_(child.weight.data)
                new_bn.bias.data.copy_(child.bias.data)
            if child.track_running_stats:
                new_bn.running_mean.data.copy_(child.running_mean.data)
                new_bn.running_var.data.copy_(child.running_var.data)
                new_bn.num_batches_tracked.data.copy_(child.num_batches_tracked.data)
            setattr(module, name, new_bn)
        elif isinstance(child, (nn.Hardswish, nn.ReLU, nn.ReLU6)):
            setattr(module, name, HoloGELU())
        elif isinstance(child, nn.Hardsigmoid):
            setattr(module, name, nn.Sigmoid())
        elif isinstance(child, nn.Dropout):
            setattr(module, name, nn.Identity())
        else:
            replace_with_holo_modules(child)

def get_holo_mobilenet_v3(num_classes=21):
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    replace_with_holo_modules(model)
    return model
