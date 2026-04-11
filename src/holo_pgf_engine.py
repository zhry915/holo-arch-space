"""
Holo-PGF Engine: machine-precision landscape coefficient extraction via Cauchy FFT.
Modified for Hugging Face Spaces deployment.
"""

import torch
import numpy as np
import math
from typing import Callable, List

def holo_cross_entropy(logits, target):
    """
    Holomorphic CrossEntropy for ComplexDouble.
    """
    logits_real = logits.real
    max_val = torch.max(logits_real, dim=-1, keepdim=True)[0]
    logits_stable = logits - max_val.to(logits.dtype)
    
    log_sum_exp = torch.log(torch.exp(logits_stable).sum(dim=-1, keepdim=True))
    log_probs = logits_stable - log_sum_exp
    
    return -log_probs[torch.arange(logits.shape[0]), target].mean()

class HoloPGFEngine:
    def __init__(self, model: torch.nn.Module, M: int = 16, eta: float = 1e-3):
        self.model = model
        self.M = M
        self.eta = eta
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def extract_spectrum(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        loss_fn: Callable,
        v: List[torch.Tensor] = None,
        target_module: torch.nn.Module = None,
        trainable_only: bool = True
    ) -> torch.Tensor:
        if target_module:
            params_to_perturb = list(target_module.parameters())
        else:
            params_to_perturb = list(self.model.parameters())
            
        if trainable_only:
            params_to_perturb = [p for p in params_to_perturb if p.requires_grad]
        
        if v is None:
            v = [torch.randn_like(p) for p in params_to_perturb]
            v_norm = torch.sqrt(sum((vi**2).sum() for vi in v))
            v = [vi / (v_norm + 1e-12) for vi in v]

        orig_dtype = next(self.model.parameters()).dtype
        orig_params = [p.clone() for p in self.model.parameters()]
        
        self.model.to(torch.complex128)
        
        angles = torch.linspace(0, 2 * np.pi, self.M + 1, device=self.device, dtype=torch.float64)[:-1]
        z_samples = self.eta * torch.exp(1j * angles)
        
        loss_samples = []
        
        current_all_params = list(self.model.parameters())
        if target_module:
            current_perturb_params = list(target_module.parameters())
        else:
            current_perturb_params = current_all_params
            
        if trainable_only:
            current_perturb_params = [p for p in current_perturb_params if p.requires_grad]
        
        orig_perturb_params_complex = [p.to(torch.complex128) for p in current_perturb_params]
        
        for z in z_samples:
            for p, p_orig_c, vi in zip(current_perturb_params, orig_perturb_params_complex, v):
                p.copy_(p_orig_c + z * vi.to(torch.complex128))
            
            logits = self.model(x.to(torch.complex128))
            
            if torch.is_complex(logits):
                loss = holo_cross_entropy(logits, y)
            else:
                loss = loss_fn(logits, y)
            loss_samples.append(loss.to(torch.complex128))
            
        for p, p_orig in zip(self.model.parameters(), orig_params):
            p.copy_(p_orig.to(torch.complex128))
        self.model.to(orig_dtype)
                
        loss_samples = torch.stack(loss_samples)
        
        psi_raw = torch.fft.fft(loss_samples) / self.M
        
        ks = torch.arange(self.M, device=self.device, dtype=torch.float64)
        factorials = torch.tensor([float(math.factorial(int(k))) for k in ks], 
                                  device=self.device, dtype=torch.float64)
        
        psi_k = (psi_raw * factorials) / (self.eta ** ks)
        
        return psi_k
