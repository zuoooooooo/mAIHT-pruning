import math
import time

import torch
import torch.nn as nn
import transformers
from .PCG import PCG
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class APPruner:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.shape[0]
        self.columns = layer.weight.shape[1]
        self.XX = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        X = inp.reshape(-1, inp.shape[-1]).float()
        self.XX += X.T.matmul(X)
        
    def Hard_threshold(W,lam):
        mask = torch.abs(W) >= lam
        W = W * mask
        return W
        
    def projection(tensor, k):
        flat_tensor = tensor.reshape(-1)
        abs_tensor = flat_tensor.abs()
        _, topk_indices = torch.topk(abs_tensor, k)
        mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
        mask[topk_indices] = True
        flat_tensor[~mask] = 0
        result = flat_tensor.view(tensor.shape)
        return result

    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, percdamp=.1 ,  iters=20, per_out=False,lam=0.00001,alpha=0.001
    ):
        XX = self.XX  
        k=int(self.rows*self.columns*(1-sparsity))
        print(XX)
        norm = torch.diag(XX).sqrt() + 1e-8
        print(norm.min(), norm.max())
        XX = XX / norm
        XX = (XX.T / norm).T
        W = (self.layer.weight.float().detach() * norm).T

        error = np.zeros(iters)
        e=torch.trace(W.T@XX@W)



        if alpha==0:
            eigenvalues, eigenvectors = torch.linalg.eigh(XX)
            max_eigenvalue = eigenvalues[-1]
            alpha=0.95/max_eigenvalue
        print(alpha)

        
        sorted_tensor, _ = torch.sort(W.reshape(-1).abs(), descending=True)
        index_95_percentile = int(0.95 * len(sorted_tensor))
        value_at_90_percentile = sorted_tensor[index_95_percentile]

        lam=(value_at_90_percentile*value_at_90_percentile/2/alpha).item()
        W=APPruner.Hard_threshold(W,math.sqrt(2*lam*alpha))
        W_0=W.clone()
        W_1=W.clone()
        t_1=1
        t_0=0
    
        for i in range(iters):
            s_true=(1-W_1.count_nonzero()/self.columns/self.rows).item()
            lam=lam*(1+sparsity*0.96-s_true)
            
            V=W_1+(t_0-1)/(t_1)*(W_1-W_0)
            W_0=W_1
            W_1=APPruner.Hard_threshold(V-alpha*torch.matmul(XX,V-W),math.sqrt(2*lam*alpha))
            t_0=t_1
            t_1=(math.sqrt(4*(t_1**2)+1)+1)/2

        flat_tensor = W_1.reshape(-1)
        abs_tensor = flat_tensor.abs()
        _, topk_indices = torch.topk(abs_tensor, k)
        mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
        mask[topk_indices] = 1
        mask=mask.view(W_1.shape)
        W_1=W_1*mask

        # W_1=PCG(XX,W,W_1,mask)
        
        out=(W_1.T)/norm
        out=(APPruner.projection(W_1,k).T/norm)
        self.XX = None
        del XX
        print((out == 0).sum().item() / out.numel())
        self.layer.weight.data = out.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.XX = None
        torch.cuda.empty_cache()

