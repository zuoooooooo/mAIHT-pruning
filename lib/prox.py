import math
import time
import numpy as np
from scipy.sparse.linalg import eigs

import torch
import torch.nn as nn
import transformers
from datetime import datetime

from .APM import APPruner
from .PCG import PCG
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class ProxPruner:
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

    
    def support_diff_size(tensor1, tensor2):
        def set_nonzero_to_one(tensor):
              # Create a boolean mask where the tensor elements are non-zero
              mask = tensor != 0
              # Create a tensor of the same shape filled with zeros
              result = torch.zeros_like(tensor, dtype=torch.float)
              # Set the positions of the non-zero elements to 1
              result[mask] = 1
              return result
        tensor1=set_nonzero_to_one(tensor1)
        tensor2=set_nonzero_to_one(tensor2)
        return torch.sum((tensor2-tensor1).abs())

    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, percdamp=.1 ,  iters=20, per_out=False,lam=0.00001,alpha=0,tol=1
    ):
        
        XX = self.XX  
        k=int(self.rows*self.columns*(1-sparsity))
        norm = torch.diag(XX).sqrt() + 1e-8
        print(norm.min(), norm.max())
        XX = XX / norm
        XX = (XX.T / norm).T
        
        # Riged term
        rho0 = 0
        XX = XX+rho0*torch.eye(XX.shape[0]).to("cuda")
        W = (self.layer.weight.float().detach() * norm).T
        
        if alpha==0:
            XX1 = XX.cpu().numpy()
            values, _ = eigs(XX1, k=1, which='LM')
            max_eigenvalue = np.max(np.real(values))
            del XX1
            # eigenvalues, eigenvectors = torch.linalg.eigh(XX)
            # max_eigenvalue = eigenvalues[-1]
            alpha=0.95/max_eigenvalue

        # del eigenvectors
        # del eigenvalues
        
        sorted_tensor, _ = torch.sort(W.reshape(-1).abs(), descending=True)
        index = int(0.95 * len(sorted_tensor))
        value = sorted_tensor[index]

        lam=(value**2/2/alpha).item()
        
        W_0=W.clone()
        W_1=W.clone()
        Z=W.clone()
        t_1=1
        t_0=0


        error = np.zeros(iters)
        sparsities = np.zeros(iters)
        e=torch.trace(W.T@XX@W)


        for i in range(iters):
            s_true=(1-W_1.count_nonzero()/self.columns/self.rows).item()
            
            lam=lam*(1+sparsity*0.97-s_true)
            
            Y=W_1+(t_0-1)/(t_1)*(W_1-W_0)+t_0/t_1*(Z-W_1)
            Z=APPruner.Hard_threshold(Y-alpha*torch.matmul(XX,Y-W),math.sqrt(2*lam*alpha))
            V=APPruner.Hard_threshold(W_1-alpha*torch.matmul(XX,W_1-W),math.sqrt(2*lam*alpha))
            t_0=t_1
            t_1=(math.sqrt(4*(t_1**2)+1)+1)/2

          
            # W_0=W_1
            # W_1=V


            a=(torch.trace((2*W-V-Z).T@XX@(V-Z))/2) <= (lam*(torch.count_nonzero(V)-torch.count_nonzero(Z)))
            if a:
                W_0=W_1
                W_1=Z
            else:
                W_0=W_1
                W_1=V

            
            error[i]=torch.trace((W-W_1).T@XX@(W-W_1))/e
            sparsities[i]=(1-W_1.count_nonzero()/self.columns/self.rows).item()
                
        del W_0
        del Z
        del Y
        del V
        torch.cuda.empty_cache()
                
        # Get the current time and format it as a string for the file name
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"out/mAIHT/error{current_time}.txt"
        np.savetxt(file_name, error)

        file_name = f"out/mAIHT/sparsity{current_time}.txt"
        np.savetxt(file_name, sparsities)
      
        flat_tensor = W_1.reshape(-1)
        abs_tensor = flat_tensor.abs()
        _, topk_indices = torch.topk(abs_tensor, k)
        mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
        mask[topk_indices] = 1
        mask=mask.view(W_1.shape)
        W_1=W_1*mask

        XX = XX - rho0*torch.eye(XX.shape[0]).to("cuda")
        W_1=PCG(XX,W,W_1,mask)
        
        out=(W_1.T/norm)
        self.XX = None
       
        print((out == 0).sum().item() / out.numel())
        self.layer.weight.data = out.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.XX = None
        torch.cuda.empty_cache()

