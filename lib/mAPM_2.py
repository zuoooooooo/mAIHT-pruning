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

class mAPPruner:
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
            mask = tensor != 0
            result = torch.zeros_like(tensor, dtype=torch.float)
            result[mask] = 1
            return result
        tensor1 = set_nonzero_to_one(tensor1)
        tensor2 = set_nonzero_to_one(tensor2)
        return torch.sum((tensor2-tensor1).abs())

    def fasterprune(
    self, sparsity, prune_n=0, prune_m=0, percdamp=.1, iters=20, per_out=False,
    lam=0.00001, alpha=0.001, tol=1
):
        XX = self.XX  
        k=int(self.rows*self.columns*(1-sparsity)+1)
        norm = torch.diag(XX).sqrt() + 1e-8
        XX = XX / norm
        XX = (XX.T / norm).T
        # Riged term
        rho0 = 0.1
        XX = XX+rho0*torch.eye(XX.shape[0]).to("cuda")
        W = (self.layer.weight.float().detach() * norm).T
        
        if alpha==0:
            XX1 = XX.cpu().numpy()
            values, _ = eigs(XX1, k=1, which='LM')
            max_eigenvalue = np.max(np.real(values))
            del XX1
            alpha=0.97/max_eigenvalue

        mask = None  # 初始化mask
        def generate_nm_mask(weight_tensor, n, m):
            with torch.no_grad():
                # 转置并reshape为(..., m)的组
                groups = weight_tensor.T.reshape(-1, m)
                # 找到每组中绝对值最大的n个元素
                _, topk_indices = torch.topk(groups.abs(), n, dim=1)
                # 生成mask
                mask = torch.zeros_like(groups, dtype=torch.float)
                mask.scatter_(1, topk_indices, 1)
                # 恢复原始形状并转置
                return mask.reshape(weight_tensor.T.shape).T

        # 主优化循环
        W_0 = W.clone()
        W_1 = W.clone()
        Z = W.clone()
        t_1 = 1
        t_0 = 0
        
        for i in range(iters):
            # 核心更新步骤
            Y = W_1 + (t_0-1)/t_1*(W_1 - W_0) + t_0/t_1*(Z - W_1)
            
            # 使用动态mask替代硬阈值算子
            if prune_n != 0:
                S1 = Y - alpha * torch.matmul(XX, Y - W)
                mask = generate_nm_mask(S1, prune_n, prune_m)
                Z = (Y - alpha * torch.matmul(XX, Y - W)) * mask
                
                S1 = W_1 - alpha * torch.matmul(XX, W_1 - W)
                mask = generate_nm_mask(S1, prune_n, prune_m)
                V = (W_1 - alpha * torch.matmul(XX, W_1 - W))* mask
            else:
                # 原有全局稀疏逻辑
                Z = APPruner.Hard_threshold(Y - alpha*torch.matmul(XX, Y-W), math.sqrt(2*lam*alpha))
                V = APPruner.Hard_threshold(W_1 - alpha*torch.matmul(XX, W_1-W), math.sqrt(2*lam*alpha))

            # 更新参数
            t_0, t_1 = t_1, (math.sqrt(4*t_1**2 +1)+1)/2
            
            # 选择更新方向
            if (torch.trace((2*W - V - Z).T@XX@(V - Z))/2 <= 0):
                W_0, W_1 = W_1, Z
            else:
                W_0, W_1 = W_1, V

        # 最终mask强化
        if prune_n != 0:
            mask = generate_nm_mask(W_1, prune_n, prune_m)
            W_1 = W_1 * mask
        else:
            abs_tensor = (W_1.reshape(-1)).abs()
            _, topk_indices = torch.topk(abs_tensor, k)
            mask = torch.zeros_like(abs_tensor, dtype=torch.bool)
            mask[topk_indices] = 1
            mask = mask.view(W_1.shape)
            W_1 = W_1 * mask

        for i in range(30):
            W_1 = (W_1 - alpha * torch.matmul(XX, W_1 - W)) * mask

        out = (W_1.T / norm)
        print((out == 0).sum().item() / out.numel())
        self.layer.weight.data = out.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        self.XX = None

    def free(self):
        self.XX = None
        torch.cuda.empty_cache()
