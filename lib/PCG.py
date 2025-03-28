import math
import time

import torch
import torch.nn as nn
import transformers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def PCG(XX, W, W0, S, tol=1e-6, max_iter=20):
    # Initialize variables
    R0 = XX @ (W - W0) *S
    M_inv = torch.diag(1 / torch.diag(XX))
    Zt = M_inv @ R0
    W_t = W0.clone()
    R_t = R0
    P_t = Zt.clone()

    for t in range(max_iter):
        # Compute alpha_t
        alpha_t = torch.trace(R_t.T @ Zt) / torch.trace(P_t.T @ XX @ P_t)
        # Update W_t
        W_t = W_t + alpha_t * P_t
        # Update R_t+1
        R_t_new = R_t - alpha_t * (XX @ P_t)
        R_t_new = R_t_new * S  # Project onto support S
        # Check convergence
        if torch.norm(R_t_new) < tol:
            break
        # Update Z_t+1
        Z_t_new = M_inv @ R_t_new
        # Compute beta_t
        beta_t = torch.trace(R_t_new.T @ Z_t_new) / torch.trace(R_t.T @ Zt)
        P_t = Z_t_new + beta_t * P_t
        R_t = R_t_new
        Zt = Z_t_new
    return W_t




