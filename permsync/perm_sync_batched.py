import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from permsync.perm_sync import perm_sync

def error_batched(T: torch.tensor, N: int, n: int):
    B = T.shape(0)
    device = T.device
    errors = torch.zeros(B, device=device)
    
    for b in range(B):
        for i in range(N):
            for j in range(N):
                if i < j:
                    errors[b] += torch.sum(torch.abs(T[b, i, j] - torch.eye(n, device=device))) / 2
    return errors

def error_against_ground_truth_batched(tau: torch.tensor, gt_perms:  torch.tensor):
    B, N, _, n, _ = tau.shape
    device = tau.device
    
    total_errs = torch.zeros(B, device=device)
    
    for b in range(B):
        for i in range(N):
            for j in range(N):
                if i < j:
                    gt_P_ij = gt_perms[b, i] @ torch.linalg.inv(gt_perms[b, j])
                    total_errs[b] += torch.sum(torch.abs(tau[b, i, j] - gt_P_ij)) / 2
    return total_errs

def perm_sync_batched(T: torch.tensor, N: int, n: int):
    B, N, _, n, _ = T.shape
    device = T.device
    
    tau = torch.zeros_like(T, device=device)
    for b in range(B):
        tau[b] = perm_sync(T[b], N, n)
    return tau