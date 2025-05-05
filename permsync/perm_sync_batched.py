import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from permsync.perm_sync import perm_sync

def error_batched(T: torch.tensor, N: int, n: int):
    B = T.shape[0]
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
                    gt_P_ij = gt_perms[b, i] @ gt_perms[b, j].T
                    total_errs[b] += torch.sum(torch.abs(tau[b, i, j] - gt_P_ij)) / 2
    return total_errs

def hungarian_matching(cost_matrix: torch.Tensor) -> torch.Tensor:
    N, M = cost_matrix.shape
    assert N == M, "Hungarian requires square matrices"
    perm = torch.zeros_like(cost_matrix)
    row, col = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    perm[row, col] = 1.0
    return perm

def perm_sync_batched(T: torch.Tensor) -> torch.Tensor:
    """
    Batched permutation synchronization via eigen-decomposition and Hungarian matching.
    
    Args:
        T: Tensor of shape (B, N, N, n, n), where
            - B = batch size
            - N = number of elements/nodes
            - n = permutation matrix size
    
    Returns:
        tau: Tensor of shape (B, N, N, n, n)
    """
    B, N, _, n, _ = T.shape
    device = T.device
    taus = []

    for b in range(B):
        T_b = T[b]  # Shape: (N, N, n, n)
        
        # Build big matrix T_
        T_ = torch.zeros((N * n, N * n), device=device)
        for i in range(N):
            for j in range(N):
                T_[i*n:(i+1)*n, j*n:(j+1)*n] = T_b[i, j]

        # Eigendecomposition
        T_ += 1e-4 * torch.eye(N * n, device=device)
        values, vectors = torch.linalg.eigh(T_)
        values = - torch.abs(values)
        sorted_index = torch.argsort(values)
        U = torch.sqrt(torch.tensor(float(N), device=device)) * vectors[:, sorted_index[:n]]
        
        # Build sigma list
        sigma = []
        for i in range(N):
            A = U[i * n : (i + 1) * n, :]
            B = U[:n, :].transpose(0, 1)
            P = (A @ B).real
            cost_matrix = torch.max(P) - P
            perm = hungarian_matching(cost_matrix)
            sigma.append(perm)
        
        # Reconstruct tau from sigma
        tau_b = torch.zeros((N, N, n, n), device=device)
        for i in range(N):
            for j in range(N):
                tau_b[i, j] = torch.matmul(sigma[i], torch.linalg.inv(sigma[j]))
        taus.append(tau_b)
    
    return torch.stack(taus, dim=0)  # Shape: (B, N, N, n, n)