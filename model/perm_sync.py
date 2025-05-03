import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def error(T: torch.tensor, N: int, n: int):
    error = 0
    for i in range(N):
        for j in range(N):
            if i < j:
                error += torch.sum(torch.abs(T[i, j] - torch.eye(n))) / 2
    return error.item()

def error_against_ground_truth(tau, gt_perms):
    N = len(gt_perms)
    n = gt_perms[0].shape[0]
    total_err = 0
    for i in range(N):
        for j in range(N):
            if i < j:
                gt_P_ij = gt_perms[i] @ torch.linalg.inv(gt_perms[j])
                total_err += torch.sum(torch.abs(tau[i, j] - gt_P_ij)) / 2
    return total_err

def hungarian_matching(cost_matrix: torch.tensor):
    N, M = cost_matrix.shape
    assert N == M, "Hungarian requires square matrices"
    
    perm = torch.zeros_like(cost_matrix)
    cost = cost_matrix.detach().cpu().numpy()
    row, col = linear_sum_assignment(cost)
    perm[row, col] = 1.0
    return perm


def perm_sync(T: torch.tensor, N: int, n: int):
    T_ = torch.zeros((N*n, N*n))
    for i in range(N):
        for j in range(N):
            T_[i * n : (i + 1) * n, j * n : (j + 1) * n] = T[i, j]

    values, vectors = torch.linalg.eigh(T_ + 1e-4 * torch.eye(N*n))
    values = - torch.abs(values)
    sorted_index = torch.argsort(values)
    
    U = torch.sqrt(torch.tensor(float(N))) * vectors[:, sorted_index[:n]]
    sigma = []
    
    for i in range(N):
        A = U[i * n : (i + 1) * n, :]
        B = U[: n, : n].transpose(0, 1)
        P = (A @ B).real
        cost_matrix = torch.max(P) - P
        perm = hungarian_matching(cost_matrix)
        sigma.append(perm)

    tau = torch.zeros((N, N, n, n))
    for i in range(N):
        for j in range(N):
            tau[i, j] = torch.matmul(sigma[i], torch.linalg.inv(sigma[j]))
    
    return tau