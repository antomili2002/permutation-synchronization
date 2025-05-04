import torch
import pytest
from permsync.perm_sync import perm_sync, error, error_against_ground_truth
from scipy.optimize import linear_sum_assignment

def generate_ground_truth(N, n):
    """Ground-truth permutation matrices"""
    perms = []
    for _ in range(N):
        idx = torch.randperm(n)
        P = torch.zeros(n, n)
        P[torch.arange(n), idx] = 1.0
        perms.append(P)
    return perms

def add_noise(P, noise_level=0.3):
    """Randomly corrupt a permutation matrix"""
    n = P.size(0)
    P_noisy = P.clone()
    num_swaps = int(noise_level * n)
    for _ in range(num_swaps):
        i, j = torch.randint(0, n, (2,))
        if i != j:
            P_noisy[[i, j]] = P_noisy[[j, i]]
    return P_noisy

def generate_pairwise(P_list):
    """Build T[i, j] = P[i] @ P[j].T"""
    N = len(P_list)
    n = P_list[0].size(0)
    T = torch.zeros((N, N, n, n))
    for i in range(N):
        for j in range(N):
            T[i, j] = P_list[i] @ P_list[j].T
    return T

def compute_accuracy(tau, perms):
    """Compute accuracy of synchronized result vs ground truth"""
    N = len(perms)
    n = perms[0].size(0)
    correct = 0
    total = 0
    for i in range(N):
        for j in range(N):
            if i >= j:
                continue
            pred = tau[i, j]
            gt = perms[i] @ perms[j].T
            # Compare argmax (permutation vector) row-wise
            pred_vec = torch.argmax(pred, dim=1)
            gt_vec = torch.argmax(gt, dim=1)
            correct += (pred_vec == gt_vec).sum().item()
            total += n
    return correct / total

@pytest.mark.parametrize("N,n", [
    (3, 4),
    (4, 5),
    (5, 7),
    (6, 8),
    (8, 10),
    (10, 12),
    (12, 15)
])
@pytest.mark.parametrize("noise", [0.0, 0.1, 0.2, 0.4])
def test_perm_sync_accuracy_varied_noise(N, n, noise):
    gt_perms = generate_ground_truth(N, n)
    noisy_perms = [add_noise(P, noise_level=noise) for P in gt_perms]
    T = generate_pairwise(noisy_perms)

    tau = perm_sync(T, N, n)

    acc = compute_accuracy(tau, gt_perms)
    err = error_against_ground_truth(tau, gt_perms)

    print(f"[N={N}, n={n}, noise={noise}] Accuracy = {acc:.4f}, Error = {err:.4f}")

    # Assertions depend on noise level
    if noise == 0.0:
        assert acc == 1.0, "Perfect input should yield perfect accuracy"
        assert err == 0.0, "Error should be zero for perfect input"
    elif noise <= 0.1:
        assert acc > 0.95, "Accuracy should be high with low noise"
    elif noise <= 0.2:
        assert acc > 0.85, "Accuracy should degrade but stay acceptable"
    elif noise <= 0.4:
        assert acc > 0.7, "Still acceptable accuracy under moderate noise"