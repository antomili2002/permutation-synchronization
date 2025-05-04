import torch
import pytest
from permsync.perm_sync_batched import perm_sync_batched, error_against_ground_truth_batched


def generate_ground_truth_batch(B, N, n):
    gt_perms = torch.zeros(B, N, n, n)
    for b in range(B):
        for i in range(N):
            idx = torch.randperm(n)
            P = torch.zeros(n, n)
            P[torch.arange(n), idx] = 1.0
            gt_perms[b, i] = P
    return gt_perms

def add_noise_batch(P_batch, noise_level=0.3):
    B, N, n, _ = P_batch.shape
    P_noisy = P_batch.clone()
    for b in range(B):
        for i in range(N):
            num_swaps = int(noise_level * n)
            for _ in range(num_swaps):
                i1, i2 = torch.randint(0, n, (2,))
                if i1 != i2:
                    tmp = P_noisy[b, i, i1].clone()
                    P_noisy[b, i, i1] = P_noisy[b, i, i2]
                    P_noisy[b, i, i2] = tmp
    return P_noisy

def generate_pairwise_batch(P_batch):
    B, N, n, _ = P_batch.shape
    T = torch.zeros(B, N, N, n, n)
    for b in range(B):
        for i in range(N):
            for j in range(N):
                T[b, i, j] = P_batch[b, i] @ P_batch[b, j].T
    return T

def compute_accuracy_batched(tau: torch.Tensor, gt_perms: torch.Tensor) -> torch.Tensor:
    """
    Computes accuracy of tau (B, N, N, n, n) against ground truth permutations (B, N, n, n).
    Accuracy is measured by comparing argmax rows of predicted and true pairwise permutation matrices.

    Returns:
        accuracies: Tensor of shape (B,) with accuracy per batch
    """
    B, N, _, n, _ = tau.shape
    accuracies = torch.zeros(B, device=tau.device)

    for b in range(B):
        correct = 0
        total = 0
        for i in range(N):
            for j in range(N):
                if i >= j:
                    continue
                pred = tau[b, i, j]
                gt = gt_perms[b, i] @ gt_perms[b, j].T
                pred_vec = torch.argmax(pred, dim=1)
                gt_vec = torch.argmax(gt, dim=1)
                correct += (pred_vec == gt_vec).sum().item()
                total += n
        accuracies[b] = correct / total if total > 0 else 0.0
    return accuracies

@pytest.mark.parametrize("B,N,n", [
    (2, 4, 5),
    (3, 5, 6),
])
@pytest.mark.parametrize("noise", [0.0, 0.1, 0.2])
def test_perm_sync_batched(B, N, n, noise):
    gt_perms = generate_ground_truth_batch(B, N, n)
    noisy_perms = add_noise_batch(gt_perms, noise_level=noise)
    T = generate_pairwise_batch(noisy_perms)

    tau = perm_sync_batched(T)

    accs = compute_accuracy_batched(tau, gt_perms)
    errs = error_against_ground_truth_batched(tau, gt_perms)

    for b in range(B):
        print(f"[B={b}, N={N}, n={n}, noise={noise}] Accuracy = {accs[b]:.4f}, Error = {errs[b]:.4f}")

        if noise == 0.0:
            assert accs[b] == 1.0, "Perfect input should yield perfect accuracy"
            assert errs[b] == 0.0, "Error should be zero for perfect input"
        elif noise <= 0.1:
            assert accs[b] > 0.95, "Accuracy should be high with low noise"
        elif noise <= 0.2:
            assert accs[b] > 0.60, "Accuracy should degrade but stay acceptable"
