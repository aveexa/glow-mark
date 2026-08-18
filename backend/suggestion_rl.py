"""Offline REINFORCE helpers for suggestion ranking (contract v1).

Metrics, sampling, and policy-gradient loss used when training the suggestion
ranker with listwise RL — not imported on the /analyze serve path.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

TOP_K = 4
RELEVANCE_GAINS = (3.0, 2.0, 1.0, 1.0)


def gold_relevance(
    priority_ids: Sequence[str],
    vocab: Sequence[str],
    gains: Sequence[float] = RELEVANCE_GAINS,
) -> torch.Tensor:
    """Build graded relevance vector shape (K,) from gold priority order."""
    index = {s: i for i, s in enumerate(vocab)}
    rel = torch.zeros(len(vocab), dtype=torch.float32)
    for rank, sid in enumerate(priority_ids):
        if rank >= len(gains):
            break
        if sid in index:
            rel[index[sid]] = float(gains[rank])
    return rel


def ndcg_at_k(
    pred_ids: Sequence[int],
    relevance: torch.Tensor,
    k: int = TOP_K,
) -> float:
    """NDCG@k for one predicted index list vs a graded relevance vector."""
    k = min(k, len(pred_ids))
    if k <= 0:
        return 0.0
    dcg = 0.0
    for t in range(k):
        idx = int(pred_ids[t])
        if 0 <= idx < relevance.numel():
            dcg += float(relevance[idx]) / math.log2(t + 2)
    # Ideal: top-k relevances by value
    top_rel, _ = torch.topk(relevance, k=min(k, int(relevance.numel())))
    idcg = 0.0
    for t, r in enumerate(top_rel.tolist()):
        if r <= 0:
            break
        idcg += float(r) / math.log2(t + 2)
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def map_at_k(
    pred_ids: Sequence[int],
    relevant_set: set[int],
    k: int = TOP_K,
) -> float:
    """Average precision @k under binary relevance (MAP component for one query)."""
    if not relevant_set:
        return 0.0
    k = min(k, len(pred_ids))
    hit_count = 0
    precisions: list[float] = []
    for rank in range(k):
        idx = int(pred_ids[rank])
        if idx in relevant_set:
            hit_count += 1
            precisions.append(hit_count / (rank + 1))
    if not precisions:
        return 0.0
    return float(sum(precisions) / len(precisions))


def greedy_top_k_indices(logits: torch.Tensor, k: int = TOP_K) -> torch.Tensor:
    """Deterministic top-k indices; logits (B, K) → LongTensor (B, k)."""
    k = min(k, logits.size(-1))
    _, indices = torch.topk(logits, k=k, dim=-1)
    return indices


def plackett_luce_sample(
    logits: torch.Tensor,
    k: int = TOP_K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sequential softmax sample without replacement (Plackett–Luce list policy).

    Returns:
        indices: (B, k) long
        log_prob: (B,) sum of step log-probs
        entropy: (B,) mean per-step entropy
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, K), got {tuple(logits.shape)}")
    batch, n_labels = logits.shape
    k = min(k, n_labels)
    masked = logits
    chosen: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for _ in range(k):
        dist = torch.distributions.Categorical(logits=masked)
        idx = dist.sample()  # (B,)
        log_probs.append(dist.log_prob(idx))
        entropies.append(dist.entropy())
        chosen.append(idx)
        # Mask selected positions with -inf for next step
        mask = torch.full_like(masked, 0.0)
        mask.scatter_(1, idx.unsqueeze(1), float("-inf"))
        masked = masked + mask
    indices = torch.stack(chosen, dim=1)
    log_prob = torch.stack(log_probs, dim=1).sum(dim=1)
    entropy = torch.stack(entropies, dim=1).mean(dim=1)
    return indices, log_prob, entropy


def reinforce_loss(
    log_prob: torch.Tensor,
    rewards: torch.Tensor,
    entropy: torch.Tensor,
    entropy_coef: float = 0.01,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """REINFORCE with batch-mean baseline + entropy bonus.

    L = -mean(w * (R - b) * log_pi) - entropy_coef * mean(H)
    When sample_weights is set (e.g. human_v1 trust), scale the PG term only.
    """
    if log_prob.shape != rewards.shape:
        raise ValueError("log_prob and rewards must share shape")
    b = rewards.mean()
    advantage = rewards - b
    pg_terms = -(advantage.detach() * log_prob)
    if sample_weights is not None:
        if sample_weights.shape != log_prob.shape:
            raise ValueError("sample_weights must match log_prob shape")
        w = sample_weights.to(dtype=pg_terms.dtype, device=pg_terms.device)
        pg = (pg_terms * w).mean()
    else:
        pg = pg_terms.mean()
    ent = entropy.mean()
    loss = pg - float(entropy_coef) * ent
    stats = {
        "loss": float(loss.detach()),
        "reward_mean": float(rewards.mean().detach()),
        "advantage_mean": float(advantage.mean().detach()),
        "entropy_mean": float(ent.detach()),
        "baseline": float(b.detach()),
    }
    return loss, stats


def batch_ndcg_rewards(
    pred_indices: torch.Tensor,
    relevance: torch.Tensor,
    k: int = TOP_K,
) -> torch.Tensor:
    """Compute NDCG@k reward per row for RL. pred (B,k), relevance (B,K) → (B,)."""
    rewards = []
    for i in range(pred_indices.size(0)):
        rewards.append(ndcg_at_k(pred_indices[i].tolist(), relevance[i], k=k))
    return torch.tensor(rewards, dtype=torch.float32, device=pred_indices.device)


@torch.no_grad()
def eval_ndcg_map(
    logits: torch.Tensor,
    relevance: torch.Tensor,
    k: int = TOP_K,
) -> tuple[float, float]:
    """Greedy top-k mean NDCG@k and MAP@k over a batch/set (offline eval)."""
    top = greedy_top_k_indices(logits, k=k)
    ndcgs: list[float] = []
    maps: list[float] = []
    for i in range(top.size(0)):
        pred = top[i].tolist()
        rel = relevance[i]
        ndcgs.append(ndcg_at_k(pred, rel, k=k))
        relevant = {j for j in range(rel.numel()) if float(rel[j]) > 0}
        maps.append(map_at_k(pred, relevant, k=k))
    n = max(len(ndcgs), 1)
    return sum(ndcgs) / n, sum(maps) / n
