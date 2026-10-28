"""Shared utilities for contrastive multi-agent algorithms.

Pairs of per-agent embeddings are scored by their (episode_id, step_id) tags:

- positive: same episode AND ``|Δstep| <= w_pos``
- excluded (gap zone): same episode AND ``w_pos < |Δstep| <= w_neg``
- negative: different episode, OR same episode AND ``|Δstep| > w_neg``

Self-pairs (same slot, same agent) are always excluded. The gap zone is
discarded from both numerator and denominator so adjacent-in-time frames
neither pull together nor push apart.

Two loss formulations are supported:

- ``sigmoid``: per-pair binary classification with ±1 targets (default)
- ``nt_xent``: multi-positive InfoNCE — ``logsumexp(positives) − logsumexp(valid)``
"""

import torch
import torch.nn as nn
from torchrl.modules import MultiAgentMLP


def compute_contrastive_loss(
    embeddings: torch.Tensor,
    episode_id: torch.Tensor,
    step_id: torch.Tensor,
    *,
    w_pos: int,
    w_neg: int,
    temperature: float,
    normalize: bool,
    mode: str,
) -> torch.Tensor:
    """Gap-zone contrastive loss over per-agent embeddings.

    Args:
        embeddings: tensor of shape ``(..., n_agents, embed_dim)``.
        episode_id: long tensor of shape ``(...,)`` — unique per trajectory.
        step_id: long tensor of shape ``(...,)`` — monotonic per-env step.
        w_pos: positive-pair half-window (in env steps).
        w_neg: negative-pair lower bound (in env steps). Must be ``>= w_pos``.
        temperature: divisor applied to similarity logits.
        normalize: if True, L2-normalize embeddings along the last dim.
        mode: ``"sigmoid"`` (default) or ``"nt_xent"``.

    Returns:
        Scalar loss tensor. Returns zero if there are fewer than 2 agents
        or no positive pairs exist in the batch.
    """
    if mode not in {"nt_xent", "sigmoid"}:
        raise ValueError(
            f"contrastive mode must be 'nt_xent' or 'sigmoid', got {mode!r}."
        )
    if w_neg < w_pos:
        raise ValueError(f"w_neg ({w_neg}) must be >= w_pos ({w_pos}).")

    if embeddings.shape[-2] < 2:
        return embeddings.new_zeros(())

    n_agents = embeddings.shape[-2]
    embed_dim = embeddings.shape[-1]
    flat = embeddings.reshape(-1, n_agents, embed_dim)
    ep_flat = episode_id.reshape(-1).to(flat.device)
    step_flat = step_id.reshape(-1).to(flat.device)

    n_positions = flat.shape[0]
    if n_positions == 0 or ep_flat.shape[0] != n_positions:
        return embeddings.new_zeros(())

    if normalize:
        flat = torch.nn.functional.normalize(flat, dim=-1)

    logits = torch.einsum("pad,qbd->paqb", flat, flat) / temperature

    same_ep = ep_flat.unsqueeze(0) == ep_flat.unsqueeze(1)  # (N, N)
    dt = (step_flat.unsqueeze(0) - step_flat.unsqueeze(1)).abs()  # (N, N)

    pos_frame = same_ep & (dt <= w_pos)
    gap_frame = same_ep & (dt > w_pos) & (dt <= w_neg)
    valid_frame = ~gap_frame

    pos_pair = pos_frame.view(n_positions, 1, n_positions, 1).expand(
        n_positions, n_agents, n_positions, n_agents
    )
    valid_pair = valid_frame.view(n_positions, 1, n_positions, 1).expand(
        n_positions, n_agents, n_positions, n_agents
    )

    slot_eye = torch.eye(n_positions, device=flat.device, dtype=torch.bool).view(
        n_positions, 1, n_positions, 1
    )
    agent_eye = torch.eye(n_agents, device=flat.device, dtype=torch.bool).view(
        1, n_agents, 1, n_agents
    )
    self_pair = slot_eye & agent_eye

    positive_mask = pos_pair & ~self_pair
    valid_mask = valid_pair & ~self_pair

    if not positive_mask.any():
        return embeddings.new_zeros(())

    if mode == "sigmoid":
        return _sigmoid_loss(logits, positive_mask, valid_mask)
    return _nt_xent_loss(logits, positive_mask, valid_mask)


def _nt_xent_loss(
    logits: torch.Tensor, positive_mask: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
    pos_logits = torch.where(positive_mask, logits, neg_inf)
    all_logits = torch.where(valid_mask, logits, neg_inf)

    pos_logsumexp = torch.logsumexp(pos_logits, dim=(-2, -1))
    denom_logsumexp = torch.logsumexp(all_logits, dim=(-2, -1))
    has_positive = positive_mask.any(dim=(-2, -1))
    loss = -(pos_logsumexp - denom_logsumexp)
    loss = loss[has_positive]
    if loss.numel() == 0:
        return logits.new_zeros(())
    return loss.mean()


def _sigmoid_loss(
    logits: torch.Tensor, positive_mask: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    labels = torch.where(
        positive_mask,
        torch.ones_like(logits),
        -torch.ones_like(logits),
    )
    loss = -torch.nn.functional.logsigmoid(labels * logits)
    loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
    loss = loss.sum(dim=(-2, -1))
    loss = loss[valid_mask.any(dim=(-2, -1))]
    if loss.numel() == 0:
        return logits.new_zeros(())
    return loss.mean()


class TensorHead(nn.Module):
    """Linear head over per-agent features, optionally with shared params."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_agents: int,
        share_params: bool,
        device,
    ):
        super().__init__()
        self.net = MultiAgentMLP(
            n_agent_inputs=in_features,
            n_agent_outputs=out_features,
            n_agents=n_agents,
            centralized=False,
            share_params=share_params,
            num_cells=[],
            activation_class=nn.Identity,
            device=device,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)
