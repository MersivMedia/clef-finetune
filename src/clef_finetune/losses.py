"""Per-question losses.

total = CE_label_smoothed + brier_weight * Brier + ordinal_weight * EMD (score only)

The ordinal term is a *supervised approximation* of the "adjacent credit" idea in
Cloudflare's RLCD stage: it penalises probability mass by its distance from the
true level, so predicting 3 when the truth is 2 costs less than predicting 0.
It is NOT RLCD (no rewards, no exact-record bonus, no reference/KL penalty).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def brier(logits: torch.Tensor, target: int) -> torch.Tensor:
    probs = logits.float().softmax(-1)
    onehot = F.one_hot(torch.tensor(target, device=logits.device), probs.shape[-1]).float()
    return ((probs - onehot) ** 2).sum()


def smoothed_ce(logits: torch.Tensor, target: int, smoothing: float) -> torch.Tensor:
    return F.cross_entropy(
        logits.float().unsqueeze(0),
        torch.tensor([target], device=logits.device),
        label_smoothing=smoothing,
    )


def emd_ordinal(logits: torch.Tensor, target: int, squared: bool = True) -> torch.Tensor:
    """Earth mover's distance between predicted and one-hot CDFs, normalised by (n-1)."""
    probs = logits.float().softmax(-1)
    n = probs.shape[-1]
    if n < 2:
        return probs.sum() * 0
    onehot = F.one_hot(torch.tensor(target, device=logits.device), n).float()
    diff = torch.cumsum(probs, -1) - torch.cumsum(onehot, -1)
    d = diff.pow(2) if squared else diff.abs()
    return d.sum() / (n - 1)


def question_loss(
    logits: torch.Tensor,
    target: int,
    qtype: str,
    label_smoothing: float = 0.05,
    brier_weight: float = 1.0,
    ordinal_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = smoothed_ce(logits, target, label_smoothing)
    br = brier(logits, target)
    loss = ce + brier_weight * br
    parts = {"ce": ce.item(), "brier": br.item()}
    if qtype == "score" and ordinal_weight > 0:
        od = emd_ordinal(logits, target)
        loss = loss + ordinal_weight * od
        parts["ordinal"] = od.item()
    return loss, parts
