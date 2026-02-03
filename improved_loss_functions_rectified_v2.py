import os
import torch
import torch.nn.functional as F

MAPE_EPSILON = 1e-7
DEFAULT_TORCH_DTYPE = torch.float32


# ----------------------------
# Helpers: log-target support
# ----------------------------
def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_use_log_target(use_log_target: bool | None) -> bool:
    # If caller doesn't specify, allow enabling via environment variable
    # (useful if callsites import functions directly).
    if use_log_target is None:
        return _env_flag("USE_LOG_TARGET", False) or _env_flag("LOG_TARGET", False)
    return bool(use_log_target)


def _v0_to_log(v0: torch.Tensor, log_eps: float, log_clamp_min: float) -> torch.Tensor:
    v0_mag = torch.clamp(torch.abs(v0), min=log_clamp_min) + log_eps
    return torch.log(v0_mag)


def _log_to_vref(logv: torch.Tensor, log_eps: float) -> torch.Tensor:
    """Train-time inverse (smooth, strictly positive).

    exp(logv) corresponds to (|V| + log_eps) because the forward log transform is
    log(|V| + log_eps). We do NOT subtract log_eps here during training to avoid a
    clamp-induced zero-gradient region near V~=0.
    """
    return torch.exp(logv)

def _log_to_vref_report(logv: torch.Tensor, log_eps: float) -> torch.Tensor:
    """Reporting-only inverse that returns the physical magnitude V>=0."""
    return torch.clamp(torch.exp(logv) - log_eps, min=0.0)


def _reconstruct_vref(
    delta: torch.Tensor,
    v0: torch.Tensor,
    *,
    use_log_target: bool | None,
    log_eps: float,
    log_clamp_min: float,
) -> torch.Tensor:
    use_log_target = _resolve_use_log_target(use_log_target)

    if use_log_target:
        logv0 = _v0_to_log(v0, log_eps=log_eps, log_clamp_min=log_clamp_min)
        return _log_to_vref(logv0 + delta, log_eps=log_eps)

    # Linear-delta mode
    return v0 + delta


def _ensure_col(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(-1) if x.ndim == 1 else x


def _shape_pack(pred_vref: torch.Tensor, target_vref: torch.Tensor, R_internal_batch: torch.Tensor):
    R_internal_batch = R_internal_batch.to(pred_vref.device)
    pred_vref = _ensure_col(pred_vref)
    target_vref = _ensure_col(target_vref)
    R_internal_batch = _ensure_col(R_internal_batch)
    return pred_vref, target_vref, R_internal_batch


# ----------------------------
# Losses (now log-target aware)
# ----------------------------
def magnitude_aware_loss(
    pred_delta_v,
    target_delta_v,
    R_internal_batch,
    V0_batch,
    graph_cutoff_for_weighting,
    negative_penalty_factor=5.0,
    *,
    use_log_target: bool | None = None,
    log_eps: float = 1e-5,
    log_clamp_min: float = 0.0,
):
    """
    Loss that explicitly penalizes magnitude errors.
    If use_log_target=True, pred_delta_v/target_delta_v are interpreted as ?logV.
    """
    if pred_delta_v.numel() == 0:
        return torch.tensor(0.0, device=pred_delta_v.device, dtype=torch.float32, requires_grad=True)

    pred_vref = _reconstruct_vref(pred_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)
    target_vref = _reconstruct_vref(target_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)

    pred_vref, target_vref, R_internal_batch = _shape_pack(pred_vref, target_vref, R_internal_batch)

    squared_error = (target_vref - pred_vref) ** 2
    ratio = torch.abs(target_vref - pred_vref) / (torch.abs(target_vref) + MAPE_EPSILON)
    log_ratio_penalty = torch.log1p(ratio)

    target_magnitude = torch.abs(target_vref)
    pred_magnitude = torch.abs(pred_vref)
    magnitude_error = (target_magnitude - pred_magnitude) ** 2

    negative_mask = torch.sigmoid(-50.0 * pred_vref)
    negative_penalty = negative_mask * (pred_vref ** 2) * negative_penalty_factor

    weights = 1.0 + (R_internal_batch / graph_cutoff_for_weighting)

    base_loss = torch.mean(weights * squared_error) * 10.0
    relative_loss = torch.mean(weights * log_ratio_penalty) * 5.0
    magnitude_loss = torch.mean(weights * magnitude_error) * 8.0
    penalty_loss = torch.mean(negative_penalty) * 0.5

    return base_loss + relative_loss + magnitude_loss + penalty_loss


def step_penalty_loss(
    pred_delta_v,
    target_delta_v,
    R_internal_batch,
    V0_batch,
    graph_cutoff_for_weighting,
    negative_penalty_factor=10.0,
    *,
    use_log_target: bool | None = None,
    log_eps: float = 1e-5,
    log_clamp_min: float = 0.0,
):
    """
    L1 + relative error + smooth penalty for negative Vref.
    If use_log_target=True, deltas are ?logV and we reconstruct Vref in linear space.
    """
    if pred_delta_v.numel() == 0:
        return torch.tensor(0.0, device=pred_delta_v.device, dtype=torch.float32, requires_grad=True)

    pred_vref = _reconstruct_vref(pred_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)
    target_vref = _reconstruct_vref(target_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)

    pred_vref, target_vref, R_internal_batch = _shape_pack(pred_vref, target_vref, R_internal_batch)

    abs_error_vref = torch.abs(target_vref - pred_vref)

    steepness = 50.0
    step_function = torch.sigmoid(-steepness * pred_vref)  # ~1 for negative, ~0 for positive
    negative_penalty = step_function * torch.abs(pred_vref) * negative_penalty_factor

    weights = 1.0 + (R_internal_batch / graph_cutoff_for_weighting)
    relative_error = abs_error_vref / (torch.abs(target_vref) + MAPE_EPSILON)

    base_loss = torch.mean(weights * abs_error_vref)
    relative_loss = torch.mean(weights * relative_error) * 0.5
    penalty_loss = torch.mean(negative_penalty) * 0.3

    return base_loss + relative_loss + penalty_loss


def huber_step_penalty_loss(
    pred_delta_v,
    target_delta_v,
    R_internal_batch,
    V0_batch,
    graph_cutoff_for_weighting,
    negative_penalty_factor=10.0,
    huber_delta=0.01,
    *,
    use_log_target: bool | None = None,
    log_eps: float = 1e-5,
    log_clamp_min: float = 0.0,
):
    """
    Huber(Vref) + log-relative-error + smooth penalty for negative Vref.
    If use_log_target=True, deltas are ?logV and we reconstruct Vref in linear space.
    """
    if pred_delta_v.numel() == 0:
        return torch.tensor(0.0, device=pred_delta_v.device, dtype=DEFAULT_TORCH_DTYPE, requires_grad=True)

    pred_vref = _reconstruct_vref(pred_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)
    target_vref = _reconstruct_vref(target_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)

    pred_vref, target_vref, R_internal_batch = _shape_pack(pred_vref, target_vref, R_internal_batch)

    huber_loss = F.huber_loss(pred_vref, target_vref, reduction="none", delta=huber_delta)

    steepness = 50.0
    step_function = torch.sigmoid(-steepness * pred_vref)
    negative_penalty = step_function * torch.abs(pred_vref) * negative_penalty_factor

    weights = 1.0 + (R_internal_batch / graph_cutoff_for_weighting)

    relative_error = torch.abs(target_vref - pred_vref) / (torch.abs(target_vref) + MAPE_EPSILON)
    log_relative_error = torch.log1p(relative_error)

    base_loss = torch.mean(weights * huber_loss)
    relative_loss = torch.mean(weights * log_relative_error) * 1.0
    penalty_loss = torch.mean(negative_penalty) * 0.5

    return base_loss + relative_loss + penalty_loss


def adaptive_loss_with_step_penalty(
    pred_delta_v,
    target_delta_v,
    R_internal_batch,
    V0_batch,
    graph_cutoff_for_weighting,
    negative_penalty_factor=15.0,
    epoch=None,
    max_epochs=2000,
    *,
    use_log_target: bool | None = None,
    log_eps: float = 1e-5,
    log_clamp_min: float = 0.0,
):
    """
    Curriculum version of MSE(Vref) + relative(Vref) + penalty, with epoch-dependent weights.
    If use_log_target=True, deltas are ?logV and we reconstruct Vref in linear space.
    """
    if pred_delta_v.numel() == 0:
        return torch.tensor(0.0, device=pred_delta_v.device, dtype=DEFAULT_TORCH_DTYPE, requires_grad=True)

    pred_vref = _reconstruct_vref(pred_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)
    target_vref = _reconstruct_vref(target_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)

    pred_vref, target_vref, R_internal_batch = _shape_pack(pred_vref, target_vref, R_internal_batch)

    if epoch is not None and max_epochs is not None:
        progress = min(epoch / (max_epochs * 0.5), 1.0)
        adaptive_penalty = negative_penalty_factor * (0.5 + 0.5 * progress)
    else:
        adaptive_penalty = negative_penalty_factor

    mse_loss = (pred_vref - target_vref) ** 2

    steepness = 50.0
    step_function = torch.sigmoid(-steepness * pred_vref)
    negative_penalty = step_function * (pred_vref ** 2) * adaptive_penalty

    weights = 1.0 + (R_internal_batch / graph_cutoff_for_weighting)

    percentage_error = torch.abs(target_vref - pred_vref) / (torch.abs(target_vref) + MAPE_EPSILON)

    if epoch is not None and max_epochs is not None and epoch < max_epochs * 0.2:
        base_loss = torch.mean(weights * mse_loss) * 10.0
        relative_loss = torch.mean(weights * percentage_error) * 1.0
        penalty_loss = torch.mean(negative_penalty) * 0.2
    else:
        base_loss = torch.mean(weights * mse_loss) * 10.0
        relative_loss = torch.mean(weights * percentage_error) * 2.0
        penalty_loss = torch.mean(negative_penalty) * 0.5

    return base_loss + relative_loss + penalty_loss


# The remaining losses are also made log-target aware for completeness.
def focal_loss_with_penalty(
    pred_delta_v,
    target_delta_v,
    R_internal_batch,
    V0_batch,
    graph_cutoff_for_weighting,
    negative_penalty_factor=10.0,
    gamma=2.0,
    *,
    use_log_target: bool | None = None,
    log_eps: float = 1e-5,
    log_clamp_min: float = 0.0,
):
    if pred_delta_v.numel() == 0:
        return torch.tensor(0.0, device=pred_delta_v.device, dtype=DEFAULT_TORCH_DTYPE, requires_grad=True)

    pred_vref = _reconstruct_vref(pred_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)
    target_vref = _reconstruct_vref(target_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)

    pred_vref, target_vref, R_internal_batch = _shape_pack(pred_vref, target_vref, R_internal_batch)

    mse = (target_vref - pred_vref) ** 2
    with torch.no_grad():
        max_error = torch.max(mse) + MAPE_EPSILON
    normalized_error = mse / max_error
    focal_weight = (1 - torch.exp(-normalized_error)) ** gamma
    focal_mse = focal_weight * mse

    steepness = 50.0
    step_function = torch.sigmoid(-steepness * pred_vref)
    negative_penalty = step_function * (pred_vref ** 2) * negative_penalty_factor

    weights = 1.0 + (R_internal_batch / graph_cutoff_for_weighting)

    base_loss = torch.mean(weights * focal_mse) * 10.0
    penalty_loss = torch.mean(negative_penalty) * 0.3
    return base_loss + penalty_loss


def log_cosh_loss(
    pred_delta_v,
    target_delta_v,
    R_internal_batch,
    V0_batch,
    graph_cutoff_for_weighting,
    negative_penalty_factor=10.0,
    *,
    use_log_target: bool | None = None,
    log_eps: float = 1e-5,
    log_clamp_min: float = 0.0,
):
    if pred_delta_v.numel() == 0:
        return torch.tensor(0.0, device=pred_delta_v.device, dtype=DEFAULT_TORCH_DTYPE, requires_grad=True)

    pred_vref = _reconstruct_vref(pred_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)
    target_vref = _reconstruct_vref(target_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)

    pred_vref, target_vref, R_internal_batch = _shape_pack(pred_vref, target_vref, R_internal_batch)

    error = target_vref - pred_vref
    log_cosh = torch.log(torch.cosh(error + MAPE_EPSILON))

    steepness = 50.0
    step_function = torch.sigmoid(-steepness * pred_vref)
    negative_penalty = step_function * (pred_vref ** 2) * negative_penalty_factor

    weights = 1.0 + (R_internal_batch / graph_cutoff_for_weighting)
    relative_error = torch.abs(error) / (torch.abs(target_vref) + MAPE_EPSILON)

    base_loss = torch.mean(weights * log_cosh) * 5.0
    relative_loss = torch.mean(weights * relative_error) * 1.0
    penalty_loss = torch.mean(negative_penalty) * 0.3
    return base_loss + relative_loss + penalty_loss


def quantile_loss(
    pred_delta_v,
    target_delta_v,
    R_internal_batch,
    V0_batch,
    graph_cutoff_for_weighting,
    negative_penalty_factor=10.0,
    quantile=0.5,
    *,
    use_log_target: bool | None = None,
    log_eps: float = 1e-5,
    log_clamp_min: float = 0.0,
):
    if pred_delta_v.numel() == 0:
        return torch.tensor(0.0, device=pred_delta_v.device, dtype=DEFAULT_TORCH_DTYPE, requires_grad=True)

    pred_vref = _reconstruct_vref(pred_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)
    target_vref = _reconstruct_vref(target_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)

    pred_vref, target_vref, R_internal_batch = _shape_pack(pred_vref, target_vref, R_internal_batch)

    error = target_vref - pred_vref
    quantile_loss_val = torch.where(error >= 0, quantile * error, (quantile - 1) * error)

    steepness = 50.0
    step_function = torch.sigmoid(-steepness * pred_vref)
    negative_penalty = step_function * (pred_vref ** 2) * (negative_penalty_factor * 2.0)

    weights = 1.0 + (R_internal_batch / graph_cutoff_for_weighting)

    base_loss = torch.mean(weights * quantile_loss_val) * 10.0
    penalty_loss = torch.mean(negative_penalty) * 0.5
    return base_loss + penalty_loss


def distance_weighted_loss(
    pred_delta_v,
    target_delta_v,
    R_internal_batch,
    V0_batch,
    graph_cutoff_for_weighting,
    negative_penalty_factor=2.0,
    eps=1e-3,
    softplus_beta=1.0,
    magnitude_alpha=0.5,
    *,
    use_log_target: bool | None = None,
    log_eps: float = 1e-5,
    log_clamp_min: float = 0.0,
):
    if pred_delta_v.numel() == 0:
        return torch.tensor(0.0, device=pred_delta_v.device, dtype=torch.float32, requires_grad=True)

    pred_vref = _reconstruct_vref(pred_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)
    target_vref = _reconstruct_vref(target_delta_v, V0_batch, use_log_target=use_log_target, log_eps=log_eps, log_clamp_min=log_clamp_min)

    pred_vref, target_vref, R_internal_batch = _shape_pack(pred_vref, target_vref, R_internal_batch)

    distance_weights = 1.0 + (R_internal_batch / graph_cutoff_for_weighting)

    magnitude_epsilon = 1e-8
    magnitude_weight = 1.0 / (target_vref.abs() + magnitude_epsilon) ** magnitude_alpha
    magnitude_weight = torch.clamp(magnitude_weight, max=100.0)

    final_weights = distance_weights * magnitude_weight

    abs_error = torch.abs(target_vref - pred_vref)
    base_loss = torch.mean(final_weights * abs_error)

    denom = torch.clamp(torch.abs(target_vref), min=eps)
    rel_error = abs_error / denom
    relative_loss = torch.mean(distance_weights * rel_error)

    penalty_loss = torch.mean(torch.relu(-pred_vref) * negative_penalty_factor)

    return base_loss + 0.05 * relative_loss + 1.00 * penalty_loss


def log_delta_huber_loss(
    pred_delta_v: torch.Tensor,
    target_delta_v: torch.Tensor,
    R_internal_batch: torch.Tensor,
    V0_batch: torch.Tensor,
    graph_cutoff_for_weighting: float,
    *,
    beta: float = 0.02,
    weight_by_distance: bool = False,
    use_log_target: bool | None = None,
    log_eps: float = 1e-5,
    log_clamp_min: float = 0.0,
) -> torch.Tensor:
    """Huber loss directly on delta targets.

    If use_log_target=True, pred_delta_v and target_delta_v are interpreted as Δlog(|V|+eps) and the
    loss is computed directly in log-space:
        L = Huber(pred_delta_log - target_delta_log)

    Parameters
    ----------
    beta:
        SmoothL1/Huber transition point (in delta units). For log-deltas, values ~0.01–0.05 are typical.
    weight_by_distance:
        If True, weight each sample by (1 + R_internal / graph_cutoff_for_weighting).

    Notes
    -----
    - R_internal_batch, V0_batch are accepted to match the training-loop signature.
    - When use_log_target=False, this reduces to a SmoothL1 loss on linear ΔV.
    """
    if pred_delta_v.numel() == 0:
        return torch.tensor(0.0, device=pred_delta_v.device, dtype=torch.float32, requires_grad=True)

    # Resolve the flag for consistency (no branching needed: we always operate on the deltas provided)
    _ = _resolve_use_log_target(use_log_target)

    pred = _ensure_col(pred_delta_v)
    target = _ensure_col(target_delta_v)

    per = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")

    if weight_by_distance:
        _, _, Rb = _shape_pack(pred, target, R_internal_batch)
        weights = 1.0 + (Rb / float(graph_cutoff_for_weighting))
        per = per * weights

    return per.mean()
