from __future__ import annotations

from typing import Any, Dict

import torch


def _token_scores(attribution: torch.Tensor) -> torch.Tensor:
    if attribution.dim() > 2:
        return torch.linalg.vector_norm(attribution, dim=-1)
    return attribution.abs()


def _energy_attribution(inputs: torch.Tensor) -> torch.Tensor:
    """Gradient magnitude of the activation-energy surrogate objective."""
    return _token_scores(2.0 * inputs.abs())


def noise_ensemble_attribution(
    outputs: torch.Tensor,
    method: str,
    *,
    samples: int = 16,
    noise_std: float = 0.1,
    seed: int = 0,
) -> torch.Tensor:
    """Compute activation-space SmoothGrad, NoiseGrad, or VarGrad scores."""
    samples = max(2, min(int(samples), 128))
    noise_std = max(0.0, min(float(noise_std), 2.0))
    base = outputs.detach().float()
    generator = torch.Generator(device=base.device)
    generator.manual_seed(int(seed))
    scale = base.detach().std().clamp_min(1e-6)
    attributions = []

    for _ in range(samples):
        noise = torch.randn(
            base.shape,
            generator=generator,
            device=base.device,
            dtype=base.dtype,
        )
        if method == "noisegrad":
            perturbed = base * (1.0 + noise * noise_std)
        else:
            perturbed = base + noise * noise_std * scale
        attributions.append(_energy_attribution(perturbed))

    stacked = torch.stack(attributions, dim=0)
    if method == "vargrad":
        return stacked.var(dim=0, unbiased=False)
    return stacked.mean(dim=0)


def attribution_sanity_report(
    outputs: torch.Tensor,
    *,
    samples: int = 16,
    noise_std: float = 0.1,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compare baselines and report perturbation/randomization sensitivity."""
    base = outputs.detach().float()
    original = _energy_attribution(base).flatten()
    smooth = noise_ensemble_attribution(
        base, "smoothgrad", samples=samples, noise_std=noise_std, seed=seed
    ).flatten()

    mean_baseline = base.mean(dim=-2, keepdim=True)
    zero_delta = _token_scores(base).flatten()
    mean_delta = _token_scores(base - mean_baseline).flatten()

    generator = torch.Generator(device=base.device)
    generator.manual_seed(int(seed) + 1)
    randomized_activations = torch.randn(
        base.shape,
        generator=generator,
        device=base.device,
        dtype=base.dtype,
    )
    randomized_activations = randomized_activations * base.std().clamp_min(1e-6)
    randomized_activations = randomized_activations + base.mean()
    randomized = _energy_attribution(randomized_activations).flatten()

    def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
        if left.numel() == 0 or right.numel() == 0:
            return 0.0
        denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        if float(denom) == 0.0:
            return 0.0
        return float(torch.dot(left, right).div(denom).item())

    randomization_similarity = cosine(original, randomized)
    return {
        "finite_fraction": float(torch.isfinite(original).float().mean().item()),
        "smoothgrad_similarity": cosine(original, smooth),
        "randomization_similarity": randomization_similarity,
        "randomization_passed": randomization_similarity < 0.95,
        "zero_baseline_mean": float(zero_delta.mean().item()),
        "mean_baseline_mean": float(mean_delta.mean().item()),
        "samples": int(samples),
        "noise_std": float(noise_std),
        "scope": "activation-space perturbation checks",
    }
