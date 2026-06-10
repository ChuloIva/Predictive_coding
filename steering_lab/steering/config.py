"""Defaults for the steering-vector pilot. Override via function kwargs in the notebook."""

from __future__ import annotations

from dataclasses import dataclass, field


# Base model — the default we run steering / metastability / basin experiments on. Override per
# notebook via the `MODEL_NAME` cell or the STEER_MODEL env var. `google/gemma-4-12B-it` is the
# encoder-free unified multimodal Gemma 4 (June 2026); it runs text-only here and steers like any
# causal LM. (Earlier pilots used Qwen/Qwen2.5-3B-Instruct — switch back by setting this.)
MODEL_NAME = "google/gemma-4-12B-it"


@dataclass
class GenConfig:
    """vLLM generation settings (Phase 1)."""
    model_name: str = MODEL_NAME
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 200          # length of each persona-conditioned completion
    n_samples: int = 1             # completions per (mechanism, variant, prompt)
    seed: int = 0
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.90
    dtype: str = "bfloat16"


@dataclass
class ExtractConfig:
    """repeng extraction settings (Phase 2)."""
    model_name: str = MODEL_NAME
    # "pca_center": center each pos/neg pair by its mean, then PCA(1) over all centered points.
    # More robust than "pca_diff" when pos/neg aren't token-aligned (our completions differ), since
    # it finds the dominant axis separating the two clouds rather than trusting each paired diff.
    method: str = "pca_center"
    batch_size: int = 32
    # Which residual-stream layers to extract. None => repeng default (all but embedding).
    # A middle band is usually where persona/style directions steer most cleanly.
    hidden_layers: list[int] | None = None
    dtype: str = "bfloat16"
    # Truncation: turn each completion into several "model is mid-generation" snapshots.
    trunc_min_tokens: int = 4      # shortest truncation (tokens of the completion)
    trunc_stride: int = 6          # step between truncation points
    trunc_max_points: int = 8      # cap truncations per (prompt, sample) pair


# Default output locations (relative to repo root / notebook cwd).
GEN_OUTPUT = "deep/steering/out/generations.jsonl"
VECTORS_OUTPUT = "deep/steering/out/control_vectors.pkl"
META_OUTPUT = "deep/steering/out/metadata.json"
