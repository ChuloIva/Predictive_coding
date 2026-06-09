"""Steering-vector pilot: extract repeng control vectors for atomic cognitive mechanisms.

Pipeline:
    1. `generate`  — vLLM generates persona-conditioned completions over a neutral prompt bank.
    2. `extract`   — build contrastive (ON vs matched-baseline) pairs from those completions and
                     train repeng ControlVectors (PCA on last-token hidden-state differences).

The mechanism definitions live in `personas` and mirror `context/pilot_personas.md`.
See `context/mechanism_syndrome_map.md` for the clinical grounding.
"""

from . import config, personas

__all__ = ["config", "personas"]
