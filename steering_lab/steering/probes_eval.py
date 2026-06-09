"""Stimuli + steering conditions for the trimmed metastability probe (Experiment ③).

Two string families, kept deliberately minimal (the false-inference batteries — congruent/incongruent,
ambiguous, questionnaire — are deferred; see `context/metastability_prereg.md`):

  A. GEN_PROMPTS  — content-free continuation prompts. The steered model *talks*; we read the
                    residual-stream trajectory off its completion (flexibility + drift).
  B. READ_BATTERY — short, ordinary, neutral passages. The model *reads* these fixed strings
                    identically under every condition; we read surprisal / next-token entropy / drift
                    off aligned tokens. Mundane on purpose: no topical charge to mask the steer, and
                    identical tokens make every condition directly comparable.

Conditions (matched magnitude): clean · looping · REBUS · random. The `random` arm — a random
direction with per-layer norm matched to the looping vector — is the keystone control that separates
"therapy / a meaningful steer" from "any perturbation of the same size".
"""

from __future__ import annotations

import numpy as np

from .personas import NEUTRAL_PROMPTS

__all__ = ["GEN_PROMPTS", "READ_BATTERY", "random_control_vector", "build_conditions"]


# --- Family A: generative prompts (for the trajectory) --------------------------------------------
# A small content-free subset of the shared neutral bank; enough completions for stable trajectory
# metrics without a long GPU run. Override by passing your own list to the runner.
GEN_PROMPTS: list[str] = list(NEUTRAL_PROMPTS[:12])


# --- Family B: read-only everyday passages (for surprisal / entropy / drift on fixed tokens) -------
# ~10 short, flat, ordinary passages. No emotional or topical charge; each ~2-3 sentences so the
# forced pass has enough next-token predictions to average over.
READ_BATTERY: list[str] = [
    "The bus arrives at the corner stop a little after eight most mornings. People queue without "
    "much fuss and find seats by the window. The ride downtown takes about twenty minutes.",

    "To make the rice, rinse one cup under cold water until it runs clear. Add it to a pot with two "
    "cups of water and a pinch of salt. Bring it to a boil, cover, and let it simmer for fifteen "
    "minutes.",

    "The weather today is mild with a light breeze from the west. Clouds are expected to clear by "
    "the afternoon. Temperatures will stay around eighteen degrees through the evening.",

    "She kept the spare key in a small dish by the front door. On weekends she watered the plants on "
    "the sill and wiped down the counter. The apartment was quiet and easy to keep tidy.",

    "The library opens at nine and closes at six on weekdays. Returns go in the slot by the entrance. "
    "The reading room upstairs has long tables and good light.",

    "He packed a sandwich, an apple, and a bottle of water for the walk. The trail followed the river "
    "for a few miles before turning uphill. Signposts marked the way at each junction.",

    "The meeting was moved to the second-floor room with the projector. We went through the agenda "
    "item by item and noted the action points. It wrapped up a few minutes before lunch.",

    "The hardware store on Main Street sells paint, nails, and garden tools. The owner knows most of "
    "the regulars by name. On Saturdays it gets busy with people starting weekend projects.",

    "After dinner they cleared the table and washed the dishes together. One rinsed while the other "
    "dried and stacked the plates in the cupboard. Then they sat down to watch the news.",

    "The train was a few minutes late but the platform was sheltered from the rain. A board overhead "
    "listed the next departures. When it pulled in, the doors opened and the carriage was half empty.",
]


# --- conditions -----------------------------------------------------------------------------------
def random_control_vector(reference, seed: int = 0):
    """A `ControlVector` of random directions, per-layer norm matched to `reference`.

    The keystone control: same per-layer magnitude as the looping vector, but a meaningless direction.
    If a real steer's effects (flexibility loss, overconfidence, reversibility) reproduce under a
    norm-matched random push, they're an artifact of perturbation size, not of the mechanism.
    """
    from repeng import ControlVector  # lazy

    rng = np.random.default_rng(seed)
    directions: dict[int, np.ndarray] = {}
    for layer, vec in reference.directions.items():
        v = np.asarray(vec, dtype=np.float32)
        r = rng.standard_normal(v.shape).astype(np.float32)
        r *= (np.linalg.norm(v) / (np.linalg.norm(r) + 1e-8))   # match this layer's norm
        directions[int(layer)] = r
    return ControlVector(model_type=reference.model_type, directions=directions)


def build_conditions(looping_vec, rebus_vec=None, *, coeff: float = 8.0, seed: int = 0):
    """The four-condition list `[(label, vector_or_None, coeff), ...]` for the runner.

    clean   : no steer (coeff 0) — baseline.
    looping : +coeff on the looping pathology vector (rumination / circular_inference).
    REBUS   : −coeff on the over-precise-prior vector (loosen the prior = the therapy arm). Defaults
              to `looping_vec` with a negative coeff if no `rebus_vec` (e.g. PC `prior_precision_high`)
              is supplied — a coarse "undo the steer" fallback, documented so it isn't over-read.
    random  : +coeff on a norm-matched random direction — the keystone control.
    """
    rebus = rebus_vec if rebus_vec is not None else looping_vec
    return [
        ("clean", None, 0.0),
        ("looping", looping_vec, float(coeff)),
        ("REBUS", rebus, -float(coeff)),
        ("random", random_control_vector(looping_vec, seed=seed), float(coeff)),
    ]
