"""Experiment ③ — metastability probe, *trimmed robust core* (pure-numpy metrics).

Motivation (NOT a claim the numbers carry): Kotler & Friston (2026) frame the looping pathologies —
rumination, worry, intrusive re-experiencing — as a *loss of metastability*: the system stops fluidly
switching among semi-stable states and gets trapped in an attractor; therapy (REBUS = a negative
steer) restores the switching. We test the falsifiable shadow of that idea in a steered LLM:
**steering toward looping pathologies should reduce the flexibility of the layer-L residual-stream
trajectory and warp the internal representation, reversibly, and distinguishably from a random
perturbation of matched magnitude.** The Friston framing is the motivation that predicted this; the
headline is the measured, defensible statement above.

This module is the pure-numpy measurement core — no torch, no model — so every metric is unit-testable
on synthetic arrays (`smoke_test()`). The model-facing forward/generation logic lives in
`experiment.py`, which feeds these functions the trajectories and log-probs.

Robust core only. The earlier broad version's knob-dependent metrics (k-means transition_rate /
mean_dwell / n_microstates) and ambiguous ones (flow_determinism, redundancy/effective_rank) were
deliberately stripped; the false-inference stimulus batteries were deferred. See
`context/metastability_prereg.md` for the locked scope and predictions.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "trajectory_metrics", "drift_from_prior", "surface_repetition", "output_metrics",
    "aggregate", "smoke_test",
    "TRAJ_KEYS", "OUTPUT_KEYS", "SICK_DIRECTION",
]

# Keys grouped by where they come from, for table layout / aggregation.
TRAJ_KEYS = ("participation_ratio", "lag1_autocorr", "mean_step")
OUTPUT_KEYS = ("surprisal", "next_token_entropy")

# +1 ⇒ a HIGHER value is the "sicker"/stuck direction; -1 ⇒ a higher value is healthier.
# (`drift_from_prior` is a steer-cost magnitude, not a health axis — read jointly, so it's omitted.)
SICK_DIRECTION = {
    "participation_ratio": -1,   # higher = more flexible (healthy)
    "lag1_autocorr": +1,         # higher = stuck (each state echoes the last)
    "mean_step": -1,             # higher = moving (healthy)
    "next_token_entropy": -1,    # lower = overconfident/over-precise (sick); higher = healthier
    "surprisal": 0,              # exploratory on neutral text — no pre-registered direction
}


# --------------------------------------------------------------------------------------------------
# trajectory geometry (generative completion, layer L, steer ON)
# --------------------------------------------------------------------------------------------------
def trajectory_metrics(traj: np.ndarray) -> dict:
    """Flexibility of a residual-stream trajectory `traj` of shape [T, d]. All parameter-free.

    participation_ratio : effective dimensionality the trajectory explores, (Σλ)²/Σλ² ∈ [1, d], from
                          the eigenvalues of the temporal covariance. High ⇒ visits many directions
                          (flexible); low ⇒ collapsed onto a basin.
    lag1_autocorr       : mean per-dim lag-1 autocorrelation of the centered trajectory. High ⇒ each
                          state just echoes the previous one (stuck).
    mean_step           : mean consecutive L2 step, normalized by mean state norm (raw movement).
    """
    traj = np.asarray(traj, dtype=np.float64)
    T = traj.shape[0]
    if T < 3:
        return {k: float("nan") for k in TRAJ_KEYS} | {"n_tokens": int(T)}

    Xc = traj - traj.mean(0)
    ev = np.linalg.eigvalsh(Xc.T @ Xc / T)
    ev = ev[ev > 1e-12]
    pr = float(ev.sum() ** 2 / (ev**2).sum()) if ev.size else 1.0

    steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    scale = float(np.linalg.norm(traj, axis=1).mean()) + 1e-8
    mean_step = float(steps.mean() / scale)

    a, b = Xc[:-1], Xc[1:]
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0)) + 1e-12
    lag1 = float(((a * b).sum(0) / den).mean())

    return {
        "n_tokens": int(T),
        "participation_ratio": pr,
        "lag1_autocorr": lag1,
        "mean_step": mean_step,
    }


# --------------------------------------------------------------------------------------------------
# complexity (same text, steer ON vs OFF) — the cleanest, fully-aligned number
# --------------------------------------------------------------------------------------------------
def drift_from_prior(traj_on: np.ndarray, traj_off: np.ndarray) -> float:
    """Mean per-token distance between the steered and clean representation of the *same* tokens.

    The KL[q‖p] analogue: how far the imposed prior (steer) drags the layer-L activations off where
    they'd naturally sit. Aligned position-by-position, parameter-free. Symmetric across steer sign —
    a magnitude, not a health axis — so read it jointly with the flexibility/output metrics.
    """
    on = np.asarray(traj_on, dtype=np.float64)
    off = np.asarray(traj_off, dtype=np.float64)
    m = min(len(on), len(off))
    if m == 0:
        return float("nan")
    d = np.linalg.norm(on[:m] - off[:m], axis=1)
    scale = float(np.linalg.norm(off[:m], axis=1).mean()) + 1e-8
    return float(d.mean() / scale)


# --------------------------------------------------------------------------------------------------
# surface-repetition guard (generative completion tokens)
# --------------------------------------------------------------------------------------------------
def surface_repetition(token_ids) -> dict:
    """Cheap lexical-repetition controls, so geometric stickiness can be shown to exceed plain
    word-repetition.

    distinct_2    : #unique bigrams / #bigrams ∈ (0, 1]. Low ⇒ repetitive surface text.
    token_entropy : Shannon entropy (nats) of the unigram token distribution. Low ⇒ few token types.
    """
    ids = [int(t) for t in token_ids]
    n = len(ids)
    if n < 2:
        return {"distinct_2": float("nan"), "token_entropy": float("nan")}

    bigrams = list(zip(ids[:-1], ids[1:]))
    distinct_2 = len(set(bigrams)) / len(bigrams)

    _, counts = np.unique(ids, return_counts=True)
    p = counts / counts.sum()
    token_entropy = float(-(p * np.log(p)).sum())

    return {"distinct_2": float(distinct_2), "token_entropy": token_entropy}


# --------------------------------------------------------------------------------------------------
# output measures (read-only forced pass) — from log-probs, so torch isn't needed here
# --------------------------------------------------------------------------------------------------
def output_metrics(logprobs: np.ndarray, target_ids) -> dict:
    """Accuracy & precision read off the next-token distribution over a fixed passage.

    `logprobs` : [P, V] log-softmax rows; row t is the distribution predicting token t+1.
    `target_ids` : the realized tokens, length P+? — we align so that target_ids[t+1] is scored by
                   logprobs[t]; the first token has no predictor and is skipped.

    surprisal          : mean −log p(realized next token) (nats) = the model's prediction error =
                         the *accuracy* term of free energy. (Exploratory on neutral text.)
    next_token_entropy : mean Shannon entropy (nats) of each next-token distribution = the model's
                         *precision/gain*. Low ⇒ sharp, overconfident posterior (over-precise).
    """
    lp = np.asarray(logprobs, dtype=np.float64)
    ids = [int(t) for t in target_ids]
    P = lp.shape[0]
    # logprobs[t] predicts ids[t+1]; need P predictors for ids[1..P].
    m = min(P, len(ids) - 1)
    if m <= 0:
        return {"surprisal": float("nan"), "next_token_entropy": float("nan")}

    rows = lp[:m]                                   # [m, V]
    nxt = np.asarray(ids[1:m + 1], dtype=int)       # [m]
    surprisal = float(-rows[np.arange(m), nxt].mean())

    p = np.exp(rows)
    p = p / p.sum(axis=1, keepdims=True)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1)
    return {"surprisal": surprisal, "next_token_entropy": float(ent.mean())}


# --------------------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------------------
def aggregate(rows: list[dict], *, by: str = "condition", baseline: str = "clean") -> dict:
    """Per-group mean of every numeric metric, plus a delta-vs-baseline for each.

    Returns {group: {metric: mean, ..., "n": int, "delta": {metric: mean − baseline_mean}}}.
    No z-scoring or composite index — the trimmed core reports raw, interpretable per-condition means
    and their signed change from `clean`, read against the pre-registered predictions table.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[by], []).append(r)

    metric_keys: list[str] = []
    for rs in groups.values():
        for r in rs:
            for k, v in r.items():
                if k != by and isinstance(v, (int, float)) and k not in metric_keys:
                    metric_keys.append(k)

    means = {
        g: {k: float(np.nanmean([r[k] for r in rs if k in r])) for k in metric_keys} | {"n": len(rs)}
        for g, rs in groups.items()
    }
    base = means.get(baseline)
    for g, mv in means.items():
        mv["delta"] = {
            k: float(mv[k] - base[k]) for k in metric_keys
        } if base is not None else {}
    return means


# --------------------------------------------------------------------------------------------------
# CPU smoke test (no model) — validates each metric moves in the predicted direction
# --------------------------------------------------------------------------------------------------
def smoke_test() -> None:
    """Synthetic ground-truth checks for every pure-numpy metric. Raises AssertionError on failure."""
    rng = np.random.default_rng(0)
    d, T = 16, 120

    # sticky: slow drift in a thin subspace (one shallow basin); flexible: hops among 5 basins
    sticky = np.cumsum(rng.normal(0, 0.05, size=(T, d)), axis=0)
    sticky[:, 3:] *= 0.05
    centers = rng.normal(0, 3, size=(5, d))
    flexible = centers[rng.integers(0, 5, size=T)] + rng.normal(0, 0.2, size=(T, d))

    ms, mf = trajectory_metrics(sticky), trajectory_metrics(flexible)
    assert ms["participation_ratio"] < mf["participation_ratio"], (ms, mf)
    assert ms["lag1_autocorr"] > mf["lag1_autocorr"], (ms, mf)
    assert ms["mean_step"] < mf["mean_step"], (ms, mf)

    # drift: a warped copy of a trajectory must be > 0 and grow with the warp
    base = rng.normal(0, 1, size=(T, d))
    near = base + rng.normal(0, 0.1, size=(T, d))
    far = base + rng.normal(0, 0.8, size=(T, d))
    assert drift_from_prior(base, base) < 1e-6
    assert 0 < drift_from_prior(base, near) < drift_from_prior(base, far)

    # surface repetition: a looping token stream is less distinct / lower entropy than a varied one
    looped = ([1, 2, 3] * 40)
    varied = list(rng.integers(0, 50, size=120))
    rl, rv = surface_repetition(looped), surface_repetition(varied)
    assert rl["distinct_2"] < rv["distinct_2"], (rl, rv)
    assert rl["token_entropy"] < rv["token_entropy"], (rl, rv)

    # output metrics: a confident (sharp) model has lower entropy AND lower surprisal on the truth
    V, P = 50, 40
    truth = rng.integers(0, V, size=P + 1)
    confident = np.full((P, V), -10.0)
    diffuse = np.full((P, V), float(np.log(1.0 / V)))
    for t in range(P):
        confident[t, truth[t + 1]] = 0.0                    # almost all mass on the true next token
    confident = confident - np.log(np.exp(confident).sum(1, keepdims=True))
    oc, od = output_metrics(confident, truth), output_metrics(diffuse, truth)
    assert oc["next_token_entropy"] < od["next_token_entropy"], (oc, od)
    assert oc["surprisal"] < od["surprisal"], (oc, od)

    # aggregate: shape + delta-vs-clean
    rows = [
        {"condition": "clean", "participation_ratio": 3.0, "lag1_autocorr": 0.1},
        {"condition": "looping", "participation_ratio": 1.5, "lag1_autocorr": 0.8},
    ]
    agg = aggregate(rows)
    assert abs(agg["looping"]["delta"]["participation_ratio"] - (-1.5)) < 1e-9
    assert abs(agg["looping"]["delta"]["lag1_autocorr"] - 0.7) < 1e-9

    print("metastability.smoke_test: OK")
    print(f"  sticky   PR={ms['participation_ratio']:.2f} ac={ms['lag1_autocorr']:+.2f} "
          f"step={ms['mean_step']:.3f}")
    print(f"  flexible PR={mf['participation_ratio']:.2f} ac={mf['lag1_autocorr']:+.2f} "
          f"step={mf['mean_step']:.3f}")
    print(f"  drift near={drift_from_prior(base, near):.3f} far={drift_from_prior(base, far):.3f}")
    print(f"  output  confident H={oc['next_token_entropy']:.2f} surp={oc['surprisal']:.2f} | "
          f"diffuse H={od['next_token_entropy']:.2f} surp={od['surprisal']:.2f}")


if __name__ == "__main__":
    smoke_test()
