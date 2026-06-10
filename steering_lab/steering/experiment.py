"""Experiment ③ runner — model-facing glue for the trimmed metastability probe.

All torch / forward-pass / generation logic lives here so `metastability.py` stays pure-numpy and
CPU-testable. The flow, per (string × condition):

  generative (GEN_PROMPTS): generate a completion under the steer, then read the SAME completion twice
    — once with the steer ON (the trajectory we score) and once OFF (the clean reference). Gives
    trajectory flexibility + drift_from_prior + surface_repetition + cross_surprise (the clean/reset
    model's surprisal on what the steer made it say — how far off-manifold it went).

  read-only (READ_BATTERY): teacher-force a fixed passage as the assistant turn, reading states+logits
    ON and OFF. Gives surprisal + next_token_entropy (precision) on aligned tokens + drift_from_prior.

A single model does everything: the steer-OFF pass *is* the clean/reset model (repeng `reset()`),
so no second model is needed for drift or cross-surprise.

See `probes_eval.build_conditions` for the four arms (clean/looping/REBUS/random) and
`context/metastability_prereg.md` for the locked predictions.
"""

from __future__ import annotations

import numpy as np

from . import metastability as M
from .probes_eval import GEN_PROMPTS, READ_BATTERY, build_conditions
from .steer import _chat   # one chat-template helper (handles Gemma 4 thinking mode) for the whole lab

__all__ = [
    "read_states_and_logits", "run_generative", "run_readonly", "run_experiment",
    "READ_PROMPT",
]

# Constant neutral framing for the read-only battery: the mundane passages are plausible assistant
# turns under this instruction, and it's identical across every condition so surprisal/entropy stay
# comparable. (The passage itself, not the framing, is what we score.)
READ_PROMPT = "Write a few plain sentences about something ordinary."


def read_states_and_logits(
    cmodel, tokenizer, prompt: str, completion: str, *,
    layer: int, vector=None, coeff: float = 0.0,
    system: str | None = None, normalize: bool = False,
):
    """One forward pass over (chat prefix + completion); read layer-`layer` states + next-token
    log-probs over the completion positions, under the given steer.

    Returns (traj[T,d] float32, comp_logprobs[T,V] float32, comp_ids list[int]) where:
      - traj[i]         = hidden state (hidden_states[layer]) at the i-th completion token
      - comp_logprobs[i]= log-softmax distribution that PREDICTS the i-th completion token
      - comp_ids[i]     = the i-th completion token id
    `layer` indexes `out.hidden_states` directly (0 = embeddings; i = output of decoder layer i-1),
    so pass a value at/after the steered band to capture the accumulated steer.
    """
    import torch

    prefix = _chat(tokenizer, prompt, system)
    prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"]
    full = tokenizer(prefix + completion, return_tensors="pt").to(cmodel.device)
    start = int(prefix_ids.shape[1])
    seq = int(full["input_ids"].shape[1])
    if seq <= start:                                    # empty completion
        d = cmodel.model.config.hidden_size
        return (np.zeros((0, d), np.float32), np.zeros((0, 0), np.float32), [])

    cmodel.reset()
    if vector is not None and coeff != 0.0:
        cmodel.set_control(vector, coeff, normalize=normalize)
    try:
        with torch.no_grad():
            out = cmodel(**full, output_hidden_states=True)
    finally:
        cmodel.reset()

    # hidden states over the completion tokens
    hs = out.hidden_states[layer][0]                    # [seq, d]
    traj = hs[start:seq].to(torch.float32).cpu().numpy()

    # predictors: logits[i-1] predicts token i; rows [start-1 : seq-1] predict completion [start:seq]
    logits = out.logits[0][start - 1:seq - 1].to(torch.float32)   # [T, V]
    comp_logprobs = torch.log_softmax(logits, dim=-1).cpu().numpy()
    comp_ids = full["input_ids"][0][start:seq].tolist()
    return traj, comp_logprobs, comp_ids


def _output_from_completion(comp_logprobs, comp_ids) -> dict:
    """surprisal + next_token_entropy over a completion whose log-probs are row-aligned to its ids.

    `comp_logprobs[i]` predicts `comp_ids[i]`. `metastability.output_metrics` uses the offset
    convention (row t predicts target[t+1]), so we prepend a dummy target to align them.
    """
    if len(comp_ids) == 0:
        return {"surprisal": float("nan"), "next_token_entropy": float("nan")}
    return M.output_metrics(comp_logprobs, [0] + list(comp_ids))


def run_generative(cmodel, tokenizer, prompts, conditions, *, read_layer, system=None, gen=None):
    """Per prompt × condition: generate under the steer, then measure the trajectory."""
    from .steer import generate_steered

    gen = dict(gen or {})
    rows = []
    for p in prompts:
        for label, vec, c in conditions:
            completion = generate_steered(cmodel, tokenizer, p, vec, c, system=system, **gen)
            traj_on, _, ids = read_states_and_logits(
                cmodel, tokenizer, p, completion, layer=read_layer, vector=vec, coeff=c, system=system)
            traj_off, lp_off, ids_off = read_states_and_logits(
                cmodel, tokenizer, p, completion, layer=read_layer, vector=None, coeff=0.0, system=system)
            row = {
                "family": "generative", "condition": label, "prompt": p, "coeff": c,
                "completion": completion,
                **M.trajectory_metrics(traj_on),
                "drift_from_prior": M.drift_from_prior(traj_on, traj_off),
                **M.surface_repetition(ids),
                # clean/reset model's surprisal on the steered text = how far off-manifold it went
                "cross_surprise": _output_from_completion(lp_off, ids_off)["surprisal"],
            }
            rows.append(row)
    return rows


def run_readonly(cmodel, tokenizer, battery, conditions, *, read_layer, prompt=READ_PROMPT, system=None):
    """Per passage × condition: teacher-force the fixed passage, measure output precision + drift."""
    rows = []
    for passage in battery:
        for label, vec, c in conditions:
            traj_on, lp_on, ids = read_states_and_logits(
                cmodel, tokenizer, prompt, passage, layer=read_layer, vector=vec, coeff=c, system=system)
            traj_off, _, _ = read_states_and_logits(
                cmodel, tokenizer, prompt, passage, layer=read_layer, vector=None, coeff=0.0, system=system)
            row = {
                "family": "readonly", "condition": label, "passage": passage[:60], "coeff": c,
                **_output_from_completion(lp_on, ids),
                "drift_from_prior": M.drift_from_prior(traj_on, traj_off),
            }
            rows.append(row)
    return rows


def run_experiment(
    cmodel, tokenizer, *, looping_vec, rebus_vec=None, read_layer,
    coeff: float = 8.0, seed: int = 0, system=None, gen=None,
    prompts=None, battery=None,
) -> dict:
    """Full trimmed Experiment ③: build the four conditions, run both batteries, aggregate.

    Returns {"conditions": [labels], "generative": rows, "readonly": rows,
             "agg_generative": {...}, "agg_readonly": {...}}.
    """
    conditions = build_conditions(looping_vec, rebus_vec, coeff=coeff, seed=seed)
    prompts = GEN_PROMPTS if prompts is None else prompts
    battery = READ_BATTERY if battery is None else battery

    gen_rows = run_generative(cmodel, tokenizer, prompts, conditions, read_layer=read_layer,
                              system=system, gen=gen)
    read_rows = run_readonly(cmodel, tokenizer, battery, conditions, read_layer=read_layer,
                             system=system)
    return {
        "conditions": [c[0] for c in conditions],
        "generative": gen_rows,
        "readonly": read_rows,
        "agg_generative": M.aggregate(gen_rows),
        "agg_readonly": M.aggregate(read_rows),
    }
