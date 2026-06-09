# Pre-registration — Metastability probe (Experiment ③, trimmed robust core)

*Locked before any GPU run. Scope deliberately narrow; the speculative extensions are listed under
"Deferred" and are out of scope until the core lands.*

## Motivation (framing, not a claim the numbers carry)

Kotler & Friston (2026) frame the looping pathologies — rumination, worry, intrusive
re-experiencing — as a **loss of metastability**: the system stops fluidly switching among
semi-stable states and gets trapped in an attractor; over-precise priors deepen the basin so
evidence can't push the state out, and therapy (REBUS = relaxing those priors, here a *negative*
steer) restores the switching. We don't claim to measure metastability in Friston's formal
(Kuramoto order-parameter) sense. The Friston story is the **motivation that predicted** the
following measurable, defensible headline:

> **Steering a language model toward looping pathologies reduces the flexibility of its layer-L
> residual-stream trajectory and warps its internal representation — reversibly (a negative steer
> restores flexibility), and distinguishably from a random perturbation of matched magnitude.**

Everything below tests exactly that sentence.

## Mechanistic setup

As the model generates, the residual stream at a chosen layer L traces a trajectory (one activation
vector per token). A steering vector is a fixed direction added into that stream at a band of middle
layers on every forward pass — a stand-in for an over-confident prior. We measure how the trajectory
*moves* (flexibility), how far the steer *drags* the representation (drift), and what the steer does
to the model's *output distribution* (precision/accuracy). A single model suffices: the steer-OFF
forward pass is the clean reference (repeng `reset()`), so drift ("same text, steer on vs off") and
cross-surprise ("clean model's surprise at steered text") need no second model.

## Metrics (robust core — all parameter-free or near-it)

| metric | source | definition | sick direction |
|---|---|---|---|
| `participation_ratio` | generative trajectory, steer ON | (Σλ)²/Σλ² of temporal covariance — effective dims explored | **lower** = collapsed/stuck |
| `lag1_autocorr` | generative trajectory, steer ON | mean per-dim lag-1 autocorrelation | **higher** = each state echoes the last |
| `mean_step` | generative trajectory, steer ON | mean L2 step, norm-scaled | **lower** = not moving |
| `drift_from_prior` | same completion, ON vs OFF | mean per-token ‖on−off‖ / scale | magnitude only (read jointly) |
| `next_token_entropy` | read-only passage, steer ON | mean entropy of next-token dist = precision/gain | **lower** = overconfident/over-precise |
| `surprisal` | read-only passage, steer ON | mean −log p(realized token) = accuracy | exploratory (no pre-reg direction on neutral text) |
| `cross_surprise` | steered completion scored under clean | clean model's mean NLL on the steered text | **higher** = off the clean manifold |
| `distinct_2`, `token_entropy` | generative completion | surface-repetition guard | lower = repetitive (sanity, not the claim) |

`drift_from_prior` is a steer-cost magnitude, not a health axis (any steer drifts), so it is read
jointly with the others, never alone. `surprisal` on neutral text is exploratory — the sharp
surprisal test (prior-congruent vs incongruent passages) was deferred. Code:
`steering/metastability.py` (pure-numpy, with `smoke_test()`).

## Conditions (matched magnitude `|coeff|`)

| arm | steer | role |
|---|---|---|
| `clean` | none (coeff 0) | baseline |
| `looping` | +coeff on a looping vector (`rumination` or `circular_inference`) | the pathology |
| `REBUS` | −coeff on the over-precise-prior vector (PC `prior_precision_high`) | the therapy arm |
| `random` | +coeff on a random direction, per-layer norm matched to the looping vector | **keystone control** |

The `random` arm is what makes the result falsifiable: it separates "a meaningful steer / therapy"
from "any perturbation of the same size." REBUS defaults to a coarse `−coeff` on the looping vector
if the PC bundle (with `prior_precision_high`) isn't extracted yet — documented so it isn't
over-read. Code: `steering/probes_eval.py`.

## Stimuli (two families only)

- **A — generative prompts** (`GEN_PROMPTS`): content-free continuation prompts (the shared
  `personas.NEUTRAL_PROMPTS`). The steered model talks; we read the trajectory. Aim ~120–200
  generated tokens for stable geometry.
- **B — read-only battery** (`READ_BATTERY`): ~10 short, flat, ordinary passages (weather, recipe,
  mundane narrative), teacher-forced identically under every condition. Identical tokens make
  surprisal/entropy/drift directly comparable; mundane content avoids masking the steer.

## Predictions (the table the GPU run is checked against)

| metric | looping vs clean | REBUS vs clean | random vs clean |
|---|---|---|---|
| participation_ratio | ↓ | ↑ (≥ clean) | ~ / diffuse |
| lag1_autocorr | ↑ | ↓ | ~ |
| mean_step | ↓ | ↑ | ↑ (but incoherent) |
| drift_from_prior | ↑ | ↑ (it is a steer) | ↑ |
| next_token_entropy | ↓ (overconfident) | ↑ | ↑ (diffuse) |
| cross_surprise | ↑ | moderate | ↑↑ |
| distinct_2 | ↓ (repetitive) | ~ clean | ↓ |

**Key discriminator (the whole point of two axes + the random arm):**
- **looping** = participation_ratio ↓ + lag1_autocorr ↑ + entropy ↓, with **coherence intact**.
- **random** = drift ↑ + entropy ↑ + cross_surprise ↑↑, with **coherence broken**.
- **REBUS** = flexibility restored (PR ↑, autocorr ↓) **without** the random signature.

A clean pass = the looping and random arms are *distinguishable* and REBUS reverses the looping
effect. That, not any single number, is the result.

## Deferred (explicitly out of scope here)

- Prior-congruent vs incongruent passage pairs; ambiguous passages (the sharp false-inference /
  surprise-asymmetry tests).
- Clinical-questionnaire behavioral anchor.
- The demoted/stripped metrics: `transition_rate`, `mean_dwell`, `n_microstates` (k-means knobs);
  `flow_determinism` (underdetermined + #PC knob); `redundancy`/`effective_rank` (ambiguous alone).
- The "dreaming / synaptic-pruning" extension and the full 2×2 complexity×flexibility narrative.

These run only **after** the core above replicates.

## Verification

- **CPU (no model):** `python -c "from steering.metastability import smoke_test; smoke_test()"` —
  each metric separates synthetic sticky vs flexible / confident vs diffuse. Plus a stub-`ControlVector`
  check of `random_control_vector` (per-layer norm match) and `build_conditions`.
- **GPU (notebook):** `steer_mechanisms.ipynb` → "Experiment ③" cell. Needs `out/control_vectors.pkl`
  (clinical) loaded; `control_vectors_pc.pkl` supplies the REBUS `prior_precision_high` if available.
  Run `experiment.run_experiment(...)` over A + B across the four arms; check the predictions table.

## Companion (qualitative, exploratory — not part of the locked claim)
A basin-flow visualizer draws the same effect as a *picture* instead of a table: the grid-teleport
flow-field method of Fernando & Guitchounts (arXiv 2502.12131), run with the steer active, so the
looping basin can be *seen* to deepen and REBUS to re-open it. Spec + citations:
`context/basin_flow.md`; code `steering/basins.py`; notebook §8. The falsifiable result stays here.

## Files
- `steering/metastability.py` — pure-numpy metrics + `smoke_test()`.
- `steering/probes_eval.py` — `GEN_PROMPTS`, `READ_BATTERY`, `random_control_vector`, `build_conditions`.
- `steering/experiment.py` — `read_states_and_logits`, `run_generative`, `run_readonly`, `run_experiment`.
- `steer_mechanisms.ipynb` — the GPU cell.
