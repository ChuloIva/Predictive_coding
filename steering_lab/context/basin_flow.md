# Basin-flow visualizer — drawing the metastability landscape

*A companion to the metastability probe (`metastability_prereg.md`). Where Experiment ③ measures
trajectory flexibility as **numbers**, this draws the residual-stream basin as a **picture** — and
overlays the steer so you watch the looping pathology deepen and narrow the basin, then watch REBUS
re-open it.*

## Why this exists

The metastability claim is geometric: a looping pathology = the residual-stream state trapped in a
deepened basin; therapy = restoring flexible state-switching. The prereg measures that indirectly
(participation ratio, autocorrelation, drift). This module makes the basin itself visible — a flow
field over a 2-D slice of the residual stream — so the claim can be *seen*, not only tabulated. It is
a qualitative, exploratory companion; the falsifiable result still lives in the prereg's predictions
table and the `random` keystone control.

## Sources (method reused; none ship code — all reimplemented from the papers)

1. **Fernando & Guitchounts (2025), "Transformer Dynamics: A neuroscientific approach to
   interpretability of large language models"** — arXiv:2502.12131. *The core method.* Treats the
   residual stream as a dynamical system, reduces it to 2-D by **PCA via SVD** (also a 4096→2
   compressing autoencoder, which we skip), then builds an **n×n grid** in that space, **inverts**
   each point to full dimension, **injects it at the input to a decoder layer**, and reads where the
   dynamics carry it. Their seed prompt was *"I'm sorry, Dave. I'm afraid I can't do that."*; their
   model Llama-3.1-8B, injection layers 0/7/15/23/31, grid n≈10. We retarget to Qwen2.5-3B and run
   the injection **with a repeng steer active**.
   - Reimplemented in `basins.py`: `PCA2` (SVD, project/invert), `flow_field` (grid teleport).

2. **Zhang, Dong & Li (2025), "Latent Trajectory Dynamics in Large Language Models" (DMET,
   Dynamical Manifold Evolution Theory)** — arXiv:2505.20340. Models *generation* (across time, not
   layers) as evolution on a low-dim semantic manifold with three phases — initialization (drift),
   expansion (directed evolution), convergence (settling into a semantic attractor) — and three
   proxy metrics: **state continuity (C)**, **attractor clustering quality (Q)**, **topological
   persistence (P)**. The convergence phase = the basin forming during generation. Our prereg's
   `lag1_autocorr` / `participation_ratio` / `mean_step` already cover the time-axis version; we cite
   DMET as the parallel framing and keep the layer-axis flow field here.

3. **Chia, Wong & Pan (2025), "Probing Latent Subspaces in LLM for AI Security: Identifying and
   Manipulating Adversarial States"** — arXiv:2503.09066. Identifies semi-stable attractor states in
   LLM latents and derives a **perturbation vector** that pushes a safe state into a jailbroken one,
   probing how the shift propagates across layers. This is the "how steep is the basin / how hard to
   escape" question operationalized — it motivates our **dose-response basin probe** (`basin_profile`:
   teleport outward at growing radii, measure the return-to-basin fraction).

*Caveat we are honoring:* the Medium "Persistent Symbolic Attractor Basins" white paper that circulates
on this topic is not peer-reviewed and makes cross-session-persistence claims we treat as unsupported;
nothing here builds on it.

## What `steering/basins.py` does

| function | role |
|---|---|
| `collect_residual(...)` | run prompts, gather layer-L residual activations to fit the basis (fit **clean** so every arm shares coordinates) |
| `PCA2.fit / project / invert` | SVD to 2-D and back — the exact Fernando reduction |
| `flow_field(...)` | n×n grid → invert → inject at decoder layer input (`forward_pre_hook` replaces the activation) → read layer output (`forward_hook`) → flow = read_z − grid_z, batched as one forward. Runs with `vector`/`coeff` so the steer is live. |
| `basin_profile(...)` | dose-response: teleport outward from a center at growing radii along `n_dirs`, score each flow as inward/outward; `inward_fraction(r)` crossing 0.5 = basin width, its slope = steepness |
| `quiver_plot / plot_basin_profile` | matplotlib renders |
| `smoke_test()` | CPU: PCA round-trips (recon < 1e-5, PC1 > PC2), contractive flow scores fully inward, repulsive fully outward |

**The single integration that earns its place:** injection happens while a repeng `ControlVector` is
active, so the *same* grid and basis are teleported under `clean`, `looping`, and `REBUS`. Predicted
reading, consistent with the prereg:

- **looping** vs clean → flow field contracts toward a tighter region; `basin_profile` inward-fraction
  stays high out to larger radii (deeper, narrower basin = the metastability loss, drawn).
- **REBUS** → basin widens / flow loosens back toward the clean field (flexibility restored).
- **random** (norm-matched) → field is disorganized, not a coherently deepened basin (the control
  that separates "a real attractor" from "any perturbation of the same size").

## How to read it (mechanistic)

The residual stream at layer L is a point in a 2048-d space; across a forward pass each token's state
flows layer to layer. PCA gives a 2-D window onto that space; the grid asks "if the state were *here*,
which way would one decoder layer push it?" Arrows converging on a region = an attractor (a belief the
model snaps back to); arrows fanning out = a ridge it slides off. A steer that adds an over-precise
prior should pull the arrows inward around the pathology's basin — visibly steeper than clean, and
distinct from the random push.

## Verification

- **CPU:** `python3 -c "from steering.basins import smoke_test; smoke_test()"` (PCA + inward scoring).
- **GPU (notebook):** `steer_mechanisms.ipynb` → "Experiment ③·b — basin landscape": fit `PCA2` on
  clean activations, then `flow_field` per condition on a shared grid; `quiver_plot` the three panels
  and `plot_basin_profile` the return curves. Needs `out/control_vectors.pkl` loaded (same bundle as
  Experiment ③).

## Scope / honesty

This is the **picture**, not the proof. It is exploratory and qualitative — the 2-D PCA window
discards most of the variance (the `explained` fractions are reported on the axes so this is never
hidden), the layer-axis flow is a one-step linearization, and "basin width" is a proxy. The
falsifiable claim remains the prereg's predictions table with the `random` control. Deferred, as in
the prereg: the autoencoder bottleneck (cleaner 2-D but another trained component), per-step (DMET
time-axis) basins, and any cross-session-persistence interpretation.
