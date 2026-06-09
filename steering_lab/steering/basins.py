"""Basin-flow visualizer — a *picture* of the residual-stream landscape, with the steer on top.

This re-implements the grid-teleport flow-field method of Fernando & Guitchounts, "Transformer
Dynamics: A neuroscientific approach to interpretability of large language models" (arXiv 2502.12131)
— the paper ships no code, so this is a faithful reimplementation of its published recipe, retargeted
from their Llama-3.1-8B to our Qwen2.5-3B + repeng `ControlModel`. The companion ideas (per-step
trajectory phases; perturb-between-attractors) come from Zhang et al. "Latent Trajectory Dynamics"
(DMET, arXiv 2505.20340) and Chia et al. "Probing Latent Subspaces in LLM for AI Security"
(arXiv 2503.09066). See `context/basin_flow.md` for the mapping and citations.

The Fernando recipe, exactly:
  1. Collect residual-stream activations across a corpus → reshape to a matrix.
  2. PCA via SVD; project onto the first two principal components (Z = Xc @ V[:, :2]).
  3. Build an n×n uniform grid across [rmin, rmax] in that 2-D space.
  4. Invert each grid point back to full dimension (x = z @ V[:, :2].T + mu) and INJECT it by
     replacing the activation at the input to a chosen decoder layer.
  5. Run the layer; read where the dynamics carried it; flow = read_z - grid_z. Quiver that.

The one addition that earns this module its place in *this* lab: step 4's injection runs while a
repeng control vector is active, so the same grid is teleported under `clean`, `looping`, and
`REBUS`. The flow field you get back is the basin — and you watch the looping steer deepen and
narrow it, then watch REBUS open it back up. That is Experiment ③'s metastability claim, drawn as a
landscape instead of a table.

Model-facing (torch); the PCA math and the synthetic `smoke_test()` are pure-numpy and CPU-testable.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = [
    "PCA2", "collect_residual", "flow_field", "basin_profile",
    "quiver_plot", "plot_basin_profile", "smoke_test",
]


# --- layer plumbing (reuses repeng's robust layer-list finder) ------------------------------------
def _layers(cmodel):
    """The decoder-layer ModuleList, whether `cmodel` is a repeng ControlModel or a bare HF model."""
    from repeng.control import model_layer_list  # llama/mistral/gemma/qwen/gpt-2 aware
    return model_layer_list(cmodel)


def _hidden_of(output):
    """Residual-stream tensor from a decoder layer's output (tuple for qwen/llama, else tensor)."""
    return output[0] if isinstance(output, tuple) else output


def _set_steer(cmodel, vector, coeff):
    """Activate an additive repeng steer for the whole forward pass (no-op for clean/coeff 0)."""
    cmodel.reset()
    if vector is not None and coeff != 0.0:
        cmodel.set_control(vector, coeff)


# --- PCA (SVD), faithful to the paper -------------------------------------------------------------
@dataclasses.dataclass
class PCA2:
    """First-two-PC projector fit on residual-stream activations. `Z = (X - mu) @ V` (V is d×2)."""
    mu: np.ndarray            # [d] column mean
    V: np.ndarray             # [d, 2] top-2 right singular vectors
    explained: np.ndarray     # [2] fraction of variance per component

    @classmethod
    def fit(cls, X: np.ndarray) -> "PCA2":
        X = np.asarray(X, dtype=np.float64)
        mu = X.mean(axis=0)
        Xc = X - mu
        # economy SVD; columns of Vt.T are principal axes, ordered by singular value
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = S ** 2
        explained = (var[:2] / var.sum()).astype(np.float32)
        return cls(mu=mu.astype(np.float32), V=Vt[:2].T.astype(np.float32), explained=explained)

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mu) @ self.V                       # [N, 2]

    def invert(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float32)
        return Z @ self.V.T + self.mu                       # [N, d]


# --- collect activations to fit the PCA -----------------------------------------------------------
def collect_residual(
    cmodel, tokenizer, prompts, *, layer: int, token_pos=None,
    vector=None, coeff: float = 0.0, system: str | None = None, max_tokens: int = 64,
) -> np.ndarray:
    """Gather layer-`layer` residual-stream activations over `prompts` to fit the PCA basis.

    `layer` indexes `out.hidden_states` (0 = embeddings; i = output of decoder layer i-1).
    `token_pos=None` keeps every (non-special) position; an int keeps just that position per prompt.
    A steer may be active during collection, but the basis is usually fit CLEAN (coeff 0) so every
    condition is drawn in the *same* coordinates — pass that PCA2 to `flow_field` for each arm.
    """
    import torch
    from .steer import _chat

    _set_steer(cmodel, vector, coeff)
    chunks: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for p in prompts:
                text = _chat(tokenizer, p, system)
                enc = tokenizer(text, return_tensors="pt").to(cmodel.device)
                out = cmodel(**enc, output_hidden_states=True)
                hs = out.hidden_states[layer][0]            # [T, d]
                hs = hs[-1:] if token_pos == -1 else (hs if token_pos is None else hs[token_pos:token_pos + 1])
                chunks.append(hs.to(torch.float32).cpu().numpy())
    finally:
        cmodel.reset()
    return np.concatenate(chunks, axis=0)


# --- the grid-teleport flow field -----------------------------------------------------------------
def flow_field(
    cmodel, tokenizer, pca: PCA2, *, layer: int, seed_prompt: str,
    grid_n: int = 12, extent: float = 2.5, token_pos: int = -1,
    vector=None, coeff: float = 0.0, system: str | None = None,
) -> dict:
    """Teleport an n×n grid of activations into decoder `layer` and read where the dynamics push them.

    For each grid point z (in PCA coords): invert to full dim, REPLACE the residual at `token_pos`
    on the *input* to decoder `layer`, run that layer (with any steer active), read its *output* at
    `token_pos`, project back → the flow vector `read_z - z` (one decoder-layer step). The whole grid
    runs in a single batched forward (grid as the batch dimension).

    `extent` scales the grid half-width in units of the PCA-projected activation std, so it adapts to
    the basis. `layer` here is the decoder-layer index (0-based into the ModuleList), i.e.
    `hidden_states[layer]` is this layer's *output*. Returns a dict ready for `quiver_plot`:
      {grid_z[G,2], flow_z[G,2], speed[G], extent, grid_n, layer, coeff, explained}.
    """
    import torch

    layers = _layers(cmodel)
    block = layers[layer]
    dev, dt = cmodel.device, cmodel.model.dtype

    # grid spans ±extent (PCA-coordinate units) around the projected-activation mean (origin ~0 in
    # centered PCA coords). Derive `extent` from the projected data range, e.g. ~2-3× its std.
    span = float(extent)
    axis = np.linspace(-span, span, grid_n, dtype=np.float32)
    gz = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1).reshape(-1, 2)   # [G, 2]
    g_full = torch.tensor(pca.invert(gz), device=dev, dtype=dt)                       # [G, d]
    G = gz.shape[0]

    # batch the seed prompt G times; each row gets its own injected activation at token_pos
    from .steer import _chat
    enc = tokenizer(_chat(tokenizer, seed_prompt, system), return_tensors="pt").to(dev)
    ids = enc["input_ids"].repeat(G, 1)
    attn = enc.get("attention_mask")
    attn = attn.repeat(G, 1) if attn is not None else None
    pos = token_pos if token_pos >= 0 else ids.shape[1] + token_pos

    captured = {}

    def pre_hook(_module, args, kwargs):
        hs = args[0].clone()
        hs[:, pos, :] = g_full                              # teleport: replace input activation
        return (hs,) + args[1:], kwargs

    def post_hook(_module, _args, output):
        captured["out"] = _hidden_of(output)[:, pos, :].detach()
        return output

    _set_steer(cmodel, vector, coeff)
    h1 = block.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = block.register_forward_hook(post_hook)
    try:
        with torch.no_grad():
            kw = {"attention_mask": attn} if attn is not None else {}
            cmodel(input_ids=ids, **kw)
    finally:
        h1.remove(); h2.remove(); cmodel.reset()

    read_full = captured["out"].to(torch.float32).cpu().numpy()                       # [G, d]
    read_z = pca.project(read_full)
    flow = read_z - gz
    return {
        "grid_z": gz, "flow_z": flow, "speed": np.linalg.norm(flow, axis=1),
        "extent": float(span), "grid_n": grid_n, "layer": layer, "coeff": float(coeff),
        "explained": pca.explained,
    }


# --- dose-response: basin width & steepness -------------------------------------------------------
def basin_profile(
    cmodel, tokenizer, pca: PCA2, *, layer: int, seed_prompt: str,
    center_z=(0.0, 0.0), radii=None, n_dirs: int = 16, seed: int = 0, token_pos: int = -1,
    vector=None, coeff: float = 0.0, system: str | None = None,
) -> dict:
    """Teleport outward from a center at growing radii; measure how often the flow points back in.

    For each radius r, place `n_dirs` probes evenly around the center (in PCA coords), teleport each,
    read the flow, and score it as *inward* if the flow has a component back toward the center
    (flow · (center - probe) > 0). `inward_fraction(r)` is a return-to-basin curve: the radius where
    it crosses 0.5 is the basin width; how sharply it drops reads out steepness (Friston precision =
    local curvature). A deeper/narrower basin under `looping` than `clean` is the metastability-loss
    signature; `REBUS` should widen it back.

    Returns {radii, inward_fraction, mean_inward_speed, center_z, layer, coeff}.
    """
    center = np.asarray(center_z, dtype=np.float32)
    radii = np.asarray(radii if radii is not None else np.linspace(0.25, 4.0, 8), dtype=np.float32)
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False) + rng.uniform(0, 2 * np.pi)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)      # [n_dirs, 2]

    # probe points for every (radius, direction), teleported in one field call via a custom grid
    inward_frac, mean_inward = [], []
    for r in radii:
        probes = center[None, :] + r * dirs                                          # [n_dirs, 2]
        field = _teleport_points(cmodel, tokenizer, pca, probes, layer=layer, seed_prompt=seed_prompt,
                                 token_pos=token_pos, vector=vector, coeff=coeff, system=system)
        flow = field["flow_z"]
        toward = center[None, :] - probes                                            # inward direction
        toward /= (np.linalg.norm(toward, axis=1, keepdims=True) + 1e-8)
        proj = (flow * toward).sum(axis=1)                                            # >0 = pulled back
        inward_frac.append(float((proj > 0).mean()))
        mean_inward.append(float(proj.mean()))
    return {
        "radii": radii, "inward_fraction": np.asarray(inward_frac, np.float32),
        "mean_inward_speed": np.asarray(mean_inward, np.float32),
        "center_z": center, "layer": layer, "coeff": float(coeff),
    }


def _teleport_points(cmodel, tokenizer, pca, points_z, *, layer, seed_prompt,
                     token_pos=-1, vector=None, coeff=0.0, system=None) -> dict:
    """Like `flow_field` but for an arbitrary list of PCA-coord points (not a regular grid)."""
    import torch
    from .steer import _chat

    layers = _layers(cmodel)
    block = layers[layer]
    dev, dt = cmodel.device, cmodel.model.dtype
    pz = np.asarray(points_z, dtype=np.float32)
    p_full = torch.tensor(pca.invert(pz), device=dev, dtype=dt)
    P = pz.shape[0]

    enc = tokenizer(_chat(tokenizer, seed_prompt, system), return_tensors="pt").to(dev)
    ids = enc["input_ids"].repeat(P, 1)
    attn = enc.get("attention_mask")
    attn = attn.repeat(P, 1) if attn is not None else None
    pos = token_pos if token_pos >= 0 else ids.shape[1] + token_pos
    captured = {}

    def pre_hook(_m, args, kwargs):
        hs = args[0].clone(); hs[:, pos, :] = p_full
        return (hs,) + args[1:], kwargs

    def post_hook(_m, _a, output):
        captured["out"] = _hidden_of(output)[:, pos, :].detach()
        return output

    _set_steer(cmodel, vector, coeff)
    h1 = block.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = block.register_forward_hook(post_hook)
    try:
        with torch.no_grad():
            kw = {"attention_mask": attn} if attn is not None else {}
            cmodel(input_ids=ids, **kw)
    finally:
        h1.remove(); h2.remove(); cmodel.reset()

    read_z = pca.project(captured["out"].to(torch.float32).cpu().numpy())
    return {"grid_z": pz, "flow_z": read_z - pz, "speed": np.linalg.norm(read_z - pz, axis=1)}


# --- plotting (lazy matplotlib) -------------------------------------------------------------------
def quiver_plot(field: dict, ax=None, *, title=None, cmap="viridis"):
    """Quiver/streamline render of a `flow_field` result. Returns the matplotlib Axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    gz, fl, sp = field["grid_z"], field["flow_z"], field["speed"]
    ax.quiver(gz[:, 0], gz[:, 1], fl[:, 0], fl[:, 1], sp, cmap=cmap,
              angles="xy", scale_units="xy", scale=1.0, width=0.004)
    ev = field.get("explained")
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})" if ev is not None else "PC1")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})" if ev is not None else "PC2")
    ax.set_title(title or f"layer {field['layer']}  coeff {field['coeff']:+g}")
    ax.set_aspect("equal")
    return ax


def plot_basin_profile(profiles: dict, ax=None, *, title="basin return curve"):
    """Plot inward_fraction vs radius for one or more labelled `basin_profile` results.

    `profiles` = {label: profile_dict}. The 0.5 crossing marks the basin width per condition.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    for label, pr in profiles.items():
        ax.plot(pr["radii"], pr["inward_fraction"], marker="o", label=label)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_xlabel("teleport radius (PCA units)")
    ax.set_ylabel("inward fraction (pulled back to basin)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    return ax


# --- CPU smoke test (PCA math + synthetic contractive flow) ---------------------------------------
def smoke_test() -> None:
    """Validate the PCA basis and the inward-flow scoring on synthetic data — no model, CPU only."""
    rng = np.random.default_rng(0)

    # (1) PCA round-trip: data living on a 2-D plane in 8-D should reconstruct near-exactly.
    basis = rng.standard_normal((8, 2))
    coords = rng.standard_normal((500, 2)) * np.array([3.0, 1.0])     # PC1 wider than PC2
    X = coords @ basis.T + rng.standard_normal(8) * 5                  # offset mean
    pca = PCA2.fit(X)
    Z = pca.project(X)
    Xr = pca.invert(Z)
    recon = np.linalg.norm(X - Xr) / np.linalg.norm(X - X.mean(0))
    assert recon < 1e-5, f"PCA reconstruction error too high: {recon}"
    assert pca.explained[0] > pca.explained[1], "PC1 should explain more than PC2"
    assert pca.explained.sum() > 0.99, f"2 PCs should capture ~all variance: {pca.explained.sum()}"
    assert abs(pca.project(pca.invert(Z)) - Z).max() < 1e-3, "project∘invert not identity"

    # (2) inward scoring: a contractive field x -> 0.5 x pulls every probe toward center.
    center = np.zeros(2, np.float32)
    for r in (0.5, 2.0, 5.0):
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        probes = r * np.stack([np.cos(angles), np.sin(angles)], 1).astype(np.float32)
        flow = -0.5 * probes                                          # contractive
        toward = center[None] - probes
        toward /= np.linalg.norm(toward, axis=1, keepdims=True) + 1e-8
        inward = ((flow * toward).sum(1) > 0).mean()
        assert inward == 1.0, f"contractive flow should be fully inward, got {inward}"

    # (3) a repulsive field x -> 1.5 x should be fully outward (basin escape).
    probes = 2.0 * np.stack([np.cos(angles), np.sin(angles)], 1).astype(np.float32)
    flow = 0.5 * probes
    toward = -probes / (np.linalg.norm(probes, axis=1, keepdims=True) + 1e-8)
    outward = ((flow * toward).sum(1) > 0).mean()
    assert outward == 0.0, f"repulsive flow should be fully outward, got {outward}"

    print("basins.smoke_test: OK")
    print(f"  PCA recon error={recon:.2e}  explained={pca.explained}  (PC1>{pca.explained[1]:.3f})")
    print("  contractive flow inward=1.0  repulsive flow inward=0.0")


if __name__ == "__main__":
    smoke_test()
