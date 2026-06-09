"""Experiment ① — are the clinical atoms linear combinations of the PC primitives?

Given two extracted vector bundles (the clinical set from `personas.py` and the predictive-coding
primitive set from `personas_pc.py`), this asks two quantitative questions:

  (a) GEOMETRY — is the PC-primitive basis more mutually orthogonal / higher-rank than the clinical
      basis? If the PC dials are more fundamental, they should be *less* redundant with each other.
      → `basis_geometry`, `compare_geometry`.

  (b) RECONSTRUCTION — can each clinical-atom (and syndrome) direction be expressed as a linear
      combination of the PC-primitive directions? I.e. does the clinical vector live *inside* the
      PC subspace, and do the primitives that carry the most weight match the pre-registered
      predictions in `personas_pc.CLINICAL_AS_PC`?  → `reconstruct`, `reconstruction_report`.

Sign convention: repeng orients each ControlVector so the +direction is the mechanism-ON side, so a
non-negative (NNLS) reconstruction of an ON clinical vector from ON PC vectors is meaningful. We
report BOTH the orientation-free subspace fit (OLS R², the strongest "is it in the span" claim) and
the NNLS loadings (directional, tests the signed hypothesis). Chance R² for a random target on a
random k-dim subspace of a d-dim space is ≈ k/d — reported as a baseline so R² is interpretable.
"""

from __future__ import annotations

import numpy as np

from .extract import load_bundle
from .personas import SYNDROME_RECIPES
from .personas_pc import CLINICAL_AS_PC, SYNDROME_AS_PC

try:  # NNLS is optional; subspace R² (OLS) works without scipy
    from scipy.optimize import nnls as _scipy_nnls
except Exception:  # pragma: no cover
    _scipy_nnls = None


# --------------------------------------------------------------------------------------------------
# bundle helpers
# --------------------------------------------------------------------------------------------------
def _dirs(bundle: dict, layer: int) -> dict[str, np.ndarray]:
    """{mechanism: direction vector} at one layer, as float64."""
    return {
        m: np.asarray(d[layer], dtype=np.float64)
        for m, d in bundle["vectors"].items()
        if layer in d
    }


def common_layers(*bundles: dict) -> list[int]:
    """Layers present for *every* mechanism in *every* bundle."""
    sets = []
    for b in bundles:
        for d in b["vectors"].values():
            sets.append(set(d.keys()))
    return sorted(set.intersection(*sets)) if sets else []


def pick_layer(*bundles: dict) -> int:
    """A sensible default: the median common layer (mid-band steers most cleanly)."""
    layers = common_layers(*bundles)
    if not layers:
        raise ValueError("bundles share no common layer")
    return layers[len(layers) // 2]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def _stack(dirs: dict[str, np.ndarray], labels: list[str], *, unit: bool = True) -> np.ndarray:
    """[k, d] matrix, one row per label (optionally unit-normalized)."""
    rows = [(_unit(dirs[m]) if unit else dirs[m]) for m in labels]
    return np.vstack(rows)


# --------------------------------------------------------------------------------------------------
# (a) geometry
# --------------------------------------------------------------------------------------------------
def basis_geometry(bundle: dict, layer: int) -> dict:
    """Redundancy / spread of a set of direction vectors at one layer.

    mean_abs_offdiag_cos : average |cosine| between distinct vectors (0 = orthogonal, 1 = collinear).
    participation_ratio  : effective number of dimensions the vectors span, (Σσ²)²/Σσ⁴ ∈ [1, k].
                           Close to k ⇒ near-orthogonal independent dials; ≪ k ⇒ redundant.
    """
    dirs = _dirs(bundle, layer)
    labels = list(dirs)
    M = _stack(dirs, labels, unit=True)          # [k, d], unit rows
    G = M @ M.T                                   # cosine matrix
    k = len(labels)
    off = G[~np.eye(k, dtype=bool)]
    s = np.linalg.svd(M, compute_uv=False)
    pr = float((s**2).sum() ** 2 / (s**4).sum()) if s.size else 0.0
    return {
        "n": k,
        "labels": labels,
        "mean_abs_offdiag_cos": float(np.abs(off).mean()),
        "max_abs_offdiag_cos": float(np.abs(off).max()),
        "participation_ratio": pr,
        "pr_fraction": pr / k if k else 0.0,   # 1.0 = perfectly orthogonal basis
        "cosine_matrix": G,
    }


def compare_geometry(clinical_bundle: dict, pc_bundle: dict, layer: int | None = None) -> dict:
    """Side-by-side geometry of the clinical vs PC-primitive bases."""
    layer = layer if layer is not None else pick_layer(clinical_bundle, pc_bundle)
    return {
        "layer": layer,
        "clinical": basis_geometry(clinical_bundle, layer),
        "pc": basis_geometry(pc_bundle, layer),
    }


# --------------------------------------------------------------------------------------------------
# (b) reconstruction
# --------------------------------------------------------------------------------------------------
def reconstruct(target: np.ndarray, basis_M: np.ndarray) -> dict:
    """Express `target` (d,) as a linear combination of basis rows `basis_M` (k, d).

    Returns OLS loadings + subspace R² (orientation-free), and NNLS loadings if scipy is present.
    """
    A = basis_M.T                                  # [d, k]
    t = target.astype(np.float64)
    ols, *_ = np.linalg.lstsq(A, t, rcond=None)
    recon = A @ ols
    ss = float(t @ t) or 1.0
    r2 = 1.0 - float((t - recon) @ (t - recon)) / ss   # = ||proj||²/||t||² for OLS

    out = {"ols": ols, "r2": r2}
    if _scipy_nnls is not None:
        nn, _ = _scipy_nnls(A, t)
        recon_nn = A @ nn
        out["nnls"] = nn
        out["r2_nnls"] = 1.0 - float((t - recon_nn) @ (t - recon_nn)) / ss
    return out


def _topk_hit(loadings: np.ndarray, labels: list[str], predicted: list[str]) -> dict:
    """Do the predicted primitives occupy the top-|predicted| loadings (by magnitude)?"""
    order = [labels[i] for i in np.argsort(-np.abs(loadings))]
    k = len(predicted)
    top = order[:k]
    hit = len(set(top) & set(predicted))
    return {"predicted": predicted, "top": top, "precision_at_k": hit / k if k else None}


def reconstruction_report(
    clinical_bundle: dict | str,
    pc_bundle: dict | str,
    layer: int | None = None,
    *,
    include_syndromes: bool = True,
    verbose: bool = True,
) -> dict:
    """Full Experiment-① readout. Accepts bundles or paths to pickles."""
    if isinstance(clinical_bundle, str):
        clinical_bundle = load_bundle(clinical_bundle)
    if isinstance(pc_bundle, str):
        pc_bundle = load_bundle(pc_bundle)

    layer = layer if layer is not None else pick_layer(clinical_bundle, pc_bundle)
    pc_dirs = _dirs(pc_bundle, layer)
    cl_dirs = _dirs(clinical_bundle, layer)
    pc_labels = list(pc_dirs)
    basis_M = _stack(pc_dirs, pc_labels, unit=True)            # [k, d]
    d = basis_M.shape[1]
    chance_r2 = len(pc_labels) / d                            # random target on random k-subspace

    geom = compare_geometry(clinical_bundle, pc_bundle, layer)

    atoms: dict[str, dict] = {}
    for atom, vec in cl_dirs.items():
        rec = reconstruct(vec, basis_M)
        load = rec.get("nnls", rec["ols"])
        loadings = {pc_labels[i]: float(load[i]) for i in range(len(pc_labels))}
        predicted = CLINICAL_AS_PC.get(atom, [])
        atoms[atom] = {
            "r2": rec["r2"],
            "r2_nnls": rec.get("r2_nnls"),
            "loadings": loadings,
            **_topk_hit(load, pc_labels, predicted),
        }

    syndromes: dict[str, dict] = {}
    if include_syndromes:
        for syn, atom_ids in SYNDROME_RECIPES.items():
            present = [a for a in atom_ids if a in cl_dirs]
            if not present:
                continue
            target = np.sum([cl_dirs[a] for a in present], axis=0)   # syndrome = sum of its atoms
            rec = reconstruct(target, basis_M)
            load = rec.get("nnls", rec["ols"])
            syndromes[syn] = {
                "r2": rec["r2"],
                "r2_nnls": rec.get("r2_nnls"),
                "loadings": {pc_labels[i]: float(load[i]) for i in range(len(pc_labels))},
                **_topk_hit(load, pc_labels, SYNDROME_AS_PC.get(syn, [])),
            }

    report = {
        "layer": layer,
        "d_model": d,
        "chance_r2": chance_r2,
        "pc_labels": pc_labels,
        "geometry": geom,
        "atoms": atoms,
        "syndromes": syndromes,
        "mean_atom_r2": float(np.mean([a["r2"] for a in atoms.values()])) if atoms else None,
        "mean_precision_at_k": (
            float(np.mean([a["precision_at_k"] for a in atoms.values()
                           if a["precision_at_k"] is not None])) if atoms else None
        ),
    }
    if verbose:
        _print_report(report)
    return report


def _print_report(r: dict) -> None:
    g = r["geometry"]
    print(f"=== Experiment ① — clinical atoms vs PC-primitive basis  (layer {r['layer']}, d={r['d_model']}) ===\n")
    print("GEOMETRY  (lower off-diag cos & higher PR-fraction = more orthogonal / fundamental)")
    print(f"  {'basis':10s} {'n':>3s} {'mean|cos|':>10s} {'max|cos|':>9s} {'PR':>6s} {'PR/n':>6s}")
    for name in ("clinical", "pc"):
        b = g[name]
        print(f"  {name:10s} {b['n']:>3d} {b['mean_abs_offdiag_cos']:>10.3f} "
              f"{b['max_abs_offdiag_cos']:>9.3f} {b['participation_ratio']:>6.2f} {b['pr_fraction']:>6.2f}")
    nnls = any(a.get("r2_nnls") is not None for a in r["atoms"].values())
    print(f"\nRECONSTRUCTION  (R² of clinical vector inside PC subspace; chance R² ≈ {r['chance_r2']:.2f})")
    hdr_r2 = "R²(nnls)" if nnls else "R²(ols)"
    print(f"  {'clinical atom':26s} {hdr_r2:>9s} {'P@k':>5s}   predicted → top-k loadings")
    for atom, a in r["atoms"].items():
        r2 = a["r2_nnls"] if nnls and a["r2_nnls"] is not None else a["r2"]
        pk = a["precision_at_k"]
        pk_s = f"{pk:.2f}" if pk is not None else "  - "
        print(f"  {atom:26s} {r2:>9.2f} {pk_s:>5s}   {a['predicted']} → {a['top']}")
    if r["syndromes"]:
        print(f"\nSYNDROMES (clinical-atom sum, reconstructed from PC basis)")
        for syn, s in r["syndromes"].items():
            r2 = s["r2_nnls"] if nnls and s["r2_nnls"] is not None else s["r2"]
            print(f"  {syn:12s} R²={r2:>5.2f}  P@k={s['precision_at_k']:.2f}   {s['predicted']} → {s['top']}")
    print(f"\nSUMMARY  mean atom R² = {r['mean_atom_r2']:.3f}   mean P@k = {r['mean_precision_at_k']:.3f}")
    print("Read: high R² ≫ chance ⇒ clinical atoms live inside the PC subspace (PC is the deeper basis).")
    print("      high P@k ⇒ the *specific* pre-registered decomposition holds, not just generic span.")
