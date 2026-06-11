"""Phase 2 — build contrastive pairs from generations and train repeng control vectors.

We turn each ON/baseline completion pair into several "model is mid-generation" snapshots by
truncating the completion at multiple token counts, then build repeng `DatasetEntry(positive,
negative)` pairs where:
    positive = <ON system> + <neutral prompt> + <ON completion truncated at k tokens>
    negative = <baseline system> + <same prompt> + <baseline completion truncated at k tokens>

repeng reads the **last token** hidden state of each string and runs PCA(1) on the
(positive − negative) differences per layer. Pairing pos/neg on the *same* neutral prompt makes
the shared prompt content cancel, leaving the mechanism axis.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np

from .config import ExtractConfig
from .personas import PERSONA_BY_ID


def load_records(path: str) -> list[dict]:
    """Load generation JSONL, skipping the leading `_config` line if present."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_config" in obj and "mechanism" not in obj:
                continue
            records.append(obj)
    return records


def _chat_prefix(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # `enable_thinking=False`: keep the Phase-2 prefix free of any thinking block so the read-back
    # hidden states align with the Phase-1 (thinking-suppressed) completion. Templates lacking the
    # flag raise — fall back to a plain render.
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _truncation_points(n_tokens: int, cfg: ExtractConfig) -> list[int]:
    """Token counts at which to snapshot a completion of length `n_tokens`."""
    if n_tokens < 1:
        return []
    pts = list(range(cfg.trunc_min_tokens, n_tokens + 1, cfg.trunc_stride))
    if not pts:
        pts = [n_tokens]
    if len(pts) > cfg.trunc_max_points:
        idx = np.linspace(0, len(pts) - 1, cfg.trunc_max_points).round().astype(int)
        pts = [pts[i] for i in sorted(set(idx.tolist()))]
    return pts


def build_pairs(
    records: list[dict], tokenizer, cfg: ExtractConfig | None = None,
    persona_by_id: dict | None = None,
) -> dict:
    """Group records by mechanism and build repeng DatasetEntry lists.

    Returns {mechanism_id: list[DatasetEntry]}.

    `persona_by_id` selects the persona registry to read ON/baseline system prompts from; defaults to
    the clinical set (`personas.PERSONA_BY_ID`). Pass `personas_pc.PC_PERSONA_BY_ID` to extract the
    predictive-coding primitive vectors instead.
    """
    from repeng import DatasetEntry  # lazy

    cfg = cfg or ExtractConfig()
    persona_by_id = persona_by_id if persona_by_id is not None else PERSONA_BY_ID

    # index: mechanism -> (prompt_idx, sample) -> variant -> completion
    index: dict[str, dict[tuple, dict[str, str]]] = {}
    prompt_by_idx: dict[int, str] = {}
    for r in records:
        m = r["mechanism"]
        key = (r["prompt_idx"], r.get("sample", 0))
        index.setdefault(m, {}).setdefault(key, {})[r["variant"]] = r["completion"]
        prompt_by_idx.setdefault(r["prompt_idx"], r["prompt"])

    pairs: dict[str, list] = {}
    for mech, by_key in index.items():
        persona = persona_by_id[mech]
        on_prefix_cache: dict[int, str] = {}
        base_prefix_cache: dict[int, str] = {}
        entries = []
        for (prompt_idx, _sample), variants in by_key.items():
            if "on" not in variants or "baseline" not in variants:
                continue
            user = prompt_by_idx[prompt_idx]
            if prompt_idx not in on_prefix_cache:
                on_prefix_cache[prompt_idx] = _chat_prefix(tokenizer, persona.on, user)
                base_prefix_cache[prompt_idx] = _chat_prefix(tokenizer, persona.baseline, user)
            on_toks = tokenizer(variants["on"], add_special_tokens=False)["input_ids"]
            base_toks = tokenizer(variants["baseline"], add_special_tokens=False)["input_ids"]
            n = min(len(on_toks), len(base_toks))
            for k in _truncation_points(n, cfg):
                pos = on_prefix_cache[prompt_idx] + tokenizer.decode(on_toks[:k])
                neg = base_prefix_cache[prompt_idx] + tokenizer.decode(base_toks[:k])
                entries.append(DatasetEntry(positive=pos, negative=neg))
        pairs[mech] = entries
    return pairs


def extract_vectors(model, tokenizer, pairs: dict, cfg: ExtractConfig | None = None) -> dict:
    """Train one repeng ControlVector per mechanism. Returns {mechanism_id: ControlVector}."""
    from repeng import ControlVector  # lazy

    cfg = cfg or ExtractConfig()
    hidden_layers = cfg.hidden_layers
    if not hidden_layers:
        # repeng's own default reads model.config.num_hidden_layers, which doesn't exist on
        # multimodal wrapper configs (Qwen3.5 keeps it in text_config; same for Gemma 4) — build
        # the same "all layers but the first" range from the real decoder stack instead.
        from .steer import decoder_layers
        hidden_layers = list(range(-1, -len(decoder_layers(model)), -1))
    vectors = {}
    for mech, entries in pairs.items():
        if not entries:
            print(f"[skip] {mech}: no pairs")
            continue
        print(f"[train] {mech}: {len(entries)} pairs")
        vectors[mech] = ControlVector.train(
            model,
            tokenizer,
            entries,
            method=cfg.method,
            batch_size=cfg.batch_size,
            hidden_layers=hidden_layers,
        )
    return vectors


def save_bundle(vectors: dict, path: str, *, model_name: str, cfg: ExtractConfig,
                pairs: dict | None = None, meta_path: str | None = None) -> None:
    """Save vectors as a plain-dict pickle (loadable without repeng) + a metadata.json."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model_type = next(iter(vectors.values())).model_type if vectors else None
    bundle = {
        "model_name": model_name,
        "model_type": model_type,
        "method": cfg.method,
        "hidden_layers": cfg.hidden_layers,
        "mechanisms": list(vectors.keys()),
        "pair_counts": {m: len(p) for m, p in (pairs or {}).items()},
        # decouple from repeng: store raw direction arrays keyed by int layer
        "vectors": {
            m: {int(l): np.asarray(d, dtype=np.float32) for l, d in v.directions.items()}
            for m, v in vectors.items()
        },
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved {len(vectors)} vectors → {path}")

    meta_path = meta_path or (os.path.splitext(path)[0].replace("control_vectors", "metadata") + ".json")
    meta = {k: v for k, v in bundle.items() if k != "vectors"}
    meta["layers_per_vector"] = {m: sorted(d.keys()) for m, d in bundle["vectors"].items()}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata → {meta_path}")


def load_bundle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def bundle_to_control_vectors(bundle: dict) -> dict:
    """Reconstruct repeng ControlVector objects from a saved bundle (needs repeng installed)."""
    from repeng import ControlVector
    return {
        m: ControlVector(model_type=bundle["model_type"], directions=dict(dirs))
        for m, dirs in bundle["vectors"].items()
    }


def cosine_matrix(bundle_or_vectors, layer: int):
    """Cosine-similarity matrix between mechanism directions at a given layer.

    Accepts a saved bundle (dict) or a {mech: ControlVector} dict.
    Returns (labels, matrix) where matrix[i, j] = cos(v_i, v_j).
    """
    if "vectors" in bundle_or_vectors and isinstance(next(iter(bundle_or_vectors["vectors"].values())), dict):
        dirs = {m: d.get(layer) for m, d in bundle_or_vectors["vectors"].items()}
    else:  # {mech: ControlVector}
        dirs = {m: v.directions.get(layer) for m, v in bundle_or_vectors.items()}
    labels = [m for m, d in dirs.items() if d is not None]
    mat = np.zeros((len(labels), len(labels)), dtype=np.float32)
    vecs = [np.asarray(dirs[m], dtype=np.float32) for m in labels]
    for i, a in enumerate(vecs):
        for j, b in enumerate(vecs):
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
            mat[i, j] = float(a @ b) / denom
    return labels, mat
