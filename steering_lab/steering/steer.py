"""Phase 3 — steer with the trained control vectors and run composition tests.

Wraps a HF model in repeng's ControlModel (applied to a layer band), adds a mechanism vector to
the residual stream at generation time, and sweeps the coefficient. Compositions (syndromes,
compounds) are just sums of mechanism vectors — repeng ControlVector supports + and *.
"""

from __future__ import annotations

import json
import re

from .extract import bundle_to_control_vectors, load_bundle  # re-exported for convenience

__all__ = [
    "load_bundle", "bundle_to_control_vectors",
    "make_control_model", "generate_steered", "sweep", "compose", "judge_text",
]


def make_control_model(model, layer_ids):
    """Wrap `model` in a repeng ControlModel over `layer_ids`. NOTE: this mutates `model`."""
    from repeng import ControlModel
    return ControlModel(model, list(layer_ids))


def _chat(tokenizer, prompt: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_steered(
    cmodel, tokenizer, prompt: str, vector=None, coeff: float = 0.0, *,
    system: str | None = None, normalize: bool = False,
    max_new_tokens: int = 200, do_sample: bool = False,
    temperature: float = 0.7, repetition_penalty: float = 1.1,
) -> str:
    """Generate a completion with an optional control vector applied. Resets control afterward.

    coeff=0 (or vector=None) => clean baseline generation.
    """
    import torch

    text = _chat(tokenizer, prompt, system)
    enc = tokenizer(text, return_tensors="pt").to(cmodel.device)

    cmodel.reset()
    if vector is not None and coeff != 0.0:
        cmodel.set_control(vector, coeff, normalize=normalize)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        out = cmodel.generate(**enc, **gen_kwargs)
    cmodel.reset()

    new_tokens = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def sweep(cmodel, tokenizer, vector, prompts: list[str], coeffs: list[float], **gen) -> list[dict]:
    """Generate over prompts × coeffs. Returns rows {prompt, coeff, text}."""
    rows = []
    for p in prompts:
        for c in coeffs:
            rows.append({
                "prompt": p,
                "coeff": c,
                "text": generate_steered(cmodel, tokenizer, p, vector, c, **gen),
            })
    return rows


def compose(vectors: dict, atom_ids: list[str], coeffs: list[float] | None = None):
    """Sum mechanism ControlVectors (optionally weighted) into one composed vector."""
    coeffs = coeffs or [1.0] * len(atom_ids)
    combined = None
    for atom, w in zip(atom_ids, coeffs):
        v = vectors[atom] * w
        combined = v if combined is None else combined + v
    return combined


def judge_text(
    text: str, mechanism_desc: str, *,
    model: str = "anthropic/claude-sonnet-4.6",
    base_url: str = "https://openrouter.ai/api/v1",
) -> dict:
    """Optional LLM judge (via OpenRouter): rate mechanism presence and coherence, 0-10 each.

    Requires OPENROUTER_API_KEY in the environment and the `openai` package.
    """
    import os
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=os.environ["OPENROUTER_API_KEY"])
    sys = (
        "You rate a text on two 0-10 scales and reply with ONLY compact JSON "
        '{"presence": <int 0-10>, "coherence": <int 0-10>}. '
        "presence = how strongly the text exhibits the described cognitive pattern (10 = textbook, "
        "0 = absent). coherence = how fluent and well-formed it is (10 = perfectly natural, "
        "0 = broken/gibberish/repetitive)."
    )
    usr = f"Cognitive pattern:\n{mechanism_desc}\n\nText:\n{text}"
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        max_tokens=60, temperature=0,
    )
    raw = r.choices[0].message.content or ""
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except Exception:
        d = {}
    return {"presence": d.get("presence"), "coherence": d.get("coherence"), "raw": raw}
