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
    "load_model_and_tokenizer", "decoder_layers",
    "make_control_model", "generate_steered", "sweep", "compose", "judge_text",
]


def decoder_layers(model):
    """The text decoder `ModuleList` for `model`, handling multimodal wrappers (e.g. Gemma 3 12B).

    Picks the language-model decoder stack, never the vision tower. Use `len(decoder_layers(model))`
    instead of `config.num_hidden_layers` so layer counts are correct inside multimodal models too.
    """
    import torch

    cands = [
        (k, v) for k, v in model.named_modules()
        if isinstance(v, torch.nn.ModuleList) and k.endswith("layers") and "vision" not in k
    ]
    if not cands:
        raise ValueError(f"could not locate a decoder layer list in {type(model).__name__}")
    for suffix in ("language_model.layers", "model.layers"):     # prefer the text decoder
        for k, v in cands:
            if k.endswith(suffix):
                return v
    return max(cands, key=lambda kv: len(kv[1]))[1]              # else the longest stack


def _load_tokenizer(model_name, hf_token):
    """A real tokenizer (with a chat template) for `model_name`, even for multimodal repos.

    Multimodal models (Gemma 4 / Gemma 3) expose tokenization + the chat template through an
    `AutoProcessor`; we unwrap its `.tokenizer` and, if needed, copy the processor's chat template
    onto it, so the rest of the codebase can keep calling plain-tokenizer APIs.
    """
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if getattr(tok, "chat_template", None):
            return tok
    except Exception:
        tok = None

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_name, token=hf_token)
    inner = getattr(proc, "tokenizer", None) or tok
    if inner is None:
        raise ValueError(f"could not obtain a tokenizer for {model_name!r}")
    if not getattr(inner, "chat_template", None) and getattr(proc, "chat_template", None):
        inner.chat_template = proc.chat_template
    return inner


def _model_loader_classes(multimodal_first: bool):
    """Auto classes to try, newest-multimodal-aware and tolerant of older transformers."""
    import transformers
    names = (
        ["AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"]
        if multimodal_first else
        ["AutoModelForCausalLM", "AutoModelForMultimodalLM", "AutoModelForImageTextToText"]
    )
    return [getattr(transformers, n) for n in names if hasattr(transformers, n)]


def load_model_and_tokenizer(
    model_name, *, load_in_4bit=False, dtype="bfloat16", device_map="auto", hf_token=None,
):
    """Load an HF model + tokenizer ready for repeng steering — big-model and Gemma-4/3 aware.

    - `device_map="auto"`; full precision by default (`dtype="bfloat16"`). `load_in_4bit=True` is
      available for small GPUs but is OFF by default — we run 12B in bf16.
    - Multimodal checkpoints (Gemma 4 12B is encoder-free unified; Gemma 3 4b/12b/27b) load via
      `AutoModelForMultimodalLM` / `AutoModelForImageTextToText`. Encoder-free Gemma 4 runs text
      through the same decoder and exposes `.logits` + `.hidden_states` for text-only input, so
      steering and the trajectory/basin reads work unchanged.
    - Registers `model.repeng_layers` so repeng's `ControlModel` finds the text decoder even when it
      is nested inside a multimodal wrapper.
    """
    import torch

    tok = _load_tokenizer(model_name, hf_token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                # repeng's control masking assumes left padding

    td = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    kw = dict(device_map=device_map, token=hf_token)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=td, bnb_4bit_use_double_quant=True,
        )
    else:
        kw["torch_dtype"] = td

    # gemma-4 (all sizes, encoder-free multimodal) and gemma-3 4b/12b/27b need a multimodal class.
    name_l = model_name.lower()
    multimodal_first = "gemma-4" in name_l or ("gemma-3" in name_l and "-1b" not in name_l)
    model, last_err = None, None
    for loader in _model_loader_classes(multimodal_first):
        try:
            model = loader.from_pretrained(model_name, **kw)
            break
        except Exception as e:                # wrong auto-class for this checkpoint — try the next
            last_err = e
    if model is None:
        raise last_err

    model.repeng_layers = decoder_layers(model)   # explicit override for repeng's layer finder
    return model, tok


def make_control_model(model, layer_ids):
    """Wrap `model` in a repeng ControlModel over `layer_ids`. NOTE: this mutates `model`."""
    from repeng import ControlModel
    return ControlModel(model, list(layer_ids))


def _chat(tokenizer, prompt: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    # `enable_thinking=False` suppresses Gemma 4's thinking block (it would pollute the trajectory /
    # surprisal reads); templates that don't define the flag simply ignore it.
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


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
