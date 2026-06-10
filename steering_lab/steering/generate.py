"""Phase 1 — generate persona-conditioned completions with vLLM.

For every (mechanism, variant ∈ {on, baseline}, neutral prompt) we generate one (or more)
completions. The output JSONL is both:
  * the raw material for vector extraction (Phase 2), and
  * Tier-0 behavioral data you can eyeball/score to check the personas actually work.

vLLM is used only for fast generation — it does NOT expose hidden states, so extraction
(Phase 2) re-loads the model in plain HF transformers via repeng.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from .config import GenConfig
from .personas import NEUTRAL_PROMPTS, PERSONAS, Persona


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    """Render a system+user turn into the model's chat format, ending with the assistant cue."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # `enable_thinking=False` suppresses Gemma 4's thinking block — otherwise the chain-of-thought
    # would land in the generated completion and pollute the contrastive training data. Templates
    # that don't define the flag raise, so fall back to a plain render.
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _save_records(records: list[dict], cfg: GenConfig, out_path: str | None) -> None:
    """Write records as JSONL (first line = config header), making parent dirs as needed."""
    if not out_path:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_config": asdict(cfg)}) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records → {out_path}")


def _tasks(prompts: list[str], personas: list[Persona]):
    """Flatten into a deterministic list of generation tasks."""
    for p in personas:
        for variant in ("on", "baseline"):
            system = p.on if variant == "on" else p.baseline
            for prompt_idx, user in enumerate(prompts):
                yield {
                    "mechanism": p.id,
                    "variant": variant,
                    "prompt_idx": prompt_idx,
                    "prompt": user,
                    "system": system,
                }


def generate_dataset(
    cfg: GenConfig | None = None,
    *,
    prompts: list[str] | None = None,
    personas: list[Persona] | None = None,
    out_path: str | None = None,
    llm=None,
    tokenizer=None,
) -> list[dict]:
    """Run vLLM generation and return (and optionally save) a list of records.

    Each record: {mechanism, variant, prompt_idx, prompt, sample, completion}.
    Pass an existing `llm`/`tokenizer` to reuse a loaded engine; otherwise they're created here.
    """
    cfg = cfg or GenConfig()
    prompts = prompts if prompts is not None else NEUTRAL_PROMPTS
    personas = personas if personas is not None else PERSONAS

    from vllm import LLM, SamplingParams  # imported lazily (heavy, GPU-only)

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if llm is None:
        llm = LLM(
            model=cfg.model_name,
            dtype=cfg.dtype,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
        )

    tasks = list(_tasks(prompts, personas))
    prompt_strs = [build_chat_prompt(tokenizer, t["system"], t["prompt"]) for t in tasks]

    sampling = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
        n=cfg.n_samples,
        seed=cfg.seed,
    )
    outputs = llm.generate(prompt_strs, sampling)

    records: list[dict] = []
    for task, out in zip(tasks, outputs):
        for sample_idx, completion in enumerate(out.outputs):
            text = completion.text.strip()
            if not text:
                continue
            records.append({
                "mechanism": task["mechanism"],
                "variant": task["variant"],
                "prompt_idx": task["prompt_idx"],
                "prompt": task["prompt"],
                "sample": sample_idx,
                "completion": text,
            })

    _save_records(records, cfg, out_path)
    return records


def generate_dataset_hf(
    cfg: GenConfig | None = None,
    *,
    prompts: list[str] | None = None,
    personas: list[Persona] | None = None,
    out_path: str | None = None,
    model=None,
    tokenizer=None,
    batch_size: int = 8,
) -> list[dict]:
    """HF-transformers fallback for Phase 1 when vLLM can't load the model (e.g. a brand-new arch
    like Gemma 4's encoder-free multimodal checkpoint, unsupported in the installed vLLM).

    Slower than vLLM but uses the same robust loader as the rest of the lab, so it works wherever
    steering works. Generates one completion per task (vLLM's `n_samples>1` is not replicated here).
    Records use the same schema as `generate_dataset`, so Phase 2 is identical.
    """
    import torch

    cfg = cfg or GenConfig()
    prompts = prompts if prompts is not None else NEUTRAL_PROMPTS
    personas = personas if personas is not None else PERSONAS

    if model is None or tokenizer is None:
        from .steer import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(cfg.model_name, dtype=cfg.dtype)

    tasks = list(_tasks(prompts, personas))
    prompt_strs = [build_chat_prompt(tokenizer, t["system"], t["prompt"]) for t in tasks]

    gen_kwargs = dict(
        max_new_tokens=cfg.max_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if cfg.temperature and cfg.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)
    else:
        gen_kwargs.update(do_sample=False)
    torch.manual_seed(cfg.seed)

    records: list[dict] = []
    for i in range(0, len(tasks), batch_size):
        bt = tasks[i:i + batch_size]
        enc = tokenizer(prompt_strs[i:i + batch_size], return_tensors="pt",
                        padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        new = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(new, skip_special_tokens=True)
        for t, txt in zip(bt, texts):
            txt = txt.strip()
            if not txt:
                continue
            records.append({
                "mechanism": t["mechanism"], "variant": t["variant"],
                "prompt_idx": t["prompt_idx"], "prompt": t["prompt"],
                "sample": 0, "completion": txt,
            })
        print(f"  generated {min(i + batch_size, len(tasks))}/{len(tasks)} tasks", end="\r")
    print()

    _save_records(records, cfg, out_path)
    return records
