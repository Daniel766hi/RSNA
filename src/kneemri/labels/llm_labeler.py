"""Offline, open-weight LLM report labeler.

Runs inside a Kaggle notebook without internet, with the model weights attached as a dataset
(e.g. an instruction-tuned Qwen / Llama / Gemma checkpoint; vLLM on 2xT4 with a 7-14B AWQ model
labels ~4.4k reports in well under an hour). Any backend that maps a list of chat prompts to
strings can be plugged in via ``generate_fn``.

Output schema (per finding): ``state`` in {present, absent, uncertain, not_mentioned} plus a
``confidence`` in [0, 1]. This follows CheXpert/CheXbert: *not mentioned* is kept apart from
*absent* (docs/research_review.md §3.3). That matters here because radiologists often leave
synovitis, OA and contusion out of reports even when the image shows them. Downstream,
:func:`to_soft_labels` turns silence into a prior with low weight, and ``refine.py`` lets the
image model fill it in.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from ..constants import ID_COL, TARGETS

STATES = ("present", "absent", "uncertain", "not_mentioned")

# Short, language-agnostic guidance. The reports are in 9+ languages; modern instruction models
# read them directly, so the prompt asks the model to reason in the report's language and
# answer in JSON.
FINDING_GUIDE: dict[str, str] = {
    "ACL": "anterior cruciate ligament injury of any grade: tear, partial tear, rupture, sprain, "
           "graft tear/insufficiency. Mucoid degeneration alone is NOT an injury.",
    "MCL": "medial collateral ligament injury of any grade: sprain, partial or complete tear, periligamentous oedema from injury.",
    "Medial Meniscus": "tear of the medial meniscus (any pattern: horizontal, radial, bucket-handle, root, complex); "
                       "intrasubstance degeneration without extension to a surface is NOT a tear.",
    "Lateral Meniscus": "tear of the lateral meniscus (same rules as medial).",
    "Medial OA": "osteoarthritis of the medial tibiofemoral compartment: cartilage loss/thinning, osteophytes, "
                 "subchondral sclerosis or cysts, joint-space narrowing in that compartment.",
    "Lateral OA": "osteoarthritis of the lateral tibiofemoral compartment (same criteria).",
    "PF OA": "patellofemoral osteoarthritis / chondropathy with degenerative change of patella or trochlea cartilage.",
    "Effusion": "joint effusion: more than physiological intra-articular fluid (small, moderate or large).",
    "Synovitis": "synovitis / synovial thickening / Hoffa (infrapatellar fat pad) synovitis or effusion-synovitis.",
    "Baker's": "Baker's (popliteal) cyst of any size, including ruptured.",
    "Contusion": "bone contusion / bone bruise / trabecular microfracture / post-traumatic bone marrow oedema.",
    "Fracture": "fracture of any knee bone: impaction, avulsion (e.g. Segond), osteochondral, stress or insufficiency fracture.",
}

SYSTEM_PROMPT = (
    "You are a musculoskeletal radiologist extracting structured labels from a knee MRI report. "
    "The report may be in any language. For each finding decide its state:\n"
    "- present: the report affirms it (any grade),\n"
    "- absent: the report explicitly negates it or describes the structure as normal/intact,\n"
    "- uncertain: hedged (possible, cannot exclude, suspicious for),\n"
    "- not_mentioned: the report does not address it.\n"
    "Give a confidence in [0,1] for your state. Answer with JSON only."
)


def build_messages(report: str) -> list[dict]:
    guide = "\n".join(f'- "{k}": {v}' for k, v in FINDING_GUIDE.items())
    schema = "{" + ", ".join(f'"{k}": {{"state": "...", "confidence": 0.0}}' for k in TARGETS) + "}"
    user = (
        f"Findings:\n{guide}\n\nReport:\n\"\"\"\n{report.strip()}\n\"\"\"\n\n"
        f"Return exactly this JSON object with all 12 keys:\n{schema}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_response(text: str) -> dict[str, tuple[str, float]]:
    """Parse the model's JSON (tolerating code fences, prose around it and missing keys)."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    data = {}
    if m:
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            data = {}
    out = {}
    lowered = {re.sub(r"[^a-z]", "", str(k).lower()): v for k, v in data.items()}
    for t in TARGETS:
        v = lowered.get(re.sub(r"[^a-z]", "", t.lower()), {})
        state = str(v.get("state", "not_mentioned")).strip().lower().replace(" ", "_") if isinstance(v, dict) else "not_mentioned"
        if state not in STATES:
            state = "not_mentioned"
        try:
            conf = float(v.get("confidence", 0.5)) if isinstance(v, dict) else 0.5
        except (TypeError, ValueError):
            conf = 0.5
        out[t] = (state, float(np.clip(conf, 0.0, 1.0)))
    return out


# (label, sample weight, explicitness) per state; "present"/"absent" scale with confidence.
DEFAULT_PRIORS = {t: 0.1 for t in TARGETS}


def to_soft_labels(
    parsed: dict[str, tuple[str, float]], priors: dict[str, float] | None = None, silent_weight: float = 0.3
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Map states to (soft label, sample weight, explicitness) per finding."""
    priors = priors or DEFAULT_PRIORS
    y, w, e = {}, {}, {}
    for t, (state, conf) in parsed.items():
        if state == "present":
            y[t], w[t], e[t] = 0.5 + 0.5 * conf, 1.0, 1.0
        elif state == "absent":
            y[t], w[t], e[t] = 0.5 - 0.5 * conf, 1.0, 1.0
        elif state == "uncertain":
            y[t], w[t], e[t] = 0.5, 0.6, 0.5
        else:
            y[t], w[t], e[t] = priors.get(t, 0.1), silent_weight, 0.0
    return y, w, e


def label_reports(
    df: pd.DataFrame,
    generate_fn: Callable[[list[list[dict]]], list[str]],
    batch_size: int = 64,
    report_col: str = "Report",
) -> pd.DataFrame:
    """Label every report; returns one row per study with ``<target>`` (soft label),
    ``<target>__w`` (weight), ``<target>__e`` (explicitness) and ``<target>__state``."""
    rows = []
    records = df[[ID_COL, report_col]].fillna("").to_dict("records")
    for i in range(0, len(records), batch_size):
        chunk = records[i: i + batch_size]
        texts = generate_fn([build_messages(r[report_col]) for r in chunk])
        for r, txt in zip(chunk, texts):
            parsed = parse_response(txt)
            y, w, e = to_soft_labels(parsed)
            row = {ID_COL: r[ID_COL]}
            for t in TARGETS:
                row[t], row[f"{t}__w"], row[f"{t}__e"], row[f"{t}__state"] = y[t], w[t], e[t], parsed[t][0]
            rows.append(row)
    return pd.DataFrame(rows)


def vllm_generate_fn(model_path: str, max_tokens: int = 700, **llm_kwargs) -> Callable[[list[list[dict]]], list[str]]:
    """Build a batched offline generator with vLLM (install from a wheel dataset on Kaggle)."""
    from vllm import LLM, SamplingParams  # noqa: PLC0415 - optional dependency

    llm = LLM(model=model_path, **{"tensor_parallel_size": 2, "max_model_len": 4096, "dtype": "half", **llm_kwargs})
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def fn(batch: list[list[dict]]) -> list[str]:
        outs = llm.chat(batch, params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    return fn


def transformers_generate_fn(model_path: str, max_new_tokens: int = 700) -> Callable[[list[list[dict]]], list[str]]:
    """Slower fallback with plain transformers (one prompt at a time)."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

    def fn(batch: Iterable[list[dict]]) -> list[str]:
        res = []
        for msgs in batch:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
            res.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        return res

    return fn
