"""Read radiology reports (any language) with a local open-weights instruct LLM and get 12 finding labels.

The model answers with compact JSON ({"ACL":0,"MCL":1,...}). Because the value tokens are generated one by one, the
probability of "1" vs "0" at each value position is taken from the generation scores, which gives a soft label per finding
in a single pass. Nothing leaves the machine: the model runs locally (on the Kaggle GPU) and only labels are written out.
"""
import re
import time
import numpy as np
import torch
from pseudo_labels import LABELS

DEFS = {
    "ACL": "anterior cruciate ligament injury (tear, rupture, sprain)",
    "MCL": "medial collateral ligament injury (tear, rupture, sprain)",
    "Medial Meniscus": "medial meniscus tear",
    "Lateral Meniscus": "lateral meniscus tear",
    "Medial OA": "osteoarthritis of the medial tibiofemoral compartment (cartilage loss, degeneration, osteophytes)",
    "Lateral OA": "osteoarthritis of the lateral tibiofemoral compartment (cartilage loss, degeneration, osteophytes)",
    "PF OA": "patellofemoral osteoarthritis (patellofemoral cartilage loss, degeneration, chondromalacia)",
    "Effusion": "joint effusion / excess joint fluid",
    "Synovitis": "synovitis (inflammation or thickening of the joint lining)",
    "Baker's": "Baker's cyst (popliteal cyst)",
    "Contusion": "bone contusion / bone bruise / bone marrow edema",
    "Fracture": "fracture",
}
SYSTEM = ("You are a musculoskeletal radiologist. You read a knee MRI report, which may be written in any language, "
          "and decide for each finding whether the report states that it is PRESENT.")
PREFILL = '{"ACL":'


# Prompt variants. v0: the first binary prompt. v1: graded answer (0 absent / 1 possible or mild / 2 clearly present).
# v2: binary with detailed per-finding guidance (synonyms, implicit mentions, generic osteoarthritis statements).
VARIANT_LEVELS = {"v0": 2, "v1": 3, "v2": 2}
GUIDE_V2 = {
    "ACL": "ACL tear, rupture, partial tear, sprain, discontinuity or disruption of the fibres, graft tear",
    "MCL": "medial collateral ligament sprain (any grade), tear, rupture or periligamentous injury",
    "Medial Meniscus": "tear of the medial meniscus of any type (horizontal, radial, longitudinal, complex, bucket-handle, flap, root, displaced fragment). "
                       "Degenerative signal or grade 1-2 signal without a tear is 0",
    "Lateral Meniscus": "tear of the lateral meniscus of any type (horizontal, radial, longitudinal, complex, bucket-handle, flap, root, displaced fragment). "
                        "Degenerative signal or grade 1-2 signal without a tear is 0",
    "Medial OA": "osteoarthritis of the medial femorotibial compartment: cartilage loss, thinning, fissuring or defect, chondromalacia/chondropathy, "
                 "osteophytes, joint space narrowing, subchondral sclerosis/cysts, or the words osteoarthritis/arthrosis/degenerative change for that compartment. "
                 "Tricompartmental, diffuse or generalised osteoarthritis counts for all three compartments",
    "Lateral OA": "osteoarthritis of the lateral femorotibial compartment (same signs as for the medial compartment). "
                  "Tricompartmental, diffuse or generalised osteoarthritis counts for all three compartments",
    "PF OA": "patellofemoral osteoarthritis: patellar or trochlear cartilage loss, thinning, fissuring or defect, chondromalacia patellae, patellofemoral osteophytes "
             "or degenerative change. Tricompartmental, diffuse or generalised osteoarthritis counts too",
    "Effusion": "joint effusion or hemarthrosis, including small or mild effusion; 0 if the report says there is no fluid",
    "Synovitis": "synovitis, synovial thickening/proliferation/hypertrophy/enhancement, reactive synovial changes, or inflammatory change/oedema of Hoffa's (infrapatellar) fat pad",
    "Baker's": "Baker's cyst, popliteal cyst, distension of the semimembranosus-gastrocnemius bursa",
    "Contusion": "bone contusion, bone bruise, traumatic or stress-related bone marrow oedema (impaction pattern)",
    "Fracture": "any fracture including avulsion (e.g. Segond), osteochondral, stress and tibial plateau fractures; surgical microfracture is not a fracture",
}


def build_messages(report, max_chars=5000, variant="v0"):
    body = f"Report:\n{str(report)[:max_chars]}"
    if variant == "v2":
        items = "\n".join(f'- "{k}": {GUIDE_V2[k]}' for k in LABELS)
        user = (f"Findings and what counts as present:\n{items}\n\n"
                "Rules: answer 1 if the report states or clearly describes the finding (also when mild, small or partial); "
                "answer 0 if the report says it is absent or normal, or does not mention it. Be careful with negations in the report's "
                "language, and do not count items of a template that only lists normal structures. Reply with ONLY compact JSON containing "
                'exactly these 12 keys in this order, values 0 or 1, for example {"ACL":0,"MCL":0,"Medial Meniscus":1,...}.\n\n' + body)
    elif variant == "v1":
        items = "\n".join(f'- "{k}": {DEFS[k]}' for k in LABELS)
        user = (f"Findings:\n{items}\n\n"
                "For each finding answer with one digit: 0 = absent, normal or not mentioned; 1 = possible, suspected, equivocal or only minimal/mild; "
                "2 = clearly present. Be careful with negations in the report's language. Reply with ONLY compact JSON containing exactly these "
                '12 keys in this order, values 0, 1 or 2, for example {"ACL":0,"MCL":2,"Medial Meniscus":1,...}.\n\n' + body)
    else:
        items = "\n".join(f'- "{k}": {DEFS[k]}' for k in LABELS)
        user = (f"Findings:\n{items}\n\n"
                "Rules: answer 1 only if the report says the finding is present (also when it is mild, small or partial); "
                "answer 0 if the report says it is absent, normal, or does not mention it. Be careful with negations in the "
                "report's language. Reply with ONLY compact JSON containing exactly these 12 keys in this order, values 0 or 1, "
                'for example {"ACL":0,"MCL":0,"Medial Meniscus":1,...}.\n\n' + body)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _single_token_ids(tok, s):
    ids = set()
    for v in (s, " " + s):
        t = tok.encode(v, add_special_tokens=False)
        if len(t) == 1:
            ids.add(t[0])
    return sorted(ids)


def extract_levels(gen_ids, scores, tok, level_ids):
    """Distribution over the answer levels at each of the first 12 value tokens (digits) of the generated JSON.
    -> array (12, n_levels), or None if fewer than 12 values were generated."""
    rows = []
    for j, tid in enumerate(gen_ids):
        if j >= len(scores):
            break
        if tok.decode([tid]).strip() in tuple("0123456789"):
            lg = scores[j].float()
            per = torch.stack([torch.logsumexp(lg[ids], 0) for ids in level_ids])
            rows.append(torch.softmax(per, 0).detach().cpu().numpy())   # the scores live on the GPU
            if len(rows) == len(LABELS):
                return np.stack(rows).astype(np.float32)
    return None


def extract_probs(gen_ids, scores, tok, zero_ids, one_ids):
    """Binary special case: P(value = 1) per value token, or None."""
    lv = extract_levels(gen_ids, scores, tok, [zero_ids, one_ids])
    return None if lv is None else lv[:, 1]


def soft_from_levels(lv):
    """(..., n_levels) -> soft label in [0,1]: P(1) for binary answers; P(2) + 0.5 * P(1) for the 0/1/2 scale."""
    return lv[..., 1] if lv.shape[-1] == 2 else lv[..., 2] + 0.5 * lv[..., 1]


def parse_json_values(text, n_levels=2):
    """Hard answers from '{"ACL":0,"MCL":1,...' (order-independent, tolerant); missing keys -> 0. Hard = highest level."""
    found = dict(re.findall(r'"([^"]+)"\s*:\s*([0-9])', text))
    return np.array([int(int(found.get(k, 0)) >= n_levels - 1) for k in LABELS], np.int8), len(found)


def _truncate(tok, report, max_report_tokens):
    """Cut a report to at most max_report_tokens tokens (Greek / Turkish text takes many more tokens per character)."""
    r = str(report)
    ids = tok(r, add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:max_report_tokens]) if len(ids) > max_report_tokens else r


@torch.no_grad()
def label_reports(model, tok, reports, bs=8, max_new_tokens=100, max_chars=5000, log_every=10,
                  max_report_tokens=1200, max_batch_tokens=5000, variant="v0", return_levels=False):
    """-> (hard (n,12) int8, soft (n,12) float32, n_parsed (n,)) [+ levels (n,12,n_levels) if return_levels].
    Returns in the order of `reports`. Batches are built so that batch size x longest prompt <= max_batch_tokens (the attention
    matrix grows with the square of the length); a batch that still runs out of memory is split."""
    n_levels = VARIANT_LEVELS[variant]
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    level_ids = [_single_token_ids(tok, str(i)) for i in range(n_levels)]
    prompts = [tok.apply_chat_template(build_messages(_truncate(tok, r, max_report_tokens), max_chars, variant),
                                       tokenize=False, add_generation_prompt=True) + PREFILL for r in reports]
    lens = [len(x) for x in tok(prompts, add_special_tokens=False)["input_ids"]]
    order = np.argsort(lens)                                   # similar lengths together: less padding
    batches, cur = [], []
    for k in order:                                            # ascending: the newest item is the longest one
        if cur and (len(cur) >= bs or (len(cur) + 1) * lens[k] > max_batch_tokens):
            batches.append(cur); cur = []
        cur.append(int(k))
    if cur:
        batches.append(cur)
    hard = np.zeros((len(prompts), len(LABELS)), np.int8)
    levels = np.zeros((len(prompts), len(LABELS), n_levels), np.float32)
    levels[..., 0] = 1.0
    n_parsed = np.zeros(len(prompts), np.int16)

    def run(idx):
        enc = tok([prompts[k] for k in idx], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, output_scores=True,
                             return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        gen = out.sequences[:, enc["input_ids"].shape[1]:]
        for r, k in enumerate(idx):
            ids = gen[r].tolist()
            text = PREFILL + tok.decode(ids, skip_special_tokens=True)
            hard[k], n_parsed[k] = parse_json_values(text, n_levels)
            lv = extract_levels(ids, [s[r] for s in out.scores], tok, level_ids)
            if lv is None:                                                    # fall back to the parsed answer
                lv = np.zeros((len(LABELS), n_levels), np.float32)
                lv[np.arange(len(LABELS)), hard[k].astype(int) * (n_levels - 1)] = 1.0
            levels[k] = lv

    def run_safe(idx):
        try:
            run(idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(idx) > 1:
                h = len(idx) // 2
                run_safe(idx[:h]); run_safe(idx[h:])
            else:
                print(f"out of memory on a single report (prompt tokens {lens[idx[0]]}); labels left at 0", flush=True)

    t0, done = time.time(), 0
    for b, idx in enumerate(batches):
        run_safe(idx)
        done += len(idx)
        if b % log_every == 0:
            print(f"labeled {done}/{len(prompts)}  {time.time() - t0:.0f}s", flush=True)
    soft = soft_from_levels(levels).astype(np.float32)
    return (hard, soft, n_parsed, levels) if return_levels else (hard, soft, n_parsed)
