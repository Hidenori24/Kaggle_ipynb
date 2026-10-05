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


def build_messages(report, max_chars=5000):
    items = "\n".join(f'- "{k}": {DEFS[k]}' for k in LABELS)
    user = (f"Findings:\n{items}\n\n"
            "Rules: answer 1 only if the report says the finding is present (also when it is mild, small or partial); "
            "answer 0 if the report says it is absent, normal, or does not mention it. Be careful with negations in the "
            "report's language. Reply with ONLY compact JSON containing exactly these 12 keys in this order, values 0 or 1, "
            'for example {"ACL":0,"MCL":0,"Medial Meniscus":1,...}.\n\n'
            f"Report:\n{str(report)[:max_chars]}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _single_token_ids(tok, s):
    ids = set()
    for v in (s, " " + s):
        t = tok.encode(v, add_special_tokens=False)
        if len(t) == 1:
            ids.add(t[0])
    return sorted(ids)


def extract_probs(gen_ids, scores, tok, zero_ids, one_ids):
    """P(value = 1) at each of the first 12 value tokens ('0' / '1') of the generated JSON, or None if fewer were generated."""
    probs = []
    for j, tid in enumerate(gen_ids):
        if j >= len(scores):
            break
        if tok.decode([tid]).strip() in ("0", "1"):
            lg = scores[j].float()
            two = torch.stack([torch.logsumexp(lg[zero_ids], 0), torch.logsumexp(lg[one_ids], 0)])
            probs.append(float(torch.softmax(two, 0)[1]))
            if len(probs) == len(LABELS):
                return np.array(probs, np.float32)
    return None


def parse_json_values(text):
    """Hard 0/1 values from '{"ACL":0,"MCL":1,...' (order-independent, tolerant); missing keys -> 0."""
    found = dict(re.findall(r'"([^"]+)"\s*:\s*([01])', text))
    return np.array([int(found.get(k, 0)) for k in LABELS], np.int8), len(found)


def _truncate(tok, report, max_report_tokens):
    """Cut a report to at most max_report_tokens tokens (Greek / Turkish text takes many more tokens per character)."""
    r = str(report)
    ids = tok(r, add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:max_report_tokens]) if len(ids) > max_report_tokens else r


@torch.no_grad()
def label_reports(model, tok, reports, bs=8, max_new_tokens=100, max_chars=5000, log_every=10,
                  max_report_tokens=1200, max_batch_tokens=5000):
    """-> (hard (n,12) int8, prob (n,12) float32, n_parsed (n,)). Returns in the order of `reports`.
    Batches are built so that batch size x longest prompt <= max_batch_tokens (the attention matrix grows with the square of the
    length: bs=8 with long Greek/Turkish reports ran out of GPU memory); a batch that still runs out of memory is split."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    zero_ids, one_ids = _single_token_ids(tok, "0"), _single_token_ids(tok, "1")
    prompts = [tok.apply_chat_template(build_messages(_truncate(tok, r, max_report_tokens), max_chars),
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
    prob = np.zeros((len(prompts), len(LABELS)), np.float32)
    n_parsed = np.zeros(len(prompts), np.int16)

    def run(idx):
        enc = tok([prompts[k] for k in idx], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, output_scores=True,
                             return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        gen = out.sequences[:, enc["input_ids"].shape[1]:]
        for r, k in enumerate(idx):
            ids = gen[r].tolist()
            text = PREFILL + tok.decode(ids, skip_special_tokens=True)
            hard[k], n_parsed[k] = parse_json_values(text)
            p = extract_probs(ids, [s[r] for s in out.scores], tok, zero_ids, one_ids)
            prob[k] = p if p is not None else hard[k].astype(np.float32)      # fall back to the parsed hard answer

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
    return hard, prob, n_parsed
