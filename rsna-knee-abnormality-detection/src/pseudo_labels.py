"""Derive weak 12-label targets from free-text radiology reports (English keywords only)."""
import re
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

RULES = {
    "ACL": r"\bACL\b|anterior cruciate",
    "MCL": r"\bMCL\b|medial collateral",
    "Medial Meniscus": r"medial meniscus",
    "Lateral Meniscus": r"lateral meniscus",
    "Medial OA": r"medial (tibiofemoral|compartment).{0,60}(osteoarth|degenerat|cartilage (loss|thinning))",
    "Lateral OA": r"lateral (tibiofemoral|compartment).{0,60}(osteoarth|degenerat|cartilage (loss|thinning))",
    "PF OA": r"patellofemoral.{0,60}(osteoarth|degenerat|cartilage (loss|thinning))",
    "Effusion": r"effusion",
    "Synovitis": r"synovitis",
    "Baker's": r"baker'?s?\s+cyst",
    "Contusion": r"contusion|bone bruise|marrow edema",
    "Fracture": r"fracture",
}
NEG = re.compile(r"\b(no|without|negative for|absence of)\b[^.;]{0,40}$", re.I)
INJURY = {"ACL": r"tear|rupture|sprain|injur", "MCL": r"tear|sprain|injur",
          "Medial Meniscus": r"tear", "Lateral Meniscus": r"tear"}


def label_report(text: str) -> dict:
    text = text or ""
    out = {}
    for name, pat in RULES.items():
        hit = 0
        for m in re.finditer(pat, text, re.I):
            start = max(0, text.rfind(".", 0, m.start()) + 1)
            if NEG.search(text[start:m.start()]):
                continue
            if name in INJURY:
                end = text.find(".", m.end())
                sent = text[start: end if end != -1 else len(text)]
                if not re.search(INJURY[name], sent, re.I):
                    continue
            hit = 1
            break
        out[name] = hit
    return out


def add_pseudo_labels(train: pd.DataFrame) -> pd.DataFrame:
    """Keep provided labels where present; fill the rest from the report."""
    pl = pd.DataFrame([label_report(r) for r in train["Report"]], index=train.index)
    out = train.copy()
    for c in LABELS:
        out[c] = out[c].where(out[c].notna(), pl[c]) if c in out else pl[c]
    return out


if __name__ == "__main__":
    print(label_report("Complete ACL tear. Small joint effusion. No fracture. Baker's cyst."))
