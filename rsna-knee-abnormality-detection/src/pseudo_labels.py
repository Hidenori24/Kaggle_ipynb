"""Derive weak 12-label targets from free-text radiology reports (English keywords; other languages are in multilingual_labels.py)."""
import re
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

_DEG = (r"osteoarth|arthrosis|degenerat|chondromalacia|chondropath|cartilage (loss|thinning|defect|damage|wear|fissur)|"
        r"(full|partial)[- ]thickness (cartilage|chondral)|chondral (loss|defect|wear|fissur)|osteophyt|"
        r"joint space (narrowing|loss)|eburnation|subchondral (sclerosis|cyst)")
RULES = {
    "ACL": r"\bACL\b|anterior cruciate",
    "MCL": r"\bMCL\b|medial collateral|tibial collateral",
    "Medial Meniscus": r"medial menisc",
    "Lateral Meniscus": r"lateral menisc",
    "Medial OA": rf"medial (tibiofemoral|femorotibial|compartment|femoral condyle|tibial plateau)[^.;]{{0,80}}({_DEG})",
    "Lateral OA": rf"lateral (tibiofemoral|femorotibial|compartment|femoral condyle|tibial plateau)[^.;]{{0,80}}({_DEG})",
    "PF OA": rf"(patellofemoral|femoropatellar|retropatellar|patellar|trochlea\w*)[^.;]{{0,80}}({_DEG})",
    "Effusion": r"effusion|hemarthrosis|hydrops",
    "Synovitis": r"synovitis|synovial (thickening|hypertrophy|proliferation|inflammation)",
    "Baker's": r"baker'?s?\s+cyst|popliteal (fossa )?cyst",
    "Contusion": r"contusion|bone bruise|marrow (edema|oedema)|bone marrow lesion|subchondral (edema|oedema)",
    "Fracture": r"(?<!micro)fracture|avulsion|segond",
}
# negation words before the finding (same sentence) or confirming absence right after it
NEG = re.compile(r"\b(no|without|negative for|absence of|free of|not|rules? out|ruled out|exclud\w*)\b[^.;]{0,50}$", re.I)
NEG_POST = re.compile(r"^[^.;]{0,30}\b(not (seen|identified|demonstrated|evident|visuali[sz]ed|present|appreciated)|"
                      r"is absent|are absent|ruled out|excluded)\b", re.I)
INJURY = {
    "ACL": r"tear|torn|ruptur|sprain|injur|disrupt|discontinu|avuls|insufficien|laxity",
    "MCL": r"tear|torn|ruptur|sprain|strain|injur|disrupt|avuls|grade (ii|iii|2|3)",
    "Medial Meniscus": r"tear|torn|ruptur|bucket|flap|displaced|fissur|grade (iii|3)",
    "Lateral Meniscus": r"tear|torn|ruptur|bucket|flap|displaced|fissur|grade (iii|3)",
}
_OA = ("Medial OA", "Lateral OA", "PF OA")


def label_report(text: str) -> dict:
    text = text or ""
    out = {}
    for name, pat in RULES.items():
        hit = 0
        for m in re.finditer(pat, text, re.I):
            start = max(0, text.rfind(".", 0, m.start()) + 1)
            end = text.find(".", m.end())
            sent = text[start: end if end != -1 else len(text)]
            if NEG.search(text[start:m.start()]) or NEG_POST.search(text[m.end():m.end() + 40]):
                continue
            if name in INJURY and not re.search(INJURY[name], sent, re.I):
                continue
            # a meniscal-degeneration sentence is not osteoarthritis unless cartilage/bone changes are named
            if name in _OA and re.search("menisc", sent, re.I) and not re.search("cartilage|chondr|osteoarth|osteophyt", sent, re.I):
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
