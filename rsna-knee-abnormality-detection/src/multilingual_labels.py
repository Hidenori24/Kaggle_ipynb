"""Keyword pseudo-labels for the non-English reports (Turkish, Greek, Bulgarian) plus language detection.

Text and patterns are folded the same way (lower-case, accents/diacritics removed, dotless i -> i,
final sigma -> sigma), so patterns can be written in the natural spelling.
Negation: Turkish puts it after the finding (verb-final), Greek/Bulgarian before it.
These rules are hand-written and have only been checked on invented sentences; compare the per-language
positive rates with English before trusting them (the notebook prints that table).
"""
import re
import unicodedata
import pandas as pd
from pseudo_labels import LABELS, label_report


def fold(s):
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("ı", "i").replace("ς", "σ")


def _p(*alts):
    return re.compile(fold("|".join(alts)))


# ---- Turkish -------------------------------------------------------------------------------------
TR_INJ = r"yirtik|ruptur|devamsizlik|kopuk|kopma|zedelen|hasar|grade (3|iii)"
TR_DEG = r"artroz|kartilaj\w* (incel|kayb|defekt|harap)|osteofit|kondromalazi|dejenerati|dejenerasyon"
TR = {
    "rules": {
        "ACL": _p(r"\bacl\b", r"on capraz"),
        "MCL": _p(r"\bmcl\b", r"medial kol+ateral", r"tibial kol+ateral"),
        "Medial Meniscus": _p(r"medial menisk"),
        "Lateral Meniscus": _p(r"lateral menisk"),
        "Medial OA": _p(rf"medial (femorotibial|tibiofemoral|kompartman)[^.;]{{0,80}}({TR_DEG})"),
        "Lateral OA": _p(rf"lateral (femorotibial|tibiofemoral|kompartman)[^.;]{{0,80}}({TR_DEG})"),
        "PF OA": _p(rf"(patellofemoral|femoropatellar|femoro-patellar)[^.;]{{0,80}}({TR_DEG})"),
        "Effusion": _p(r"\bsivi", r"efuzyon"),
        "Synovitis": _p(r"sinovit", r"sinovyal (kalinlas|proliferasyon|inflamasyon)"),
        "Baker's": _p(r"\bbaker", r"popliteal kist"),
        "Contusion": _p(r"kontuzyon", r"kemik ilig\w* odem", r"kemik ili\w* sinyal artis", r"bone bruise"),
        "Fracture": _p(r"\bkirik", r"fraktur", r"fissur"),
    },
    "inj": {k: _p(TR_INJ) for k in ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus")},
    "neg": _p(r"yoktur", r"izlenmedi", r"izlenmemis", r"izlenmez", r"gorulmedi", r"gorulmemis", r"saptanmadi",
              r"saptanmamis", r"normal", r"dogal", r"fizyolojik", r"mevcut degil", r"bulunmamaktadir"),
    "neg_after": True,
}

# ---- Greek ---------------------------------------------------------------------------------------
EL_INJ = r"ρηξ|ρωγμ|διακοπ|διαρρηξ|σχισμ|βαθμου 3|grade 3"
EL_DEG = r"οστεοαρθριτ|εκφυλιστ|χονδρ|οστεοφυτ|αραιωση"
EL = {
    "rules": {
        "ACL": _p(r"\bacl\b", r"προσθι\w+ χιαστ", r"πχσ", r"πχα"),
        "MCL": _p(r"\bmcl\b", r"εσω πλαγι", r"εσω παραπλευρ"),
        "Medial Meniscus": _p(r"εσω μηνισκ"),
        "Lateral Meniscus": _p(r"εξω μηνισκ"),
        "Medial OA": _p(rf"εσω διαμερισμα[^.;]{{0,80}}({EL_DEG})", rf"({EL_DEG})[^.;]{{0,40}}εσω διαμερισμα"),
        "Lateral OA": _p(rf"εξω διαμερισμα[^.;]{{0,80}}({EL_DEG})", rf"({EL_DEG})[^.;]{{0,40}}εξω διαμερισμα"),
        "PF OA": _p(rf"(επιγονατιδομηριαι|μηροεπιγονατιδ|patellofemoral)[^.;]{{0,80}}({EL_DEG})"),
        "Effusion": _p(r"συλλογη υγρου", r"εξιδρωμα", r"εκχυση", r"υγρου εντος"),
        "Synovitis": _p(r"συνοβιτιδ", r"υμενιτιδ", r"συνοβι\w+ παχυνση"),
        "Baker's": _p(r"\bbaker", r"ιγνυακ\w+ κυστ"),
        "Contusion": _p(r"θλασ", r"οιδημα[^.;]{0,25}μυελ", r"μωλωπ"),
        "Fracture": _p(r"καταγμ", r"fraktur"),
    },
    "inj": {k: _p(EL_INJ) for k in ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus")},
    "neg": _p(r"δεν (παρατηρ|ανιχν|υπαρχ|διακριν|ευρεθ|απεικον|φαινετ)", r"χωρις", r"αρνητικ", r"απουσι"),
    "neg_after": False,
}

# ---- Bulgarian -----------------------------------------------------------------------------------
BG_INJ = r"руптур|разкъс|прекъсн|разрив|скъсан|нарушена цялост|grade 3|3 степен|iii степен"
BG_DEG = r"артроз|дегенерат|остеофит|хрущял\w* (изтънен|загуб|дефект)"
BG = {
    "rules": {
        "ACL": _p(r"\bacl\b", r"предн\w* кръст", r"пкл"),
        "MCL": _p(r"\bmcl\b", r"медиал\w* колатерал", r"вътрешн\w* колатерал", r"мкл"),
        "Medial Meniscus": _p(r"медиал\w* менискус", r"вътрешн\w* менискус"),
        "Lateral Meniscus": _p(r"латерал\w* менискус", r"външн\w* менискус"),
        "Medial OA": _p(rf"медиал\w* (феморотибиал\w* )?(компартмент|отдел)[^.;]{{0,80}}({BG_DEG})"),
        "Lateral OA": _p(rf"латерал\w* (феморотибиал\w* )?(компартмент|отдел)[^.;]{{0,80}}({BG_DEG})"),
        "PF OA": _p(rf"(пателофеморал|феморопателар)\w*[^.;]{{0,80}}({BG_DEG})"),
        "Effusion": _p(r"излив", r"течност в ставата", r"свободна течност"),
        "Synovitis": _p(r"синовит", r"синовиална (пролиферация|задебеляване)"),
        "Baker's": _p(r"бейкър", r"\bbaker", r"подколенн\w* кист"),
        "Contusion": _p(r"контузи", r"костен оток", r"оток[^.;]{0,15}мозък"),
        "Fracture": _p(r"фрактур", r"счупван", r"фисур", r"счупен"),
    },
    "inj": {k: _p(BG_INJ) for k in ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus")},
    "neg": _p(r"не се (установ|вижд|наблюдав|визуализ|открив|констат)", r"\bбез\b", r"липсва", r"отсъства", r"няма"),
    "neg_after": False,
}
LANGS = {"tr": TR, "el": EL, "bg": BG}
_TR_MARK = re.compile(r"\b(capraz|eklem\w*|yirtik|devamsizlik|izlen\w*|yoktur|normaldir|bulgular\w*|menisku\w*|"
                      r"sinyal artis\w*|tetkik\w*|kemik|ligaman\w*|sivi)\b")
# ASCII-only text can still be Spanish/Portuguese/French/German/Italian: treat as 'other' when >=2 such words
_OTHER_MARK = re.compile(r"\b(del|con|sin|los|las|una|derrame|rotura|ligamento|menisco|nao|sem|ruptura|avec|sans|"
                         r"epanchement|und|mit|keine|der|die|das|ohne|erguss|riss|nella|della|senza|versamento|"
                         r"lesione|lesion|rodilla|joelho|genou|knie)\b")
_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+")


def detect_lang(text):
    """'en' | 'tr' | 'el' | 'bg' | 'other' (other = e.g. Latin-script languages we have no rules for)."""
    t = str(text)
    letters = [c for c in t if c.isalpha()]
    n = max(len(letters), 1)
    greek = sum(0x370 <= ord(c) <= 0x3FF or 0x1F00 <= ord(c) <= 0x1FFF for c in letters) / n
    cyr = sum(0x400 <= ord(c) <= 0x4FF for c in letters) / n
    if greek > 0.3:
        return "el"
    if cyr > 0.3:
        return "bg"
    if len(_TR_MARK.findall(fold(t))) >= 2:
        return "tr"
    ascii_share = sum(c.isascii() for c in t) / max(len(t), 1)
    if ascii_share > 0.97 and len(_OTHER_MARK.findall(fold(t))) < 2:
        return "en"
    return "other"


def label_report_lang(text, lang):
    if lang == "en":
        return label_report(text)
    cfg = LANGS.get(lang)
    if cfg is None:
        return {c: 0 for c in LABELS}
    out = {c: 0 for c in LABELS}
    for sent in _SPLIT.split(fold(text)):
        for name in LABELS:
            if out[name]:
                continue
            for m in cfg["rules"][name].finditer(sent):
                inj = cfg["inj"].get(name)
                if inj is not None and not inj.search(sent):
                    continue
                scope = sent[m.end():] if cfg["neg_after"] else sent[max(0, m.start() - 70):m.start()]
                if cfg["neg"].search(scope):
                    continue
                out[name] = 1
                break
    return out


def add_pseudo_labels_ml(train):
    """Adds a 'lang' column; true labels are kept, the rest filled from the per-language rules.
    Languages without rules ('other') keep NaN pseudo-labels -> select with train.lang.isin([...])."""
    out = train.copy()
    out["lang"] = out["Report"].fillna("").map(detect_lang)
    pl = pd.DataFrame([label_report_lang(r, l) if l != "other" else {c: float("nan") for c in LABELS}
                       for r, l in zip(out["Report"].fillna(""), out["lang"])], index=out.index)
    for c in LABELS:
        out[c] = out[c].where(out[c].notna(), pl[c])
    return out
