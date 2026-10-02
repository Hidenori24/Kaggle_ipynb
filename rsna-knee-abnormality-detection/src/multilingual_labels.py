"""Keyword pseudo-labels for the non-English reports (Turkish, Greek, Bulgarian, Spanish, German, French) + language detection.

Text and patterns are folded the same way (lower-case, accents/diacritics removed, dotless i -> i, final sigma -> sigma,
sharp s -> ss), so patterns can be written in the natural spelling.
Negation scope differs by language: Turkish puts it after the finding (verb-final), Greek/Bulgarian/Spanish before it,
German both. For ACL / MCL / meniscus labels a tear/injury word is required in the same sentence.
The rules are hand-written; they were written from word-frequency lists of the real reports (no report text) and checked
only on invented sentences. Compare per-language positive rates with English before trusting them (the notebook prints them).
"""
import re
import unicodedata
import pandas as pd
from pseudo_labels import LABELS, label_report


def fold(s):
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("ı", "i").replace("ς", "σ").replace("ß", "ss").replace("œ", "oe").replace("æ", "ae")


def _p(*alts):
    return re.compile(fold("|".join(alts)))


_LIG = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus")
_OA = ("Medial OA", "Lateral OA", "PF OA")


def _inj(ligament, meniscus):
    return {"ACL": _p(ligament), "MCL": _p(ligament), "Medial Meniscus": _p(meniscus), "Lateral Meniscus": _p(meniscus)}


# ---- Turkish -------------------------------------------------------------------------------------
TR_DEG = (r"artroz|kikirdak\w* (incel|kayb|harap|defekt)|kartilaj\w* (incel|kayb|harap|defekt)|kayb|incelme|daralma|"
          r"osteofit|kondromalazi|dejenerati|dejenerasyon|subkondral (skleroz|kist|odem)")
_TR_COMP = r"(medial|medyal)[^.;]{0,40}(femorotibial|tibiofemoral|kompartman|eklem)"
_TR_LCOMP = r"lateral[^.;]{0,40}(femorotibial|tibiofemoral|kompartman|eklem)"
TR = {
    "rules": {
        "ACL": _p(r"\bacl\b", r"on capraz"),
        "MCL": _p(r"\bmcl\b", r"(medial|medyal) kol+ateral", r"tibial kol+ateral", r"kol+ateral bag"),
        "Medial Meniscus": _p(r"(medial|medyal) menisk"),
        "Lateral Meniscus": _p(r"lateral menisk"),
        "Medial OA": _p(rf"{_TR_COMP}[^.;]{{0,80}}({TR_DEG})"),
        "Lateral OA": _p(rf"{_TR_LCOMP}[^.;]{{0,80}}({TR_DEG})"),
        "PF OA": _p(rf"(patellofemoral|femoropatellar|femoro-patellar)[^.;]{{0,80}}({TR_DEG})"),
        "Effusion": _p(r"\bsivi", r"efuzyon"),
        "Synovitis": _p(r"sinovit", r"sinovyal (kalinlas|proliferasyon|inflamasyon)"),
        "Baker's": _p(r"\bbaker", r"popliteal kist", r"popliteal fossa\w* kist"),
        "Contusion": _p(r"kontuzyon", r"kemik ilig\w*[^.;]{0,25}odem", r"kemik ilik\w*[^.;]{0,25}odem",
                        r"kemik ilig\w* sinyal artis", r"subkondral odem", r"bone bruise"),
        "Fracture": _p(r"\bkirik", r"fraktur", r"fissur"),
    },
    "inj": _inj(r"yirtik|ruptur|devamsizlik|kopuk|kopma|zedelen|hasar|sprain|burkul|grade (2|3|ii|iii)",
                r"yirtik|ruptur|devamsizlik|kopuk|kopma|grade (3|iii)"),
    "excl": _p(r"menisk"),
    "neg_pre": None,
    "neg_post": _p(r"yoktur", r"izlenmedi", r"izlenmemis", r"izlenmez", r"gorulmedi", r"gorulmemis", r"saptanmadi",
                   r"saptanmamis", r"normal", r"dogal", r"fizyolojik", r"mevcut degil", r"bulunmamaktadir"),
}

# ---- Greek ---------------------------------------------------------------------------------------
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
    "inj": _inj(r"ρηξ|ρωγμ|διακοπ|διαρρηξ|σχισμ|βαθμου 3|grade 3", r"ρηξ|ρωγμ|διακοπ|διαρρηξ|σχισμ|βαθμου 3|grade 3"),
    "excl": None,
    "neg_pre": _p(r"δεν (παρατηρ|ανιχν|υπαρχ|διακριν|ευρεθ|απεικον|φαινετ)", r"χωρις", r"αρνητικ", r"απουσι"),
    "neg_post": None,
}

# ---- Bulgarian -----------------------------------------------------------------------------------
BG_INJ = r"руптур|разкъс|прекъсн|разрив|скъсван|скъсан|увред|нарушена цялост|grade 3|3 степен|iii степен"
BG_DEG = r"артроз|дегенерат|остеофит|хрущял\w* (изтънен|загуб|дефект)|изтънен"
BG = {
    "rules": {
        "ACL": _p(r"\bacl\b", r"предн\w* кръст", r"пкл"),
        "MCL": _p(r"\bmcl\b", r"медиал\w* колатерал", r"вътрешн\w* колатерал", r"мкл"),
        "Medial Meniscus": _p(r"медиал\w* менискус", r"вътрешн\w* менискус"),
        "Lateral Meniscus": _p(r"латерал\w* менискус", r"външн\w* менискус"),
        "Medial OA": _p(rf"медиал\w*[^.;]{{0,60}}({BG_DEG})", rf"({BG_DEG})[^.;]{{0,60}}медиал\w*"),
        "Lateral OA": _p(rf"латерал\w*[^.;]{{0,60}}({BG_DEG})", rf"({BG_DEG})[^.;]{{0,60}}латерал\w*"),
        "PF OA": _p(rf"(патела|пателофемор|феморопател|пателар)\w*[^.;]{{0,60}}({BG_DEG})",
                    rf"({BG_DEG})[^.;]{{0,60}}(патела|пателофемор)\w*"),
        "Effusion": _p(r"излив", r"течност в ставата", r"свободна течност"),
        "Synovitis": _p(r"синовит", r"синовиална (пролиферация|задебеляване)"),
        "Baker's": _p(r"бейкър", r"бекер", r"\bbaker", r"подколенн\w* кист", r"поплитеал\w* кист",
                      r"кист\w*[^.;]{0,20}поплитеал"),
        "Contusion": _p(r"контузи", r"костен оток", r"оток[^.;]{0,15}мозък", r"костномозъчен едем",
                        r"едем[^.;]{0,25}(костн|мозък)"),
        "Fracture": _p(r"фрактур", r"счупван", r"фисур", r"счупен"),
    },
    "inj": _inj(BG_INJ, BG_INJ),
    "excl": _p(r"менискус"),
    "neg_pre": _p(r"не се (установ|вижд|наблюдав|визуализ|открив|констат)", r"\bбез\b", r"липсва", r"отсъства", r"няма"),
    "neg_post": None,
}

# ---- Spanish -------------------------------------------------------------------------------------
ES_DEG = (r"condropatia|condromalacia|artrosis|gonartrosis|osteofit|pinzamiento|geodas|degenerativ|"
          r"perdida de cartilago|adelgazamiento|ulcera|erosion|condral")
_ES_MED = r"((compartimiento|compartimento|femorotibial\w*|condilo( femoral)?|platillo( tibial)?|meseta( tibial)?)[^.;]{0,20}(interno|interna|medial))"
_ES_LAT = r"((compartimiento|compartimento|femorotibial\w*|condilo( femoral)?|platillo( tibial)?|meseta( tibial)?)[^.;]{0,20}(externo|externa|lateral))"
_ES_PF = r"(femoropatelar\w*|patelofemoral\w*|femoro-patelar|rotula|rotuliano|rotuliana|faceta|troclea)"
ES_INJ = r"rotura|ruptura|desgarro|lesion|elongacion|distension|insuficiencia|laxitud|perdida de continuidad|esguince"
ES_INJ_M = r"rotura|ruptura|desgarro|fisura|grado (iii|3)|asa de cubo|flap|lesion compleja"
ES = {
    "rules": {
        "ACL": _p(r"\blca\b", r"\bacl\b", r"cruzado anterior"),
        "MCL": _p(r"\bllm\b", r"\bmcl\b", r"colateral medial", r"colateral interno", r"lateral interno", r"\blli\b"),
        "Medial Meniscus": _p(r"menisco interno", r"menisco medial"),
        "Lateral Meniscus": _p(r"menisco externo", r"menisco lateral"),
        "Medial OA": _p(rf"{_ES_MED}[^.;]{{0,80}}({ES_DEG})", rf"({ES_DEG})[^.;]{{0,60}}{_ES_MED}"),
        "Lateral OA": _p(rf"{_ES_LAT}[^.;]{{0,80}}({ES_DEG})", rf"({ES_DEG})[^.;]{{0,60}}{_ES_LAT}"),
        "PF OA": _p(rf"{_ES_PF}[^.;]{{0,80}}({ES_DEG})", rf"({ES_DEG})[^.;]{{0,60}}{_ES_PF}"),
        "Effusion": _p(r"derrame", r"liquido (libre )?(intra)?articular", r"hidrartros"),
        "Synovitis": _p(r"sinovitis", r"proliferacion sinovial", r"hiperplasia sinovial", r"sinovial (engrosada|hipertrof)"),
        "Baker's": _p(r"\bbaker", r"quistes? poplit"),
        "Contusion": _p(r"contusion", r"edema (oseo|medular|de medula)", r"edema[^.;]{0,15}medula osea"),
        "Fracture": _p(r"fractura", r"fisura osea"),
    },
    "inj": _inj(ES_INJ, ES_INJ_M),
    "excl": _p(r"menisc"),
    "neg_pre": _p(r"\bsin\b", r"\bno (se )?\w+", r"\bausencia\b", r"\bdescart", r"\bnegativ", r"\blibre de"),
    "neg_post": None,
}

# ---- German --------------------------------------------------------------------------------------
DE_DEG = (r"chondropathie|arthrose|gonarthrose|osteophyt|knorpelschaden|knorpelverlust|knorpeldefekt|chondromalazie|"
          r"degenerativ|gelenkspaltverschmalerung|subchondral\w* (sklerose|zyste|odem)")
DE_INJ = r"riss|ruptur|ausriss|zerrung|lasion|insuffizienz|distorsion|elongation|kontinuitatsunterbrechung"
DE_INJ_M = r"riss|ruptur|korbhenkel|lasion|grad (iii|3)"
DE = {
    "rules": {
        "ACL": _p(r"vorder\w* kreuzband", r"\bvkb\b", r"\bacl\b"),
        "MCL": _p(r"innenband", r"medial\w* kollateral", r"medial\w* seitenband", r"\bmcl\b"),
        "Medial Meniscus": _p(r"innenmeniskus", r"medial\w* meniskus"),
        "Lateral Meniscus": _p(r"aussenmeniskus", r"lateral\w* meniskus"),
        "Medial OA": _p(rf"(medial\w*|innen)[^.;]{{0,20}}(kompartiment|femorotibial\w*)[^.;]{{0,80}}({DE_DEG})",
                        rf"(kompartiment|femorotibial\w*)[^.;]{{0,20}}medial\w*[^.;]{{0,80}}({DE_DEG})"),
        "Lateral OA": _p(rf"(lateral\w*|aussen)[^.;]{{0,20}}(kompartiment|femorotibial\w*)[^.;]{{0,80}}({DE_DEG})",
                         rf"(kompartiment|femorotibial\w*)[^.;]{{0,20}}lateral\w*[^.;]{{0,80}}({DE_DEG})"),
        "PF OA": _p(rf"(retropatellar\w*|femoropatellar\w*|patellofemoral\w*|patellargleitlager|trochlea)[^.;]{{0,80}}({DE_DEG})"),
        "Effusion": _p(r"gelenkerguss", r"\berguss", r"ergus"),
        "Synovitis": _p(r"synovitis", r"synovialitis", r"synovial\w* (verdick|proliferation)"),
        "Baker's": _p(r"\bbaker", r"poplite\w* zyste", r"zyste\w*[^.;]{0,20}poplite"),
        "Contusion": _p(r"knochenmark\w*[- ]?odem", r"kontusion", r"bone bruise", r"knochenmarkskontusion"),
        "Fracture": _p(r"fraktur", r"knochenbruch", r"infraktion"),
    },
    "inj": _inj(DE_INJ, DE_INJ_M),
    "excl": _p(r"meniskus"),
    "neg_pre": _p(r"\bkein\w*", r"\bohne\b", r"\bnicht\b", r"ausschluss"),
    "neg_post": _p(r"nicht (nachweisbar|abgrenzbar|erkennbar|vorhanden|gesichert|abzugrenzen)", r"ausgeschlossen"),
}


# ---- French --------------------------------------------------------------------------------------
FR_DEG = r"chondropathie|arthrose|osteophyt|pincement|amincissement|perte de cartilage|degeneratif|chondra|usure|fibrillation"
_FR_MED = r"((compartiment|femoro-?tibial\w*|condyle( femoral)?|plateau( tibial)?)[^.;]{0,20}(interne|medial\w*))"
_FR_LAT = r"((compartiment|femoro-?tibial\w*|condyle( femoral)?|plateau( tibial)?)[^.;]{0,20}(externe|lateral\w*))"
_FR_PF = r"(femoro-?patellaire|patellofemoral\w*|rotule|trochlee|facette)"
FR = {
    "rules": {
        "ACL": _p(r"croise anterieur", r"\blca\b", r"\bacl\b"),
        "MCL": _p(r"collateral\w* (medial|interne|tibial)", r"lateral interne", r"\bllm\b", r"\bmcl\b"),
        "Medial Meniscus": _p(r"menisque (interne|medial)"),
        "Lateral Meniscus": _p(r"menisque (externe|lateral)"),
        "Medial OA": _p(rf"{_FR_MED}[^.;]{{0,80}}({FR_DEG})", rf"({FR_DEG})[^.;]{{0,60}}{_FR_MED}"),
        "Lateral OA": _p(rf"{_FR_LAT}[^.;]{{0,80}}({FR_DEG})", rf"({FR_DEG})[^.;]{{0,60}}{_FR_LAT}"),
        "PF OA": _p(rf"{_FR_PF}[^.;]{{0,80}}({FR_DEG})", rf"({FR_DEG})[^.;]{{0,60}}{_FR_PF}"),
        "Effusion": _p(r"epanchement", r"liquide articulaire", r"hydarthrose"),
        "Synovitis": _p(r"synovite"),
        "Baker's": _p(r"\bbaker", r"kyste\w* poplit"),
        "Contusion": _p(r"contusion", r"oedeme (osseux|medullaire|de la moelle)", r"oedeme[^.;]{0,15}moelle"),
        "Fracture": _p(r"fracture"),
    },
    "inj": _inj(r"dechirure|rupture|lesion|entorse|desinsertion|elongation|distension",
                r"dechirure|fissure|rupture|lesion complexe|grade (3|iii)|anse de seau|flap|transfixiante"),
    "excl": _p(r"menisque"),
    "neg_pre": _p(r"\bsans\b", r"\baucun\w*", r"\bpas d", r"\bpas de\b", r"\babsence\b", r"\bexclu", r"\bnegati"),
    "neg_post": None,
}

LANGS = {"tr": TR, "el": EL, "bg": BG, "es": ES, "de": DE, "fr": FR}
_TR_MARK = re.compile(r"\b(capraz|eklem\w*|yirtik|devamsizlik|izlen\w*|yoktur|normaldir|bulgular\w*|menisku\w*|"
                      r"sinyal artis\w*|tetkik\w*|kemik|ligaman\w*|sivi)\b")
_ES_MARK = re.compile(r"\b(rotura|derrame|rodilla|menisco\w*|ligamento\w*|hallazgos|impresion|senal|cuadricipital|"
                      r"antecedentes|conservada|normales|tendones|cruzados|colaterales)\b")
_DE_MARK = re.compile(r"\b(gelenkerguss|innenmeniskus|aussenmeniskus|kreuzband\w*|kein\w*|ohne|riss\w*|regelrecht\w*|"
                      r"darstellung|intakt\w*|weichteile|knochenmark\w*|knie\w*|unauffall\w*)\b")
_FR_MARK = re.compile(r"\b(anterieur|posterieur|epanchement|menisque|croise|dechirure|genou|articulaire|sans|avec|aucun\w*)\b")
_PT_MARK = re.compile(r"\b(nao|sem|joelho|lesao|articulacao|alteracoes)\b")
# other Latin-script languages we have no rules for (Portuguese, French, Italian, ...): need >=2 distinct words
_OTHER_MARK = re.compile(r"\b(nao|sem|ruptura|joelho|nella|della|senza|versamento|lesione|ginocchio)\b")
_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+")


def detect_lang(text):
    """'en' | 'tr' | 'el' | 'bg' | 'es' | 'de' | 'fr' | 'other'."""
    t = str(text)
    letters = [c for c in t if c.isalpha()]
    n = max(len(letters), 1)
    if sum(0x370 <= ord(c) <= 0x3FF or 0x1F00 <= ord(c) <= 0x1FFF for c in letters) / n > 0.3:
        return "el"
    if sum(0x400 <= ord(c) <= 0x4FF for c in letters) / n > 0.3:
        return "bg"
    f = fold(t)
    if len(set(_TR_MARK.findall(f))) >= 2:
        return "tr"
    if _PT_MARK.search(f):  # Portuguese shares words with Spanish (ligamento, derrame) but not its negation: keep out
        return "other"
    if len(set(_FR_MARK.findall(f))) >= 2:
        return "fr"
    es, de = len(set(_ES_MARK.findall(f))), len(set(_DE_MARK.findall(f)))
    if es >= 2 and es >= de:
        return "es"
    if de >= 2:
        return "de"
    if len(set(_OTHER_MARK.findall(f))) >= 2:
        return "other"
    ascii_share = sum(c.isascii() for c in t) / max(len(t), 1)
    return "en" if ascii_share > 0.97 else "other"


def label_report_lang(text, lang):
    if lang == "en":
        return label_report(text)
    cfg = LANGS.get(lang)
    out = {c: 0 for c in LABELS}
    if cfg is None:
        return out
    for sent in _SPLIT.split(fold(text)):
        for name in LABELS:
            if out[name]:
                continue
            if name in _OA and cfg["excl"] is not None and cfg["excl"].search(sent):
                continue
            for m in cfg["rules"][name].finditer(sent):
                inj = cfg["inj"].get(name)
                if inj is not None and not inj.search(sent):
                    continue
                if cfg["neg_pre"] is not None and cfg["neg_pre"].search(sent[max(0, m.start() - 70):m.start()]):
                    continue
                if cfg["neg_post"] is not None and cfg["neg_post"].search(sent[m.end():m.end() + 150]):
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
