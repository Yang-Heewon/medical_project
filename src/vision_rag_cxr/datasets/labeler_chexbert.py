"""CheXpert/CheXbert 14-label labeler.

이 모듈은 두 가지를 제공한다.

1. ``CHEXBERT_LABELS``: 프레임워크 전체가 공유하는 고정 순서 14 label.
   RAG DB build, retriever, evaluation 모두 이 순서를 신뢰한다.
2. ``CheXbertLikeLabeler``: CheXbert 설치 전에도 pipeline을 끝까지 돌릴 수 있게 하는
   keyword 기반 fallback labeler. 실제 논문/보고 결과는 외부 CheXbert backend로 생성해
   ``scripts/10_merge_chexbert_labels.py``로 merge하는 것을 권장한다.

라벨링 정책 (docs/PROJECT_FRAMEWORK_PROMPT_KO.md 기준):
- label space: CheXpert/CheXbert 14 labels
- main label type: multi-label (top-1로 축소하지 않는다)
- uncertain policy: U-Ones (uncertain은 binary에서 1로, raw에는 -1로 보존)
- negation은 0으로 처리한다.
"""

from __future__ import annotations

import re
from typing import Iterable

# CheXpert/CheXbert 표준 14 label, 고정 순서.
CHEXBERT_LABELS: list[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# 각 label을 켜는 keyword/정규식 패턴. 단어 경계를 쓰되 의학 표현 변형을 일부 허용한다.
# "No Finding"은 별도 로직으로 처리하므로 여기 포함하지 않는다.
_LABEL_PATTERNS: dict[str, list[str]] = {
    "Enlarged Cardiomediastinum": [
        r"enlarged cardiomediastin\w*",
        r"widen\w* mediastin\w*",
        r"mediastinal widening",
        r"cardiomediastinal silhouette is enlarged",
        r"prominent mediastin\w*",
    ],
    "Cardiomegaly": [
        r"cardiomegaly",
        r"enlarged heart",
        r"cardiac enlargement",
        r"enlarged cardiac silhouette",
        r"heart size is enlarged",
        r"enlarged cardiac contour",
    ],
    "Lung Opacity": [
        r"opacit\w+",
        r"opacification",
        r"airspace disease",
        r"air space disease",
        r"infiltrate\w*",
        r"reticular\w*",
    ],
    "Lung Lesion": [
        r"nodul\w+",
        r"\bmass(es)?\b",
        r"\blesion\w*",
        r"granuloma\w*",
    ],
    "Edema": [
        r"edema",
        r"oedema",
        r"vascular congestion",
        r"pulmonary congestion",
    ],
    "Consolidation": [
        r"consolidat\w+",
    ],
    "Pneumonia": [
        r"pneumonia",
        r"infectio\w+",
        r"bronchopneumonia",
    ],
    "Atelectasis": [
        r"atelecta\w+",
        r"\bcollapse\w*",
    ],
    "Pneumothorax": [
        r"pneumothora\w+",
    ],
    "Pleural Effusion": [
        r"effusion\w*",
        r"pleural fluid",
    ],
    "Pleural Other": [
        r"pleural thickening",
        r"pleural scar\w*",
        r"pleural plaque\w*",
        r"\bfibrosis\b",
        r"pleural calcification",
    ],
    "Fracture": [
        r"fractur\w+",
    ],
    "Support Devices": [
        r"\btube\b",
        r"catheter\w*",
        r"pacemaker\w*",
        r"\bpicc\b",
        r"central line",
        r"\bport-?a-?cath\w*",
        r"sternotomy wire\w*",
        r"surgical clip\w*",
        r"\bstent\w*",
        r"\bdevice\w*",
    ],
}

# negation/uncertainty cue. label keyword 주변 window에서 검사한다.
_NEGATION_CUES = [
    "no ",
    "no evidence of",
    "without",
    "negative for",
    "free of",
    "resolution of",
    "resolved",
    "ruled out",
    "rule out",
    "absence of",
    "not identified",
    "not seen",
    "unremarkable for",
]
_UNCERTAINTY_CUES = [
    "possible",
    "possibly",
    "may represent",
    "may be",
    "cannot exclude",
    "cannot be excluded",
    "suggestive of",
    "suspicious for",
    "questionable",
    "concerning for",
    "likely",
    "probable",
    "differential",
    "could represent",
    "?",
]
# "No Finding"을 켜는 normal cue.
_NORMAL_CUES = [
    "no acute",
    "no active",
    "normal",
    "clear lung",
    "lungs are clear",
    "unremarkable",
    "no evidence of acute",
    "no significant",
    "no abnormalit",
    "within normal limits",
    "no findings",
]

_WINDOW = 40  # negation/uncertainty를 keyword 앞쪽 몇 글자까지 볼지.


# PadChest-GR 24-finding 키워드 패턴 (실제 grounded_reports 문장 어휘에 근거).
# key는 PADCHEST_GR_FINDINGS 이름과 정확히 일치. "No Finding"은 파생, "Other"는 잔여라 패턴 없음.
_PADCHEST_GR_PATTERNS: dict[str, list[str]] = {
    "Aortic elongation": [r"aortic elongation", r"elongat\w*\s+aort", r"elongation of the aorta",
                          r"unfold\w*\s+aort", r"aortic unfolding"],
    "Cardiomegaly": [r"cardiomegal\w*", r"enlarged heart", r"cardiac enlargement",
                     r"increased? cardiac silhouette", r"enlarged cardiac silhouette"],
    "Nodule": [r"nodul\w*", r"granulom\w*"],
    "Pleural effusion": [r"pleural effusion", r"\beffusion\b", r"costophrenic angle",
                         r"blunting of the .{0,20}costophrenic", r"\bpleural fluid\b"],
    "Scoliosis": [r"scolio\w*"],
    "Vertebral degenerative changes": [r"spondyl\w*", r"osteophyt\w*",
                                       r"degenerative\s+\w*\s*(change|column|spine|vertebr)",
                                       r"vertebral degenerative", r"\bdiscopath\w*"],
    "Hyperinflated lung": [r"air trapping", r"hyperinflat\w*", r"emphysem\w*", r"hyperinsufflat\w*"],
    "Vascular hilar enlargement": [r"hilar enlargement", r"enlargement of the .{0,15}hil",
                                   r"congested hil\w*", r"hilar congestion", r"prominent hil\w*",
                                   r"vascular hil\w*"],
    "Atelectasis": [r"atelecta\w*"],
    "Aortic atheromatosis": [r"aortic calcification", r"calcified aort\w*", r"aortic atheromatosis",
                            r"atheromatosis", r"calcified aortic"],
    "Pleural thickening": [r"pleural thickening", r"apical (cap|thickening)", r"biapical\s+\w*\s*thickening"],
    "Interstitial pattern": [r"interstitial pattern", r"interstitial infiltrate", r"interstitial-alveolar"],
    "Alveolar pattern": [r"alveolar pattern", r"alveolar infiltrate", r"air\s?space\s+(disease|involvement|infiltrate)"],
    "Electrical device": [r"pacemaker\w*", r"electrical device", r"\bholter\b", r"defibrillat\w*",
                          r"\bicd\b", r"pacing (lead|wire)"],
    "Hemidiaphragm elevation": [r"elevation of the .{0,15}hemidiaphragm", r"hemidiaphragm\w*\s+elevat\w*",
                                r"elevat\w*\s+\w*\s*(hemidiaphragm|diaphragm)", r"diaphragm\w*\s+elevat\w*"],
    "Fracture": [r"fractur\w+", r"\bcallus\w*", r"bone callus"],
    "Hypoexpansion": [r"volume loss", r"hypoexpansion", r"hypoventilation", r"loss of volume",
                      r"reduced lung volume", r"poor inspiration"],
    "Central venous catheter": [r"central venous catheter", r"central line", r"central venous",
                                r"\bpicc\b", r"jugular .{0,15}cath", r"subclavian .{0,15}(vein|line|cath)"],
    "Hiatal hernia": [r"hiat\w*\s+hernia"],
    "Endotracheal tube": [r"endotracheal tube", r"tracheostomy\w*", r"\bett\b", r"intubat\w*", r"orotracheal"],
    "NSG tube": [r"nasogastric\w*", r"\bng tube\b", r"\bnsg tube\b", r"feeding tube", r"\bsng\b"],
    "Bronchiectasis": [r"bronchiectasi\w*", r"bronchiectatic"],
    "Goiter": [r"goit\w*re?", r"\bgoiter\b"],
    "Osteopenia": [r"osteopen\w*", r"osteoporo\w*", r"demineraliz\w*", r"bone demineralisation"],
}

# label_space -> 패턴 세트 (plug-in/out: 데이터셋 라벨 공간마다 채점기 패턴을 끼운다).
PATTERN_SETS: dict[str, dict[str, list[str]]] = {
    "chexbert_14": _LABEL_PATTERNS,
    "padchest_gr_24": _PADCHEST_GR_PATTERNS,
    "padchest_gr": _PADCHEST_GR_PATTERNS,
}


def _has_cue(window_text: str, cues: Iterable[str]) -> bool:
    return any(cue in window_text for cue in cues)


class CheXbertLikeLabeler:
    """keyword 기반 CheXbert-like fallback labeler.

    실제 CheXbert 모델이 아니라, 설치 전/smoke 단계에서 pipeline을 끝까지 돌리고
    label 컬럼 계약(`chexbert_labels_binary`/`chexbert_labels_raw`)을 채우기 위한 adapter다.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        # uncertain_policy: u_ones(기본) -> uncertain을 binary 1로. u_zeros -> 0으로.
        self.uncertain_policy = str(self.config.get("uncertain_policy", "u_ones")).lower()
        # plug-in/out: label_space마다 라벨 목록 + 키워드 패턴 세트를 끼운다.
        self.label_space = str(self.config.get("label_space", "chexbert_14"))
        patterns = PATTERN_SETS.get(self.label_space, _LABEL_PATTERNS)
        if self.label_space in ("padchest_gr_24", "padchest_gr"):
            from vision_rag_cxr.datasets.label_spaces import PADCHEST_GR_LABELS  # lazy: 순환 import 회피
            self.labels = list(PADCHEST_GR_LABELS)
        elif self.label_space == "chexbert_14":
            self.labels = list(CHEXBERT_LABELS)
        else:
            from vision_rag_cxr.datasets.label_spaces import resolve_labels  # lazy
            self.labels = resolve_labels({"label_space": self.label_space})
        self._compiled = {
            label: [re.compile(p, flags=re.IGNORECASE) for p in pats]
            for label, pats in patterns.items()
        }

    # --- 단일 텍스트 ---------------------------------------------------------
    def _raw_label_text(self, text: str) -> dict[str, int]:
        """raw label dict를 만든다. 값은 1(positive)/-1(uncertain)/0(neg or absent)."""
        raw = {label: 0 for label in self.labels}
        if not text:
            return raw
        norm = " " + re.sub(r"\s+", " ", str(text).lower()).strip() + " "

        for label, patterns in self._compiled.items():
            for pat in patterns:
                m = pat.search(norm)
                if not m:
                    continue
                start = max(0, m.start() - _WINDOW)
                window = norm[start : m.start()]
                full_window = norm[start : m.end() + 5]
                if _has_cue(window, _NEGATION_CUES):
                    # 명시적 negation은 0으로 둔다 (이미 0이므로 skip).
                    continue
                if _has_cue(full_window, _UNCERTAINTY_CUES):
                    raw[label] = -1 if raw[label] == 0 else raw[label]
                else:
                    raw[label] = 1
                if raw[label] == 1:
                    break  # 확정 positive면 추가 패턴 검사 불필요.
        return raw

    def _binary_from_raw(self, raw: dict[str, int]) -> dict[str, int]:
        """raw(-1/0/1) -> binary(0/1). uncertain_policy 적용."""
        binary = {}
        for label, v in raw.items():
            if v == 1:
                binary[label] = 1
            elif v == -1:
                binary[label] = 1 if self.uncertain_policy == "u_ones" else 0
            else:
                binary[label] = 0
        # No Finding: 다른 abnormal label이 하나도 없으면 1.
        abnormal_positive = any(binary[l] == 1 for l in self.labels if l != "No Finding")
        binary["No Finding"] = 0 if abnormal_positive else 1
        return binary

    def label_report(self, impression: str, findings: str = "") -> tuple[dict[str, int], dict[str, int]]:
        """primary=impression, secondary=findings+impression 정책으로 (binary, raw) 반환.

        impression이 비어 있거나 너무 짧으면 findings+impression을 secondary로 쓴다.
        """
        impression = str(impression or "").strip()
        findings = str(findings or "").strip()
        primary_text = impression
        min_len = int(self.config.get("min_primary_chars", 3))
        if len(primary_text) < min_len:
            primary_text = (findings + " " + impression).strip()

        raw = self._raw_label_text(primary_text)
        # findings는 secondary audit으로 합쳐 positive 보강 (negation은 그대로 0 유지).
        if self.config.get("use_findings_secondary", True) and findings:
            sec = self._raw_label_text(findings)
            for label in self.labels:
                if raw[label] == 0 and sec[label] == 1:
                    raw[label] = 1
                elif raw[label] == 0 and sec[label] == -1:
                    raw[label] = -1
        binary = self._binary_from_raw(raw)
        return binary, raw

    # --- batch ---------------------------------------------------------------
    def label_texts(self, texts: Iterable[str]) -> list[dict[str, int]]:
        """text list -> binary label dict list. (단위 테스트가 쓰는 API)"""
        out = []
        for t in texts:
            binary, _ = self.label_report(t)
            out.append(binary)
        return out


def labels_to_binary_vector(label_dict: dict[str, int]) -> list[int]:
    """label dict를 CHEXBERT_LABELS 순서의 0/1 vector로 변환한다."""
    return [int(label_dict.get(label, 0)) for label in CHEXBERT_LABELS]


# label_space -> 사람이 읽는 채점기 카탈로그 (vrag list / plug-in 문서용).
LABELER_CATALOG: dict[str, str] = {
    "chexbert_14": "CheXbert-like 14-label keyword labeler (IU/CheXpert 계열)",
    "padchest_gr_24": "PadChest-GR 24-finding keyword labeler (No Finding + 24 + Other)",
}


def build_labeler(config: dict | None = None) -> CheXbertLikeLabeler:
    """label_space에 맞는 keyword labeler를 만든다 (plug-in/out).

    config["label_space"]로 채점기 라벨 목록+패턴을 끼운다. 미지원 space는 CheXbert-14로 fallback.
    실제 CheXbert 모델 등 다른 채점기를 붙이려면 config["labeler"]로 분기를 추가한다.
    """
    cfg = dict(config or {})
    name = str(cfg.get("labeler", "keyword")).lower()
    if name in ("keyword", "chexbert_like", "chexbert-like", ""):
        return CheXbertLikeLabeler(cfg)
    raise ValueError(f"지원하지 않는 labeler입니다: {name}")
