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
        self.labels = list(CHEXBERT_LABELS)
        self._compiled = {
            label: [re.compile(p, flags=re.IGNORECASE) for p in patterns]
            for label, patterns in _LABEL_PATTERNS.items()
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
