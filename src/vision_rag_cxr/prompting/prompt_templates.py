"""Prompt templates.

STYLE_PROFILE은 TextGrad가 최적화할 수 있는 부분이다.
모델 weight를 바꾸지 않고 prompt instruction fragment만 수정한다.

modality-parameterized: 데이터셋이 chest가 아닐 수 있으므로(ROCO 등 멀티-모달리티) 프롬프트에
'chest X-ray'를 박지 않는다. {modality}는 sample/dataset의 modality로 채우고(없으면 generic),
모델에게 '실제 보이는 모달리티/해부 부위를 먼저 식별하고 그에 맞는 소견을 기술'하라고 지시한다.
"""

DEFAULT_MODALITY = "medical image"

BASE_STYLE_PROFILE = """
You are a careful radiology assistant.
First identify the imaging modality and anatomical region actually shown in the image,
and report findings appropriate to that modality and region — do not assume a specific body part.
Generate a concise impression in the dataset's reporting style.
Do not claim normal findings if abnormal findings are visible.
Do not invent unsupported findings.
"""

IMPRESSION_PROMPT = """
{style_profile}

Task:
Given the {modality} (and optional support examples), generate the most clinically faithful impression.
Report findings appropriate to the modality and anatomy actually shown; do NOT assume the image is a
chest X-ray unless it clearly is.

Output JSON schema:
{{
  "modality": "<imaging modality / anatomical region you actually see>",
  "impression": "<concise impression>",
  "mentioned_findings": ["<finding 1>", "..."],
  "uncertainty_phrases": ["<uncertain phrase if any>"],
  "no_finding_claim": true/false
}}

{context_examples}
"""

LOCALIZATION_PROMPT = """
{style_profile}

Task:
Given the {modality} (and optional support examples), identify suspected lesions and their approximate
bounding boxes. Report only findings appropriate to the modality/anatomy actually shown.

Output JSON schema:
{{
  "modality": "<imaging modality / anatomical region you actually see>",
  "lesions": [
    {{
      "label": "<the dataset's finding label if applicable>",
      "anatomy": "<anatomical location>",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.0,
      "evidence": "<short reason>"
    }}
  ],
  "global_impression_optional": "<optional impression>"
}}

Important:
If no lesion is visible, return an empty lesions list.
Do not invent a lesion just because support examples contain one.

{context_examples}
"""


def render_prompt(template: str, *, style_profile: str, context_examples: str = "",
                  modality: str | None = None) -> str:
    """프롬프트 템플릿을 modality와 함께 렌더링한다.

    modality는 sample['modality'] 또는 dataset 기본값. 비어 있으면 generic 'medical image'.
    (chest 고정이 아니라 데이터셋-불문으로 동작하게 하는 핵심 헬퍼)
    """
    m = str(modality).strip() if modality else ""
    return template.format(style_profile=style_profile, context_examples=context_examples,
                           modality=(m or DEFAULT_MODALITY))
