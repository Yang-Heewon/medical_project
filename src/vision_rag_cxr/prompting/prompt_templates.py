"""Prompt templates.

STYLE_PROFILE은 TextGrad가 최적화할 수 있는 부분이다.
모델 weight를 바꾸지 않고 prompt instruction fragment만 수정한다.

modality-parameterized(e2e): 'chest X-ray'를 코드에 박지 않고 {modality}를 sample/dataset의
modality 필드로 채운다(PadChest='chest X-ray', ROCO='radiology (mixed modality)', 없으면 generic).
프롬프트는 "This is a {modality}. 판독 impression을 써라"로 모달리티는 맥락으로 알려주되(유출 아님),
이미지 묘사(view/quality)는 금지하고 소견 기술을 강제한다. → 한 템플릿이 모든 데이터셋에 적용(e2e).
"""

DEFAULT_MODALITY = "medical image"

BASE_STYLE_PROFILE = """
You are an expert radiologist.
Write a concise clinical impression in the dataset's reporting style, naming the findings
you observe with precise radiology terminology.
Do not claim normal findings if abnormal findings are visible.
Do not invent unsupported findings.
"""

IMPRESSION_PROMPT = """
{style_profile}

Task:
This is a {modality}. Examine it and write the radiology IMPRESSION.
- State the abnormal findings you observe, concisely, in standard radiology terminology.
- Write ONLY the impression (the findings). Do NOT describe the image type, projection/view, or quality.
- Do not default to a normal impression when abnormalities are visible; do not invent findings.

Output JSON schema:
{{
  "impression": "<concise radiology impression of the findings>",
  "mentioned_findings": ["<finding 1>", "..."],
  "no_finding_claim": true/false
}}

{context_examples}
"""

LOCALIZATION_PROMPT = """
{style_profile}

Task:
This is a {modality}. Identify suspected lesions and their approximate bounding boxes.
State only the findings you observe; do not describe the image type/view/quality.

Output JSON schema:
{{
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
