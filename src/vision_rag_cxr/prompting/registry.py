"""Prompt(STYLE_PROFILE) plug-in/plug-out 레지스트리.

base 프롬프트(STYLE_PROFILE)를 dataset/generator/encoder/critic/labeler처럼 '이름으로 끼우고 빼게' 한다.
- STYLE_PROFILE_CATALOG: 이름 → base 프롬프트 텍스트 (modality_aware / chest_xray / generic_radiology)
- build_style_profile(spec): spec이 (1) 카탈로그 이름 (2) 파일 경로 (3) 리터럴 텍스트 중 무엇이든 해석.
- resolve_style_profile(config): inference/TextGrad가 쓰는 단일 진입점.
    우선순위: optimized_style_profile_path(TextGrad 산출) > style_profile/prompt_profile(spec) > default.

새 base 프롬프트를 붙이려면: 아래 CATALOG에 한 줄 추가하거나, --prompt-profile에 파일 경로/텍스트를 직접 준다.
"""

from __future__ import annotations

from pathlib import Path

from vision_rag_cxr.prompting.prompt_templates import BASE_STYLE_PROFILE

# 흉부 전용(흉부 데이터셋에서 modality를 명시적으로 chest로 고정하고 싶을 때).
CHEST_XRAY_PROFILE = """
You are a careful chest radiograph assistant.
Systematically review lungs, pleura, heart/mediastinum, bones and devices, and report the
clinically important findings using precise radiology terminology.
Do not claim normal findings if abnormal findings are visible.
Do not invent unsupported findings.
"""

# 모달리티 불문(비-chest 포함). 모델이 실제 모달리티/부위를 먼저 식별하게 한다 = 기본값.
MODALITY_AWARE_PROFILE = BASE_STYLE_PROFILE

# 일반 radiology(모달리티 식별 + 간결 소견).
GENERIC_RADIOLOGY_PROFILE = """
You are a careful radiology assistant.
First identify the imaging modality and anatomical region shown, then report the clinically
important findings appropriate to that modality using precise terminology.
Do not claim normal findings if abnormal findings are visible.
Do not invent unsupported findings.
"""

STYLE_PROFILE_CATALOG: dict[str, str] = {
    "modality_aware": MODALITY_AWARE_PROFILE,      # 기본(비-chest 안전)
    "chest_xray": CHEST_XRAY_PROFILE,
    "generic_radiology": GENERIC_RADIOLOGY_PROFILE,
}

# 사람이 읽는 카탈로그(vrag list / 문서용).
PROMPT_CATALOG: dict[str, str] = {
    "modality_aware": "모달리티 자기식별형 base 프롬프트(비-chest 안전, 기본값)",
    "chest_xray": "흉부 전용 base 프롬프트(흉부 데이터셋에서 modality 고정)",
    "generic_radiology": "일반 radiology base 프롬프트(모달리티 식별 + 간결)",
}


def build_style_profile(spec: str | None = None, default: str | None = None) -> str:
    """STYLE_PROFILE spec을 해석한다: 카탈로그 이름 | 파일 경로 | 리터럴 텍스트.

    spec이 비면 default(없으면 modality_aware 기본)를 반환한다.
    """
    if not spec or not str(spec).strip():
        return default if default is not None else MODALITY_AWARE_PROFILE
    spec = str(spec)
    if spec in STYLE_PROFILE_CATALOG:        # (1) 카탈로그 이름
        return STYLE_PROFILE_CATALOG[spec]
    p = Path(spec)
    if p.exists() and p.is_file():           # (2) 파일 경로
        return p.read_text(encoding="utf-8")
    return spec                              # (3) 리터럴 텍스트 그대로


def resolve_style_profile(config: dict, default: str | None = None) -> str:
    """inference/TextGrad 공통 진입점.

    우선순위: optimized_style_profile_path(TextGrad 산출) > style_profile/prompt_profile(spec) > default.
    """
    opt = config.get("optimized_style_profile_path")
    if opt and Path(opt).exists():
        return Path(opt).read_text(encoding="utf-8")
    spec = config.get("style_profile") or config.get("prompt_profile")
    return build_style_profile(spec, default)
