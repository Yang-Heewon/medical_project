"""RAG k-shot context를 prompt에 넣기 위한 formatter."""

from __future__ import annotations


def build_context_examples_text(examples: list[dict]) -> str:
    """support examples를 VLM prompt에 넣을 문자열로 변환한다.

    실제 image tensor를 few-shot으로 넣는 모델 adapter에서는 ``context_examples`` 원본 list를 같이 받는다.
    이 텍스트 formatter는 모든 VLM에서 공통으로 사용할 수 있는 report-side context를 만든다.
    """
    # RAG 토글: 검색된 example이 없으면(=No-RAG) 빈 문자열 -> 프롬프트에 RAG 블록 자체가 없음.
    # 있으면(=Vision-RAG) 'Support examples:' 헤더와 함께 블록 전체가 등장.
    if not examples:
        return ""
    chunks = []
    for i, ex in enumerate(examples, start=1):
        chunks.append(
            f"[Support Example {i}]\n"
            f"UID: {ex.get('uid')}\n"
            f"Frontal image: {ex.get('frontal_path', '')}\n"
            f"Lateral image: {ex.get('lateral_path', '')}\n"
            f"Impression: {ex.get('impression', '')}\n"
            f"Findings: {ex.get('findings', '')}\n"
            f"Labels: {ex.get('chexbert_labels_binary', '')}\n"
            f"Retrieval score: {ex.get('retrieval_score', '')}\n"
        )
    return "Support examples (retrieved related cases):\n\n" + "\n\n".join(chunks)
