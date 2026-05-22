"""RAG k-shot context를 prompt에 넣기 위한 formatter."""

from __future__ import annotations


def build_context_examples_text(examples: list[dict]) -> str:
    """support examples를 VLM prompt에 넣을 문자열로 변환한다.

    실제 image tensor를 few-shot으로 넣는 모델 adapter에서는 ``context_examples`` 원본 list를 같이 받는다.
    이 텍스트 formatter는 모든 VLM에서 공통으로 사용할 수 있는 report-side context를 만든다.
    """
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
    return "\n\n".join(chunks)
