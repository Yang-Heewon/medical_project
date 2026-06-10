"""BioViL-T (Microsoft BiomedVLP-BioViL-T) CXR-특화 image-text encoder adapter.

MIMIC-CXR + radiology report로 학습된 흉부 X-ray 특화 인코더. health_multimodal(hi-ml-multimodal)
패키지로 image/text inference engine을 만들어 128-d joint embedding을 낸다.
- 이미지: ImageInferenceEngine.get_projected_global_embedding(Path) -> 128-d
- 텍스트: TextInferenceEngine.get_embeddings_from_prompt([..]) -> [N,128]
retrieval(FAISS cosine)을 위해 L2 normalize한다. (BiomedCLIP보다 흉부 검색 이웃 품질이 좋음)

설치: pip install --no-deps hi-ml-multimodal pydicom SimpleITK scikit-image lazy_loader imageio tifffile networkx
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vision_rag_cxr.models.base import BaseVisionEncoder


class BioViLEncoder(BaseVisionEncoder):
    def __init__(self, config: dict):
        super().__init__(config)
        self.device = config.get("device", "cuda")
        self.embedding_dim = int(config.get("embedding_dim", 128))
        self.img_engine = None
        self.txt_engine = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        import warnings
        warnings.filterwarnings("ignore")
        from health_multimodal.image import get_image_inference
        from health_multimodal.image.utils import ImageModelType

        from vision_rag_cxr.utils.devices import resolve_device

        self.device = resolve_device(self.device)
        self.img_engine = get_image_inference(ImageModelType.BIOVIL_T)
        self.img_engine.to(self.device)
        try:
            from health_multimodal.text import get_bert_inference
            from health_multimodal.text.utils import BertEncoderType
            self.txt_engine = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
            self.txt_engine.to(self.device)
        except Exception as e:  # 텍스트 인코더 실패해도 이미지 검색은 동작
            print(f"[biovil] text engine load warn: {e}", flush=True)
            self.txt_engine = None
        self._loaded = True

    @staticmethod
    def _to_np(v) -> np.ndarray:
        return v.detach().float().cpu().numpy() if hasattr(v, "detach") else np.asarray(v, dtype="float32")

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype="float32").reshape(-1)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def _emb_image(self, path: str) -> np.ndarray:
        return self._to_np(self.img_engine.get_projected_global_embedding(Path(str(path)))).reshape(-1)

    def encode_image(self, frontal_path: str, lateral_path: str | None = None) -> np.ndarray:
        self.load()
        vecs = [self._emb_image(frontal_path)]
        if lateral_path and Path(str(lateral_path)).exists():
            vecs.append(self._emb_image(str(lateral_path)))
        return self._normalize(np.mean(vecs, axis=0))

    @staticmethod
    def _clean_text(text) -> str:
        try:
            import pandas as pd
            if pd.isna(text):
                return ""
        except Exception:
            if text is None:
                return ""
        return str(text)

    def encode_text(self, text: str) -> np.ndarray:
        self.load()
        if self.txt_engine is None:
            return self._normalize(np.zeros(self.embedding_dim, dtype="float32"))
        arr = self._to_np(self.txt_engine.get_embeddings_from_prompt([self._clean_text(text)]))
        return self._normalize(arr.reshape(arr.shape[0], -1)[0])

    def encode_text_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        self.load()
        from tqdm import tqdm
        if self.txt_engine is None:
            return np.zeros((len(texts), self.embedding_dim), dtype="float32")
        out = []
        clean = [self._clean_text(t) for t in texts]
        for start in tqdm(range(0, len(clean), batch_size), desc="BioViL-T encode texts"):
            chunk = clean[start:start + batch_size]
            arr = self._to_np(self.txt_engine.get_embeddings_from_prompt(chunk)).reshape(len(chunk), -1)
            arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
            out.append(arr.astype("float32"))
        return np.vstack(out).astype("float32")

    def encode_image_batch(self, rows: list[dict], batch_size: int = 32) -> np.ndarray:
        self.load()
        from tqdm import tqdm
        out = []
        for row in tqdm(rows, desc="BioViL-T encode images"):
            out.append(self.encode_image(row["frontal_path"], row.get("lateral_path")))
        return np.vstack(out).astype("float32")
