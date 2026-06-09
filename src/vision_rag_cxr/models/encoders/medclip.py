"""BiomedCLIP/MedCLIP 계열 vision-text encoder adapter.

이 클래스는 support set을 vector DB로 만들 때 쓰는 plug-and-play encoder다.
기본값은 Hugging Face에 공개된 BiomedCLIP이며, config의 ``model_name_or_path``만
바꾸면 같은 interface로 다른 OpenCLIP 호환 medical CLIP 계열 모델을 붙일 수 있다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from vision_rag_cxr.models.base import BaseVisionEncoder


class MedCLIPEncoder(BaseVisionEncoder):
    """OpenCLIP 호환 medical image-text encoder.

    frontal/lateral이 모두 있으면 각각 embedding을 만든 뒤 평균한다.
    평균 후 다시 L2 normalize해서 FAISS cosine 검색에 바로 사용할 수 있게 한다.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name_or_path = config.get(
            "model_name_or_path",
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )
        self.device = config.get("device", "cuda")
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """모델/전처리/tokenizer를 lazy loading한다.

        DB build와 query retrieval 모두 이 encoder를 만들기 때문에, 실제 호출 전까지
        weight를 올리지 않으면 테스트와 config 검증이 훨씬 가볍다.
        """
        if self._loaded:
            return

        import open_clip

        from vision_rag_cxr.utils.devices import resolve_device

        # cuda 없으면 mps(Apple)→cpu 자동 폴백
        self.device = resolve_device(self.device)

        self.model, self.preprocess = open_clip.create_model_from_pretrained(self.model_name_or_path)
        self.tokenizer = open_clip.get_tokenizer(self.model_name_or_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype="float32").reshape(-1)
        return vec / (np.linalg.norm(vec) + 1e-8)

    @staticmethod
    def _open_rgb(path: str) -> Image.Image:
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"image path not found: {path}")
        return Image.open(path).convert("RGB")

    def encode_image(self, frontal_path: str, lateral_path: str | None = None) -> np.ndarray:
        """frontal/lateral image embedding을 만든다."""
        self.load()
        import torch

        images = [self._open_rgb(frontal_path)]
        if lateral_path and Path(str(lateral_path)).exists():
            images.append(self._open_rgb(str(lateral_path)))

        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_image(tensors)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            feature = features.mean(dim=0)
            feature = feature / (feature.norm() + 1e-8)
        return self._normalize(feature.detach().float().cpu().numpy())

    @staticmethod
    def _clean_text(text) -> str:
        """NaN/None/float가 tokenizer에 들어가지 않도록 빈 문자열로 정리한다."""
        try:
            import pandas as pd

            if pd.isna(text):
                return ""
        except Exception:
            if text is None:
                return ""
        return str(text)

    def encode_text(self, text: str) -> np.ndarray:
        """report/impression text embedding을 만든다."""
        self.load()
        import torch

        tokens = self.tokenizer([self._clean_text(text)]).to(self.device)
        with torch.inference_mode():
            feature = self.model.encode_text(tokens)
            feature = feature / (feature.norm(dim=-1, keepdim=True) + 1e-8)
        return self._normalize(feature[0].detach().float().cpu().numpy())

    def encode_text_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """여러 report/impression을 batch로 encoding한다."""
        self.load()
        import torch

        from tqdm import tqdm

        vectors = []
        clean_texts = [self._clean_text(text) for text in texts]
        for start in tqdm(range(0, len(clean_texts), batch_size), desc="Encoding support texts"):
            chunk = clean_texts[start:start + batch_size]
            tokens = self.tokenizer(chunk).to(self.device)
            with torch.inference_mode():
                features = self.model.encode_text(tokens)
                features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            vectors.append(features.detach().float().cpu().numpy())
        return np.vstack(vectors).astype("float32")

    def encode_image_batch(self, rows: list[dict], batch_size: int = 32) -> np.ndarray:
        """frontal/lateral image embedding을 batch로 만든다.

        각 sample은 frontal embedding과 lateral embedding을 평균해서 하나의 vector로 만든다.
        batch 내부에서는 view 단위로 펼쳐서 한 번에 encode한 뒤 sample별로 다시 묶는다.
        """
        self.load()
        import torch

        from tqdm import tqdm

        outputs = []
        for start in tqdm(range(0, len(rows), batch_size), desc="Encoding support images"):
            chunk = rows[start:start + batch_size]
            images = []
            owner = []
            for local_idx, row in enumerate(chunk):
                images.append(self._open_rgb(row["frontal_path"]))
                owner.append(local_idx)
                lateral_path = row.get("lateral_path")
                if lateral_path and Path(str(lateral_path)).exists():
                    images.append(self._open_rgb(str(lateral_path)))
                    owner.append(local_idx)

            tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
            with torch.inference_mode():
                features = self.model.encode_image(tensors)
                features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)

            grouped = [[] for _ in chunk]
            for feature, local_idx in zip(features, owner):
                grouped[local_idx].append(feature)
            for feats in grouped:
                sample_feature = torch.stack(feats).mean(dim=0)
                sample_feature = sample_feature / (sample_feature.norm() + 1e-8)
                outputs.append(sample_feature.detach().float().cpu().numpy())
        return np.vstack(outputs).astype("float32")
