"""CPU multilingual E5 embeddings. Download is explicit; inference is offline."""

import os
import hashlib
from pathlib import Path
import numpy as np

MODEL = "voiceops/multilingual-e5-small-int8"
REPOSITORY = "intfloat/multilingual-e5-small"
MODEL_FILE = "onnx/model_qint8_avx512_vnni.onnx"


class LocalEmbedder:
    name = MODEL

    def __init__(self, cache_dir: Path, *, allow_download: bool = False):
        import onnxruntime
        onnxruntime.disable_telemetry_events()
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType
        if MODEL not in {m["model"] for m in TextEmbedding.list_supported_models()}:
            TextEmbedding.add_custom_model(model=MODEL, pooling=PoolingType.MEAN, normalization=True,
                sources=ModelSource(hf=REPOSITORY), dim=384, model_file=MODEL_FILE)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = TextEmbedding(MODEL, cache_dir=str(cache_dir), threads=2,
                                   providers=["CPUExecutionProvider"], local_files_only=not allow_download)
        model_dir = Path(self.model.model._model_dir)
        fingerprint = hashlib.sha256()
        for path in (model_dir / MODEL_FILE, model_dir / "tokenizer.json", model_dir / "config.json"):
            fingerprint.update(path.read_bytes())
        self.fingerprint = fingerprint.hexdigest()

    def passages(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self.model.embed(["passage: " + text for text in texts], batch_size=8)), dtype=np.float32)

    def queries(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self.model.embed(["query: " + text for text in texts], batch_size=8)), dtype=np.float32)
