"""Exercise pinned Surya predictors and real tensor/cache cleanup, no downloads.

Runs on CPU CI; this is not a GPU inference or low-VRAM hardware fit test.
"""

from collections import deque
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, "/app")

import torch
from marker import models as factory
from surya.common.adetr.decoder import (
    SuryaADETRDecoderSdpaAttention, SuryaADETRDecoderSdpaCrossAttention,
)
from surya.common.surya.decoder import Qwen2RotaryEmbedding
from surya.common.surya.decoder.config import SuryaDecoderConfig
from modules.offload import _move, _clear_runtime_cache, _root
from modules.tuning import create_tuned_models


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.config = SimpleNamespace(beacon_token_interval=1, image_token_id=2,
                                      register_token_ids=[3], bbox_size=1024)
        config = SimpleNamespace(attention_dropout=0.0, hidden_size=4,
            num_attention_heads=1, head_dim=4, num_key_value_heads=1,
            attention_bias=False, rope_theta=10000, encoder_hidden_size=4)
        self.attention = SuryaADETRDecoderSdpaAttention(config, max_boxes=2)
        self.cross_attention = SuryaADETRDecoderSdpaCrossAttention(config)
        self.rope = Qwen2RotaryEmbedding(SuryaDecoderConfig(
            hidden_size=4, intermediate_size=8, num_attention_heads=1,
            num_key_value_heads=1, head_dim=4, max_position_embeddings=32,
        ))

    @property
    def device(self):
        return self.weight.device

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, input_ids, **kwargs):
        # Called by the real OCRErrorPredictor.__call__ implementation.
        return SimpleNamespace(logits=torch.ones(len(input_ids), 2, device=self.device))


class TinyProcessor:
    pad_token_id = 0
    beacon_token_id = 1
    model_device = "cpu"

    def __call__(self, texts, **kwargs):
        return SimpleNamespace(input_ids=torch.ones(len(texts), 2, dtype=torch.long),
                               attention_mask=torch.ones(len(texts), 2, dtype=torch.long))


loads = []


class TinyLoader:
    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint

    def model(self, device, dtype, attention_implementation=None):
        loads.append(self.checkpoint)
        return TinyModel().to(device=device, dtype=dtype)

    def processor(self):
        return TinyProcessor()


constructors = [factory.FoundationPredictor, factory.TableRecPredictor,
                factory.DetectionPredictor, factory.OCRErrorPredictor]
# Patch only disk/download loaders, not the real predictor constructors.
originals = [constructor.model_loader_cls for constructor in constructors]
try:
    for constructor in constructors:
        constructor.model_loader_cls = TinyLoader
    models = create_tuned_models({}, offload=True)
    assert loads == [], loads
    for predictor in models.values():
        predictor.disable_tqdm = True
    assert loads == [], loads
    assert models["layout_model"]([], batch_size=1) == []
    assert models["recognition_model"]([], bboxes=[], recognition_batch_size=1) == []
    assert models["table_rec_model"]([], batch_size=1) == []
    result = models["ocr_error_model"](["example"], batch_size=1)
    assert result.texts == ["example"]
    models["ocr_error_model"](["again"], batch_size=1)
    assert len(loads) == 4, loads  # detection unused; OCR error reused

    for key in ("layout_model", "recognition_model", "table_rec_model"):
        predictor = models[key]._load()
        root = _root(predictor)
        if hasattr(root, "kv_cache"):
            root.kv_cache = {"tensor": torch.ones(2)}
            root.prompt_queue = deque(["pending"])
        root.model.attention._setup_cache(1, "cpu", torch.float32)
        root.model.cross_attention._setup_cache(1, "cpu", torch.float32)
        root.model.attention._update_cache(torch.ones(1, 1, 1, 4), torch.ones(1, 1, 1, 4))
        root.model.cross_attention._update_cache(torch.ones(1, 1, 1, 4), torch.ones(1, 1, 1, 4))
        _move(predictor, "cpu", torch.float64)
        assert root.model.dtype == torch.float64
        assert root.model.rope.original_inv_freq.device.type == "cpu"
        if hasattr(root, "special_token_ids"):
            for name in ("special_token_ids", "device_pad_token", "device_beacon_token"):
                assert getattr(root, name).device.type == "cpu"
                assert not getattr(root, name).is_floating_point()
        _clear_runtime_cache(predictor)
        _clear_runtime_cache(predictor)  # safe after success, partial setup or repeat
        assert root.model.attention.key_states is None
        assert root.model.cross_attention.value_states is None
        assert getattr(root, "kv_cache", None) is None

    # Execute the CUDA call boundary using real CPU tensors as a stand-in.
    # Only device transfer is mapped to CPU; real Surya OCR-error inference runs.
    proxy = models["ocr_error_model"]
    proxy._device, proxy._dtype = "cuda", torch.float32
    moves = []

    def cpu_transfer(predictor, device, dtype=None):
        moves.append(device)
        _move(predictor, "cpu", dtype)

    with patch("modules.offload._move", side_effect=cpu_transfer):
        assert proxy(["boundary"], batch_size=1).texts == ["boundary"]
    assert moves == ["cuda", "cpu"], moves
finally:
    for constructor, original in zip(constructors, originals):
        constructor.model_loader_cls = original

print("Offload check passed: real Surya constructors, lazy reuse, token tensors, table caches and call boundary (CPU stand-in).")
