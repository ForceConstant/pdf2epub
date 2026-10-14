"""Lazy, call-scoped CUDA residency for marker 1.10.2 / Surya 0.17.1.

Marker passes predictors by reference and may reuse them (OCR, equations,
tables). Keep those references stable, but construct on CPU only on first use
and return weights and auxiliary tensors to CPU after every inference call.
"""

from threading import RLock


def _root(predictor):
    return getattr(predictor, "foundation_predictor", predictor)


def _move(predictor, device, dtype=None):
    root = _root(predictor)
    options = {"device": device}
    if dtype is not None:
        options["dtype"] = dtype
    root.model.to(**options)
    # Dynamic RoPE keeps an unregistered copy alongside its inv_freq buffer.
    for module in root.model.modules():
        original = getattr(module, "original_inv_freq", None)
        if original is not None:
            module.original_inv_freq = original.to(device=device)
    # Surya's BasePredictor.to() only moves weights; FoundationPredictor.to()
    # additionally moves special_token_ids, but misses these two token tensors.
    for name in ("device_pad_token", "device_beacon_token", "special_token_ids"):
        tensor = getattr(root, name, None)
        if tensor is not None:
            setattr(root, name, tensor.to(device=device))
    processor = getattr(root, "processor", None)
    if processor is not None and hasattr(processor, "model_device"):
        processor.model_device = device


def _clear_runtime_cache(predictor):
    root = _root(predictor)
    if hasattr(root, "kv_cache"):
        root.kv_cache = None
        root.prompt_queue.clear()
        root.batch_prompt_mapping = None
    # Table attention caches are plain attributes, not registered buffers, so
    # Module.to() cannot free them. None is safe even before the first call or
    # after an interrupted inference; Surya recreates them in _setup_cache().
    for module in root.model.modules():
        for name in ("key_states", "value_states"):
            if hasattr(module, name):
                setattr(module, name, None)


class OffloadedPredictor:
    """Internal Marker predictor proxy; inference enters through __call__."""

    def __init__(self, key, loader, device, dtype, lock, torch_module):
        self._key = key
        self._loader = loader
        self._device = device
        self._dtype = dtype
        self._lock = lock
        self._torch = torch_module
        self._predictor = None
        self._pending = {}

    def _load(self):
        if self._predictor is None:
            print(f"Loading {self._key} on cpu (on demand)...", flush=True)
            predictor = self._loader()
            for name, value in self._pending.items():
                setattr(predictor, name, value)
            self._predictor = predictor
            self._pending.clear()
        return self._predictor

    def __getattr__(self, name):
        with self._lock:
            if name in self._pending:
                return self._pending[name]
            return getattr(self._load(), name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            with self._lock:
                if self._predictor is None:
                    self._pending[name] = value
                else:
                    setattr(self._predictor, name, value)

    def __call__(self, *args, **kwargs):
        # One shared lock serializes complete calls across all five proxies.
        # Marker invokes them sequentially; no second model remains on CUDA.
        with self._lock:
            predictor = self._load()
            cuda = self._device == "cuda"
            try:
                if cuda:
                    print(f"Moving {self._key} to cuda for inference...", flush=True)
                    _move(predictor, "cuda", self._dtype)
                return predictor(*args, **kwargs)
            finally:
                _clear_runtime_cache(predictor)
                if cuda:
                    _move(predictor, "cpu")
                    self._torch.cuda.empty_cache()
                    print(f"Offloaded {self._key} to cpu.", flush=True)


def create_offloaded_models(factory, devices):
    import torch

    settings = factory.surya_settings
    targets = {key: devices.get(key, settings.TORCH_DEVICE_MODEL) for key in (
        "layout_model", "recognition_model", "table_rec_model",
        "detection_model", "ocr_error_model",
    )}
    if any(device not in {"cpu", "cuda"} for device in targets.values()):
        raise ValueError("PDF2EPUB_MODEL_OFFLOAD=1 supports only cpu and cuda model devices")
    if "cuda" in targets.values() and not torch.cuda.is_available():
        raise ValueError("PDF2EPUB_MODEL_OFFLOAD=1 requires CUDA for cuda-selected models")
    compile_flags = ("COMPILE_ALL", "COMPILE_FOUNDATION", "COMPILE_LAYOUT",
                     "COMPILE_DETECTOR", "COMPILE_TABLE_REC", "COMPILE_OCR_ERROR")
    if any(getattr(settings, name, False) for name in compile_flags):
        raise ValueError("PDF2EPUB_MODEL_OFFLOAD=1 requires Surya model compilation disabled")

    def load(constructor, checkpoint=None, wrapper=None):
        # float32 avoids Surya remapping explicit float16 foundation loads to
        # bfloat16. Cast on transfer using the eventual GPU's native dtype.
        options = {"device": "cpu", "dtype": torch.float32}
        if checkpoint is not None:
            options["checkpoint"] = checkpoint
        predictor = constructor(**options)
        return wrapper(predictor) if wrapper is not None else predictor

    loaders = {
        "layout_model": lambda: load(factory.FoundationPredictor,
            settings.LAYOUT_MODEL_CHECKPOINT, factory.LayoutPredictor),
        "recognition_model": lambda: load(factory.FoundationPredictor,
            settings.RECOGNITION_MODEL_CHECKPOINT, factory.RecognitionPredictor),
        "table_rec_model": lambda: load(factory.TableRecPredictor),
        "detection_model": lambda: load(factory.DetectionPredictor),
        "ocr_error_model": lambda: load(factory.OCRErrorPredictor),
    }
    lock = RLock()
    models = {}
    for key, loader in loaders.items():
        device = targets[key]
        dtype = torch.float32
        if device == "cuda":
            dtype = torch.float16
            if key in {"layout_model", "recognition_model"} and torch.cuda.is_bf16_supported(
                including_emulation=False
            ):
                dtype = torch.bfloat16
        models[key] = OffloadedPredictor(key, loader, device, dtype, lock, torch)
    return models
