"""Opt-in runtime tuning; unset variables preserve marker's defaults."""

import os
import re


BATCH_SETTINGS = {
    "LAYOUT": "layout_batch_size",
    "RECOGNITION": "recognition_batch_size",
    "DETECTION": "detection_batch_size",
    "TABLE": "table_rec_batch_size",
    "EQUATION": "equation_batch_size",
    "OCR_ERROR": "ocr_error_batch_size",
}
MODEL_SETTINGS = {
    "LAYOUT": "layout_model",
    "RECOGNITION": "recognition_model",
    "TABLE": "table_rec_model",
    "DETECTION": "detection_model",
    "OCR_ERROR": "ocr_error_model",
}


def read_tuning(environ=None):
    """Return validated (converter config, model devices) before loading models.

    A stage-specific value overrides the global value. Empty values are unset;
    an explicit per-model 'auto' restores marker's default device for that model.
    """
    env = os.environ if environ is None else environ
    config, devices = {}, {}

    def batch(name):
        value = env.get(name, "")
        if not value:
            return None
        if not re.fullmatch(r"[0-9]+", value) or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer (got {value!r})")
        return int(value)

    def device(name):
        value = env.get(name, "")
        if not value:
            return None
        if value not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"{name} must be auto, cpu, cuda, or mps (got {value!r})")
        return value

    default_batch = batch("PDF2EPUB_BATCH_SIZE")
    for stage, key in BATCH_SETTINGS.items():
        value = batch(f"PDF2EPUB_{stage}_BATCH_SIZE")
        if value is None:
            value = default_batch
        if value is not None:
            config[key] = value

    default_device = device("PDF2EPUB_MODEL_DEVICE")
    for stage, key in MODEL_SETTINGS.items():
        value = device(f"PDF2EPUB_{stage}_DEVICE") or default_device
        if value is not None and value != "auto":
            devices[key] = value
    return config, devices


def create_tuned_models(devices):
    """Load directly onto selected devices, never all on CUDA then offload.

    Mirrors marker-pdf 1.10.2's five-model factory only when overrides exist.
    CPU overrides explicitly use float32: the global CUDA settings otherwise
    select CUDA-oriented dtypes even for a predictor loaded on CPU.
    """
    from marker import models as factory

    if not devices:
        return factory.create_model_dict()

    import torch

    def kwargs(key):
        selected = devices.get(key)
        if selected is None:
            return {}
        options = {"device": selected}
        if selected == "cpu":
            options["dtype"] = torch.float32
        return options

    def load(key, constructor, checkpoint=None, wrapper=None):
        print(f"Loading {key} on {devices.get(key, 'auto')}...", flush=True)
        options = kwargs(key)
        if checkpoint is not None:
            options["checkpoint"] = checkpoint
        predictor = constructor(**options)
        return wrapper(predictor) if wrapper is not None else predictor

    return {
        "layout_model": load(
            "layout_model", factory.FoundationPredictor,
            factory.surya_settings.LAYOUT_MODEL_CHECKPOINT, factory.LayoutPredictor,
        ),
        "recognition_model": load(
            "recognition_model", factory.FoundationPredictor,
            factory.surya_settings.RECOGNITION_MODEL_CHECKPOINT, factory.RecognitionPredictor,
        ),
        "table_rec_model": load("table_rec_model", factory.TableRecPredictor),
        "detection_model": load("detection_model", factory.DetectionPredictor),
        "ocr_error_model": load("ocr_error_model", factory.OCRErrorPredictor),
    }
