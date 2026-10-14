"""Lifecycle tests without CUDA hardware or model downloads."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import sys
import types
from threading import Event

import pytest

from modules import offload, tuning, pdf2md


@pytest.fixture
def runtime(monkeypatch):
    events, predictors = [], []
    torch = types.SimpleNamespace(
        float32="float32", float16="float16", bfloat16="bfloat16",
        cuda=types.SimpleNamespace(is_available=lambda: True,
            is_bf16_supported=lambda **kwargs: False,
            empty_cache=lambda: events.append("empty_cache")),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Tensor:
        def __init__(self):
            self.device, self.dtype = "cpu", "int64"

        def to(self, device):
            self.device = device
            return self

    class Model:
        def __init__(self, key):
            self.key, self.device, self.dtype = key, "cpu", "float32"
            self.key_states = None
            self.value_states = None
            self.original_inv_freq = Tensor()
            self.fail_transfer = False

        def modules(self):
            return [self]

        def to(self, device, dtype=None):
            events.append((self.key, "move", device))
            self.device = device
            if dtype is not None:
                self.dtype = dtype
            if device == "cuda" and self.fail_transfer:
                raise RuntimeError("transfer OOM")

    class Predictor:
        def __init__(self, key, **kwargs):
            assert kwargs["device"] == "cpu"
            assert kwargs["dtype"] == "float32"
            events.append((key, "load"))
            self.model = Model(key)
            self.processor = types.SimpleNamespace(model_device="cpu")
            self.device_pad_token = Tensor()
            self.device_beacon_token = Tensor()
            self.special_token_ids = Tensor()
            self.kv_cache = None
            self.prompt_queue = deque()
            self.batch_prompt_mapping = None
            self.fail_inference = False
            self.action = None
            predictors.append(self)

        def __call__(self, value, **kwargs):
            events.append((self.model.key, "call", self.model.device, value, kwargs))
            # Every other GPU-selected model must already be on CPU.
            assert all(p is self or p.model.device == "cpu" for p in predictors)
            assert self.processor.model_device == self.model.device
            assert self.model.original_inv_freq.device == self.model.device
            for name in ("device_pad_token", "device_beacon_token", "special_token_ids"):
                assert getattr(self, name).device == self.model.device
                assert getattr(self, name).dtype == "int64"
            self.kv_cache = object()
            self.model.key_states = object()
            self.model.value_states = object()
            self.prompt_queue.append("pending")
            self.batch_prompt_mapping = {0: "active"}
            if self.action:
                self.action()
            if self.fail_inference:
                raise RuntimeError("inference OOM")
            return [value]

    factory = types.SimpleNamespace(
        surya_settings=types.SimpleNamespace(TORCH_DEVICE_MODEL="cuda",
            LAYOUT_MODEL_CHECKPOINT="layout", RECOGNITION_MODEL_CHECKPOINT="recognition"),
        FoundationPredictor=lambda checkpoint, **kwargs: Predictor(checkpoint, **kwargs),
        LayoutPredictor=lambda predictor: predictor,
        RecognitionPredictor=lambda predictor: predictor,
        TableRecPredictor=lambda **kwargs: Predictor("table", **kwargs),
        DetectionPredictor=lambda **kwargs: Predictor("detection", **kwargs),
        OCRErrorPredictor=lambda **kwargs: Predictor("ocr_error", **kwargs),
    )
    marker = types.ModuleType("marker")
    marker.models = factory
    monkeypatch.setitem(sys.modules, "marker", marker)
    return types.SimpleNamespace(factory=factory, torch=torch, events=events, predictors=predictors)


@pytest.mark.parametrize("value,expected", [("", False), ("0", False), ("1", True)])
def test_offload_setting(value, expected):
    assert tuning.read_offload({"PDF2EPUB_MODEL_OFFLOAD": value}) is expected


@pytest.mark.parametrize("value", ["true", "false", "2", "yes", " 1"])
def test_invalid_offload_setting(value):
    with pytest.raises(ValueError, match="PDF2EPUB_MODEL_OFFLOAD"):
        tuning.read_tuning({"PDF2EPUB_MODEL_OFFLOAD": value})


def test_lazy_cpu_loading_and_no_download_for_config(runtime):
    models = tuning.create_tuned_models({}, offload=True)
    models["recognition_model"].disable_tqdm = True
    assert models["recognition_model"].disable_tqdm is True
    assert runtime.events == []
    assert models["recognition_model"]("text", batch_size=1) == ["text"]
    assert runtime.predictors[0].disable_tqdm is True
    assert len(runtime.predictors) == 1


def test_models_swap_and_reuse_without_reloading(runtime):
    models = tuning.create_tuned_models({}, offload=True)
    for key in ["layout_model", "detection_model", "recognition_model",
                "table_rec_model", "recognition_model"]:
        assert models[key](key) == [key]
        assert all(p.model.device == "cpu" for p in runtime.predictors)
        for p in runtime.predictors:
            assert p.kv_cache is None
            assert p.model.key_states is None and p.model.value_states is None
            assert not p.prompt_queue and p.batch_prompt_mapping is None
            assert p.device_pad_token.device == p.device_beacon_token.device == "cpu"
            assert p.special_token_ids.device == "cpu"
            assert p.model.original_inv_freq.device == "cpu"
    assert len(runtime.predictors) == 4  # OCR error never used; OCR reused.
    assert sum(e == "empty_cache" for e in runtime.events) == 5
    assert all(p.model.dtype == "float16" for p in runtime.predictors)


def test_wrapper_moves_foundation_not_just_wrapper(runtime):
    root = runtime.factory.FoundationPredictor(checkpoint="layout", device="cpu", dtype="float32")
    wrapper = types.SimpleNamespace(foundation_predictor=root)
    offload._move(wrapper, "cuda", "float16")
    assert root.model.device == root.device_pad_token.device == "cuda"
    root.kv_cache = object()
    offload._clear_runtime_cache(wrapper)
    offload._move(wrapper, "cpu")
    assert root.kv_cache is None and root.model.device == "cpu"


def test_explicit_cpu_models_never_transfer(runtime):
    models = tuning.create_tuned_models({"table_rec_model": "cpu"}, offload=True)
    models["layout_model"]("layout")
    models["table_rec_model"]("table")
    assert ("table", "move", "cuda") not in runtime.events
    assert runtime.predictors[1].model.dtype == "float32"


@pytest.mark.parametrize("failure", ["fail_inference", "fail_transfer"])
def test_errors_release_partial_weights_and_caches(runtime, failure):
    model = tuning.create_tuned_models({}, offload=True)["recognition_model"]
    root = model._load()
    setattr(root if failure == "fail_inference" else root.model, failure, True)
    with pytest.raises(RuntimeError, match="OOM"):
        model("bad")
    assert root.model.device == root.special_token_ids.device == "cpu"
    assert root.device_pad_token.device == root.device_beacon_token.device == "cpu"
    assert root.kv_cache is None and root.model.key_states is None
    assert runtime.events[-1] == "empty_cache"
    root.fail_inference = root.model.fail_transfer = False
    assert model("retry") == ["retry"]


def test_bfloat16_only_for_supported_foundation_gpu(runtime):
    runtime.torch.cuda.is_bf16_supported = lambda **kwargs: True
    models = tuning.create_tuned_models({}, offload=True)
    models["layout_model"]("layout")
    models["table_rec_model"]("table")
    assert runtime.predictors[0].model.dtype == "bfloat16"
    assert runtime.predictors[1].model.dtype == "float16"


@pytest.mark.parametrize("device", ["mps", "xla"])
def test_unsupported_devices_fail_before_model_loading(runtime, device):
    with pytest.raises(ValueError, match="only cpu and cuda"):
        tuning.create_tuned_models({"layout_model": device}, offload=True)
    assert runtime.events == []


def test_missing_cuda_fails_before_model_loading(runtime):
    runtime.torch.cuda.is_available = lambda: False
    with pytest.raises(ValueError, match="requires CUDA"):
        tuning.create_tuned_models({}, offload=True)
    assert runtime.events == []
    devices = {key: "cpu" for key in tuning.MODEL_SETTINGS.values()}
    models = tuning.create_tuned_models(devices, offload=True)
    assert models["layout_model"]("cpu") == ["cpu"]


def test_compiled_models_rejected_before_loading(runtime):
    runtime.factory.surya_settings.COMPILE_ALL = True
    with pytest.raises(ValueError, match="compilation disabled"):
        tuning.create_tuned_models({}, offload=True)
    assert runtime.events == []


def test_parallel_calls_do_not_overlap_gpu_residency(runtime):
    models = tuning.create_tuned_models({}, offload=True)
    entered, release, second_attempt = Event(), Event(), Event()
    root = models["layout_model"]._load()

    def hold():
        entered.set()
        assert release.wait(5)

    root.action = hold

    def second():
        second_attempt.set()
        return models["recognition_model"]("second")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(models["layout_model"], "first")
        assert entered.wait(5)
        other = pool.submit(second)
        assert second_attempt.wait(5)
        try:
            assert len(runtime.predictors) == 1  # second waits before loading
        finally:
            release.set()
        assert first.result(5) == ["first"]
        assert other.result(5) == ["second"]


def test_converter_uses_lazy_predictors_for_repeated_stages(runtime, monkeypatch, tmp_path):
    for name in list(tuning.os.environ):
        if name.startswith("PDF2EPUB_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PDF2EPUB_MODEL_OFFLOAD", "1")
    monkeypatch.setenv("PDF2EPUB_BATCH_SIZE", "1")

    def converter(artifact_dict, config):
        assert runtime.events == []  # converter built before any model load
        assert config["recognition_batch_size"] == 1

        def render(path):
            artifact_dict["recognition_model"].disable_tqdm = True
            for key in ["layout_model", "recognition_model", "table_rec_model", "recognition_model"]:
                artifact_dict[key](path, batch_size=1)
            return types.SimpleNamespace(markdown="Swapped", metadata={}, images={})
        return render

    module = types.ModuleType("marker.converters.pdf")
    module.PdfConverter = converter
    monkeypatch.setitem(sys.modules, "marker.converters.pdf", module)
    pdf2md.convert_pdf("book.pdf", tmp_path)
    assert (tmp_path / "book.md").read_text() == "Swapped"
    assert len(runtime.predictors) == 3
    assert all(p.model.device == "cpu" for p in runtime.predictors)
