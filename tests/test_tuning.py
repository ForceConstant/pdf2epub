"""Runtime configuration and placement tests; no ML models are downloaded."""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import tuning, pdf2md


def test_unset_and_empty_preserve_upstream_defaults():
    assert tuning.read_tuning({}) == ({}, {})
    assert tuning.read_tuning({"PDF2EPUB_BATCH_SIZE": "", "PDF2EPUB_MODEL_DEVICE": "auto"}) == ({}, {})


def test_global_batch_with_stage_override():
    config, devices = tuning.read_tuning({
        "PDF2EPUB_BATCH_SIZE": "1", "PDF2EPUB_RECOGNITION_BATCH_SIZE": "3",
    })
    assert set(config) == set(tuning.BATCH_SETTINGS.values())
    assert config.pop("recognition_batch_size") == 3
    assert set(config.values()) == {1}
    assert devices == {}


@pytest.mark.parametrize("stage,key", tuning.BATCH_SETTINGS.items())
def test_individual_batch_only_changes_its_stage(stage, key):
    assert tuning.read_tuning({f"PDF2EPUB_{stage}_BATCH_SIZE": "2"}) == ({key: 2}, {})


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "two", "1; echo nope", " 1"])
def test_invalid_batches_are_rejected(value):
    with pytest.raises(ValueError, match="PDF2EPUB_BATCH_SIZE must be a positive integer"):
        tuning.read_tuning({"PDF2EPUB_BATCH_SIZE": value})


def test_invalid_specific_setting_does_not_fall_back_to_global():
    with pytest.raises(ValueError, match="PDF2EPUB_TABLE_BATCH_SIZE"):
        tuning.read_tuning({"PDF2EPUB_BATCH_SIZE": "1", "PDF2EPUB_TABLE_BATCH_SIZE": "0"})


def test_global_device_with_override_and_auto_escape():
    _, devices = tuning.read_tuning({
        "PDF2EPUB_MODEL_DEVICE": "cpu", "PDF2EPUB_LAYOUT_DEVICE": "cuda",
        "PDF2EPUB_RECOGNITION_DEVICE": "auto",
    })
    assert devices == {
        "layout_model": "cuda", "table_rec_model": "cpu",
        "detection_model": "cpu", "ocr_error_model": "cpu",
    }


@pytest.mark.parametrize("stage,key", tuning.MODEL_SETTINGS.items())
def test_individual_cpu_device(stage, key):
    assert tuning.read_tuning({f"PDF2EPUB_{stage}_DEVICE": "cpu"}) == ({}, {key: "cpu"})


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="PDF2EPUB_TABLE_DEVICE"):
        tuning.read_tuning({"PDF2EPUB_TABLE_DEVICE": "gpu"})


@pytest.mark.parametrize("name,value", [("PDF2EPUB_BATCH_SIZE", "0"),
                                       ("PDF2EPUB_TABLE_DEVICE", "gpu")])
def test_cli_rejects_bad_tuning_before_reading_inputs(monkeypatch, capsys, name, value):
    import main as cli
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["pdf2epub", "nonexistent.pdf"])

    def no_queue(*args):
        pytest.fail("Input processing must not run for invalid tuning")

    monkeypatch.setattr(cli.pdf2md, "add_pdfs_to_queue", no_queue)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert name in capsys.readouterr().err


@pytest.fixture
def model_factory(monkeypatch):
    calls = []
    factory = types.SimpleNamespace(
        surya_settings=types.SimpleNamespace(
            LAYOUT_MODEL_CHECKPOINT="layout-checkpoint",
            RECOGNITION_MODEL_CHECKPOINT="recognition-checkpoint",
        ),
        create_model_dict=lambda: "original-models",
    )

    def constructor(label):
        def create(**kwargs):
            calls.append((label, kwargs))
            # No .to() exists: models must load directly onto chosen devices.
            return types.SimpleNamespace(label=label, options=kwargs)
        return create

    factory.FoundationPredictor = constructor("foundation")
    factory.LayoutPredictor = lambda predictor: predictor
    factory.RecognitionPredictor = lambda predictor: predictor
    factory.TableRecPredictor = constructor("table")
    factory.DetectionPredictor = constructor("detection")
    factory.OCRErrorPredictor = constructor("ocr_error")
    marker = types.ModuleType("marker")
    marker.models = factory
    monkeypatch.setitem(sys.modules, "marker", marker)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float32="float32"))
    return calls


def test_no_device_override_uses_original_factory(model_factory):
    assert tuning.create_tuned_models({}) == "original-models"
    assert model_factory == []


def test_cpu_offload_loads_on_cpu_without_staging_on_gpu(model_factory):
    models = tuning.create_tuned_models({"recognition_model": "cpu", "table_rec_model": "cpu"})
    assert models["recognition_model"].options == {
        "checkpoint": "recognition-checkpoint", "device": "cpu", "dtype": "float32",
    }
    assert models["table_rec_model"].options == {"device": "cpu", "dtype": "float32"}
    assert models["layout_model"].options == {"checkpoint": "layout-checkpoint"}
    assert models["detection_model"].options == {}
    assert len(model_factory) == 5


def test_all_cpu_models_use_float32(model_factory):
    _, devices = tuning.read_tuning({"PDF2EPUB_MODEL_DEVICE": "cpu"})
    tuning.create_tuned_models(devices)
    assert all(kwargs["device"] == "cpu" and kwargs["dtype"] == "float32"
               for _, kwargs in model_factory)


def test_conversion_forwards_batches_and_preserves_page_range(monkeypatch, tmp_path):
    for name in list(tuning.os.environ):
        if name.startswith("PDF2EPUB_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PDF2EPUB_BATCH_SIZE", "1")
    monkeypatch.setenv("PDF2EPUB_TABLE_DEVICE", "cpu")
    captured = {}

    def load(devices):
        captured["devices"] = devices
        return {"sentinel": "models"}

    def converter(**kwargs):
        captured.update(kwargs)
        return lambda path: types.SimpleNamespace(markdown="Converted", metadata={}, images={})

    monkeypatch.setattr(pdf2md, "create_tuned_models", load)
    module = types.ModuleType("marker.converters.pdf")
    module.PdfConverter = converter
    monkeypatch.setitem(sys.modules, "marker.converters.pdf", module)
    pdf2md.convert_pdf("book.pdf", tmp_path, max_pages=2, start_page=3)
    assert captured["config"]["page_range"] == [3, 4]
    assert all(captured["config"][key] == 1 for key in tuning.BATCH_SETTINGS.values())
    assert captured["devices"] == {"table_rec_model": "cpu"}
    assert captured["artifact_dict"] == {"sentinel": "models"}
    assert (tmp_path / "book.md").read_text() == "Converted"
