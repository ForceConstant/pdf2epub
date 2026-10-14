"""Check real marker configuration/constructor contracts without model downloads."""

import inspect
import sys

sys.path.insert(0, "/app")

import torch
from marker import models as factory
from marker.builders.layout import LayoutBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.processors.table import TableProcessor
from marker.processors.equation import EquationProcessor
from modules.tuning import read_tuning

config, devices = read_tuning({"PDF2EPUB_BATCH_SIZE": "1", "PDF2EPUB_TABLE_DEVICE": "cpu"})
assert devices == {"table_rec_model": "cpu"}
predictor = object()
layout = LayoutBuilder(predictor, config=config)
line = LineBuilder(predictor, predictor, config=config)
ocr = OcrBuilder(predictor, config=config)
table = TableProcessor(predictor, predictor, predictor, config=config)
equation = EquationProcessor(predictor, config=config)
sizes = [layout.get_batch_size(), line.get_detection_batch_size(),
         line.get_ocr_error_batch_size(), ocr.get_recognition_batch_size(),
         table.get_table_rec_batch_size(), table.get_detection_batch_size(),
         table.get_recognition_batch_size(), equation.get_batch_size()]
assert sizes == [1] * 8, sizes
# Check that the pinned Surya constructors accept direct CPU/float32 placement.
for constructor in [factory.FoundationPredictor, factory.TableRecPredictor,
                    factory.DetectionPredictor, factory.OCRErrorPredictor]:
    inspect.signature(constructor).bind(device="cpu", dtype=torch.float32)
for wrapper in [factory.LayoutPredictor, factory.RecognitionPredictor]:
    inspect.signature(wrapper).bind(predictor)
print("Tuning check passed: real Marker batch overrides and direct-device constructor contracts.")
