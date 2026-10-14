"""Launcher contract checks with a fake Docker CLI; no containers are started."""

import os
from pathlib import Path
import subprocess

import pytest


LAUNCHER = Path(__file__).resolve().parent.parent / "scripts" / "pdf2epub.sh"


@pytest.fixture
def launch(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        '#!/bin/sh\nprintf "%s\\0" "$@" > "$DOCKER_ARGS"\n'
        'exit "${DOCKER_EXIT:-0}"\n'
    )
    docker.chmod(0o755)
    input_file = tmp_path / "A book.pdf"
    input_file.touch()
    log = tmp_path / "docker-args"

    def run(*args, **overrides):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("PDF2EPUB_") and key not in
               {"PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"}}
        env.update(PATH=f"{bin_dir}:{env['PATH']}", DOCKER_ARGS=str(log),
                   PDF2EPUB_CACHE_DIR=str(tmp_path / "model cache"))
        env.update(overrides)
        result = subprocess.run(
            ["bash", str(LAUNCHER), str(input_file), *args],
            cwd=tmp_path, env=env, capture_output=True, text=True,
        )
        argv = (log.read_bytes().decode().rstrip("\0").split("\0")
                if log.exists() else [])
        return result, argv

    return run


def test_standard_mode_remains_opt_in(launch):
    result, argv = launch()
    assert result.returncode == 0, result.stderr
    assert argv[:3] == ["run", "--rm", "--init"]
    assert "--pull=always" in argv
    assert "--gpus" not in argv
    assert argv[-4:] == ["ghcr.io/forceconstant/pdf2epub:latest",
                         "/input/A book.pdf", "/output", "--yes"]


def test_nvidia_selects_package_and_preserves_metadata(launch):
    result, argv = launch("converted books", "--title", "A book; $(literal)",
                          "--author", "{stem}", PDF2EPUB_GPU="nvidia")
    assert result.returncode == 0, result.stderr
    assert "--pull=always" in argv
    index = argv.index("--gpus")
    assert argv[index + 1] == "all"
    assert argv[-8:] == ["ghcr.io/forceconstant/pdf2epub-nvidia:latest",
                         "/input/A book.pdf", "/output", "--yes", "--title",
                         "A book; $(literal)", "--author", "{stem}"]


def test_nvidia_allows_image_override(launch):
    result, argv = launch(PDF2EPUB_GPU="nvidia", PDF2EPUB_IMAGE="custom:cuda")
    assert result.returncode == 0, result.stderr
    assert "--gpus" in argv
    assert "custom:cuda" in argv


def test_image_override_does_not_expose_gpu(launch):
    result, argv = launch(PDF2EPUB_IMAGE="custom:cuda")
    assert result.returncode == 0, result.stderr
    assert "--pull=always" in argv
    assert "--gpus" not in argv


def test_invalid_gpu_mode_fails_before_docker(launch):
    result, argv = launch(PDF2EPUB_GPU="amd")
    assert result.returncode == 1
    assert "PDF2EPUB_GPU must be none or nvidia" in result.stderr
    assert not argv


def test_docker_failure_propagates(launch):
    result, _ = launch(PDF2EPUB_GPU="nvidia", DOCKER_EXIT="42")
    assert result.returncode == 42


def test_tuning_is_forwarded_literally_before_image(launch):
    settings = {
        "PDF2EPUB_MODEL_OFFLOAD": "1",
        "PDF2EPUB_BATCH_SIZE": "1", "PDF2EPUB_MODEL_DEVICE": "auto",
        "PDF2EPUB_RECOGNITION_BATCH_SIZE": "2", "PDF2EPUB_TABLE_DEVICE": "cpu",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64; $(literal)",
        "PDF2EPUB_UNKNOWN": "must-not-be-forwarded",
    }
    result, argv = launch(PDF2EPUB_GPU="nvidia", **settings)
    assert result.returncode == 0, result.stderr
    image_index = argv.index("ghcr.io/forceconstant/pdf2epub-nvidia:latest")
    for name, value in settings.items():
        token = f"{name}={value}"
        if name == "PDF2EPUB_UNKNOWN":
            assert token not in argv
        else:
            index = argv.index(token)
            assert argv[index - 1] == "--env"
            assert index < image_index


def test_every_stage_control_is_forwarded(launch):
    settings = {f"PDF2EPUB_{stage}_BATCH_SIZE": "1" for stage in
                ["LAYOUT", "RECOGNITION", "DETECTION", "TABLE", "EQUATION", "OCR_ERROR"]}
    settings.update({f"PDF2EPUB_{stage}_DEVICE": "cpu" for stage in
                     ["LAYOUT", "RECOGNITION", "DETECTION", "TABLE", "OCR_ERROR"]})
    result, argv = launch(**settings)
    assert result.returncode == 0, result.stderr
    assert all(f"{name}={value}" in argv for name, value in settings.items())


def test_unset_tuning_adds_no_implicit_batch_or_device_override(launch):
    result, argv = launch()
    assert result.returncode == 0, result.stderr
    assert not any(value.startswith("PDF2EPUB_") or value.startswith("PYTORCH_") for value in argv)
