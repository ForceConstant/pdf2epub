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
               if not key.startswith("PDF2EPUB_")}
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
    assert "--gpus" not in argv
    assert argv[-4:] == ["ghcr.io/forceconstant/pdf2epub:0.2.0",
                         "/input/A book.pdf", "/output", "--yes"]


def test_nvidia_selects_package_and_preserves_metadata(launch):
    result, argv = launch("converted books", "--title", "A book; $(literal)",
                          "--author", "{stem}", PDF2EPUB_GPU="nvidia")
    assert result.returncode == 0, result.stderr
    index = argv.index("--gpus")
    assert argv[index + 1] == "all"
    assert argv[-8:] == ["ghcr.io/forceconstant/pdf2epub-nvidia:0.2.0",
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
    assert "--gpus" not in argv


def test_invalid_gpu_mode_fails_before_docker(launch):
    result, argv = launch(PDF2EPUB_GPU="amd")
    assert result.returncode == 1
    assert "PDF2EPUB_GPU must be none or nvidia" in result.stderr
    assert not argv


def test_docker_failure_propagates(launch):
    result, _ = launch(PDF2EPUB_GPU="nvidia", DOCKER_EXIT="42")
    assert result.returncode == 42
