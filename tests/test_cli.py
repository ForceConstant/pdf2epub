"""Tests for the CLI layer in main.py.

`main.py` imports torch at module level, which is far too heavy to install just
to test argument handling, so a stub stands in when the real thing is absent.
"""

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Stub torch before importing main; the real package pulls in CUDA wheels.
try:  # pragma: no cover - depends on the environment
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    stub = types.ModuleType("torch")
    stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    stub.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    sys.modules["torch"] = stub

import main as cli
from modules import metadata as metadata_lib


def build_parser():
    import argparse

    parser = argparse.ArgumentParser()
    cli.add_metadata_arguments(parser)
    return parser


class TestMetadataArguments:
    def test_all_documented_flags_are_accepted(self):
        args = build_parser().parse_args([
            "--title", "Dune",
            "--author", "Frank Herbert",
            "--publisher", "Ace",
            "--language", "en",
            "--rights", "Public domain",
            "--identifier", "urn:uuid:1234",
            "--date", "1965-08-01",
        ])
        assert args.title == "Dune"
        assert args.author == "Frank Herbert"
        assert args.publisher == "Ace"
        assert args.language == "en"
        assert args.rights == "Public domain"
        assert args.identifier == "urn:uuid:1234"
        assert args.date == "1965-08-01"

    def test_defaults_are_unset(self):
        # None means "not supplied", which is distinct from an empty string.
        args = build_parser().parse_args([])
        assert args.title is None
        assert args.author is None
        assert args.date is None

    def test_collect_returns_empty_when_none_given(self):
        args = build_parser().parse_args([])
        assert cli.collect_metadata_args(args) == {}

    def test_collect_returns_only_supplied_flags(self):
        args = build_parser().parse_args(["--title", "Dune", "--author", "Frank Herbert"])
        assert cli.collect_metadata_args(args) == {
            "title": "Dune",
            "author": "Frank Herbert",
        }

    def test_collect_keeps_empty_string_as_supplied(self):
        # An explicitly empty value still counts as supplied so that
        # resolve_metadata applies it rather than a stale previous value.
        args = build_parser().parse_args(["--title", ""])
        assert cli.collect_metadata_args(args) == {"title": ""}

    def test_placeholders_pass_through_to_validation(self):
        args = build_parser().parse_args(["--title", "{stem}"])
        result = cli.metadata_from_args(args, Path("/books/dune.pdf"))
        assert result["dc:title"] == "dune"

    def test_bad_date_raises_metadata_error(self):
        args = build_parser().parse_args(["--date", "not-a-date"])
        with pytest.raises(metadata_lib.MetadataError):
            cli.metadata_from_args(args, Path("book.pdf"))


class TestStdinDetection:
    def test_returns_false_for_non_tty(self, monkeypatch):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
        assert cli.stdin_is_interactive() is False

    def test_returns_true_for_tty(self, monkeypatch):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        assert cli.stdin_is_interactive() is True

    def test_handles_stdin_without_isatty(self, monkeypatch):
        # sys.stdin can be None under some schedulers (pythonw, cron).
        monkeypatch.setattr(cli.sys, "stdin", None)
        assert cli.stdin_is_interactive() is False

    def test_survives_closed_stdin(self, monkeypatch):
        class Closed:
            def isatty(self):
                raise ValueError("I/O operation on closed file")

        monkeypatch.setattr(cli.sys, "stdin", Closed())
        assert cli.stdin_is_interactive() is False


@pytest.mark.parametrize('flags', [
    ['--articles-only'], ['--substantive-only'],
    ['--articles-only', '--substantive-only'],
])
def test_existing_epub_article_cli_does_not_run_pdf_transcription(tmp_path, monkeypatch, flags):
    from test_chapter_filter import make_epub, classifier, CONFIG
    source = tmp_path / 'Beat #250.epub'
    make_epub(source)
    original_filter = cli.chapter_filter.filter_epub
    monkeypatch.setattr(cli.vlm_lib, 'read_vlm_config',
                        lambda environ=None: CONFIG if environ is not None else None)
    monkeypatch.setattr(cli.chapter_filter, 'filter_epub',
                        lambda src, dst, cfg, **kw: original_filter(src, dst, cfg, classifier))
    def no_pdf(*args, **kwargs):
        raise AssertionError('EPUB must not run OCR')
    monkeypatch.setattr(cli.pdf2md, 'add_pdfs_to_queue', no_pdf)
    monkeypatch.setattr(sys, 'argv', ['main.py', str(source), *flags])
    cli.main()
    assert source.with_name('Beat #250-substantive.epub').is_file()
    assert len(cli.chapter_filter.read_epub(source)[3]) == 3


def test_existing_epub_requires_opt_in(tmp_path, monkeypatch):
    source = tmp_path / 'original.epub'
    source.write_bytes(b'untouched')
    monkeypatch.setattr(cli.vlm_lib, 'read_vlm_config', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['main.py', str(source)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert source.read_bytes() == b'untouched'
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize('vision', [False, True])
@pytest.mark.parametrize('flags', [
    ['--articles-only'], ['--substantive-only'],
    ['--articles-only', '--substantive-only'], [],
])
def test_pdf_filter_is_one_final_pass_independent_of_transcription(tmp_path, monkeypatch, vision, flags):
    from test_chapter_filter import make_epub, classifier, CONFIG
    source = tmp_path / 'Beat #250.pdf'
    source.write_bytes(b'%PDF-1.4\n')
    output = tmp_path / 'output'
    observed = {'transcription': [], 'filter': []}
    def config(environ=None):
        return CONFIG if environ is not None or vision else None
    monkeypatch.setattr(cli.vlm_lib, 'read_vlm_config', config)
    monkeypatch.setattr(cli.pdf2md, 'add_pdfs_to_queue', lambda path: [source])
    def transcribe(*args, **kwargs):
        assert 'articles_only' not in kwargs, 'No old completeness prefilter'
        observed['transcription'].append('vision')
        args[1].mkdir(parents=True)
    def local(*args, **kwargs):
        observed['transcription'].append('local')
        args[1].mkdir(parents=True)
    monkeypatch.setattr(cli.vlm_lib, 'convert_pdf_vlm', transcribe)
    monkeypatch.setattr(cli.pdf2md, 'convert_pdf', local)
    monkeypatch.setattr(cli.mark2epub, 'convert_to_epub',
                        lambda work, out, **kw: make_epub(out / kw['epub_filename']))
    original_filter = cli.chapter_filter.filter_epub
    def filter_once(src, dst, cfg, **kwargs):
        observed['filter'].append(str(src))
        return original_filter(src, dst, cfg, classifier)
    monkeypatch.setattr(cli.chapter_filter, 'filter_epub', filter_once)
    monkeypatch.setattr(sys, 'argv', ['main.py', str(source), str(output), '--yes', *flags])
    cli.main()
    assert observed['transcription'] == ['vision' if vision else 'local']
    assert len(observed['filter']) == bool(flags)
    result = output / 'Beat' / 'Beat #250.epub'
    assert len(cli.chapter_filter.read_epub(result)[3]) == (1 if flags else 3)


@pytest.mark.parametrize('existing_output', [False, True])
def test_failed_pdf_article_filter_never_publishes_raw_epub(tmp_path, monkeypatch, existing_output):
    from test_chapter_filter import make_epub, CONFIG
    source = tmp_path / 'Beat #250.pdf'
    source.touch()
    output = tmp_path / 'output'
    destination = output / 'Beat' / 'Beat #250.epub'
    if existing_output:
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b'previous-output')
    monkeypatch.setattr(cli.vlm_lib, 'read_vlm_config',
                        lambda environ=None: CONFIG if environ is not None else None)
    monkeypatch.setattr(cli.pdf2md, 'add_pdfs_to_queue', lambda path: [source])
    monkeypatch.setattr(cli.pdf2md, 'convert_pdf', lambda src, work, *args: work.mkdir(parents=True))
    monkeypatch.setattr(cli.mark2epub, 'convert_to_epub',
                        lambda work, out, **kw: make_epub(out / kw['epub_filename']))
    def fail_filter(*args, **kwargs):
        raise cli.vlm_lib.VlmError('classification failed')
    monkeypatch.setattr(cli.chapter_filter, 'filter_epub', fail_filter)
    monkeypatch.setattr(sys, 'argv', ['main.py', str(source), str(output), '--yes', '--articles-only'])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    if existing_output:
        assert destination.read_bytes() == b'previous-output'
    else:
        assert not destination.exists()
    assert list(output.glob('.pdf2epub-work/**/.epub-unfiltered/*.epub'))
