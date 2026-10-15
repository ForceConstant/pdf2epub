"""VLM conversion path: configuration, request shape and failure handling.

No network access and no PDF rendering: the transport is injected, and the
pdfium module is faked, so these run anywhere the rest of the suite does.
"""

import io
import json
import sys
import threading
import time
import types
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from modules import vlm


def make_config(**overrides):
    config = {
        "model": vlm.DEFAULT_MODEL,
        "base_url": vlm.DEFAULT_BASE_URL,
        "api_key": "test-key",
        "max_tokens": 4096,
        "dpi": 150,
        "concurrency": 2,
        "max_retries": 2,
        "timeout": 30,
        "max_pixels": 2000,
    }
    config.update(overrides)
    return config


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def completion(text, finish_reason="stop", reasoning="hidden thoughts"):
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": text,
                            "reasoning_content": reasoning},
            }
        ]
    }


PAGE = {"page": 7, "data_url": "data:image/png;base64,QUJD"}


class TestConfig:
    def test_disabled_without_flag(self):
        assert vlm.read_vlm_config({}) is None
        assert vlm.read_vlm_config({"PDF2EPUB_VLM": "0"}) is None

    def test_defaults_and_env_overrides(self):
        config = vlm.read_vlm_config({
            "PDF2EPUB_VLM": "1", "COMMANDCODE_TOKEN": "tok",
            "PDF2EPUB_VLM_MODEL": "custom/vision", "PDF2EPUB_VLM_DPI": "200",
            "PDF2EPUB_VLM_CONCURRENCY": "8",
        })
        assert config["model"] == "custom/vision"
        assert config["dpi"] == 200
        assert config["concurrency"] == 8
        assert config["api_key"] == "tok"
        assert config["base_url"] == vlm.DEFAULT_BASE_URL
        assert config["timeout"] == 120

    def test_explicit_key_preferred_over_fallback(self):
        config = vlm.read_vlm_config({
            "PDF2EPUB_VLM": "1", "COMMANDCODE_TOKEN": "fallback",
            "PDF2EPUB_VLM_API_KEY": "explicit",
        })
        assert config["api_key"] == "explicit"

    def test_requires_a_key(self):
        with pytest.raises(vlm.VlmError, match="No VLM API key"):
            vlm.read_vlm_config({"PDF2EPUB_VLM": "1"})

    @pytest.mark.parametrize("value", ["0", "-3", "many", "1.5"])
    def test_rejects_bad_numbers(self, value):
        with pytest.raises(vlm.VlmError, match="PDF2EPUB_VLM_DPI"):
            vlm.read_vlm_config({"PDF2EPUB_VLM": "1", "PDF2EPUB_VLM_API_KEY": "k",
                                 "PDF2EPUB_VLM_DPI": value})

    def test_surrounding_whitespace_is_tolerated(self):
        config = vlm.read_vlm_config({"PDF2EPUB_VLM": "1",
                                      "PDF2EPUB_VLM_API_KEY": " k ",
                                      "PDF2EPUB_VLM_DPI": " 200 "})
        assert config["dpi"] == 200 and config["api_key"] == "k"

    def test_flag_accepts_common_truthy_spellings(self):
        for value in ["1", "true", "TRUE", "yes"]:
            assert vlm.read_vlm_config({"PDF2EPUB_VLM": value,
                                        "PDF2EPUB_VLM_API_KEY": "k"}) is not None

    def test_rejects_non_http_base_url(self):
        with pytest.raises(vlm.VlmError, match="must be an http"):
            vlm.read_vlm_config({"PDF2EPUB_VLM": "1", "PDF2EPUB_VLM_API_KEY": "k",
                                 "PDF2EPUB_VLM_BASE_URL": "ftp://example.com"})


class TestPayload:
    def test_request_shape(self):
        payload = vlm.build_payload("model-x", "prompt text", PAGE["data_url"], 1234)
        assert payload["model"] == "model-x"
        assert payload["max_tokens"] == 1234
        assert payload["temperature"] == 0
        content = payload["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "prompt text"}
        assert content[1]["image_url"]["url"] == PAGE["data_url"]

    def test_transcript_is_markdown_with_quotes(self):
        payload = vlm.build_payload("m", 'say "hi"', PAGE["data_url"], 10)
        assert json.loads(json.dumps(payload))["messages"][0]["content"][0]["text"] == 'say "hi"'


class TestExtract:
    def test_uses_content_and_ignores_reasoning(self):
        assert vlm.extract_markdown(completion("# Title")) == "# Title"

    def test_unwraps_bare_fence(self):
        assert vlm.extract_markdown(completion("```\n# T\n```")) == "# T"

    def test_keeps_inner_fences(self):
        text = "```python\nprint(1)\n```"
        assert vlm.extract_markdown(completion(text)) == text

    def test_supports_content_parts(self):
        payload = {"choices": [{"finish_reason": "stop",
                                "message": {"content": [{"text": "A"}, {"text": "B"}]}}]}
        assert vlm.extract_markdown(payload) == "AB"

    def test_empty_with_length_hint(self):
        with pytest.raises(vlm.VlmError, match="MAX_TOKENS"):
            vlm.extract_markdown(completion("", finish_reason="length"))

    def test_bad_shape(self):
        with pytest.raises(vlm.VlmError, match="unexpected response shape"):
            vlm.extract_markdown({"nope": True})


class TestTranscribe:
    def test_success_sends_auth_and_returns_text(self):
        seen = {}

        def opener(request, timeout=None):
            seen["url"] = request.full_url
            seen["auth"] = request.get_header("Authorization")
            seen["agent"] = request.get_header("User-agent")
            seen["body"] = json.loads(request.data.decode())
            return FakeResponse(completion("# Page seven"))

        text = vlm.transcribe_page(PAGE, make_config(), opener=opener)
        assert text == "# Page seven"
        assert seen["url"].endswith("/chat/completions")
        assert seen["auth"] == "Bearer test-key"
        assert seen["agent"] == "Mozilla/5.0"
        assert seen["body"]["messages"][0]["content"][1]["image_url"]["url"] == PAGE["data_url"]

    def test_retries_rate_limit_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(vlm.time, "sleep", lambda _s: None)
        calls = {"n": 0}

        def opener(request, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError(request.full_url, 429, "slow down",
                                             {}, io.BytesIO(b"rate limited"))
            return FakeResponse(completion("ok"))

        assert vlm.transcribe_page(PAGE, make_config(max_retries=2), opener=opener) == "ok"
        assert calls["n"] == 2

    def test_retries_empty_transcription(self, monkeypatch):
        monkeypatch.setattr(vlm.time, "sleep", lambda _s: None)
        calls = {"n": 0}

        def opener(request, timeout=None):
            calls["n"] += 1
            content = "" if calls["n"] == 1 else "recovered"
            return FakeResponse(completion(content))

        assert vlm.transcribe_page(PAGE, make_config(), opener=opener) == "recovered"
        assert calls["n"] == 2

    def test_hard_deadline_aborts_a_slow_body(self):
        class SlowHandler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{")
                self.wfile.flush()
                time.sleep(0.5)
                try:
                    self.wfile.write(b"}")
                except OSError:
                    pass

            def log_message(self, _format, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), SlowHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        config = make_config(base_url=f"http://127.0.0.1:{server.server_port}",
                             timeout=0.1, max_retries=0)
        started = time.monotonic()
        try:
            with pytest.raises(vlm.VlmError, match="timed out after 0.1s"):
                vlm.transcribe_page(PAGE, config)
            assert time.monotonic() - started < 0.4
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_gives_up_after_retries(self, monkeypatch):
        monkeypatch.setattr(vlm.time, "sleep", lambda _s: None)
        calls = {"n": 0}

        def opener(request, timeout=None):
            calls["n"] += 1
            raise urllib.error.HTTPError(request.full_url, 503, "down",
                                         {}, io.BytesIO(b"unavailable"))

        with pytest.raises(vlm.VlmError, match="HTTP 503"):
            vlm.transcribe_page(PAGE, make_config(max_retries=1), opener=opener)
        assert calls["n"] == 2

    def test_client_error_is_not_retried(self, monkeypatch):
        monkeypatch.setattr(vlm.time, "sleep", lambda _s: None)
        calls = {"n": 0}

        def opener(request, timeout=None):
            calls["n"] += 1
            raise urllib.error.HTTPError(request.full_url, 400, "bad request",
                                         {}, io.BytesIO(b"bad model"))

        with pytest.raises(vlm.VlmError, match="HTTP 400"):
            vlm.transcribe_page(PAGE, make_config(max_retries=3), opener=opener)
        assert calls["n"] == 1

    def test_network_error_reported(self, monkeypatch):
        monkeypatch.setattr(vlm.time, "sleep", lambda _s: None)

        def opener(request, timeout=None):
            raise urllib.error.URLError("no route")

        with pytest.raises(vlm.VlmError, match="network error"):
            vlm.transcribe_page(PAGE, make_config(max_retries=0), opener=opener)


class TestTranscribePages:
    def test_order_preserved_with_concurrency(self):
        pages = [{"page": n, "data_url": f"data:image/png;base64,{n}"} for n in range(1, 6)]
        progress = []

        def opener(request, timeout=None):
            body = json.loads(request.data.decode())
            label = body["messages"][0]["content"][1]["image_url"]["url"].rsplit(",", 1)[-1]
            return FakeResponse(completion(f"page {label}"))

        texts = vlm.transcribe_pages(pages, make_config(concurrency=4), opener=opener,
                                     progress=lambda done, total: progress.append((done, total)))
        assert texts == [f"page {n}" for n in range(1, 6)]
        assert progress == [(n, 5) for n in range(1, 6)]

    def test_single_page_and_empty_input(self):
        def opener(request, timeout=None):
            return FakeResponse(completion("only"))

        assert vlm.transcribe_pages([PAGE], make_config(concurrency=4), opener=opener) == ["only"]
        assert vlm.transcribe_pages([], make_config(), opener=opener) == []


class FakePdfium:
    """Stands in for pypdfium2 so page-range maths can be tested offline."""

    def __init__(self, total=5):
        self.pages = [types.SimpleNamespace(
            render=lambda scale, index=i: types.SimpleNamespace(
                to_pil=lambda: FakeImage(index))
        ) for i in range(total)]

    def PdfDocument(self, _path):  # noqa: N802 - mirrors pdfium/pypdfium2
        outer = self

        class Doc:
            def __len__(self):
                return len(outer.pages)

            def __getitem__(self, index):
                return outer.pages[index]

            def close(self):
                outer.closed = True

        return Doc()


class FakeImage:
    def __init__(self, index):
        self.size = (10 + index, 8)
        self.mode = "RGB"

    def convert(self, mode):
        return self

    def resize(self, size, resample=None):
        self.size = size
        return self

    def save(self, buffer, format=None):
        buffer.write(b"\x89PNG\r\n\x1a\n" + bytes([len(self.size)]))


class TestRenderPages:
    def test_page_range_and_count(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pypdfium2", FakePdfium(total=5))
        progress = []
        pages = vlm.render_pages("book.pdf", start_page=1, max_pages=2, dpi=144,
                                 progress=lambda done, total: progress.append((done, total)))
        assert [p["page"] for p in pages] == [2, 3]
        assert all(p["data_url"].startswith("data:image/png;base64,") for p in pages)
        assert progress == [(0, 2), (1, 2), (2, 2)]

    def test_default_renders_everything(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pypdfium2", FakePdfium(total=3))
        assert [p["page"] for p in vlm.render_pages("book.pdf")] == [1, 2, 3]

    def test_document_is_closed_even_when_range_is_invalid(self, monkeypatch):
        fake = FakePdfium(total=2)
        monkeypatch.setitem(sys.modules, "pypdfium2", fake)
        with pytest.raises(vlm.VlmError, match="past the end"):
            vlm.render_pages("book.pdf", start_page=9)
        assert getattr(fake, "closed", False) is True

    def test_start_past_end_is_an_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pypdfium2", FakePdfium(total=2))
        with pytest.raises(vlm.VlmError, match="past the end"):
            vlm.render_pages("book.pdf", start_page=9)

    def test_max_pixels_downscales(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pypdfium2", FakePdfium(total=1))
        pages = vlm.render_pages("book.pdf", dpi=600, max_pixels=2000)
        assert len(pages) == 1


class TestConvertPdfVlm:
    def test_requires_enabled_config(self, tmp_path):
        with pytest.raises(vlm.VlmError, match="not enabled"):
            vlm.convert_pdf_vlm("book.pdf", tmp_path, config=None)

    def test_writes_markdown_with_page_markers_and_progress_bars(
        self, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.setitem(sys.modules, "pypdfium2", FakePdfium(total=2))
        calls = {"n": 0}

        def opener(request, timeout=None):
            calls["n"] += 1
            return FakeResponse(completion(f"body {calls['n']}"))

        pdf = tmp_path / "issue.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        path = vlm.convert_pdf_vlm(pdf, tmp_path / "out",
                                   config=make_config(concurrency=1), opener=opener)
        text = path.read_text()
        assert path.name == "issue.md"
        assert calls["n"] == 2
        assert text.count("<!-- page ") == 2
        assert text.index("<!-- page 1 -->") < text.index("<!-- page 2 -->")
        assert "body 1" in text and "body 2" in text
        output = capsys.readouterr().out
        assert "Rendering pages [########################] 2/2" in output
        assert "Transcribed pages [########################] 2/2" in output
