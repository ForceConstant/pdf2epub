"""Page-level VLM conversion: render each PDF page and transcribe it to Markdown.

This is an alternative to the Surya/marker path in :mod:`modules.pdf2md`. It
targets hosted multimodal models that are already available to this project
(CommandCode / DeepSeek V4.1 Flash), so the expensive layout and OCR work moves
off the local CPU/GPU entirely. The output layout matches the marker path, so
`modules.mark2epub` builds the EPUB unchanged.
"""

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_BASE_URL = "https://api.commandcode.ai/provider/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"

PAGE_PROMPT = (
    "Convert this single page image into GitHub-flavored Markdown.\n"
    "Transcribe every visible piece of text exactly, in natural reading order.\n"
    "Preserve headings, lists, block quotes, code blocks and inline emphasis.\n"
    "Render tables as Markdown tables. Render equations as LaTeX ($...$ inline, "
    "$$...$$ display).\n"
    "Keep image captions as [Figure: caption] on their own line.\n"
    "Do NOT add commentary, page numbers, running heads, or explanations.\n"
    "Output ONLY the Markdown for this page."
)


class VlmError(RuntimeError):
    """Raised for configuration problems and unrecoverable API failures."""


def read_vlm_config(environ=None):
    """Return validated VLM settings, or None when the VLM path is disabled.

    Enabling uses `--vlm`/`PDF2EPUB_VLM=1`; the API key falls back to
    COMMANDCODE_TOKEN so the account already configured for this gateway works.
    """
    env = os.environ if environ is None else environ
    enabled = env.get("PDF2EPUB_VLM", "").lower() in {"1", "true", "yes"}
    if not enabled:
        return None

    def positive_int(name, default):
        raw = env.get(name, "").strip()
        if not raw:
            return default
        if not raw.isdigit() or int(raw) < 1:
            raise VlmError(f"{name} must be a positive integer (got {raw!r})")
        return int(raw)

    model = env.get("PDF2EPUB_VLM_MODEL", "").strip() or DEFAULT_MODEL
    base_url = env.get("PDF2EPUB_VLM_BASE_URL", "").strip() or DEFAULT_BASE_URL
    if not base_url.startswith(("http://", "https://")):
        raise VlmError("PDF2EPUB_VLM_BASE_URL must be an http(s) URL")
    api_key = env.get("PDF2EPUB_VLM_API_KEY", "").strip() or env.get(
        "COMMANDCODE_TOKEN", ""
    ).strip()
    if not api_key:
        raise VlmError(
            "No VLM API key: set PDF2EPUB_VLM_API_KEY or COMMANDCODE_TOKEN"
        )

    return {
        "model": model,
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "max_tokens": positive_int("PDF2EPUB_VLM_MAX_TOKENS", 8192),
        "dpi": positive_int("PDF2EPUB_VLM_DPI", 150),
        "concurrency": positive_int("PDF2EPUB_VLM_CONCURRENCY", 4),
        "max_retries": positive_int("PDF2EPUB_VLM_MAX_RETRIES", 2),
        "timeout": positive_int("PDF2EPUB_VLM_TIMEOUT", 300),
        "max_pixels": positive_int("PDF2EPUB_VLM_MAX_PIXELS", 2000),
    }


def render_pages(pdf_path, start_page=0, max_pages=None, dpi=150, max_pixels=2000):
    """Render PDF pages to PNG data URLs, optionally resized to a pixel budget."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - dependency of surya
        raise VlmError("pypdfium2 is required for the VLM path") from exc

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        total = len(document)
        first = max(0, start_page or 0)
        last = total if max_pages is None else min(total, first + max_pages)
        if first >= total:
            raise VlmError(f"start page {first} is past the end of {total} pages")

        scale = dpi / 72.0
        pages = []
        for index in range(first, last):
            image = document[index].render(scale=scale).to_pil().convert("RGB")
            longest = max(image.size)
            if longest > max_pixels:
                ratio = max_pixels / longest
                image = image.resize(
                    (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
                    resample=1,  # PIL.Image.LANCZOS without importing PIL here
                )
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            pages.append({"page": index + 1,
                          "data_url": f"data:image/png;base64,{encoded}"})
        return pages
    finally:
        document.close()


def build_payload(model, prompt, data_url, max_tokens):
    """Build an OpenAI-compatible multimodal chat request body."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }


def extract_markdown(response):
    """Pull the page Markdown out of a chat-completion response.

    Reasoning models return chain-of-thought in a separate field; only the
    answer is used. A lone fenced code block is unwrapped so the page does not
    end up nested inside ``` fences in the EPUB.
    """
    try:
        choice = response["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise VlmError(f"unexpected response shape: {str(response)[:200]}") from exc

    content = choice.get("message", {}).get("content") or ""
    if isinstance(content, list):  # some providers return content parts
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    content = content.strip()

    if not content:
        reason = choice.get("finish_reason")
        if reason == "length":
            raise VlmError(
                "model hit the token limit before producing text; raise "
                "PDF2EPUB_VLM_MAX_TOKENS"
            )
        raise VlmError("model returned an empty page transcription")

    lines = content.splitlines()
    # Only a bare fence is treated as a wrapper; "```python" is a real code
    # block and must survive into the EPUB.
    if (
        len(lines) >= 3
        and lines[0].strip() == "```"
        and lines[-1].strip() == "```"
    ):
        content = "\n".join(lines[1:-1]).strip()
    return content


def transcribe_page(page, config, prompt=PAGE_PROMPT, opener=None):
    """Transcribe one rendered page, retrying transient failures."""
    payload = build_payload(
        config["model"], prompt, page["data_url"], config["max_tokens"]
    )
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{config['base_url']}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
            # The provider rejects requests without a browser-ish agent.
            "User-Agent": "Mozilla/5.0",
        },
    )
    open_url = opener or urllib.request.urlopen
    attempts = config["max_retries"] + 1
    for attempt in range(attempts):
        try:
            with open_url(request, timeout=config["timeout"]) as response:
                return extract_markdown(json.load(response))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == attempts - 1:
                raise VlmError(f"HTTP {exc.code} from {config['base_url']}: {detail}")
        except urllib.error.URLError as exc:
            if attempt == attempts - 1:
                raise VlmError(f"network error talking to {config['base_url']}: {exc}")
        except TimeoutError:
            if attempt == attempts - 1:
                raise VlmError(f"timed out after {config['timeout']}s per page")
        time.sleep(2 * (attempt + 1))
    raise VlmError("unreachable")


def transcribe_pages(pages, config, prompt=PAGE_PROMPT, opener=None, progress=None):
    """Transcribe pages concurrently, preserving page order in the result."""
    if not pages:
        return []
    workers = max(1, min(config["concurrency"], len(pages)))
    results = [None] * len(pages)
    if workers == 1:
        for index, page in enumerate(pages):
            results[index] = transcribe_page(page, config, prompt, opener)
            if progress:
                progress(index + 1, len(pages))
        return results

    def work(index):
        text = transcribe_page(pages[index], config, prompt, opener)
        if progress:
            progress(index + 1, len(pages))
        return index, text

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, text in pool.map(work, range(len(pages))):
            results[index] = text
    return results


def convert_pdf_vlm(pdf_path, output_dir, max_pages=None, start_page=None,
                    config=None, prompt=PAGE_PROMPT, opener=None):
    """Write `<stem>.md` for a PDF using the VLM path; mirrors the marker layout."""
    config = config or read_vlm_config()
    if config is None:
        raise VlmError("VLM path is not enabled")

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    print(f"VLM transcription: model={config['model']} dpi={config['dpi']} "
          f"concurrency={config['concurrency']}", flush=True)

    pages = render_pages(pdf_path, start_page=start_page, max_pages=max_pages,
                         dpi=config["dpi"], max_pixels=config["max_pixels"])
    print(f"Rendered {len(pages)} page(s) from {pdf_path.name}", flush=True)

    def report(done, total):
        print(f"  transcribed {done}/{total} pages", flush=True)

    texts = transcribe_pages(pages, config, prompt, opener, progress=report)

    chunks = []
    for page, text in zip(pages, texts):
        chunks.append(f"<!-- page {page['page']} -->\n\n{text}\n")
    markdown = "\n\n".join(chunks).strip() + "\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{pdf_path.stem}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    print(f"Markdown saved to: {markdown_path}", flush=True)
    return markdown_path
