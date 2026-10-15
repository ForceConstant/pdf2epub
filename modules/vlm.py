"""Page-level VLM conversion: render each PDF page and transcribe it to Markdown.

This is an alternative to the Surya/marker path in :mod:`modules.pdf2md`. It
targets hosted multimodal models that are already available to this project
(CommandCode / DeepSeek V4.1 Flash), so the expensive layout and OCR work moves
off the local CPU/GPU entirely. The output layout matches the marker path, so
`modules.mark2epub` builds the EPUB unchanged.
"""

import base64
import http.client
import io
import json
import os
import re
import socket
import threading
import unicodedata
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "https://api.commandcode.ai/provider/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"

PAGE_PROMPT = (
    "Convert this single page image into GitHub-flavored Markdown.\n"
    "Transcribe every visible piece of text exactly, in natural reading order.\n"
    "Preserve headings, lists, block quotes, code blocks and inline emphasis.\n"
    "Use a level-1 heading (#) for the start/title of a new article; use level-2 "
    "and lower headings for sections within an article.\n"
    "Render tables as Markdown tables. Render equations as LaTeX ($...$ inline, "
    "$$...$$ display).\n"
    "Keep image captions as [Figure: caption] on their own line.\n"
    "Do NOT add commentary, page numbers, running heads, or explanations.\n"
    "Output ONLY the Markdown for this page."
)


def _report_progress(label, done, total, width=24):
    """Write a compact in-place progress bar for CLI/container logs."""
    done = min(max(done, 0), total)
    filled = int(width * done / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{label} [{bar}] {done}/{total}", end="", flush=True)


def _slug(text, fallback="chapter"):
    """Return a short, filesystem-safe slug for generated chapter filenames."""
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return (value[:60].rstrip("-") or fallback)


def _without_cover_page(markdown):
    """Drop page 1's transcription; it is represented by the EPUB cover image."""
    return re.sub(
        r"(?ms)^<!-- page 1 -->\s*.*?(?=^<!-- page \d+ -->|\Z)", "", markdown
    ).strip()


def split_markdown_into_chapters(markdown, fallback_title):
    """Split page transcription at article-title headings, preserving section text.

    VLM output is instructed to mark article starts as H1. If it returns no H1,
    H2 headings are used as a conservative fallback; if it returns no headings,
    the body remains a single chapter.
    """
    body = _without_cover_page(markdown)
    body = re.sub(r"(?m)^<!-- page \d+ -->\s*$", "", body).strip()
    lines = body.splitlines()
    heading_re = re.compile(r"^(#{1,2})\s+(.+?)\s*#*\s*$")
    h1_indices = [i for i, line in enumerate(lines)
                  if (m := heading_re.match(line)) and len(m.group(1)) == 1]
    level = 1 if h1_indices else 2
    starts = []
    for i, line in enumerate(lines):
        match = heading_re.match(line)
        if match and len(match.group(1)) == level:
            starts.append((i, match.group(2).strip()))

    if not starts:
        return [(fallback_title, body or fallback_title)]

    chapters = []
    preamble = "\n".join(lines[:starts[0][0]]).strip()
    if preamble:
        chapters.append(("Front Matter", preamble))
    for pos, (line_index, title) in enumerate(starts):
        end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        content = "\n".join(lines[line_index:end]).strip()
        if content:
            chapters.append((title, content))
    return chapters or [(fallback_title, body or fallback_title)]


def save_cover_image(pdf_path, images_dir, dpi=150, max_pixels=2400):
    """Render the PDF's first page as the EPUB's photographic cover asset."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - dependency of marker
        raise VlmError("pypdfium2 is required to render the PDF cover") from exc

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        if not len(document):
            raise VlmError("cannot create a cover from a PDF with no pages")
        image = document[0].render(scale=dpi / 72.0).to_pil().convert("RGB")
        longest = max(image.size)
        if longest > max_pixels:
            ratio = max_pixels / longest
            image = image.resize(
                (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
                resample=1,
            )
        images_dir.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        (images_dir / "cover.jpg").write_bytes(buffer.getvalue())
    finally:
        document.close()


def save_chapter_files(markdown, output_dir, pdf_stem):
    """Write one Markdown file per inferred article and return EPUB chapter rows."""
    chapters = split_markdown_into_chapters(markdown, pdf_stem)
    rows = []
    used = set()
    for index, (title, content) in enumerate(chapters, start=1):
        base = f"{pdf_stem}-chapter-{index:03d}-{_slug(title)}"
        filename = f"{base}.md"
        while filename.lower() in used:
            base += "-2"
            filename = f"{base}.md"
        used.add(filename.lower())
        (output_dir / filename).write_text(content.rstrip() + "\n", encoding="utf-8")
        rows.append({"markdown": filename, "css": "", "title": title})
    return rows


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
        "timeout": positive_int("PDF2EPUB_VLM_TIMEOUT", 120),
        "max_pixels": positive_int("PDF2EPUB_VLM_MAX_PIXELS", 2000),
    }


def render_pages(pdf_path, start_page=0, max_pages=None, dpi=150, max_pixels=2000,
                 progress=None):
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
        page_count = last - first
        if progress:
            progress(0, page_count)
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
            if progress:
                progress(len(pages), page_count)
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
    """Transcribe one page with a hard per-attempt deadline and retries."""
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
    attempts = config["max_retries"] + 1
    for attempt in range(attempts):
        try:
            if opener:
                with opener(request, timeout=config["timeout"]) as response:
                    result = json.load(response)
            else:
                status, result = _post_json_with_deadline(
                    request, body, config["timeout"]
                )
                if status < 200 or status >= 300:
                    detail = result.decode("utf-8", "replace")[:300]
                    retryable = status == 429 or 500 <= status < 600
                    if not retryable or attempt == attempts - 1:
                        raise VlmError(
                            f"HTTP {status} from {config['base_url']}: {detail}"
                        )
                    time.sleep(2 * (attempt + 1))
                    continue
            return extract_markdown(result)
        except json.JSONDecodeError as exc:
            if attempt == attempts - 1:
                raise VlmError(
                    f"page {page['page']} returned invalid JSON after {attempts} attempts"
                ) from exc
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == attempts - 1:
                raise VlmError(f"HTTP {exc.code} from {config['base_url']}: {detail}")
        except urllib.error.URLError as exc:
            if attempt == attempts - 1:
                raise VlmError(f"network error talking to {config['base_url']}: {exc}")
        except (TimeoutError, socket.timeout) as exc:
            if attempt == attempts - 1:
                raise VlmError(
                    f"page {page['page']} timed out after {config['timeout']}s "
                    f"on each of {attempts} attempts"
                ) from exc
            print(
                f"\nPage {page['page']} timed out after {config['timeout']}s; "
                f"retrying ({attempt + 2}/{attempts})",
                flush=True,
            )
        except VlmError as exc:
            if str(exc) != "model returned an empty page transcription" or attempt == attempts - 1:
                raise
            print(
                f"\nPage {page['page']} returned an empty transcription; "
                f"retrying ({attempt + 2}/{attempts})",
                flush=True,
            )
        time.sleep(2 * (attempt + 1))
    raise VlmError("unreachable")


def _post_json_with_deadline(request, body, timeout):
    """POST JSON and abort the underlying socket at a wall-clock deadline.

    urllib's socket timeout is an inactivity timeout, so a server that trickles
    bytes can keep a request alive indefinitely. Closing the HTTP connection
    from a timer enforces an actual upper bound for both headers and body reads.
    """
    parsed = urlsplit(request.full_url)
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    expired = threading.Event()

    def abort():
        expired.set()
        if connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        connection.close()

    timer = threading.Timer(timeout, abort)
    timer.daemon = True
    timer.start()
    try:
        target = parsed.path or "/"
        if parsed.query:
            target += f"?{parsed.query}"
        headers = {
            "Authorization": request.get_header("Authorization"),
            "Content-Type": request.get_header("Content-type") or "application/json",
            "User-Agent": request.get_header("User-agent") or "Mozilla/5.0",
        }
        connection.request("POST", target, body=body, headers=headers)
        response = connection.getresponse()
        response_body = response.read()
        if expired.is_set():
            raise TimeoutError(f"request exceeded its {timeout}s deadline")
        parsed_body = (
            response_body
            if response.status < 200 or response.status >= 300
            else json.loads(response_body)
        )
        return response.status, parsed_body
    except (http.client.HTTPException, OSError) as exc:
        if expired.is_set():
            raise TimeoutError(f"request exceeded its {timeout}s deadline") from exc
        raise urllib.error.URLError(exc) from exc
    finally:
        timer.cancel()
        connection.close()


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
        return index, transcribe_page(pages[index], config, prompt, opener)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, index) for index in range(len(pages))]
        completed = 0
        for future in as_completed(futures):
            index, text = future.result()
            results[index] = text
            completed += 1
            if progress:
                progress(completed, len(pages))
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

    def report(label):
        def update(done, total):
            _report_progress(label, done, total)
        return update

    render_progress = report("Rendering pages")
    try:
        pages = render_pages(pdf_path, start_page=start_page, max_pages=max_pages,
                             dpi=config["dpi"], max_pixels=config["max_pixels"],
                             progress=render_progress)
    finally:
        print(flush=True)
    print(f"Rendered {len(pages)} page(s) from {pdf_path.name}", flush=True)

    transcribe_progress = report("Transcribed pages")
    try:
        texts = transcribe_pages(pages, config, prompt, opener,
                                 progress=transcribe_progress)
    finally:
        print(flush=True)

    chunks = []
    for page, text in zip(pages, texts):
        chunks.append(f"<!-- page {page['page']} -->\n\n{text}\n")
    markdown = "\n\n".join(chunks).strip() + "\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{pdf_path.stem}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    print(f"Markdown saved to: {markdown_path}", flush=True)

    chapter_rows = save_chapter_files(markdown, output_dir, pdf_path.stem)
    images_dir = output_dir / "images"
    save_cover_image(pdf_path, images_dir, dpi=config["dpi"])

    # mark2epub uses this manifest to distinguish generated article chapters
    # from the full transcript retained above for reference.
    description_path = output_dir / "description.json"
    try:
        description = json.loads(description_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        description = {}
    description["chapters"] = chapter_rows
    description["cover_image"] = description.get("cover_image") or "cover.jpg"
    description.setdefault("metadata", {})
    description.setdefault("default_css", ["style.css"])
    description_path.write_text(
        json.dumps(description, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Created {len(chapter_rows)} article chapter(s)", flush=True)
    print(f"Rendered EPUB cover to: {images_dir / 'cover.jpg'}", flush=True)
    return markdown_path
