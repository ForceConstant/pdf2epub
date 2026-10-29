# PDF2EPUB 📚

Convert PDF files to nicely structured Markdown and EPUB format with intelligent layout detection.

## ✨ Features

- 📖 Smart layout detection for books and academic papers
- 🔍 Advanced text extraction and OCR capabilities
- 📊 Table detection and formatting
- 🖼️ Image extraction and optimization
- 📝 Clean markdown output with preserved structure
- 📱 EPUB generation with customizable styling
- 🌐 Multi-language support
- ⚡ Non-interactive batch conversion with CLI metadata options
- 🚀 GPU acceleration support (NVIDIA & AMD)
- 🍎 Apple Silicon support

## 🛠️ Dependencies

- Python 3.10–3.14 (3.13 recommended, see below)
- PyTorch (with CUDA/ROCm support for GPU acceleration)
- marker-pdf==1.10.2
- transformers==4.57.6
- markdown==3.10.2
- latex2mathml==3.81.0

### ⚠️ Python version

Python 3.13 is recommended.

`marker-pdf` constrains `Pillow<11.0.0`, and Pillow only ships Python 3.14
wheels from 11.3.0 onward. On Python 3.10–3.13 every dependency installs as a
prebuilt wheel. On 3.14, pip has to build Pillow from source instead: this
works, but it is slower and requires a working C toolchain plus the image
library headers Pillow links against. On Debian/Ubuntu install them first:

```bash
sudo apt install libjpeg-dev zlib1g-dev libtiff-dev libfreetype6-dev libwebp-dev
```

Without these headers the install fails with
`RequiredDependencyException: The headers or library files could not be found for jpeg`.

## 💻 Installation

1. Create and activate a virtual environment.

On Linux/Mac:
```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

On Windows:
```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
```

2. Install Python dependencies (this installs PyTorch as well):
```bash
pip install -r requirements.txt
```

3. GPU acceleration (optional):

PyTorch is installed as a dependency in step 2. On Apple Silicon that wheel
already supports MPS, so no further action is needed. For a specific CUDA or
ROCm build, reinstall PyTorch using the selector at
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
For example, for AMD GPUs with ROCm:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

4. Verify GPU support:
```python
import torch
print(torch.__version__)  # PyTorch version
print(torch.cuda.is_available())  # Should return True for NVIDIA
print(torch.backends.mps.is_available())  # Should return True for Apple Silicon
print(torch.version.hip)  # Should print ROCm version for AMD
```

### 🐳 Docker

A CPU-only image can be built from the included `Dockerfile`:

```bash
docker build -t pdf2epub .
```

Run it with your PDFs mounted at `/data` and a model cache volume (marker-pdf
downloads its models on first run):

```bash
docker run --rm \
  -v "$(pwd)":/data \
  -v pdf2epub-models:/models \
  pdf2epub input.pdf
```

Metadata is prompted for interactively when a terminal is attached. Without
one, pdf2epub warns and falls back to defaults, so `-it` is optional. Passing
any of the metadata options (`--title`, `--author`, …) sets them explicitly and
skips the prompts entirely:

```bash
docker run --rm \
  -v "$(pwd)":/data \
  -v pdf2epub-models:/models \
  pdf2epub input.pdf --title "Dune" --author "Frank Herbert"
```

Tagged releases are also published to
`ghcr.io/overcuriousity/pdf2epub` by the Docker workflow.

## 🚀 Usage

### Basic Usage

Convert a single PDF file:
```bash
python main.py input.pdf
```

Convert all PDFs in a directory:
```bash
python main.py input_directory/
```

EPUB generation prompts interactively for metadata (title, author, language,
and so on; press Enter to accept each default). When a terminal is available
you can answer those prompts; otherwise pdf2epub prints a warning and uses the
default metadata, so nothing fails for lack of a TTY. Use `--skip-epub` to
produce only markdown without any prompts, or supply the metadata options below
to set the values directly (see [Batch processing](#batch-processing)).

### Metadata Options

Every EPUB metadata value can be set on the command line, which makes
non-interactive and batch conversion possible:

```bash
python main.py book.pdf --title "Dune" --author "Frank Herbert" --publisher "Ace"
```

| Option | Description | Default |
| --- | --- | --- |
| `--title` | Book title | The PDF file name |
| `--author` | Author(s); separate several names with `;` | `Unknown Author` |
| `--publisher` | Publisher | `PDF2EPUB` |
| `--language` | Language code, e.g. `en`, `de`, `fr` | `en` |
| `--rights` | Rights statement | `All rights reserved` |
| `--identifier` | Unique identifier | A stable UUID derived from the title |
| `--date` | Publication date as `YYYY-MM-DD` | Today |
| `-y`, `--yes` | Accept the defaults for everything not given | — |

Notes:

- `--author "Jane Doe; John Roe"` records two separate `dc:creator` entries.
  Commas are not treated as separators, so `"Doe, Jane"` stays one name.
- The title defaults to the PDF's file name rather than `Untitled Document`,
  so a batch run stays legible.
- `--identifier` is derived from the title and file name, so re-running the
  same conversion reproduces the same identifier instead of generating a new
  one each time.

### Batch Processing

`--title` and `--author` accept the placeholders `{filename}` (with extension)
and `{stem}` (without). They are filled in per file, so one command converts a
whole directory without the metadata being copied between books:

```bash
python main.py ./input/ --title "{stem}" --author "Unknown Author"
```

The PDF file name is also used to derive a unique `dc:identifier` per book.

Running without a terminal (cron, CI, piped input) no longer crashes with
`EOFError`: pdf2epub prints a warning and falls back to the defaults above.
Supply the metadata options to control the values, or `-y` to accept the
defaults deliberately.

### Advanced Options

```bash
python main.py [input_path] [output_path] [options]

Options:
  --max-pages INT          Maximum number of pages to process
  --start-page INT         Page number to start from
  --skip-epub              Skip EPUB generation, only create markdown
  --skip-md                Skip markdown generation, use existing markdown files
  --keep-intermediates     Keep Markdown, images, and conversion files in the work area
  --series STR             Set series directory; supports {stem} and {filename}
  --title STR              Book title (supports {stem} and {filename})
  --author STR             Author(s); separate several names with ";"
  --publisher STR          Publisher
  --language STR           Language code, e.g. en, de, fr
  --rights STR             Rights statement
  --identifier STR         Unique identifier
  --date STR               Publication date as YYYY-MM-DD
  -y, --yes                Accept default metadata instead of prompting
```

If `input_path` is omitted, all PDFs in `./input/` are processed.

### Examples

Process a specific range of pages:
```bash
python main.py book.pdf --start-page 10 --max-pages 50
```

Convert to markdown only:
```bash
python main.py thesis.pdf --skip-epub
```

Convert a directory unattended, one book per command:
```bash
python main.py ./books/ --title "{stem}" --author "Unknown Author" \
  --publisher "Archive.org" --language en < /dev/null
```

### Tests

```bash
pip install pytest
pytest tests/
```

### Output Structure

```
output_directory/
├── Make/
│   ├── Make 20250402.epub
│   └── Make 20250501.epub
```

The series directory is inferred from a recognizable trailing issue/date suffix
in the PDF filename (`Make 20250402.pdf` becomes `Make/Make 20250402.epub`).
Use `--series "Series Name"` when a filename cannot be inferred reliably.
After a successful conversion, generated Markdown and other working files are
removed from the hidden work area; `--keep-intermediates` retains them there
for review. `--skip-epub` always leaves the Markdown work files in place.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

Please ensure your code follows the existing style and includes appropriate tests.

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/overcuriousity/pdf2epub.git
cd pdf2epub
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install development dependencies:
```bash
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Known Issues

- Some image embedding might need manual adjustment
- Some complex mathematical equations might not be perfectly converted
- Certain PDF layouts with multiple columns may require manual adjustment
- Font detection might be imperfect in some cases

## 🙏 Acknowledgments

This project builds upon several excellent open-source libraries:
- [marker-pdf](https://github.com/VikParuchuri/marker) for PDF processing
- [mark2epub](https://github.com/AlexPof/mark2epub) for markdown conversion
- [PyTorch](https://pytorch.org/) for GPU acceleration
- [Transformers](https://huggingface.co/transformers) for advanced text processing

## Hosted vision-model conversion (`--vlm`)

The opt-in VLM path transcribes PDF pages with a hosted multimodal model instead
of running local layout/OCR models. It needs a CommandCode token and internet
access; the API key is passed into the local Docker container as an environment
variable. Set `PDF2EPUB_VLM_API_KEY` or `COMMANDCODE_TOKEN`, then run:

```bash
export PDF2EPUB_VLM_API_KEY=...
export PDF2EPUB_VLM_MAX_TOKENS=16384
export PDF2EPUB_VLM_CONCURRENCY=8
export PDF2EPUB_VLM_TIMEOUT=120
export PDF2EPUB_VLM_MAX_RETRIES=2
./scripts/pdf2epub.sh ./issue.pdf ./output --vlm --title '{stem}'
```

To keep substantial articles and drop blurbs, teasers, ads and contents pages,
add `--articles-only`. See [article filtering](#article-filtering-pdf-or-existing-epub)
for PDF and existing-EPUB examples.

Each page request has a 120-second hard deadline and up to two retries by
default. Timed-out pages and empty transcriptions are retried; if a page still
fails, conversion stops with its page number instead of waiting indefinitely.
Increase `PDF2EPUB_VLM_TIMEOUT` for providers that need longer per-page
responses, or set `PDF2EPUB_VLM_MAX_RETRIES` to change the retry count. Each
successful page is checkpointed in the issue's hidden work directory. If a run
fails, rerunning the same command resumes from completed pages; checkpoints are
automatically discarded when the conversion succeeds and its work directory is
cleaned up. Checkpoints are isolated by PDF contents, model, render settings,
token limit, and prompt so stale transcriptions are not reused.

The default model is `deepseek/deepseek-v4.1-flash`. On a dense 8-page test,
concurrency 8 completed in 117 seconds at about $0.003 per page. VLM output is
split into article chapters at level-1 headings (the prompt asks the model to
use level-1 headings for article starts); the complete page transcript is also
kept as `<pdf-stem>.md`. The PDF's first page is rendered as the EPUB cover and
is omitted from the article text. This path still does not extract or place
in-body figures, and complex tables depend on the model's Markdown output.

### Article filtering (PDF or existing EPUB)

`--articles-only` runs one optional **text-only chapter filter** over the finished
EPUB. `--substantive-only` remains a compatibility alias for exactly the same
filter; passing both still runs it only once. It drops product announcements/spec
blurbs, news roundups, download catalogs,
teasers, issue previews, ads, contents/navigator pages and other non-reading matter.
It keeps developed analysis/interviews, useful tutorials, and actual hands-on
reviews. Long collections of blurbs are not articles; short useful tutorials can be.

```bash
# Add to a new VLM conversion (also works after local OCR or --skip-md):
./scripts/pdf2epub.sh issue.pdf ./output --vlm --articles-only --yes

# Filter an already-converted EPUB without rendering/transcribing the PDF again:
./scripts/pdf2epub.sh './Beat Magazine #20260250.epub' ./articles --articles-only
# Result: ./articles/Beat Magazine #20260250.epub

# Native CLI, no output directory: writes issue-substantive.epub beside the source.
python main.py issue.epub --articles-only
```

Uses the existing `PDF2EPUB_VLM_*` text API settings and `COMMANDCODE_TOKEN` fallback;
this flag alone does **not** switch PDF OCR to vision. Each chapter receives a
separate bounded text request, concurrently up to `PDF2EPUB_VLM_CONCURRENCY`.
Classification uses up to 18,000 characters of opening/ending text and at least
16,384 output tokens (including reasoning). Only `kind=article` with confidence
>=0.80 is included. This confidence is a model judgment, not calibrated certainty.
A chapter does not need both its own opening and conclusion: substantive
continuation chapters can stay. This replaces the old completeness-only filter;
there is no separate earlier 90%-confidence completeness gate. Filtering also
works after local OCR or with retained Markdown, without requiring `--vlm`.

For output/cache compatibility with 0.3.9, existing EPUBs without an explicit
output directory still use the `-substantive.epub` suffix, and the adjacent
`*.substantive-filter.json` records **every chapter's decision and
reason**, survives normal work-directory cleanup, and checkpoints completed calls.
An unchanged source/model/prompt can resume without reclassifying completed
chapters. Any API/parse failure or zero retained chapters leaves the EPUB untouched.
For new PDF conversions the raw EPUB is staged in the hidden work area; only a
successfully filtered EPUB is published at the final output path. Use
`--keep-intermediates` to retain that staged raw EPUB. For EPUB input the original
is unchanged unless the output directory explicitly selects the same file
(then a non-overwriting `*.unfiltered.epub` backup is required). Reading order, EPUB 3
navigation and legacy NCX are updated together. Cover, retained article content,
styles and images are preserved. This pass cannot split merged stories or trim
blurbs inside an otherwise retained chapter. Review a representative issue before
large batch runs. Existing behavior remains unchanged without the flag.

The launcher prints the effective converter flags before starting Docker, including
its automatic `--yes`. Values are shell-quoted to make spaces and literal
placeholders unambiguous, for example:

```text
Converter flags: --yes --vlm --articles-only --title \{stem\}
```

Docker environment arguments and API-key values are not included in that diagnostic.

### Verify that article filtering completed

Capture both stdout and stderr, including launcher diagnostics and failures:

```bash
./scripts/pdf2epub.sh issue.pdf ./output --vlm --articles-only > test.log 2>&1
```

Require `Article filtering: enabled`, `Article filter enabled: reading ...`,
completed chapter counts and a `Substantive chapters: KEPT/TOTAL; report: ...`
summary. PDF runs also print `Article-filtered EPUB ready: ...` after successful
filtering. `EPUB creation complete` by itself refers to EPUB assembly, not proof
that classification succeeded. Check the process exit status and adjacent report.
Raw HTML void tags such as `<br>` in Markdown tables are normalized to valid
XHTML when generating chapters. Existing EPUBs with the same void-tag defect can
be normalized in memory for filtering without changing the original source.

Math conversion operates only on HTML text ranges, never across tags/attributes
or inside code, scripts, comments or existing MathML. Ordinary prices such as
`$10 and $20` remain currency; genuine `$x^2$` and `$$x^2$$` expressions still
become MathML. Inline math delimiters cannot border whitespace or close immediately
before a digit. If an older staged EPUB already contains corrupted MathML, rebuild
from retained Markdown with `--skip-md --articles-only`; don't retranscribe the PDF
or try to guess lost markup from the damaged EPUB.

### Article-free publications (0.3.15+)

A completed `--articles-only` pass can legitimately select no chapters (for
example, a publisher catalog made up of solicitations and previews). This now
exits successfully with `Substantive chapters: 0/N; report: ...` and
`Article filter complete: no qualifying articles; no EPUB produced; existing
files unchanged.` It does **not** create an empty book, publish the unfiltered
archive, or overwrite an existing destination. PDF conversion work files and
checkpoints remain available for review.

The adjacent `*.substantive-filter.json` records `status: "no_articles"`,
`kept_chapters: 0`, the completed chapter decisions, and `output_written: false`.
A published filtered archive instead records `status: "filtered"` and
`output_written: true`. Interrupted/API-failed passes are not completed outcomes;
a zero-selection pass with unreadable XHTML remains an error. Refiltering the
same saved EPUB reuses matching cached decisions without paid retranscription.
Automation must validate the completed report rather than treating any missing
EPUB or partial report as success.
