#!/usr/bin/env bash
# Run the fork's PDF converter in a disposable container (local Docker daemon).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: pdf2epub.sh INPUT [OUTPUT_DIR] [PDF2EPUB_OPTIONS...]

INPUT is a host PDF file or directory of PDFs.
OUTPUT_DIR defaults to ./output; results go into OUTPUT_DIR/<pdf-stem>/.
All remaining arguments are forwarded unchanged to pdf2epub.
Runs unattended (--yes), removes the container on exit, and keeps the model cache.

Examples:
  pdf2epub.sh "./Dune.pdf" ./converted --title "Dune" --author "Frank Herbert"
  pdf2epub.sh ./books ./converted --title '{stem}' --author "Unknown Author"
  pdf2epub.sh ./book.pdf --max-pages 20 --language en
  pdf2epub.sh ./book.pdf --skip-epub
  PDF2EPUB_GPU=nvidia pdf2epub.sh ./book.pdf --title "Dune" --author "Frank Herbert"

Environment overrides:
  PDF2EPUB_GPU        none (default) or nvidia (adds --gpus all)
  PDF2EPUB_IMAGE      Default: ghcr.io/forceconstant/pdf2epub:latest
                     With nvidia: ghcr.io/forceconstant/pdf2epub-nvidia:latest
                     Checks for updates each run; explicit version pins allowed
  PDF2EPUB_CACHE_DIR  Default: ${XDG_CACHE_HOME:-$HOME/.cache}/pdf2epub
  PDF2EPUB_BATCH_SIZE Global batch size; unset preserves converter defaults
  PDF2EPUB_{LAYOUT,RECOGNITION,DETECTION,TABLE,EQUATION,OCR_ERROR}_BATCH_SIZE
                     Per-stage positive integer; overrides global batch size
  PDF2EPUB_MODEL_DEVICE  auto (default), cpu, cuda, or mps
  PDF2EPUB_MODEL_OFFLOAD  1: lazy CPU loading and call-scoped CUDA residency
                        0 or unset: keep models resident (default)
  PDF2EPUB_{LAYOUT,RECOGNITION,DETECTION,TABLE,OCR_ERROR}_DEVICE
                     Per-model device; overrides global model device
  PYTORCH_ALLOC_CONF / PYTORCH_CUDA_ALLOC_CONF  Optional allocator settings

Vision-model path (hosted page transcription; use the CLI flag --vlm):
  PDF2EPUB_VLM        1 to transcribe pages with a hosted vision model
  PDF2EPUB_VLM_MODEL  Model id (default: deepseek/deepseek-v4.1-flash)
  PDF2EPUB_VLM_BASE_URL  OpenAI-compatible base (default: CommandCode)
  PDF2EPUB_VLM_API_KEY   Bearer token; falls back to COMMANDCODE_TOKEN
  PDF2EPUB_VLM_MAX_TOKENS, PDF2EPUB_VLM_DPI, PDF2EPUB_VLM_CONCURRENCY,
  PDF2EPUB_VLM_MAX_RETRIES, PDF2EPUB_VLM_TIMEOUT, PDF2EPUB_VLM_MAX_PIXELS
                     Tuning for the vision path
                     NOTE: the API key is passed as a container env var and is
                     therefore visible in the local docker command line.

Requires Bash and Docker. Run on the machine hosting your Docker daemon;
input, output, and cache paths are bind-mounted from that machine.
NVIDIA mode needs NVIDIA Container Toolkit and a compatible host driver.
Models download on first use and are reused afterward.
EOF
}

fail() { printf 'pdf2epub: %s\n' "$*" >&2; exit 1; }

if [[ $# -eq 0 ]]; then usage >&2; exit 2; fi
if [[ $1 == --help || $1 == -h ]]; then usage; exit 0; fi
command -v docker >/dev/null 2>&1 || fail 'Docker was not found in PATH.'

input=$1
shift
output=./output
if [[ $# -gt 0 && $1 != -* ]]; then
  output=$1
  shift
fi
# Optional separator before the converter's flags.
if [[ $# -gt 0 && $1 == -- ]]; then shift; fi

if [[ -d $input ]]; then
  input_mount=$(cd -- "$input" && pwd -P)
  container_input=/input
elif [[ -f $input ]]; then
  [[ $input == *.[pP][dD][fF] ]] || fail "Input is not a PDF: $input"
  input_mount=$(cd -- "$(dirname -- "$input")" && pwd -P)
  container_input="/input/$(basename -- "$input")"
else
  fail "Input file or directory does not exist: $input"
fi

gpu_args=()
case ${PDF2EPUB_GPU:-none} in
  none) default_image=ghcr.io/forceconstant/pdf2epub:latest ;;
  nvidia)
    default_image=ghcr.io/forceconstant/pdf2epub-nvidia:latest
    gpu_args=(--gpus all)
    ;;
  *) fail 'PDF2EPUB_GPU must be none or nvidia.' ;;
esac
image=${PDF2EPUB_IMAGE:-$default_image}
# Only forward documented tuning variables; arrays keep values literal.
tuning_args=()
for name in PDF2EPUB_BATCH_SIZE PDF2EPUB_MODEL_DEVICE PDF2EPUB_MODEL_OFFLOAD \
  PDF2EPUB_LAYOUT_BATCH_SIZE PDF2EPUB_RECOGNITION_BATCH_SIZE \
  PDF2EPUB_DETECTION_BATCH_SIZE PDF2EPUB_TABLE_BATCH_SIZE \
  PDF2EPUB_EQUATION_BATCH_SIZE PDF2EPUB_OCR_ERROR_BATCH_SIZE \
  PDF2EPUB_LAYOUT_DEVICE PDF2EPUB_RECOGNITION_DEVICE PDF2EPUB_DETECTION_DEVICE \
  PDF2EPUB_TABLE_DEVICE PDF2EPUB_OCR_ERROR_DEVICE \
  PDF2EPUB_VLM PDF2EPUB_VLM_MODEL PDF2EPUB_VLM_BASE_URL PDF2EPUB_VLM_API_KEY \
  PDF2EPUB_VLM_MAX_TOKENS PDF2EPUB_VLM_DPI PDF2EPUB_VLM_CONCURRENCY \
  PDF2EPUB_VLM_MAX_RETRIES PDF2EPUB_VLM_TIMEOUT PDF2EPUB_VLM_MAX_PIXELS \
  PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF; do
  if [[ -v $name ]]; then tuning_args+=(--env "$name=${!name}"); fi
done
cache=${PDF2EPUB_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/pdf2epub}
mkdir -p -- "$output" "$cache"
output_mount=$(cd -- "$output" && pwd -P)
cache_mount=$(cd -- "$cache" && pwd -P)
[[ -w $output_mount ]] || fail "Output directory is not writable: $output_mount"
[[ -w $cache_mount ]] || fail "Model cache is not writable: $cache_mount"

printf 'Image: %s\nOutput: %s\nModel cache: %s\n' \
  "$image" "$output_mount" "$cache_mount" >&2

# Arrays/quoted arguments preserve spaces, semicolons, and literal placeholders.
# Use your UID/GID so generated files are not owned by root.
# No -it: --yes intentionally makes metadata generation non-interactive.
exec docker run --rm --init --pull=always \
  "${gpu_args[@]}" \
  "${tuning_args[@]}" \
  --user "$(id -u):$(id -g)" \
  --volume "$input_mount:/input:ro" \
  --volume "$output_mount:/output" \
  --volume "$cache_mount:/models" \
  --env HOME=/models \
  --env XDG_CACHE_HOME=/models/cache \
  --env HF_HOME=/models/huggingface \
  --env TORCH_HOME=/models/torch \
  "$image" "$container_input" /output --yes "$@"
