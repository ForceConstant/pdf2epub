#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import sys
from pathlib import Path
import modules.pdf2md as pdf2md
import modules.mark2epub as mark2epub
import modules.metadata as metadata_lib
import modules.vlm as vlm_lib
import modules.chapter_filter as chapter_filter
import torch


def add_metadata_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the EPUB metadata options.

    Every option takes a value that may contain the placeholders {filename}
    and {stem}, which are replaced with the name of the PDF being processed so
    that one command line can convert a whole directory.
    """
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Book title (default: the PDF file name)'
    )
    parser.add_argument(
        '--author',
        type=str,
        default=None,
        help='Author(s); separate several names with ";" to list them separately'
    )
    parser.add_argument(
        '--publisher',
        type=str,
        default=None,
        help='Publisher (default: PDF2EPUB)'
    )
    parser.add_argument(
        '--language',
        type=str,
        default=None,
        help='Language code, e.g. en, de, fr (default: en)'
    )
    parser.add_argument(
        '--rights',
        type=str,
        default=None,
        help='Rights statement (default: All rights reserved)'
    )
    parser.add_argument(
        '--identifier',
        type=str,
        default=None,
        help='Unique identifier (default: a stable UUID derived from the title)'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Publication date as YYYY-MM-DD (default: today)'
    )


def collect_metadata_args(args: argparse.Namespace) -> dict:
    """Turn the parsed metadata options into a mapping for build_metadata.

    Returns an empty mapping when none were supplied, which lets EPUB
    generation fall back to the interactive prompts.
    """
    values = {}
    for flag, _key in metadata_lib.METADATA_FIELDS:
        value = getattr(args, flag, None)
        if value is not None:
            values[flag] = value

    return values


def metadata_from_args(args: argparse.Namespace, pdf_path: Path) -> dict:
    """Validate the metadata options for one PDF, expanding placeholders."""
    return metadata_lib.build_metadata(collect_metadata_args(args), filename=str(pdf_path))


def stdin_is_interactive() -> bool:
    """True when stdin is a terminal that can still be prompted on."""
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def infer_series_name(pdf_stem: str, explicit_series: str = None) -> str:
    """Infer a magazine series from a trailing issue/date suffix, if present."""
    if explicit_series:
        series = explicit_series.replace("{stem}", pdf_stem).strip()
    else:
        issue_suffix = re.compile(
            r"^(?P<series>.+?)[ _-]+(?:"
            r"\d{6,8}|"
            r"\d{4}(?:[-_.]\d{1,2}(?:[-_.]\d{1,2})?)?|"
            r"issue[ _-]*#?\s*\d+|"
            r"#\s*\d+)\s*$",
            re.IGNORECASE,
        )
        match = issue_suffix.match(pdf_stem)
        series = match.group("series").rstrip(" _-.") if match else pdf_stem

    if not series or series in {".", ".."} or "/" in series or "\\" in series:
        raise ValueError(
            "series name must be a non-empty single path component"
        )
    return series


def clean_working_directory(work_dir: Path, output_root: Path) -> bool:
    """Delete conversion work files and now-empty staging directories."""
    if not work_dir.is_dir():
        return False

    shutil.rmtree(work_dir)
    parent = work_dir.parent
    output_root = output_root.resolve()
    while parent.resolve() != output_root:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent
    return True


def main():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for processing.")
    elif torch.backends.mps.is_available():
        print("MPS is available. Using Apple Silicon for processing.")
    else:
        print("CUDA is not available. Using CPU for processing.")
        
    parser = argparse.ArgumentParser(
        description='Convert PDF files to EPUB format via Markdown'
    )
    parser.add_argument(
        'input_path',
        nargs='?',
        type=str,
        help='Path to input PDF file or directory (default: ./input/*.pdf)'
    )
    parser.add_argument(
        'output_path',
        nargs='?',
        type=str,
        help='Output root for series directories (default: beside the input PDF)'
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        default=None,
        help='Maximum number of pages to process'
    )
    parser.add_argument(
        '--start-page',
        type=int,
        default=None,
        help='Page number to start from'
    )
    parser.add_argument(
        '--skip-epub',
        action='store_true',
        help='Skip EPUB generation, only create markdown'
    )
    parser.add_argument(
        '--skip-md',
        action='store_true',
        help='Skip markdown generation, use existing markdown files'
    )
    parser.add_argument(
        '--keep-intermediates',
        action='store_true',
        help='Keep Markdown, images, and other conversion files in the work area'
    )
    parser.add_argument(
        '--series',
        type=str,
        default=None,
        help='Series directory name (supports {stem} and {filename}; inferred by default)'
    )
    parser.add_argument(
        '--vlm',
        action='store_true',
        help='Transcribe pages with a hosted vision model instead of local models'
    )
    parser.add_argument('--vlm-model', type=str, default=None,
                        help='Vision model id for --vlm')
    parser.add_argument('--vlm-dpi', type=int, default=None,
                        help='Page render DPI for --vlm')
    parser.add_argument(
        '--articles-only', '--substantive-only', dest='articles_only',
        action='store_true',
        help=('Keep substantive articles; remove blurbs, ads, teasers and contents '
              '(--substantive-only is a compatibility alias; also accepts an EPUB)')
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Accept the default EPUB metadata instead of prompting for it'
    )
    add_metadata_arguments(parser)

    args = parser.parse_args()
    print(f"Article filtering: {'enabled' if args.articles_only else 'disabled'}", flush=True)

    if args.vlm:
        os.environ['PDF2EPUB_VLM'] = '1'
    if args.vlm_model:
        os.environ['PDF2EPUB_VLM_MODEL'] = args.vlm_model
    if args.vlm_dpi:
        os.environ['PDF2EPUB_VLM_DPI'] = str(args.vlm_dpi)
    try:
        vlm_config = vlm_lib.read_vlm_config()
    except vlm_lib.VlmError as exc:
        parser.error(str(exc))
    filter_config = None
    if args.articles_only:
        if args.skip_epub:
            parser.error('--articles-only requires EPUB generation')
        try:
            filter_config = vlm_lib.read_vlm_config(dict(os.environ, PDF2EPUB_VLM='1'))
        except vlm_lib.VlmError as exc:
            parser.error(str(exc))
    if args.input_path and Path(args.input_path).suffix.lower() == '.epub':
        if not args.articles_only:
            parser.error('EPUB input requires --articles-only')
        if args.skip_md or args.max_pages is not None or args.start_page is not None:
            parser.error('PDF transcription options do not apply to EPUB input')
        source = Path(args.input_path)
        destination = (Path(args.output_path) / source.name if args.output_path
                       else source.with_name(source.stem + '-substantive.epub'))
        try:
            report = chapter_filter.filter_epub(source, destination, filter_config,
                                       progress=lambda done, total: vlm_lib._report_progress(
                                           'Article filter', done, total))
        except Exception as exc:
            print(f'Error filtering {source.name}: {exc}', file=sys.stderr)
            sys.exit(1)
        if report['status'] == 'filtered':
            print(f'Filtered EPUB: {destination}', flush=True)
        return

    metadata_args = collect_metadata_args(args)

    if args.skip_epub and metadata_args:
        parser.error("metadata options are ignored with --skip-epub")

    # Reading metadata from stdin is only possible on a terminal. Batch runs
    # (cron, CI, piped input) would otherwise die with an EOFError partway
    # through a directory, so fall back to defaults with a warning instead.
    unattended = False
    if metadata_args or args.yes:
        unattended = True
    elif not args.skip_epub and not stdin_is_interactive():
        print(
            "Warning: stdin is not a terminal, so metadata cannot be asked for. "
            "Using defaults; pass --title/--author/... to set them.",
            file=sys.stderr,
        )
        unattended = True

    # Get input path
    input_path = Path(args.input_path) if args.input_path else pdf2md.get_default_input_dir()
    
    # Get queue of PDFs to process
    queue = pdf2md.add_pdfs_to_queue(input_path)
    print(f"Found {len(queue)} PDF files to process")

    # A date that cannot be parsed fails every file identically, so reject it
    # before doing the expensive PDF conversion work.
    if metadata_args and args.date is not None:
        try:
            metadata_lib.normalise_date(args.date)
        except metadata_lib.MetadataError as exc:
            parser.error(str(exc))
    
    # Process each PDF
    failed = []
    for pdf_path in queue:
        print(f"\nProcessing: {pdf_path.name}")
        
        output_root = Path(args.output_path) if args.output_path else pdf_path.parent
        try:
            series_template = args.series
            if series_template:
                series_template = series_template.replace(
                    "{filename}", pdf_path.name
                )
            series_name = infer_series_name(pdf_path.stem, series_template)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            failed.append(pdf_path.name)
            continue

        series_dir = output_root / series_name
        work_root = output_root / ".pdf2epub-work" / series_name
        generated_work_dir = work_root / pdf_path.stem
        # `--skip-md` can reuse prior retained work files or the old issue-named
        # output directory created before series routing was introduced.
        if args.skip_md:
            candidates = (
                generated_work_dir,
                output_root / pdf_path.stem,
                series_dir / pdf_path.stem,
            )
            markdown_dir = next((p for p in candidates if p.is_dir()), generated_work_dir)
        else:
            markdown_dir = generated_work_dir
            
        try:
            # Check if markdown directory exists when skipping MD generation
            if args.skip_md:
                if not markdown_dir.exists():
                    print(f"Error: Markdown directory not found: {markdown_dir}", file=sys.stderr)
                    failed.append(pdf_path.name)
                    continue
                print(f"Using existing markdown files from: {markdown_dir}")
                
            # Convert PDF to Markdown unless skipped
            if not args.skip_md:
                if vlm_config:
                    print("Converting PDF to Markdown with the vision model...")
                    vlm_lib.convert_pdf_vlm(
                        str(pdf_path), markdown_dir, args.max_pages,
                        args.start_page, config=vlm_config,
                    )
                else:
                    print("Converting PDF to Markdown...")
                    pdf2md.convert_pdf(
                        str(pdf_path), markdown_dir, args.max_pages,
                        args.start_page,
                    )
            
            # Convert Markdown to EPUB unless skipped
            if not args.skip_epub:
                epub_metadata = None
                if unattended:
                    try:
                        epub_metadata = metadata_from_args(args, pdf_path)
                    except metadata_lib.MetadataError as exc:
                        print(f"Error: {exc}", file=sys.stderr)
                        failed.append(pdf_path.name)
                        continue

                print("Converting Markdown to EPUB...")
                series_dir.mkdir(parents=True, exist_ok=True)
                epub_path = series_dir / f"{pdf_path.stem}.epub"
                # A failed filter must not leave a raw EPUB at the final path,
                # where batch existence checks could mistake it for success.
                build_dir = markdown_dir / '.epub-unfiltered' if filter_config else series_dir
                build_dir.mkdir(parents=True, exist_ok=True)
                mark2epub.convert_to_epub(
                    markdown_dir,
                    build_dir,
                    metadata=epub_metadata,
                    epub_filename=epub_path.name,
                )
                if filter_config:
                    report = chapter_filter.filter_epub(
                        build_dir / epub_path.name, epub_path, filter_config, input_pdf=pdf_path,
                        progress=lambda done, total: vlm_lib._report_progress(
                            'Article filter', done, total),
                    )
                    if report['status'] == 'no_articles':
                        print(f'Article-filtered EPUB skipped: {pdf_path.name}; '
                              'no qualifying articles; conversion files retained.', flush=True)
                        continue
                    print(f'Article-filtered EPUB ready: {epub_path}', flush=True)
                if not args.keep_intermediates:
                    if epub_path.is_file() and clean_working_directory(
                        markdown_dir, output_root
                    ):
                        print(
                            f"Removed conversion files; kept EPUB: {epub_path}",
                            flush=True,
                        )
                    else:
                        print(
                            "EPUB file was not created; keeping conversion files.",
                            file=sys.stderr,
                        )
                
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {str(e)}", file=sys.stderr)
            failed.append(pdf_path.name)
            continue

    if failed:
        print(
            f"\nFailed to process {len(failed)} of {len(queue)} PDF file(s):",
            file=sys.stderr,
        )
        for name in failed:
            print(f"  - {name}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
