# XHTML recovery without retranscription

The generator normalizes raw HTML void tags and HTML character references before
writing chapter XHTML. EPUB input applies the same conservative repairs in memory
when XML parsing fails; the source archive is never rewritten.

- XML's five predefined entities and numeric references are preserved.
- HTML5 named entities (for example `&nbsp;`) become numeric references.
- Literal ampersands and unknown names (for example `S&P;`) are escaped without
  dropping their text. Existing `&amp;` references are not double-escaped.
- Escaped code, quoted attributes, comments, CDATA and processing instructions
  are preserved. Unrelated malformed markup still fails loudly.

For an undefined-entity failure, use release 0.3.13 or newer to filter the saved
`.epub-unfiltered/*.epub` into a separate output directory with `--articles-only`.
There is no need to transcribe the PDF again. Keep the original work directory
and inspect the adjacent `*.substantive-filter.json` and recovered EPUB.

A corrupted MathML/tag-boundary failure is different: rebuild retained Markdown
with `--skip-md --articles-only` instead of guessing repairs to lost markup.
