# XHTML recovery without retranscription

The generator normalizes raw HTML void tags and HTML character references before
writing chapter XHTML. EPUB input applies the same conservative repairs in memory
when XML parsing fails; the source archive is never rewritten.

- XML's five predefined entities and numeric references are preserved.
- HTML5 named entities (for example `&nbsp;`) become numeric references.
- Literal ampersands and unknown names (for example `S&P;`) are escaped without
  dropping their text. Existing `&amp;` references are not double-escaped.
- Escaped code, quoted attributes, comments, CDATA and processing instructions
  are preserved.
- OCR layout labels that Markdown passed through as raw inline HTML (`<C64>`,
  `<INTRO>`, `<INFO ... />`, `<BODY>`) are escaped as literal text. The text is
  kept; it is never dropped. Element names are matched case-sensitively, so an
  uppercase pseudo-tag is not mistaken for the real XHTML element.
- Real elements that are left open or closed out of order are balanced by
  inserting the missing end tags.
- A member that still cannot be parsed after those repairs is excluded from the
  filtered EPUB and reported: the filter prints `Skipping unrepairable XHTML
  <member>`, adds an `"kind": "unreadable"` row to the adjacent
  `*.substantive-filter.json`, and converts the rest of the book.

A completely unreadable member therefore no longer aborts the whole issue, and
a recoverable pseudo-tag no longer costs the article.

For a pseudo-tag or unbalanced-tag failure, use release 0.3.14 or newer to
filter the saved `.epub-unfiltered/*.epub` (EPUB input, `--articles-only`) into
the published directory; no PDF retranscription is needed.

For an undefined-entity failure, use release 0.3.13 or newer to filter the saved
`.epub-unfiltered/*.epub` into a separate output directory with `--articles-only`.
There is no need to transcribe the PDF again. Keep the original work directory
and inspect the adjacent `*.substantive-filter.json` and recovered EPUB.

A corrupted MathML/tag-boundary failure is different: rebuild retained Markdown
with `--skip-md --articles-only` instead of guessing repairs to lost markup.
