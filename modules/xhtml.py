"""Normalize HTML void elements carried through raw Markdown into XHTML.

Python-Markdown's XHTML serializer handles generated elements, but raw HTML is
passed through unchanged. Patch actual start/end tags using parser offsets so
attributes, prose, escaped code examples, comments and script text stay intact.
"""
from html.parser import HTMLParser

VOID_ELEMENTS = frozenset({
    'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta',
    'param', 'source', 'track', 'wbr',
})


def normalize_html_voids(source):
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    edits = []

    class Normalizer(HTMLParser):
        def start(self):
            line, column = self.getpos()
            return offsets[line - 1] + column

        def normalize_tag(self, tag):
            if tag in VOID_ELEMENTS:
                text = self.get_starttag_text()
                replacement = '<' + tag + text[1 + len(tag):]
                if not replacement.endswith('/>'):
                    replacement = replacement[:-1].rstrip() + ' />'
                start = self.start()
                if replacement != text:
                    edits.append((start, start + len(text), replacement))

        def handle_starttag(self, tag, attrs):
            self.normalize_tag(tag)

        def handle_startendtag(self, tag, attrs):
            self.normalize_tag(tag)

        def handle_endtag(self, tag):
            if tag in VOID_ELEMENTS:
                start = self.start()
                end = source.index('>', start) + 1
                edits.append((start, end, ''))

    parser = Normalizer(convert_charrefs=False)
    parser.feed(source)
    parser.close()
    for start, end, replacement in reversed(edits):
        source = source[:start] + replacement + source[end:]
    return source
