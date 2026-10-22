"""Normalize HTML void elements and character references for XML-safe XHTML.

Python-Markdown's XHTML serializer handles generated elements, but raw HTML is
passed through unchanged. Patch actual start/end tags using parser offsets so
attributes, prose, escaped code examples, comments and script text stay intact.
"""
from html.entities import html5
from html.parser import HTMLParser
import re

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


def normalize_html_entities(source):
    """Make HTML character references XML-safe without reparsing the markup.

    Preserve XML's five entities and numeric references exactly. Convert known
    HTML5 names to numeric references (including multi-codepoint entities), and
    escape unknown/literal ampersands rather than dropping publication text.
    Never decode the entire document: doing so would turn escaped code into
    markup and quotes into attribute delimiters. Comments/CDATA/PIs are literal.
    """
    pattern = re.compile(
        r'<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>|'
        r'&(?:([A-Za-z][A-Za-z0-9]*);|#(?:[0-9]+|x[0-9A-Fa-f]+);)?',
        re.DOTALL,
    )

    def replace(match):
        token = match.group()
        if not token.startswith('&'):
            return token
        name = match.group(1)
        if token.startswith('&#') or name in {'amp', 'lt', 'gt', 'quot', 'apos'}:
            return token
        value = html5.get(name + ';') if name else None
        if value is not None:
            return ''.join(f'&#{ord(char)};' for char in value)
        return '&amp;' + token[1:]

    return pattern.sub(replace, source)


def html_text_ranges(source):
    """Return text-only source ranges, excluding code, scripts and existing math.

    Adjacent entity/data callbacks are merged, but ranges never cross HTML tags
    or attributes. Offsets refer to the original string so untouched markup is
    preserved exactly rather than reserialized by an HTML parser.
    """
    offsets = [0] + [i + 1 for i, char in enumerate(source) if char == '\n']
    ranges = []
    excluded = {'pre', 'code', 'script', 'style', 'math'}

    class TextRanges(HTMLParser):
        depth = 0

        def add(self, length):
            if self.depth:
                return
            line, column = self.getpos()
            start = offsets[line - 1] + column
            end = start + length
            if ranges and ranges[-1][1] == start:
                ranges[-1] = (ranges[-1][0], end)
            else:
                ranges.append((start, end))

        def handle_starttag(self, tag, attrs):
            if tag.rsplit(':', 1)[-1] in excluded:
                self.depth += 1

        def handle_endtag(self, tag):
            if tag.rsplit(':', 1)[-1] in excluded:
                self.depth = max(0, self.depth - 1)

        def handle_startendtag(self, tag, attrs):
            pass

        def handle_data(self, data):
            self.add(len(data))

        def entity(self, name, prefix_length):
            line, column = self.getpos()
            end = offsets[line - 1] + column + prefix_length + len(name)
            self.add(prefix_length + len(name) + (source[end:end + 1] == ';'))

        def handle_entityref(self, name):
            self.entity(name, 1)

        def handle_charref(self, name):
            self.entity(name, 2)

    parser = TextRanges(convert_charrefs=False)
    parser.feed(source)
    parser.close()
    return ranges
