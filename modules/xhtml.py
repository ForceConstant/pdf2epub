"""Normalize HTML void elements and character references for XML-safe XHTML.

Python-Markdown's XHTML serializer handles generated elements, but raw HTML is
passed through unchanged. Patch actual start/end tags using parser offsets so
attributes, prose, escaped code examples, comments and script text stay intact.
"""
from html import escape
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


# Element names the generator and EPUB assembly legitimately emit. XHTML is
# case-sensitive, so anything outside this lowercase set is publication text,
# not markup.
KNOWN_ELEMENTS = frozenset("""
a abbr address area article aside audio b bdi bdo blockquote body br button
canvas caption cite code col colgroup data datalist dd del details dfn dialog
div dl dt em embed fieldset figcaption figure footer form h1 h2 h3 h4 h5 h6
head header hgroup hr html i iframe img input ins kbd label legend li link main
map mark menu meta meter nav noscript object ol optgroup option output p param
picture pre progress q rp rt ruby s samp script search section select slot small
source span strong style sub summary sup table tbody td template textarea tfoot
th thead time title tr track u ul var video wbr
svg g defs symbol use image path rect circle ellipse line polyline polygon text
tspan linearGradient radialGradient stop clipPath mask pattern marker
math maction maligngroup malignmark menclose merror mfenced mfrac mglyph mi
mlabeledtr mmultiscripts mn mo mover mpadded mphantom mroot mrow ms mspace
msqrt mstyle msub msubsup msup mtable mtd mtext mtr munder munderover
semantics annotation annotation-xml
""".split())

# Elements HTML implicitly closes before opening another of the same name.
IMPLIED_END = frozenset({'p', 'li', 'dt', 'dd', 'tr', 'td', 'th', 'option'})
# Block elements that cannot nest inside a paragraph.
BLOCK_CLOSES_P = frozenset({
    'address', 'article', 'aside', 'blockquote', 'details', 'div', 'dl',
    'fieldset', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3',
    'h4', 'h5', 'h6', 'header', 'hgroup', 'hr', 'main', 'menu', 'nav', 'ol',
    'p', 'pre', 'section', 'table', 'ul',
})

_TAG_NAME = re.compile(r'<\s*/?\s*([^\s/>]+)')


def repair_mismatched_tags(source):
    """Recover XHTML that raw OCR text turned into unbalanced markup.

    Vision/OCR transcripts contain layout labels such as ``<INFO ...>``,
    ``<INTRO>``, ``<BODY>`` and ``<C64>`` that Python-Markdown passes through
    as raw inline HTML. Those names are not XHTML elements, so they are escaped
    as literal text (text is preserved, never dropped). Real elements that are
    left open or closed out of order are balanced by inserting end tags.

    Returns ``(repaired, escaped_tags, inserted_end_tags)``. Callers still
    validate the result with an XML parser; this function asserts nothing.
    """
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    edits = []
    stack = []
    escaped = closed = 0

    def opened_at():
        line, column = parser.getpos()
        return offsets[line - 1] + column

    def escape_tag(start, end, raw):
        nonlocal escaped
        escaped += 1
        edits.append((start, end, escape(raw)))

    def close_top(position):
        nonlocal closed
        name = stack.pop()
        edits.append((position, position, f'</{name}>'))
        closed += 1

    def raw_name(raw, fallback):
        match = _TAG_NAME.match(raw)
        return match.group(1) if match else fallback

    class Repair(HTMLParser):
        def handle_starttag(self, tag, attrs):
            raw = self.get_starttag_text()
            name = raw_name(raw, tag)
            start = opened_at()
            if name not in KNOWN_ELEMENTS:
                escape_tag(start, start + len(raw), raw)
                return
            while stack and ((stack[-1] == name and name in IMPLIED_END)
                             or (stack[-1] == 'p' and name in BLOCK_CLOSES_P)):
                close_top(start)
            stack.append(name)

        def handle_startendtag(self, tag, attrs):
            raw = self.get_starttag_text()
            name = raw_name(raw, tag)
            if name not in KNOWN_ELEMENTS:
                start = opened_at()
                escape_tag(start, start + len(raw), raw)

        def handle_endtag(self, tag):
            start = opened_at()
            end = source.index('>', start) + 1
            raw = source[start:end]
            name = raw_name(raw, tag)
            if name not in KNOWN_ELEMENTS or name not in stack:
                escape_tag(start, end, raw)
                return
            while stack[-1] != name:
                close_top(start)
            stack.pop()

    parser = Repair(convert_charrefs=False)
    parser.feed(source)
    parser.close()
    position = len(source)
    while stack:
        close_top(position)
    for start, end, replacement in reversed(edits):
        source = source[:start] + replacement + source[end:]
    return source, escaped, closed
