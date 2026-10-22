"""Tests for the interactive metadata prompt in mark2epub.

These cover the defaults shown at the prompt, which are what a user accepts by
pressing Enter.
"""

import builtins
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules import mark2epub, metadata as metadata_lib


def answer(monkeypatch, *responses):
    """Feed *responses* to the metadata prompts, then answer the review prompt."""
    answers = list(responses)

    def fake_input(prompt=""):
        return answers.pop(0) if answers else ""

    monkeypatch.setattr(builtins, "input", fake_input)


# title, author, identifier, language, rights, publisher, date, review
DEFAULTS_ONLY = ("My Title", "Jane Doe", "", "", "", "", "", "n")


class TestInteractiveIdentifier:
    def test_derived_from_entered_title_and_file(
        self, monkeypatch, capsys
    ):
        answer(monkeypatch, *DEFAULTS_ONLY)
        result = mark2epub.get_metadata_from_user(
            {}, fallback_title="book", filename="/work/book"
        )
        assert result["metadata"]["dc:identifier"] == metadata_lib.default_identifier(
            "My Title", "/work/book"
        )

    def test_same_title_different_files_give_different_identifiers(
        self, monkeypatch, capsys
    ):
        # The interactive prompt must key on the file as well as the title, or
        # two books sharing a title collide on dc:identifier.
        seen = set()
        for directory in ("bookA", "bookB"):
            answer(monkeypatch, *DEFAULTS_ONLY)
            result = mark2epub.get_metadata_from_user(
                {}, fallback_title=directory, filename=f"/work/{directory}"
            )
            seen.add(result["metadata"]["dc:identifier"])

        assert len(seen) == 2

    def test_matches_unattended_path_for_same_inputs(self, monkeypatch):
        # Both paths must agree, or switching modes would change the id.
        answer(monkeypatch, *DEFAULTS_ONLY)
        interactive = mark2epub.get_metadata_from_user(
            {}, fallback_title="scan", filename="/work/scan"
        )["metadata"]

        overrides = metadata_lib.build_metadata(
            {"title": "My Title", "author": "Jane Doe"}, filename="scan"
        )
        unattended = metadata_lib.resolve_metadata(
            overrides, fallback_title="scan", filename="/work/scan"
        )

        assert interactive["dc:identifier"] == unattended["dc:identifier"]
        assert interactive["dc:title"] == unattended["dc:title"]

    def test_existing_identifier_is_offered_and_kept(self, monkeypatch):
        existing = {"metadata": {"dc:identifier": "urn:isbn:old", "dc:title": "Kept Title"}}
        answer(monkeypatch, "", "", "", "", "", "", "", "n")
        result = mark2epub.get_metadata_from_user(existing, filename="/work/book")
        assert result["metadata"]["dc:identifier"] == "urn:isbn:old"

    def test_user_supplied_identifier_wins(self, monkeypatch):
        answer(monkeypatch, "T", "A", "urn:isbn:mine", "", "", "", "", "n")
        result = mark2epub.get_metadata_from_user({}, filename="/work/book")
        assert result["metadata"]["dc:identifier"] == "urn:isbn:mine"


class TestInteractiveDefaults:
    def test_falls_back_to_file_name_for_title(self, monkeypatch):
        answer(monkeypatch, "", "", "", "", "", "", "", "n")
        result = mark2epub.get_metadata_from_user({}, fallback_title="dune")
        assert result["metadata"]["dc:title"] == "dune"

    def test_existing_title_is_offered_as_default(self, monkeypatch):
        existing = {"metadata": {"dc:title": "Old Title"}}
        answer(monkeypatch, "", "", "", "", "", "", "", "n")
        result = mark2epub.get_metadata_from_user(existing, fallback_title="new")
        assert result["metadata"]["dc:title"] == "Old Title"

    def test_list_author_from_previous_run_is_flattened(self, monkeypatch):
        # description.json may hold several authors from an unattended run.
        existing = {"metadata": {"dc:creator": ["Frank Herbert", "Brian Herbert"]}}
        answer(monkeypatch, "", "", "", "", "", "", "", "n")
        result = mark2epub.get_metadata_from_user(existing)
        assert result["metadata"]["dc:creator"] == "Frank Herbert, Brian Herbert"

    def test_bad_date_falls_back_to_default(self, monkeypatch, capsys):
        answer(monkeypatch, "T", "A", "", "", "", "", "not-a-date", "n")
        result = mark2epub.get_metadata_from_user({}, filename="/work/book")
        # The unusable value is discarded rather than written into the EPUB.
        assert result["metadata"]["dc:date"] != "not-a-date"
        assert "not-a-date" in capsys.readouterr().out


class TestInteractiveFlags:
    def test_signature_accepts_filename_keyword(self):
        # Guard against the parameter being dropped again; it is what keeps
        # same-titled documents from colliding.
        import inspect

        params = inspect.signature(mark2epub.get_metadata_from_user).parameters
        assert "filename" in params


def test_raw_html_line_breaks_in_tables_produce_parseable_xhtml(tmp_path):
    from xml.etree import ElementTree as ET
    source = '# Gear\n\n| Type | Name |\n| --- | --- |\n| Dual-Röhren-<br>Großmembran-<BR>Kondensatormikrofon | Aurora<br>Beat |\n'
    document, _ = mark2epub.get_chapter_XML(tmp_path, 'gear.md', [], content=source)
    root = ET.fromstring(document)
    assert len(root.findall('.//{http://www.w3.org/1999/xhtml}br')) == 3
    assert 'Dual-Röhren-' in document


def test_void_normalization_preserves_code_comments_attributes_and_valid_markup():
    from modules.xhtml import normalize_html_voids
    source = '<p>Text &amp; prose &lt;br&gt;</p><!-- <br> -->\n<img alt="a > b" src="a.jpg"><br /><hr>\n<pre>&lt;br&gt;</pre><script>const x = "<br>";</script>'
    output = normalize_html_voids(source)
    assert output == source.replace('<img alt="a > b" src="a.jpg">', '<img alt="a > b" src="a.jpg" />').replace('<hr>', '<hr />')


def test_kickstarter_currency_and_bold_boundaries_are_preserved(tmp_path):
    from xml.etree import ElementTree as ET
    source = ('**A total of 3,300 backers pledged $178,173!** '
              'The prior record ended at $152,028. Other tiers cost $10 or $20.')
    document, _ = mark2epub.get_chapter_XML(tmp_path, 'fundraiser.md', [], content=source)
    root = ET.fromstring(document)
    text = ''.join(root.itertext())
    assert '$178,173!' in text and '$152,028' in text
    assert '$10 or $20' in text
    assert root.find('.//{http://www.w3.org/1999/xhtml}strong').text.endswith('$178,173!')
    assert not root.findall('.//{http://www.w3.org/1998/Math/MathML}math')


def test_math_is_confined_to_text_not_markup_attributes_or_code():
    from xml.etree import ElementTree as ET
    source = ('<p><a href="https://example.invalid/$asset$">Link</a> '
              '<strong>$unfinished</strong> text $y+1$.</p>'
              '<pre><code>$x^2$ and $$y$$</code></pre><!-- $comment$ -->')
    output = mark2epub.convert_math_to_mathml(source)
    root = ET.fromstring('<root>' + output + '</root>')
    assert 'href="https://example.invalid/$asset$"' in output
    assert '<strong>$unfinished</strong>' in output
    assert '<pre><code>$x^2$ and $$y$$</code></pre>' in output
    assert '<!-- $comment$ -->' in output
    assert len(root.findall('.//{http://www.w3.org/1998/Math/MathML}math')) == 1


def test_real_inline_and_display_math_remain_supported_with_html_entities():
    from xml.etree import ElementTree as ET
    source = '<p>Inline $x &lt; y$ and $5+5$.</p><p>$$x^2 + y^2$$</p>'
    output = mark2epub.convert_math_to_mathml(source)
    root = ET.fromstring('<root>' + output + '</root>')
    maths = root.findall('.//{http://www.w3.org/1998/Math/MathML}math')
    assert len(maths) == 3
    assert [node.get('display') for node in maths] == ['inline', 'inline', 'block']
    assert root.find('div').get('class') == 'math-display'
    assert not root.findall('.//p/div')
    assert any(node.text == '<' for node in maths[0].iter())


def test_literal_escaped_dollars_and_existing_math_are_not_rewritten():
    source = (r'<p>Costs \$5 and \$10.</p>'
              '<math xmlns="http://www.w3.org/1998/Math/MathML"><mtext>$literal$</mtext></math>')
    assert mark2epub.convert_math_to_mathml(source) == source


def test_failed_math_conversion_preserves_original_text(monkeypatch):
    def invalid(*args, **kwargs):
        return '<math><unclosed></math>'
    monkeypatch.setattr(mark2epub.latex2mathml.converter, 'convert', invalid)
    source = '<p>Keep $x^2$ exactly when conversion fails.</p>'
    assert mark2epub.convert_math_to_mathml(source) == source


def test_economist_sources_and_html_entities_produce_valid_xhtml(tmp_path):
    from xml.etree import ElementTree as ET
    source = ('# Sources\n\nSources: CME Group; FactSet; S&P; '
              'Thompson Lloyd &amp; Ewart; USDA.\n\n'
              '<p title="A&nbsp;B &quot;quoted&quot; &unknown;">'
              'A&nbsp;B &copy; &NotEqualTilde; &unknown; '
              '&#169; &#xA9; &lt;br&gt;</p>')
    document, _ = mark2epub.get_chapter_XML(tmp_path, 'sources.md', [], content=source)
    root = ET.fromstring(document)
    text = ''.join(root.itertext())
    assert 'S&P; Thompson Lloyd & Ewart; USDA.' in text
    assert 'A\u00a0B © ≂̸ &unknown; © © <br>' in text
    assert root.find('.//{http://www.w3.org/1999/xhtml}p[@title]').get('title') == 'A\u00a0B "quoted" &unknown;'


def test_entity_normalization_preserves_escaped_text_attributes_and_literal_sections():
    from xml.etree import ElementTree as ET
    from modules.xhtml import normalize_html_entities
    source = ('<root><a href="https://example.invalid/?a=1&b=2" '
              'title="&quot;&lt;&gt;&apos;&amp;">R&D S&P; &unknown; '
              '&nbsp; &copy; &#160; &#xA0; &amp;nbsp;</a>'
              '<!-- &nbsp; &raw; --><![CDATA[&nbsp; &raw;]]>'
              '<?literal &nbsp; &raw;?></root>')
    output = normalize_html_entities(source)
    root = ET.fromstring(output)
    assert root.find('a').get('href') == 'https://example.invalid/?a=1&b=2'
    assert root.find('a').get('title') == '\"<>\'&'
    assert root.find('a').text == 'R&D S&P; &unknown; \u00a0 © \u00a0 \u00a0 &nbsp;'
    for literal in ('<!-- &nbsp; &raw; -->', '<![CDATA[&nbsp; &raw;]]>', '<?literal &nbsp; &raw;?>'):
        assert literal in output
    assert normalize_html_entities(output) == output
