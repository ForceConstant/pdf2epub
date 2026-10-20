"""Substance judgments and EPUB-preserving mutations, without network access."""
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.chapter_filter import filter_epub, read_epub, _prune_links, XHTML
from modules.vlm import VlmError


CONFIG = dict(model='test', base_url='https://test.invalid', max_tokens=8192,
              concurrency=2)


def make_epub(path):
    container = '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OPS/package.opf"/></rootfiles></container>'
    opf = '''<package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata/>
    <manifest><item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>
    <item id="toc" href="TOC.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="a" href="contents.xhtml" media-type="application/xhtml+xml"/>
    <item id="b" href="article.xhtml" media-type="application/xhtml+xml"/>
    <item id="c" href="blurb.xhtml" media-type="application/xhtml+xml"/>
    <item id="img" href="cover.jpg" media-type="image/jpeg"/></manifest>
    <spine toc="ncx"><itemref idref="cover"/><itemref idref="a"/><itemref idref="b"/><itemref idref="c"/></spine>
    <guide><reference type="cover" href="cover.xhtml"/><reference type="toc" href="contents.xhtml"/></guide></package>'''
    def xhtml(body):
        return f'<html xmlns="{XHTML}"><head><title>Head must not count</title></head><body>{body}</body></html>'
    files = {
        'META-INF/container.xml': container, 'OPS/package.opf': opf,
        'OPS/contents.xhtml': xhtml('<h1>Table of contents</h1><p>A long teaser disguised as an article.</p>'),
        'OPS/article.xhtml': xhtml('<h1>When producers come under fire due to the EU AI Act</h1><p>Developed analysis.</p>'),
        'OPS/blurb.xhtml': xhtml('<h1>New synth announced</h1><p>Available now. Features and price.</p>'),
        'OPS/cover.xhtml': xhtml('<img src="cover.jpg"/>'), 'OPS/cover.jpg': b'cover-bytes',
        'OPS/TOC.xhtml': xhtml('<nav xmlns:epub="http://www.idpf.org/2007/ops" epub:type="toc"><ol><li><a href="contents.xhtml">Contents</a></li><li><a href="article.xhtml">Analysis</a></li><li><a href="blurb.xhtml">Synth</a></li></ol></nav>'),
        'OPS/toc.ncx': '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap><navPoint id="one" playOrder="1"><navLabel><text>Contents</text></navLabel><content src="contents.xhtml"/></navPoint><navPoint id="two" playOrder="2"><navLabel><text>Analysis</text></navLabel><content src="article.xhtml"/></navPoint><navPoint id="three" playOrder="3"><navLabel><text>Synth</text></navLabel><content src="blurb.xhtml"/></navPoint></navMap></ncx>',
    }
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('mimetype', b'application/epub+zip')
        for name, data in files.items():
            z.writestr(name, data)


def classifier(chapters, config, **kwargs):
    title, content = chapters[0]
    assert 'Head must not count' not in content
    assert config['max_tokens'] >= 16384
    assert kwargs['excerpt_limit'] == 18000
    # Model false-positive on explicit TOC must be overridden.
    keep = 'AI Act' in title or title == 'Table of contents'
    return {0: dict(kind='article' if keep else 'non_article', complete=False,
                    confidence=.98, reason='Judged substance, not completeness.')}


def test_filter_updates_all_navigation_preserves_cover_and_source(tmp_path):
    source, dest = tmp_path / 'original.epub', tmp_path / 'filtered.epub'
    make_epub(source)
    original = source.read_bytes()
    source.chmod(0o640)
    report = filter_epub(source, dest, CONFIG, classifier)
    assert [r['included'] for r in report['chapters']] == [False, True, False]
    assert source.read_bytes() == original
    assert dest.stat().st_mode & 0o777 == 0o640
    assert [c['title'] for c in read_epub(dest)[3]] == ['When producers come under fire due to the EU AI Act']
    with zipfile.ZipFile(dest) as z, zipfile.ZipFile(source) as src:
        assert z.infolist()[0].filename == 'mimetype'
        assert z.infolist()[0].compress_type == zipfile.ZIP_STORED
        for n in ('OPS/article.xhtml', 'OPS/cover.jpg', 'OPS/cover.xhtml'):
            assert z.read(n) == src.read(n)
        for n in ('OPS/package.opf', 'OPS/TOC.xhtml', 'OPS/toc.ncx'):
            data = z.read(n)
            assert b'contents.xhtml' not in data
            assert b'blurb.xhtml' not in data
            assert b'article.xhtml' in data
            ET.fromstring(data)
        assert b'playOrder="1"' in z.read('OPS/toc.ncx')


def test_resume_and_signature_invalidation(tmp_path):
    source, dest = tmp_path / 'original.epub', tmp_path / 'filtered.epub'
    make_epub(source)
    filter_epub(source, dest, CONFIG, classifier)
    def no_call(*args, **kwargs):
        raise AssertionError('cache not reused')
    filter_epub(source, dest, CONFIG, no_call)
    with pytest.raises(AssertionError, match='cache not reused'):
        filter_epub(source, dest, dict(CONFIG, model='changed'), no_call)


def test_failure_and_empty_selection_leave_destination_unchanged(tmp_path):
    source, dest = tmp_path / 'original.epub', tmp_path / 'filtered.epub'
    make_epub(source)
    dest.write_bytes(b'existing-output')
    def reject(*args, **kwargs):
        return {0: dict(kind='non_article', complete=True, confidence=1, reason='Blurb')}
    with pytest.raises(VlmError, match='No substantive chapters'):
        filter_epub(source, dest, CONFIG, reject)
    assert dest.read_bytes() == b'existing-output'
    assert len(json.loads(dest.with_suffix('.substantive-filter.json').read_text())['chapters']) == 3


def test_in_place_backup_is_not_overwritten(tmp_path):
    source = tmp_path / 'original.epub'
    make_epub(source)
    original = source.read_bytes()
    filter_epub(source, source, CONFIG, classifier)
    assert source.with_suffix('.unfiltered.epub').read_bytes() == original
    filtered = source.read_bytes()
    with pytest.raises(VlmError, match='Refusing to overwrite backup'):
        filter_epub(source, source, CONFIG, classifier)
    assert source.read_bytes() == filtered


def test_nested_navigation_promotes_retained_entries_without_duplicates():
    root = ET.fromstring(f'<html xmlns="{XHTML}"><ol><li><a href="drop.xhtml">Drop</a><ol><li><a href="keep.xhtml">Keep</a></li><li><a href="drop2.xhtml">Drop</a></li></ol></li></ol><p><a href="drop.xhtml">See earlier</a></p></html>')
    assert _prune_links(root, 'OPS/nav.xhtml', {'OPS/drop.xhtml', 'OPS/drop2.xhtml'})
    links = root.findall(f'.//{{{XHTML}}}a')
    assert [a.get('href') for a in links] == ['keep.xhtml']
    assert root.find(f'.//{{{XHTML}}}span').text == 'See earlier'
    assert len(root.findall(f'.//{{{XHTML}}}li')) == 1


def test_cache_reapplies_inclusion_policy_without_api_calls(tmp_path, monkeypatch):
    import modules.chapter_filter as cf
    source, dest = tmp_path / 'original.epub', tmp_path / 'filtered.epub'
    make_epub(source)
    filter_epub(source, dest, CONFIG, classifier)
    monkeypatch.setattr(cf, 'THRESHOLD', .99)
    def no_call(*args, **kwargs):
        raise AssertionError('threshold change should reuse judgments')
    with pytest.raises(VlmError, match='No substantive chapters'):
        filter_epub(source, dest, CONFIG, no_call)
    report = json.loads(dest.with_suffix('.substantive-filter.json').read_text())
    assert report['threshold'] == .99
    assert not any(c['included'] for c in report['chapters'])


def test_broken_nav_never_installs_output(tmp_path):
    source, dest = tmp_path / 'original.epub', tmp_path / 'filtered.epub'
    make_epub(source)
    with zipfile.ZipFile(source) as archive:
        entries = [(i, archive.read(i)) for i in archive.infolist()]
    with zipfile.ZipFile(source, 'w') as archive:
        for info, data in entries:
            if info.filename == 'OPS/TOC.xhtml':
                data = data.replace(b'article.xhtml', b'missing.xhtml')
            archive.writestr(info, data)
    dest.write_bytes(b'old-output')
    with pytest.raises(VlmError, match='Broken EPUB navigation'):
        filter_epub(source, dest, CONFIG, classifier)
    assert dest.read_bytes() == b'old-output'


def test_existing_epub_with_raw_html_breaks_filters_without_changing_source(tmp_path, capsys):
    source, dest = tmp_path / 'original.epub', tmp_path / 'filtered.epub'
    make_epub(source)
    with zipfile.ZipFile(source) as archive:
        entries = [(i, archive.read(i)) for i in archive.infolist()]
    with zipfile.ZipFile(source, 'w') as archive:
        for info, data in entries:
            if info.filename == 'OPS/article.xhtml':
                data = data.replace(b'Developed analysis.', b'Developed<br>analysis.')
            archive.writestr(info, data)
    original = source.read_bytes()
    filter_epub(source, dest, CONFIG, classifier)
    assert source.read_bytes() == original
    assert len(read_epub(dest)[3]) == 1
    with zipfile.ZipFile(dest) as archive:
        assert b'Developed<br />analysis.' in archive.read('OPS/article.xhtml')
    output = capsys.readouterr().out
    assert 'Article filter enabled: reading' in output
    assert 'Normalized HTML void elements in OPS/article.xhtml' in output


def test_unrepairable_xhtml_logs_filter_start_before_failing(tmp_path, capsys):
    source, dest = tmp_path / 'original.epub', tmp_path / 'filtered.epub'
    make_epub(source)
    with zipfile.ZipFile(source) as archive:
        entries = [(i, archive.read(i)) for i in archive.infolist()]
    with zipfile.ZipFile(source, 'w') as archive:
        for info, data in entries:
            if info.filename == 'OPS/article.xhtml':
                data = data.replace(b'</p>', b'</wrong>')
            archive.writestr(info, data)
    with pytest.raises(VlmError, match='Invalid XHTML in OPS/article.xhtml'):
        filter_epub(source, dest, CONFIG, classifier)
    assert 'Article filter enabled: reading' in capsys.readouterr().out
    assert not dest.exists()
