"""Check real archive navigation, including readers that decode URLs early.

Also executable without pytest inside the release image before publication.
"""

import json
import posixpath
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET
from zipfile import ZIP_STORED, ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import mark2epub


def verify_navigation_output(directory):
    work = directory / "markdown"
    work.mkdir()
    filenames = [
        "mojo #20261201-chapter-001.md",
        "mojo #20261201-chapter-002.md",
        "music.2026 #50% ? café & live.md",
    ]
    titles = ["Jane Wiedlin", "A Year In The Life", "Café & Live"]
    chapters = []
    for name, title in zip(filenames, titles):
        (work / name).write_text(
            f"# {title}\n\nArticle: {title}.\n\n"
            "| Model | Type |\n| --- | --- |\n| Aurora<br>Beat | Dual-Röhren |\n",
            encoding="utf-8",
        )
        chapters.append({"markdown": name, "title": title, "css": ""})
    (work / "description.json").write_text(json.dumps({"chapters": chapters}))
    output = directory / "output"
    mark2epub.convert_to_epub(work, output, metadata={"dc:title": "Mojo #20261201"})

    opf_ns = {"o": "http://www.idpf.org/2007/opf"}
    xhtml_ns = {"h": "http://www.w3.org/1999/xhtml"}
    ncx_ns = {"n": "http://www.daisy.org/z3986/2005/ncx/"}
    with ZipFile(output / "markdown.epub") as archive:
        assert archive.testzip() is None
        assert archive.infolist()[0].filename == "mimetype"
        assert archive.infolist()[0].compress_type == ZIP_STORED
        package = ET.fromstring(archive.read("OPS/package.opf"))
        items = package.findall("o:manifest/o:item", opf_ns)
        manifest = {item.attrib["id"]: item.attrib["href"] for item in items}
        for href in manifest.values():
            assert posixpath.normpath("OPS/" + unquote(urlsplit(href).path)) in archive.namelist()

        spine = [manifest[item.attrib["idref"]] for item in package.findall("o:spine/o:itemref", opf_ns)]
        nav_item = next(item for item in items if "nav" in item.attrib.get("properties", "").split())
        nav = ET.fromstring(archive.read("OPS/" + nav_item.attrib["href"]))
        links = nav.findall(".//h:nav/h:ol/h:li/h:a", xhtml_ns)
        ncx = ET.fromstring(archive.read("OPS/" + manifest["ncx"]))
        ncx_links = ncx.findall(".//n:content", ncx_ns)
        assert [link.text for link in links] == titles
        hrefs = [link.attrib["href"] for link in links]
        assert len(hrefs) == len(set(hrefs)) == len(titles)
        assert hrefs == spine[1:] == [link.attrib["src"] for link in ncx_links]
        for href, title in zip(hrefs, titles):
            # Exercise the early-decode behavior reported by the reader.
            decoded = unquote(href)
            assert decoded == href
            assert not any(c in decoded for c in " #?%&")
            assert decoded.isascii()
            target = urlsplit(decoded)
            assert not target.fragment and not target.query
            chapter = ET.fromstring(archive.read("OPS/" + target.path))
            assert chapter.find(".//h:h1", xhtml_ns).text == title
            assert len(chapter.findall('.//h:br', xhtml_ns)) == 1


def test_navigation_targets_resolve_for_special_source_filenames(tmp_path):
    verify_navigation_output(tmp_path)


if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        verify_navigation_output(Path(tmp))
    print("EPUB navigation regression check passed")
