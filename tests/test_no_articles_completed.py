"""Real read-only batch skip predicate: never invokes Docker or the converter."""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

CHECK = Path(__file__).resolve().parent.parent / 'scripts/no-articles-completed.sh'


def fixture(tmp_path):
    pdf = tmp_path / 'Catalog #1 with spaces.pdf'
    pdf.write_bytes(b'%PDF catalog')
    report = tmp_path / 'Catalog.substantive-filter.json'
    data = dict(status='no_articles', output_written=False,
                input_pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                kept_chapters=0, total_chapters=1, unreadable_chapters=0,
                chapters=[dict(included=False, kind='non_article')])
    report.write_text(json.dumps(data))
    return pdf, report, data


def check(pdf, report):
    return subprocess.run(['bash', str(CHECK), str(pdf), str(report)],
                          capture_output=True).returncode


def test_exact_completed_source_skips_but_changed_pdf_does_not(tmp_path):
    pdf, report, _ = fixture(tmp_path)
    assert check(pdf, report) == 0
    original = pdf.read_bytes()
    assert pdf.read_bytes() == original
    pdf.write_bytes(b'%PDF changed')
    assert check(pdf, report) != 0


@pytest.mark.parametrize('field,value', [
    ('status', 'classifying'), ('output_written', True), ('kept_chapters', 1),
    ('total_chapters', 0), ('total_chapters', 2), ('unreadable_chapters', 1),
    ('input_pdf_sha256', 'wrong'), ('chapters', []),
    ('chapters', [dict(included=True, kind='article')]),
    ('chapters', [dict(included=False, kind='unreadable')]),
])
def test_partial_or_invalid_report_does_not_skip(tmp_path, field, value):
    pdf, report, data = fixture(tmp_path)
    data[field] = value
    report.write_text(json.dumps(data))
    assert check(pdf, report) != 0


def test_missing_or_malformed_report_does_not_skip(tmp_path):
    pdf, report, _ = fixture(tmp_path)
    report.write_text('{broken')
    assert check(pdf, report) != 0
    report.unlink()
    assert check(pdf, report) != 0
