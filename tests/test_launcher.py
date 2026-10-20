"""Exercise launcher diagnostics with a fake Docker executable, never containers."""
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parent.parent / 'scripts' / 'pdf2epub.sh'


@pytest.mark.parametrize('suffix,flags', [
    ('.pdf', ['--vlm', '--articles-only', '--title', '{stem}', '--author', 'One; Two']),
    ('.epub', ['--articles-only', '--title', 'Issue #1 with spaces']),
    ('.pdf', []),
])
def test_echoed_flags_match_actual_literal_arguments_without_env_values(tmp_path, suffix, flags):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    fake = bin_dir / 'docker'
    fake.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                    'with open(os.environ["LAUNCHER_RECORD"], "w") as f:\n'
                    ' json.dump(sys.argv[1:], f)\n')
    fake.chmod(0o755)
    source = tmp_path / ('An issue #1' + suffix)
    source.touch()
    record = tmp_path / 'args.json'
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
               PDF2EPUB_CACHE_DIR=str(tmp_path / 'cache'), LAUNCHER_RECORD=str(record),
               PDF2EPUB_VLM_API_KEY='test-only-key-not-to-be-logged',
               COMMANDCODE_TOKEN='test-only-fallback-not-to-be-logged')
    result = subprocess.run(['bash', str(LAUNCHER), str(source), str(tmp_path / 'out'),
                             *flags], env=env, text=True, capture_output=True, check=True)
    line = next(line for line in result.stderr.splitlines()
                if line.startswith('Converter flags:'))
    assert shlex.split(line.partition(':')[2]) == ['--yes', *flags]
    args = json.loads(record.read_text())
    first_option = args.index('/output') + 1
    assert args[first_option:] == ['--yes', *flags]
    assert '--rm' in args
    assert 'test-only-key-not-to-be-logged' not in result.stderr + result.stdout
    assert 'test-only-fallback-not-to-be-logged' not in result.stderr + result.stdout
