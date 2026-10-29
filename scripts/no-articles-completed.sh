#!/usr/bin/env bash
# Read-only batch predicate. Exit 0 only for this exact PDF's completed empty pass.
set -euo pipefail
[[ $# == 2 && -f $1 && -f $2 ]] || exit 1
command -v jq >/dev/null 2>&1 || exit 1
command -v sha256sum >/dev/null 2>&1 || exit 1
digest=$(sha256sum < "$1")
digest=${digest%% *}
jq -e --arg digest "$digest" '
  .status == "no_articles" and .output_written == false and
  .input_pdf_sha256 == $digest and .kept_chapters == 0 and
  .unreadable_chapters == 0 and (.total_chapters | type == "number") and
  .total_chapters > 0 and (.chapters | type == "array") and
  (.chapters | length) == .total_chapters and
  all(.chapters[]; .included == false and
      (.kind == "article" or .kind == "non_article" or .kind == "uncertain"))
' "$2" >/dev/null 2>&1
