#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = RELEASE_ROOT / "manifests" / "validation" / "closure_report.json"
SELF_PATH = Path(__file__).resolve()
TEXT_EXTS = {"", ".py", ".sh", ".json", ".md", ".txt"}
SKIP_DIR_NAMES = {"__pycache__", ".git"}
BANNED = [
    re.compile("/" + "Users" + "/"),
    re.compile(r"/home/[a-z]"),
    re.compile("MoReBench" + "_backup"),
    re.compile("our" + "_experiments/"),
    re.compile(r"outputs/20\d{2}-\d{2}-\d{2}"),
    re.compile(r"release_staging/data/criterion_pairs"),
    re.compile(r"code/scripts/"),
]

issues = []
scanned = 0
for path in sorted(RELEASE_ROOT.rglob("*")):
    if not path.is_file():
        continue
    if path.resolve() == SELF_PATH:
        continue
    if path.resolve() == OUT_PATH.resolve():
        continue
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        continue
    if path.suffix not in TEXT_EXTS and path.parent.name != "bin":
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    scanned += 1
    rel = path.relative_to(RELEASE_ROOT).as_posix()
    for pattern in BANNED:
        for m in pattern.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            line = text.splitlines()[line_no - 1].strip() if text.splitlines() else ""
            issues.append({
                "path": rel,
                "line": line_no,
                "pattern": pattern.pattern,
                "excerpt": line[:240],
            })

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
report = {
    "release_root": RELEASE_ROOT.name,
    "scanned_files": scanned,
    "issue_count": len(issues),
    "issues": issues,
    "status": "ok" if not issues else "failed",
}
OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"closure scan: {scanned} files, {len(issues)} issue(s)")
if issues:
    print(f"details -> {OUT_PATH}")
    sys.exit(1)
print(f"details -> {OUT_PATH}")
