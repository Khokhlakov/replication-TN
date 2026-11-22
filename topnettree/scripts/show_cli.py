#!/usr/bin/env python3
"""Show what pipeline scripts take and which tables they mention, without running them.

For each script: the start of its docstring, every argparse option (name, default, required,
help), and every string constant that names a csv or a data path. It only parses the source
(ast), so nothing is executed and no file is touched. That matters here: some pipeline scripts
write tables as soon as they start, and `--help` would run them if they do not use argparse.

    python scripts/show_cli.py scripts/41_build_feature_rows.py scripts/62_make_prediction_manifest.py
    python scripts/show_cli.py scripts/37*.py > logs/cli_37.txt
"""

from __future__ import annotations

import ast
import sys


def const(node):
    return node.value if isinstance(node, ast.Constant) else None


def inspect(path):
    print("=" * 78)
    print(path)
    src = open(path, encoding="utf-8", errors="replace").read()
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        print(f"  cannot parse: {exc}")
        return
    doc = ast.get_docstring(tree) or "(no module docstring)"
    print("\n".join(doc.splitlines()[:30]))

    options, uses_argparse = [], False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "ArgumentParser":
                uses_argparse = True
            if node.func.attr == "add_argument":
                flags = [const(a) for a in node.args if isinstance(const(a), str)]
                kw = {k.arg: const(k.value) for k in node.keywords if k.arg}
                options.append((flags, kw))
    print("\n--- options ---")
    if not options:
        print("  none found" + ("" if uses_argparse else " (no argparse: running this script takes no arguments)"))
    for flags, kw in options:
        bits = [f"{k}={kw[k]!r}" for k in ("default", "required", "choices", "action") if k in kw]
        print(f"  {' '.join(flags):<28} {', '.join(bits)}  {kw.get('help') or ''}")

    paths = sorted(
        {
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and ("data/" in n.value or n.value.endswith((".csv", ".json")))
            and len(n.value) < 140
            and "\n" not in n.value
        }
    )
    print("\n--- data paths mentioned ---")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        inspect(p)
