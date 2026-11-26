"""
Run every chapter script one after another (a quick "does everything work?" check).

Usage:
    python run_all.py              # all chapters (about 30-40 minutes on a GPU, much longer on a CPU)
    python run_all.py 4 5          # only chapters 4 and 5
    python run_all.py --list       # just list the scripts

Notes:
  * Chapters run in order: a few later scripts reuse models trained earlier (e.g. Chapter 18
    reuses Chapter 1's cat/dog model, and trains it if it's missing).
  * Scripts with a web server or an interactive part are run with --selftest so they exit by themselves.
  * Scripts that need an optional package (matplotlib, scikit-learn, ...) print a message and skip
    that part when it is missing.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXTRA_ARGS = {"06_web_app.py": ["--selftest"]}

args = [a for a in sys.argv[1:] if not a.startswith("--")]
chapters = sorted(ROOT.glob("Chapter_*"))
if args:
    chapters = [c for c in chapters if any(c.name.startswith(f"Chapter_{int(a):02d}") for a in args)]

scripts = [s for c in chapters for s in sorted(c.glob("[0-9][0-9]_*.py"))]
if "--list" in sys.argv:
    print("\n".join(str(s.relative_to(ROOT)) for s in scripts))
    raise SystemExit

results = []
for script in scripts:
    rel = script.relative_to(ROOT)
    print(f"\n{'=' * 70}\n>>> {rel}\n{'=' * 70}", flush=True)
    t0 = time.perf_counter()
    code = subprocess.call([sys.executable, script.name, *EXTRA_ARGS.get(script.name, [])], cwd=script.parent)
    results.append((rel, code, time.perf_counter() - t0))

print(f"\n{'=' * 70}\nSUMMARY")
for rel, code, secs in results:
    print(f"  {'OK  ' if code == 0 else 'FAIL'} {secs:6.1f}s  {rel}")
failed = sum(code != 0 for _, code, _ in results)
print(f"\n{len(results) - failed}/{len(results)} scripts succeeded.")
sys.exit(1 if failed else 0)
