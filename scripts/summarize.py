"""Tabla agente x vista de marcas: media ± desvío entre semillas.

    python scripts/summarize.py runs            # imprime tabla markdown
    python scripts/summarize.py runs --csv out.csv
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYS = ["success", "steps_success", "revisit_frac", "write_rate", "return_base"]

ap = argparse.ArgumentParser()
ap.add_argument("runs", type=Path)
ap.add_argument("--csv", type=Path)
args = ap.parse_args()

groups = defaultdict(list)
for f in sorted(args.runs.glob("*/eval/*/summary.json")):
    s = json.load(open(f))
    run = f.parts[-4]
    exp = run.rsplit("_s", 1)[0]          # p. ej. A3_goal
    groups[(exp, f.parent.name)].append(s)

rows = []
for (exp, view), ss in sorted(groups.items()):
    row = dict(exp=exp, eval=view, n_seeds=len(ss))
    for k in KEYS:
        v = [s[k]["mean"] for s in ss if s[k]["mean"] is not None]
        row[k] = f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "-"
    rows.append(row)

cols = ["exp", "eval", "n_seeds"] + KEYS
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for r in rows:
    print("| " + " | ".join(str(r[c]) for c in cols) + " |")
if args.csv:
    import csv
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
