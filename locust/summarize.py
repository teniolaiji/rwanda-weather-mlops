"""Turn the Locust CSV results into a README table and a chart.

Usage: python locust/summarize.py                     (local Docker runs: results/c1_stats.csv ...)
       python locust/summarize.py --prefix cloud_c    (Azure runs: results/cloud_c1_stats.csv ...)
Writes: results/<label>summary.md and results/<label>latency_vs_containers.png
"""
import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default="c")
PREFIX = parser.parse_args().prefix
LABEL = "" if PREFIX == "c" else PREFIX.rstrip("c").rstrip("_") + "_"
PATTERN = re.compile(rf"^{re.escape(PREFIX)}(\d+)_stats\.csv$")


def read(path):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["Name"] == "/predict":
                return row
    return None


rows = []
files = [(int(m.group(1)), p) for p in RESULTS.glob("*_stats.csv") if (m := PATTERN.match(p.name))]
for n, path in sorted(files):
    r = read(path)
    if not r:
        continue
    reqs, fails = int(r["Request Count"]), int(r["Failure Count"])
    rows.append({
        "containers": n,
        "requests": reqs,
        "rps": float(r["Requests/s"]),
        "avg": float(r["Average Response Time"]),
        "median": float(r["Median Response Time"]),
        "p95": float(r["95%"]),
        "p99": float(r["99%"]),
        "max": float(r["Max Response Time"]),
        "fail_pct": 100 * fails / reqs if reqs else 0,
    })

if not rows:
    raise SystemExit(f"No results/{PREFIX}<N>_stats.csv files found. Run the load tests first.")

lines = [
    "| Containers | Requests | Requests/s | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) | Max (ms) | Failures |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(f"| {r['containers']} | {r['requests']:,} | {r['rps']:.1f} | {r['avg']:.0f} | {r['median']:.0f} | "
                 f"{r['p95']:.0f} | {r['p99']:.0f} | {r['max']:.0f} | {r['fail_pct']:.1f}% |")
table = "\n".join(lines)
(RESULTS / f"{LABEL}summary.md").write_text(table + "\n")
print(table)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["containers"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(x, [r["median"] for r in rows], "o-", label="Median", color="#2563a8")
    a1.plot(x, [r["p95"] for r in rows], "o-", label="95th percentile", color="#00a1de")
    a1.set(title="Response time vs containers", xlabel="Docker containers", ylabel="Response time (ms)", xticks=x)
    a1.legend(); a1.grid(alpha=.3)
    a2.bar([str(v) for v in x], [r["rps"] for r in rows], color="#00a1de")
    a2.set(title="Throughput vs containers", xlabel="Docker containers", ylabel="Requests per second")
    a2.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{LABEL}latency_vs_containers.png", dpi=150)
    print(f"\nSaved {LABEL}summary.md and {LABEL}latency_vs_containers.png in {RESULTS}")
except ImportError:
    print("matplotlib not installed: table saved, chart skipped")
