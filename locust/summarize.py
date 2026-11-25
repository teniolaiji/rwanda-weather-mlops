"""Turn the Locust CSV results into a README table and a chart.

Usage: python locust/summarize.py            (reads results/c*_stats.csv)
Writes: results/summary.md and results/latency_vs_containers.png
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def read(path):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["Name"] == "/predict":
                return row
    return None


rows = []
for path in sorted(RESULTS.glob("c*_stats.csv"), key=lambda p: int(re.findall(r"\d+", p.stem)[0])):
    r = read(path)
    if not r:
        continue
    n = int(re.findall(r"\d+", path.stem)[0])
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
    raise SystemExit("No results/c*_stats.csv files found. Run the load tests first.")

lines = [
    "| Containers | Requests | Requests/s | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) | Max (ms) | Failures |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(f"| {r['containers']} | {r['requests']:,} | {r['rps']:.1f} | {r['avg']:.0f} | {r['median']:.0f} | "
                 f"{r['p95']:.0f} | {r['p99']:.0f} | {r['max']:.0f} | {r['fail_pct']:.1f}% |")
table = "\n".join(lines)
(RESULTS / "summary.md").write_text(table + "\n")
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
    fig.savefig(RESULTS / "latency_vs_containers.png", dpi=150)
    print(f"\nSaved {RESULTS / 'summary.md'} and {RESULTS / 'latency_vs_containers.png'}")
except ImportError:
    print("matplotlib not installed: table saved, chart skipped")
