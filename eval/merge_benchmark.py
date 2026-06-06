"""
Merge no-aug benchmark (has OR-Tools data) with aug benchmark (has pomo_aug_mean).
Run once both CSVs exist:
  python eval/merge_benchmark.py
"""
import csv, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_csv(path):
    with open(path) as f:
        return {(r["problem"], r["n"]): r for r in csv.DictReader(f)}


def val(s):
    return s if s not in ("", "None") else None


noaug_path = os.path.join(ROOT, "data", "processed", "benchmark_results_noaug.csv")
aug_path   = os.path.join(ROOT, "data", "processed", "benchmark_results.csv")
out_path   = os.path.join(ROOT, "data", "processed", "benchmark_results_full.csv")

noaug = read_csv(noaug_path)
aug   = read_csv(aug_path)

rows = []
for key in noaug:
    na, a = noaug[key], aug.get(key, {})
    ort   = val(na.get("ort_mean"))
    paug  = val(a.get("pomo_aug_mean"))
    aug_gap = None
    if ort and paug:
        aug_gap = round(100 * (float(paug) - float(ort)) / float(ort), 2)

    rows.append({
        "problem":          na["problem"],
        "n":                na["n"],
        "nn_mean":          val(na.get("nn_mean")),
        "pomo_mean":        val(na.get("pomo_mean")),
        "pomo_aug_mean":    paug,
        "ort_mean":         ort,
        "ort_n_solved":     val(na.get("ort_n_solved")),
        "pomo_gap_pct":     val(na.get("pomo_gap_pct")),
        "pomo_aug_gap_pct": aug_gap,
    })

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Merged {len(rows)} rows → {out_path}")
for r in rows:
    print(f"  {r['problem']:>5} n={r['n']:>2}  "
          f"NN={r['nn_mean'] or '—':>7}  "
          f"POMO={r['pomo_mean'] or '—':>7}  "
          f"POMO+aug={r['pomo_aug_mean'] or '—':>7}  "
          f"ORT={r['ort_mean'] or '—':>7}  "
          f"gap={r['pomo_gap_pct'] or '—':>7}  "
          f"aug_gap={r['pomo_aug_gap_pct'] or '—':>7}")
