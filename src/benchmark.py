"""
Inference cost benchmark: how much compute each classifier family actually costs.

This closes the remaining half of the base paper's second stated limitation. The
authors speculated that other models "could offer substantial improvements in speed
while maintaining similar accuracy levels" but never measured it, and our first pass
compared accuracy only.

Three things are done differently from the timing already in adaptive.py:

  1. Batch throughput and single sample latency are reported separately. Predicting
     the whole holdout in one call lets NumPy vectorise across the array, which is
     not what a deployed detector does when it scores one memory dump at a time.
     Reporting only the batch figure flatters every model equally and answers the
     wrong question.
  2. A warm up pass is discarded and the remaining runs are repeated, with the median
     reported rather than a single reading, because the first call carries lazy
     initialisation and cold caches and shared hardware is noisy.
  3. Threading is pinned to one core so the comparison reflects the work each model
     does rather than how much parallelism each library happened to grab.

Absolute timings are not comparable to the paper's, which were taken on an Intel
i5-1035G1. Ratios between families measured on one machine are, which is why the
report should lead with the ratio. Model size in kilobytes is directly comparable.

Usage:  python -m src.benchmark
"""

from __future__ import annotations

import os

# Must be set before scikit-learn, NumPy or XGBoost import their BLAS backends.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402

from src.adaptive import select_top_features, split_for_subtype  # noqa: E402
from src.comparison import build_models  # noqa: E402
from src.preprocessing import Dataset, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

REPEATS = 10
BATCH_ROWS = 10_000
SINGLE_CALLS = 200


# ----------------------------------------------------------------- hardware ----
def host_spec() -> dict:
    """Everything needed to make these timings reproducible on another machine."""
    cpu = platform.processor() or platform.machine()
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    elif sys.platform == "win32":
        # platform.processor() returns the family and model identifier on Windows,
        # so take the marketing name from the registry instead.
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass

    ram_gb = None
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    ram_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
        elif sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            ram_gb = round(status.ullTotalPhys / 1024 ** 3, 1)
        else:
            ram_gb = round(os.sysconf("SC_PAGE_SIZE")
                           * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except (OSError, ValueError, AttributeError):
        pass

    import sklearn
    spec = {
        "label": os.environ.get("MALXPLAIN_MACHINE_LABEL", "unlabelled machine"),
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "ram_gb": ram_gb,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "threads_pinned_to": 1,
        "measured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import xgboost
        spec["xgboost"] = xgboost.__version__
    except ImportError:
        spec["xgboost"] = None
    return spec


# ---------------------------------------------------------------- measuring ---
def _median_us(fn, repeats: int, per_call: int) -> tuple[float, float]:
    """Run fn repeats times after a warm up, return (median us per item, IQR)."""
    fn()  # warm up, discarded
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6 / per_call)
    samples.sort()
    q1 = statistics.median(samples[: len(samples) // 2])
    q3 = statistics.median(samples[(len(samples) + 1) // 2:])
    return statistics.median(samples), q3 - q1


def benchmark_model(clf, X_test: pd.DataFrame) -> dict:
    batch = X_test.iloc[:BATCH_ROWS].to_numpy()
    singles = [X_test.iloc[i: i + 1].to_numpy() for i in range(SINGLE_CALLS)]

    batch_us, batch_iqr = _median_us(
        lambda: clf.predict(batch), REPEATS, len(batch)
    )

    def _one_at_a_time():
        for row in singles:
            clf.predict(row)

    single_us, single_iqr = _median_us(_one_at_a_time, REPEATS, SINGLE_CALLS)

    blob = pickle.dumps(clf)
    return {
        "Batch us/sample": round(batch_us, 3),
        "Batch IQR": round(batch_iqr, 3),
        "Single us/sample": round(single_us, 2),
        "Single IQR": round(single_iqr, 2),
        "Model size (KB)": round(len(blob) / 1024, 2),
    }


def run_all(ds: Dataset | None = None) -> pd.DataFrame:
    ds = ds or prepare()
    subtypes = ds.subtypes
    models = build_models()

    rows = []
    for i, subtype in enumerate(subtypes, 1):
        X_train, y_train, X_test, _, _ = split_for_subtype(ds, subtype)
        feats = select_top_features(X_train, y_train)
        # Fit and predict on plain arrays so the timing reflects the deployed
        # path rather than pandas column validation overhead.
        Xtr = X_train[feats].to_numpy()
        Xte = X_test[feats]

        for name, proto in models.items():
            clf = clone(proto)
            # Single threaded fitting too, so the comparison is clean end to end.
            if hasattr(clf, "n_jobs"):
                clf.set_params(n_jobs=1)
            clf.fit(Xtr, y_train.to_numpy())
            rec = {"Subtype": subtype, "Classifier": name}
            rec.update(benchmark_model(clf, Xte))
            rows.append(rec)
        print(f"  [{i:2d}/{len(subtypes)}] {subtype}")

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, accuracy_source: Path | None = None) -> pd.DataFrame:
    agg = df.groupby("Classifier").agg(
        **{
            "Batch us/sample": ("Batch us/sample", "median"),
            "Single us/sample": ("Single us/sample", "median"),
            "Model size (KB)": ("Model size (KB)", "mean"),
        }
    ).reset_index()

    src = accuracy_source or (RESULTS_DIR / "upgrade2_comparison_summary.csv")
    if src.exists():
        acc = pd.read_csv(src)[["Classifier", "Mean_accuracy", "Mean_unseen_recall"]]
        agg = agg.merge(acc, on="Classifier", how="left")

    # Ratios are the hardware independent part, so they carry the headline claim.
    rf = agg.loc[agg["Classifier"] == "Random Forest"]
    if not rf.empty:
        for col, label in (("Single us/sample", "Latency vs RF"),
                           ("Model size (KB)", "Size vs RF")):
            base = rf[col].iloc[0]
            agg[label] = (agg[col] / base).round(2)

    return agg.sort_values("Single us/sample").reset_index(drop=True)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    spec = host_spec()
    print("=== Inference cost benchmark ===")
    print(f"  machine : {spec['label']}")
    print(f"  cpu     : {spec['cpu']} ({spec['logical_cores']} logical cores)")
    print(f"  ram     : {spec['ram_gb']} GB")
    print(f"  python  : {spec['python']}, scikit-learn {spec['scikit_learn']}")
    print(f"  threads : pinned to 1\n")

    df = run_all()
    label = spec["label"].replace(" ", "_").replace("/", "_")
    df.to_csv(RESULTS_DIR / f"upgrade5_benchmark_{label}.csv", index=False)

    summary = summarise(df)
    summary.insert(0, "Machine", spec["label"])
    summary.to_csv(RESULTS_DIR / f"upgrade5_benchmark_summary_{label}.csv", index=False)
    (RESULTS_DIR / f"upgrade5_host_{label}.json").write_text(json.dumps(spec, indent=2))

    print("\n" + summary.to_string(index=False))
    print(f"\n[benchmark] wrote upgrade5_benchmark_{label}.csv and its summary")
    print("[benchmark] absolute times are specific to this machine; the ratio "
          "columns are the comparable part")


if __name__ == "__main__":
    main()
