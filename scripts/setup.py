"""
MalXplain: one-command bootstrap.

Downloads the CIC-MalMem-2022 dataset and verifies the environment, so a freshly
cloned repository is ready to run the pipeline and the Streamlit prototype.

Usage:  python scripts/setup.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "Obfuscated-MalMem2022.csv"

# The dataset is distributed by the Canadian Institute for Cybersecurity and is
# mirrored in the base paper's own public repository.
DATA_URL = (
    "https://raw.githubusercontent.com/Adnane017/"
    "Detecting_new_obfuscated_malware_variants/main/Obfuscated-MalMem2022.csv"
)
EXPECTED_ROWS = 58596


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = 100 * done / total_size
    bar = "#" * int(pct // 2.5)
    sys.stdout.write(f"\r    [{bar:<40}] {pct:5.1f}%  ({done/1e6:.1f}/{total_size/1e6:.1f} MB)")
    sys.stdout.flush()


def download_dataset(force: bool = False) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    if CSV_PATH.exists() and not force:
        print(f"[1/3] Dataset already present: {CSV_PATH.name}")
        return CSV_PATH

    print(f"[1/3] Downloading CIC-MalMem-2022 ...")
    urllib.request.urlretrieve(DATA_URL, CSV_PATH, reporthook=_progress)
    print()
    return CSV_PATH


def verify_dataset(path: Path) -> bool:
    print("[2/3] Verifying dataset integrity ...")
    try:
        import pandas as pd
    except ImportError:
        print("    ! pandas not installed, run: pip install -r requirements.txt")
        return False

    df = pd.read_csv(path)
    rows_ok = len(df) == EXPECTED_ROWS
    balance_ok = (df["Class"] == "Benign").sum() == (df["Class"] == "Malware").sum() == 29298
    subtypes = df.loc[df["Class"] == "Malware", "Category"].str.split("-").str[1].nunique()

    print(f"    records            : {len(df):,} {'OK' if rows_ok else 'MISMATCH'} (expected 58,596)")
    print(f"    benign/malware 50/50: {'OK' if balance_ok else 'MISMATCH'}")
    print(f"    malware subtypes    : {subtypes} {'OK' if subtypes == 15 else 'MISMATCH'} (expected 15)")
    return rows_ok and balance_ok and subtypes == 15


def check_environment() -> bool:
    print("[3/3] Checking environment ...")
    ok = True
    for pkg in ["pandas", "numpy", "sklearn", "shap", "xgboost", "streamlit"]:
        try:
            __import__(pkg)
            print(f"    {pkg:<12} OK")
        except ImportError:
            print(f"    {pkg:<12} MISSING  -> pip install -r requirements.txt")
            ok = False

    sys.path.insert(0, str(ROOT))
    try:
        from src.reporter import list_models, ollama_available
        if ollama_available():
            models = list_models()
            print(f"    ollama       OK  (models: {', '.join(models) or 'none pulled'})")
        else:
            print("    ollama       not reachable, the app will use the template fallback.")
            print("                 Install from https://ollama.com and run: ollama pull llama3.2:3b")
    except Exception as exc:
        print(f"    ollama       check skipped ({exc})")

    return ok


def main() -> None:
    print("=" * 70)
    print("MalXplain setup")
    print("=" * 70)

    path = download_dataset()
    data_ok = verify_dataset(path)
    env_ok = check_environment()

    print("\n" + "=" * 70)
    if data_ok and env_ok:
        print("Setup complete. Next steps:\n")
        print("  python -m src.verify      # reproduce and verify against the paper")
        print("  python -m streamlit run app.py   # launch the dashboard")
    else:
        print("Setup incomplete, see the messages above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
