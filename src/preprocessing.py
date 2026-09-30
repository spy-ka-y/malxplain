"""
MalXplain: data preparation module.

Replicates Part I ("Data preparation") of Madamidola, Ngobigha & Ez-zizi (2024),
"Detecting new obfuscated malware variants: A lightweight and interpretable
machine learning approach", Intelligent Systems with Applications 25, 200472.

Steps, exactly as described in the paper (Section 3.2.1):
  1. Remove invariant features (pslist.nprocs64bit, handles.nport,
     svcscan.interactive_process_services).
  2. Parse the malware family and subtype out of the `Category` column.
  3. Label-encode the target (Benign -> 0, Malware -> 1).
  4. Min-max scale the continuous features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# The three features the paper drops because they are constant across the dataset.
INVARIANT_FEATURES = [
    "pslist.nprocs64bit",
    "handles.nport",
    "svcscan.interactive_process_services",
]

BENIGN_LABEL = "Benign"
MALWARE_LABEL = "Malware"

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Obfuscated-MalMem2022.csv"


@dataclass
class Dataset:
    """Container for the prepared dataset."""

    X: pd.DataFrame          # scaled feature matrix
    y: pd.Series             # 0 = benign, 1 = malware
    family: pd.Series        # Ransomware / Spyware / Trojan, or "Benign"
    subtype: pd.Series       # e.g. Transponder, Zeus, ... or "Benign"
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.X)

    @property
    def subtypes(self) -> list[str]:
        """The 15 malware subtype names, sorted."""
        return sorted(s for s in self.subtype.unique() if s != BENIGN_LABEL)


def load_raw(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw CIC-MalMem-2022 CSV."""
    df = pd.read_csv(path)
    if df.shape[0] != 58596:
        raise ValueError(f"Expected 58,596 records (paper Section 3.1), found {df.shape[0]}")
    return df


def parse_category(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Split the `Category` column into (family, subtype).

    Malicious rows look like  "Ransomware-Ako-<sha256>-1.raw"
    Benign rows are simply    "Benign"
    """
    parts = df["Category"].str.split("-", n=2, expand=True)
    family = parts[0]
    subtype = parts[1].fillna(BENIGN_LABEL)
    subtype = subtype.where(family != BENIGN_LABEL, BENIGN_LABEL)
    return family, subtype


def find_invariant_features(df: pd.DataFrame) -> list[str]:
    """Return the numeric columns that carry no information (a single unique value)."""
    numeric = df.select_dtypes(include="number")
    return [c for c in numeric.columns if numeric[c].nunique(dropna=False) <= 1]


def prepare(
    path: str | Path = DEFAULT_DATA_PATH,
    scale: bool = True,
    verbose: bool = True,
) -> Dataset:
    """Run the full data-preparation pipeline and return a `Dataset`."""
    df = load_raw(path)
    family, subtype = parse_category(df)

    # Target encoding: Benign -> 0, Malware -> 1
    y = (df["Class"] == MALWARE_LABEL).astype(int)
    y.name = "Class"

    features = df.drop(columns=["Category", "Class"])

    detected = find_invariant_features(features)
    to_drop = [c for c in INVARIANT_FEATURES if c in features.columns]
    # Guard: the paper names three invariant features; verify that claim holds here.
    unexpected = sorted(set(detected) - set(to_drop))
    missing = sorted(set(to_drop) - set(detected))
    if verbose:
        print(f"[prep] raw features            : {features.shape[1]}")
        print(f"[prep] invariant (detected)    : {sorted(detected)}")
        if unexpected:
            print(f"[prep] NOTE extra invariants   : {unexpected}")
        if missing:
            print(f"[prep] NOTE named-but-variant  : {missing}")

    features = features.drop(columns=to_drop)

    feature_names = list(features.columns)
    if scale:
        scaler = MinMaxScaler()
        features = pd.DataFrame(
            scaler.fit_transform(features), columns=feature_names, index=features.index
        )

    if verbose:
        print(f"[prep] features after drop     : {features.shape[1]}")
        print(f"[prep] records                 : {len(features)}")
        print(f"[prep] benign / malware        : {(y == 0).sum()} / {(y == 1).sum()}")
        print(f"[prep] malware subtypes        : {subtype[y == 1].nunique()}")

    return Dataset(
        X=features,
        y=y,
        family=family,
        subtype=subtype,
        feature_names=feature_names,
    )


if __name__ == "__main__":
    ds = prepare()
    print("\nSubtypes:", ds.subtypes)
