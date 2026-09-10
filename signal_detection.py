"""
signal_detection.py
===================
Pharmacovigilance disproportionality analysis.

Computes PRR (Proportional Reporting Ratio) and ROR (Reporting Odds Ratio)
for suspect_drug × primary_reaction pairs from the FDA adverse events dataset.

Standard signal thresholds applied:
  - count (a) >= 3
  - PRR >= 2
  - chi-square >= 4   (approximate: uses (a - E)^2 / E where E = expected count)

Reference:
  Evans SJ et al. Use of proportional reporting ratios (PRRs) for signal
  generation from spontaneous adverse drug reaction reports.
  Pharmacoepidemiol Drug Saf. 2001;10(6):483-6.
"""

import numpy as np
import pandas as pd


# ── Core computation ──────────────────────────────────────────────────────────

def compute_prr_ror(df: pd.DataFrame,
                    drug_col: str = "suspect_drug",
                    reaction_col: str = "primary_reaction",
                    min_count: int = 3) -> pd.DataFrame:
    """
    Compute PRR and ROR for every drug–reaction pair with >= min_count reports.

    Parameters
    ----------
    df          : Raw dataframe containing at least drug_col and reaction_col.
    drug_col    : Column name for the suspect drug.
    reaction_col: Column name for the primary reaction.
    min_count   : Minimum number of co-reports to include a pair.

    Returns
    -------
    DataFrame with columns:
        drug, reaction, count_a, prr, ror, chi2, is_signal
    sorted descending by prr.
    """
    # Work on clean copies
    work = df[[drug_col, reaction_col]].dropna().copy()
    work.columns = ["drug", "reaction"]

    N = len(work)                                         # total reports

    # Marginal counts
    drug_counts     = work["drug"].value_counts()         # n_drug (a+b)
    reaction_counts = work["reaction"].value_counts()     # n_reaction (a+c)

    # Pair counts (a)
    pair_counts = (
        work.groupby(["drug", "reaction"])
        .size()
        .reset_index(name="count_a")
    )
    pair_counts = pair_counts[pair_counts["count_a"] >= min_count].copy()

    # Merge marginals
    pair_counts["n_drug"]     = pair_counts["drug"].map(drug_counts)
    pair_counts["n_reaction"] = pair_counts["reaction"].map(reaction_counts)

    a = pair_counts["count_a"].values.astype(float)
    b = pair_counts["n_drug"].values.astype(float)     - a   # drug, NOT reaction
    c = pair_counts["n_reaction"].values.astype(float) - a   # reaction, NOT drug
    d = N - a - b - c                                         # neither

    # Clip to avoid div-by-zero
    b = np.maximum(b, 0.5)
    c = np.maximum(c, 0.5)
    d = np.maximum(d, 0.5)

    # PRR = (a / (a+b)) / (c / (c+d))
    prr = (a / (a + b)) / (c / (c + d))

    # ROR = (a * d) / (b * c)
    ror = (a * d) / (b * c)

    # Chi-square (simple 1-df Yates-corrected approximation)
    E = (a + b) * (a + c) / N          # expected count
    chi2 = (a - E) ** 2 / np.maximum(E, 0.5)

    pair_counts["prr"]       = np.round(prr,  3)
    pair_counts["ror"]       = np.round(ror,  3)
    pair_counts["chi2"]      = np.round(chi2, 3)

    # Standard Evans signal flag: count>=3, PRR>=2, chi2>=4
    pair_counts["is_signal"] = (
        (pair_counts["count_a"] >= min_count) &
        (pair_counts["prr"]     >= 2.0) &
        (pair_counts["chi2"]    >= 4.0)
    )

    result = (
        pair_counts[["drug", "reaction", "count_a", "prr", "ror", "chi2", "is_signal"]]
        .sort_values("prr", ascending=False)
        .reset_index(drop=True)
    )
    return result


def top_signals(signals_df: pd.DataFrame,
                n: int = 100,
                only_flagged: bool = True) -> pd.DataFrame:
    """
    Return the top-n flagged (or all) signal pairs sorted by PRR descending.
    """
    df = signals_df.copy()
    if only_flagged:
        df = df[df["is_signal"]]
    return df.head(n).reset_index(drop=True)


def drug_signals(signals_df: pd.DataFrame, drug_name: str) -> pd.DataFrame:
    """
    Return all signal pairs for a specific drug.
    """
    mask = signals_df["drug"].str.upper() == drug_name.upper()
    return signals_df[mask].reset_index(drop=True)


# ── Pre-compute & cache helper ────────────────────────────────────────────────

_CACHE: dict = {}


def get_signals(df: pd.DataFrame,
                min_count: int = 3,
                force_recompute: bool = False) -> pd.DataFrame:
    """
    Compute PRR/ROR signals and cache the result in-process to avoid
    recomputing on every Streamlit rerun.
    """
    key = f"signals_{min_count}"
    if not force_recompute and key in _CACHE:
        return _CACHE[key]
    result = compute_prr_ror(df, min_count=min_count)
    _CACHE[key] = result
    return result


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    DATA = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
    print("Loading dataset...")
    df = pd.read_csv(DATA)
    print(f"  {len(df):,} rows loaded")

    print("\nComputing PRR/ROR signals (min_count=3)...")
    sigs = compute_prr_ror(df, min_count=3)
    print(f"  Total pairs evaluated : {len(sigs):,}")
    print(f"  Flagged signals       : {sigs['is_signal'].sum():,}")

    print("\nTop 10 flagged signals by PRR:")
    top = top_signals(sigs, n=10)
    print(top[["drug", "reaction", "count_a", "prr", "ror", "chi2"]].to_string(index=False))
