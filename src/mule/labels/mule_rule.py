"""The mule label.

Rule B ("completed", the label): an account is a mule in a window if, inside that window, it
sends a laundering transaction at time t2 after receiving one at some time t1 <= t2. The receipt
may fall in an earlier window, so pass-throughs that straddle a cutoff are not lost.

Rule A ("within", reported for comparison): receipt and send must both fall inside the window.

Under both rules self-transfers count as neither receiving nor sending, but they do count as
activity, and a cycle's originator (sends first, receives at the end) is not a mule unless it
sends again later. Accounts that touch laundered money in the window without meeting the rule
(pure sources, final cash-out accounts) are flagged `adjacent`.

Because "earliest receipt <= latest send" holds exactly when some receipt precedes some send,
each rule needs one groupby per direction rather than a pairwise search.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Rule = Literal["completed", "within"]


def window_slice(tx: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Transactions with start <= ts < end; None means unbounded on that side."""
    mask = np.ones(len(tx), dtype=bool)
    if start is not None:
        mask &= (tx["ts"] >= pd.Timestamp(start)).to_numpy()
    if end is not None:
        mask &= (tx["ts"] < pd.Timestamp(end)).to_numpy()
    return tx.loc[mask]


def _laundering(tx: pd.DataFrame) -> pd.DataFrame:
    return tx.loc[(tx["is_laundering"] == 1) & ~tx["is_self"]]


def label_accounts(
    tx: pd.DataFrame, start=None, end=None, rule: Rule = "completed"
) -> pd.DataFrame:
    """One row per account active in [start, end): account_id, is_mule, adjacent."""
    window = window_slice(tx, start, end)
    active = np.union1d(window["src_id"].to_numpy(), window["dst_id"].to_numpy())

    in_window = _laundering(window)
    receipts = in_window if rule == "within" else _laundering(window_slice(tx, None, end))
    first_in = receipts.groupby("dst_id")["ts"].min()
    last_out = in_window.groupby("src_id")["ts"].max()
    both = pd.concat(
        [first_in.rename("first_in"), last_out.rename("last_out")], axis=1, join="inner"
    )
    mules = both.index[both["first_in"] <= both["last_out"]].to_numpy()
    touched = np.union1d(in_window["src_id"].to_numpy(), in_window["dst_id"].to_numpy())

    labels = pd.DataFrame({"account_id": active})
    labels["is_mule"] = np.isin(labels["account_id"].to_numpy(), mules)
    labels["adjacent"] = np.isin(labels["account_id"].to_numpy(), touched) & ~labels["is_mule"]
    return labels
