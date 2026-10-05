"""Pass-through timing: how quickly money that arrives at an account leaves it again."""

from __future__ import annotations

import pandas as pd


def forwarding_delays(tx: pd.DataFrame, laundering_only: bool = False) -> pd.DataFrame:
    """For each incoming transfer, minutes until the receiver's next outgoing transfer.

    Uses one vectorised merge_asof instead of a Python loop. `tx` must be sorted by ts.
    Self-transfers are excluded on both sides. NaN delay = nothing sent afterwards.
    """
    t = tx.loc[~tx["is_self"]]
    if laundering_only:
        t = t.loc[t["is_laundering"] == 1]
    incoming = pd.DataFrame({"ts": t["ts"].to_numpy(), "account_id": t["dst_id"].to_numpy()})
    outgoing = pd.DataFrame({"out_ts": t["ts"].to_numpy(), "account_id": t["src_id"].to_numpy()})
    merged = pd.merge_asof(
        incoming,
        outgoing,
        left_on="ts",
        right_on="out_ts",
        by="account_id",
        direction="forward",
        allow_exact_matches=True,
    )
    merged["delay_min"] = (merged["out_ts"] - merged["ts"]).dt.total_seconds() / 60
    return merged
