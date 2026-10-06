"""Group 8, pattern features: the laundering shapes the AMLworld simulator plants.

Computed per window span (non-self transfers), with the same no-future, no-label rules as
every other feature. Path-based counts use time-respecting 2-hop paths S -> B -> T: B's
first receipt from S comes no later than its last send to T.

Account features (prefix pa_):
  fan-in / fan-out bursts : most incoming / outgoing transfers in any trailing 24 hours
  scatter-gather          : (S, T) pairs joined by two or more different intermediaries;
                            counted for the account as middle, as source and as target
  3-cycles                : directed cycles S -> B -> T -> S through the account
  2-hop throughput        : time-respecting 2-hop paths through the account
Accounts whose distinct senders x distinct receivers exceeds `pair_cap` are hubs: their
middle-role counts would explode, so those are NaN (XGBoost treats NaN as "unknown").

Transaction features (prefix e_pa_, for the transaction task): hours until the receiver's
next send, hours since the sender's last receipt, transfers between the same pair, and the
scatter-gather and 3-cycle counts of the (sender, receiver) pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACCOUNT_COLUMNS = [
    "pa_log_fan_in_24h_max",
    "pa_log_fan_out_24h_max",
    "pa_log_sg_middle",
    "pa_log_sg_source",
    "pa_log_sg_target",
    "pa_log_cycle3",
    "pa_log_paths2",
]
EDGE_COLUMNS = [
    "e_pa_log_fwd_delay_h",
    "e_pa_log_src_prev_in_h",
    "e_pa_log_pair_count",
    "e_pa_log_sg",
    "e_pa_log_cycle3",
]
DAY_MIN = 1440


def _burst_max(account: np.ndarray, minutes: np.ndarray, window: int = DAY_MIN) -> pd.Series:
    """Per account, the most events inside any trailing `window` minutes (vectorised)."""
    if len(account) == 0:
        return pd.Series(dtype=float)
    order = np.lexsort((minutes, account))
    acc, mins = account[order].astype(np.int64), minutes[order].astype(np.int64)
    key = acc * (mins.max() + 2 * window + 1) + mins  # sorted, and accounts never overlap
    first = np.searchsorted(key, key - window, side="left")
    counts = np.arange(len(key)) - first + 1
    return pd.Series(counts).groupby(acc).max()


def _paths(nonself: pd.DataFrame, pair_cap: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Distinct pairs, time-respecting 2-hop paths, and the hub accounts left out."""
    pairs = (
        nonself.groupby(["src_id", "dst_id"])["ts"]
        .agg(t_first="min", t_last="max", n="size")
        .reset_index()
    )
    n_in = pairs.groupby("dst_id").size()
    n_out = pairs.groupby("src_id").size()
    product = n_in.mul(n_out, fill_value=0)
    hubs = product.index[product > pair_cap].to_numpy()
    left = pairs.loc[~pairs["dst_id"].isin(hubs), ["src_id", "dst_id", "t_first"]]
    left = left.rename(columns={"src_id": "S", "dst_id": "B", "t_first": "t_in"})
    right = pairs.loc[~pairs["src_id"].isin(hubs), ["src_id", "dst_id", "t_last"]]
    right = right.rename(columns={"src_id": "B", "dst_id": "T", "t_last": "t_out"})
    paths = left.merge(right, on="B")
    paths = paths.loc[
        (paths["S"] != paths["T"]) & (paths["t_in"] <= paths["t_out"]), ["S", "B", "T"]
    ]
    return pairs, paths.reset_index(drop=True), hubs


def pattern_tables(span: pd.DataFrame, pair_cap: int = 400):
    """Account-level pattern features for every account in the span, plus a per-pair table
    (sg, cycle3) for the transaction features. Expects span sorted by ts."""
    n = span.loc[~span["is_self"], ["src_id", "dst_id", "ts"]]
    accounts = np.union1d(n["src_id"].to_numpy(), n["dst_id"].to_numpy())
    out = pd.DataFrame(index=pd.Index(accounts, name="account_id"))
    if n.empty:
        for c in ACCOUNT_COLUMNS:
            out[c] = np.nan
        return out, pd.DataFrame(columns=["src_id", "dst_id", "n", "sg", "cycle3"]), np.array([])

    minutes = ((n["ts"] - n["ts"].min()).dt.total_seconds() // 60).to_numpy()
    out["pa_log_fan_in_24h_max"] = np.log1p(_burst_max(n["dst_id"].to_numpy(), minutes))
    out["pa_log_fan_out_24h_max"] = np.log1p(_burst_max(n["src_id"].to_numpy(), minutes))

    pairs, paths, hubs = _paths(n, pair_cap)
    middles = paths.groupby(["S", "T"])["B"].transform("size")
    sg = paths.loc[middles >= 2]
    st = sg[["S", "T"]].drop_duplicates()
    closing = paths.merge(
        pairs[["src_id", "dst_id"]].rename(columns={"src_id": "T", "dst_id": "S"}),
        on=["T", "S"],
    )
    counts = {
        "pa_log_sg_middle": sg.groupby("B").size(),
        "pa_log_sg_source": st.groupby("S").size(),
        "pa_log_sg_target": st.groupby("T").size(),
        "pa_log_cycle3": closing.groupby("B").size(),
        "pa_log_paths2": paths.groupby("B").size(),
    }
    for col, series in counts.items():
        out[col] = np.log1p(series.reindex(out.index, fill_value=0).astype(float))
    # Middle-role counts are unknown for hubs (left out of path enumeration).
    hub_rows = out.index.isin(hubs)
    for col in ("pa_log_sg_middle", "pa_log_cycle3", "pa_log_paths2"):
        out.loc[hub_rows, col] = np.nan

    pair_sg = sg.groupby(["S", "B"]).size().rename("sg")
    pair_cycle = closing.groupby(["S", "B"]).size().rename("cycle3")
    per_pair = pairs[["src_id", "dst_id", "n"]].set_index(["src_id", "dst_id"])
    per_pair = per_pair.join(pair_sg.rename_axis(["src_id", "dst_id"]))
    per_pair = per_pair.join(pair_cycle.rename_axis(["src_id", "dst_id"]))
    per_pair = per_pair.fillna({"sg": 0.0, "cycle3": 0.0}).reset_index()
    receiver_is_hub = per_pair["dst_id"].isin(hubs)
    per_pair.loc[receiver_is_hub, ["sg", "cycle3"]] = np.nan
    return out[ACCOUNT_COLUMNS], per_pair, hubs


def edge_patterns(
    tx: pd.DataFrame, start, end, lookback_hours: int, pair_cap: int = 400
) -> pd.DataFrame:
    """Transaction-level pattern features for every transaction in [start, end)."""
    from mule.labels.mule_rule import window_slice

    span_start = None if start is None else pd.Timestamp(start) - pd.Timedelta(hours=lookback_hours)
    span = window_slice(tx, span_start, end)
    window = window_slice(tx, start, end)
    _, per_pair, _ = pattern_tables(span, pair_cap)

    n = span.loc[~span["is_self"]]
    outgoing = pd.DataFrame({"out_ts": n["ts"].to_numpy(), "acct": n["src_id"].to_numpy()})
    incoming = pd.DataFrame({"in_ts": n["ts"].to_numpy(), "acct": n["dst_id"].to_numpy()})
    rows = pd.DataFrame(
        {
            "tx_id": window["tx_id"].to_numpy(),
            "ts": window["ts"].to_numpy(),
            "src": window["src_id"].to_numpy(),
            "dst": window["dst_id"].to_numpy(),
            "self": window["is_self"].to_numpy(),
        }
    )
    fwd = pd.merge_asof(
        rows.rename(columns={"dst": "acct"}),
        outgoing,
        left_on="ts",
        right_on="out_ts",
        by="acct",
        direction="forward",
    )
    prev = pd.merge_asof(
        rows.rename(columns={"src": "acct"}),
        incoming,
        left_on="ts",
        right_on="in_ts",
        by="acct",
        direction="backward",
        allow_exact_matches=False,
    )
    hours = pd.Timedelta(hours=1)
    out = pd.DataFrame({"tx_id": rows["tx_id"]})
    out["e_pa_log_fwd_delay_h"] = np.log1p(((fwd["out_ts"] - fwd["ts"]) / hours).to_numpy())
    out["e_pa_log_src_prev_in_h"] = np.log1p(((prev["ts"] - prev["in_ts"]) / hours).to_numpy())
    pair = rows[["src", "dst"]].merge(
        per_pair.rename(columns={"src_id": "src", "dst_id": "dst"}), on=["src", "dst"], how="left"
    )
    out["e_pa_log_pair_count"] = np.log1p(pair["n"].to_numpy(dtype=float))
    out["e_pa_log_sg"] = np.log1p(pair["sg"].to_numpy(dtype=float))
    out["e_pa_log_cycle3"] = np.log1p(pair["cycle3"].to_numpy(dtype=float))
    out.loc[rows["self"].to_numpy(), EDGE_COLUMNS] = np.nan  # self-transfers form no pattern
    return out.astype({c: "float32" for c in EDGE_COLUMNS})
