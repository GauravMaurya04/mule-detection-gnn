"""Tier 1 account features (groups 1-5), computed point-in-time for one label window.

Span: one per window, from `lookback_hours` before the window starts up to the window's end,
so a pass-through that began just before the window is visible. Nothing at or after the
window's end is read, and the laundering flag is never read.

Per-day rates divide counts and USD totals by the account's active days *in that same span*
(lookback + window), so numerator and denominator always cover the same transfers. Dividing
span totals by window-only active days would inflate the rate for any account that was also
active during the lookback.

Tier 1 (32 columns: vol 9, cp 6, pt 6, ch 9, xb 2) describes each account's own transfers.
Tier 2 adds 16: graph 8 (PageRank both ways, coreness, clustering, and the pass-through and ACH
habits of the accounts it trades with) and entity 8 (accounts held by the same owner, owner type).
Graph features are computed on the same span; neighbour averages use neighbours' Tier 1
features, never their labels. Pattern features (7, see features/patterns.py) count the
laundering shapes the simulator plants: fan bursts, scatter-gather, 3-cycles, 2-hop paths.
55 columns in all. Payment formats are pinned to the seven AMLworld formats so the
column set never depends on the data. Column prefixes name the group, which the
explainability stage relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mule.features.passthrough import forwarding_delays
from mule.features.patterns import pattern_tables
from mule.labels.mule_rule import window_slice

GROUPS = {
    "vol": "Volume and activity",
    "cp": "Counterparties",
    "pt": "Pass-through",
    "ch": "Channels and currency",
    "xb": "Cross-bank",
    "gr": "Graph",
    "en": "Entity",
    "pa": "Patterns",
}
TIER1 = ("vol", "cp", "pt", "ch", "xb")
FORBIDDEN = ("laundering", "mule", "adjacent", "pattern", "label")
PAYMENT_FORMATS = ("ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire")
N_FEATURES = {"vol": 9, "cp": 6, "pt": 6, "ch": 7 + 2, "xb": 2, "gr": 8, "en": 8, "pa": 7}


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def _long_format(span: pd.DataFrame) -> pd.DataFrame:
    """One row per (account, non-self transfer, direction)."""
    n = span.loc[~span["is_self"]]
    currencies = sorted(
        set(n["currency_paid"].cat.categories) | set(n["currency_received"].cat.categories)
    )
    cur_paid = pd.Categorical(n["currency_paid"].astype(str), categories=currencies).codes
    cur_recv = pd.Categorical(n["currency_received"].astype(str), categories=currencies).codes
    shared = {
        "ts": n["ts"].to_numpy(),
        "usd": n["amount_usd"].to_numpy(),
        # Unknown formats get code -1: they count as transfers but in no format share.
        "fmt": pd.Categorical(n["payment_format"].astype(str), categories=PAYMENT_FORMATS).codes,
        "cross_bank": n["cross_bank"].to_numpy(),
        "xcur": cur_paid != cur_recv,
    }
    out = pd.DataFrame(
        {
            "account_id": n["src_id"].to_numpy(),
            "cp": n["dst_id"].to_numpy(),
            "cp_bank": n["dst_bank"].to_numpy(),
            "is_out": True,
            "cur": cur_paid,
            **shared,
        }
    )
    inc = pd.DataFrame(
        {
            "account_id": n["dst_id"].to_numpy(),
            "cp": n["src_id"].to_numpy(),
            "cp_bank": n["src_bank"].to_numpy(),
            "is_out": False,
            "cur": cur_recv,
            **shared,
        }
    )
    return pd.concat([out, inc], ignore_index=True)


def _distinct(long: pd.DataFrame, col: str, mask=None) -> pd.Series:
    sub = long if mask is None else long.loc[mask]
    return sub[["account_id", col]].drop_duplicates().groupby("account_id").size()


def _tier1(span: pd.DataFrame, index: np.ndarray, forward_hours) -> pd.DataFrame:
    """Tier 1 features from the span's transfers, for the accounts in `index`."""
    long = _long_format(span)
    f = pd.DataFrame(index=pd.Index(index, name="account_id"))
    g = long.groupby("account_id")
    is_out = long["is_out"]
    n_out = long.loc[is_out].groupby("account_id").size()
    n_in = long.loc[~is_out].groupby("account_id").size()
    usd_out = long.loc[is_out].groupby("account_id")["usd"].sum()
    usd_in = long.loc[~is_out].groupby("account_id")["usd"].sum()
    n_out, n_in = n_out.reindex(f.index, fill_value=0), n_in.reindex(f.index, fill_value=0)
    usd_out, usd_in = (
        usd_out.reindex(f.index, fill_value=0.0),
        usd_in.reindex(f.index, fill_value=0.0),
    )

    # 1. Volume and activity. active_days covers the whole span, like the counts it divides.
    days = long.assign(day=long["ts"].dt.floor("D"))
    active_days = _distinct(days, "day").reindex(f.index)
    n_self = span.loc[span["is_self"]].groupby("src_id").size().reindex(f.index, fill_value=0)
    log_usd = np.log1p(long["usd"])
    f["vol_active_days"] = active_days
    f["vol_in_per_day"] = n_in / active_days
    f["vol_out_per_day"] = n_out / active_days
    f["vol_log_usd_in_per_day"] = np.log1p(usd_in / active_days)
    f["vol_log_usd_out_per_day"] = np.log1p(usd_out / active_days)
    f["vol_log_amount_mean"] = log_usd.groupby(long["account_id"]).mean()
    f["vol_log_amount_max"] = log_usd.groupby(long["account_id"]).max()
    f["vol_log_amount_std"] = log_usd.groupby(long["account_id"]).std()
    f["vol_self_share"] = n_self / (n_self + n_in + n_out).replace(0, np.nan)

    # 2. Counterparties
    pairs = long.groupby(["account_id", "cp"]).agg(n=("is_out", "size"), n_out=("is_out", "sum"))
    pairs["sends_to"] = pairs["n_out"] > 0
    pairs["receives_from"] = pairs["n_out"] < pairs["n"]
    pairs["repeat_n"] = np.where(pairs["n"] > 1, pairs["n"], 0)
    pairs["both"] = pairs["sends_to"] & pairs["receives_from"]
    pg = pairs.groupby(level="account_id")
    senders = pg["receives_from"].sum().reindex(f.index, fill_value=0)
    receivers = pg["sends_to"].sum().reindex(f.index, fill_value=0)
    f["cp_log_unique_senders"] = np.log1p(senders)
    f["cp_log_unique_receivers"] = np.log1p(receivers)
    f["cp_fan_in_ratio"] = senders / n_in.replace(0, np.nan)
    f["cp_fan_out_ratio"] = receivers / n_out.replace(0, np.nan)
    f["cp_repeat_share"] = pg["repeat_n"].sum() / (n_in + n_out).replace(0, np.nan)
    f["cp_reciprocity"] = pg["both"].mean()

    # 3. Pass-through
    total = (usd_in + usd_out).replace(0, np.nan)
    f["pt_out_in_ratio"] = (usd_out / usd_in.replace(0, np.nan)).clip(upper=10)
    f["pt_net_flow"] = (usd_in - usd_out) / total
    delays = forwarding_delays(span)
    dg = delays.groupby("account_id")["delay_min"]
    f["pt_fwd_ever_share"] = delays["delay_min"].notna().groupby(delays["account_id"]).mean()
    for h in forward_hours:
        within = (delays["delay_min"] <= h * 60).groupby(delays["account_id"]).mean()
        f[f"pt_fwd_{h}h_share"] = within
    f["pt_median_delay_h"] = dg.median() / 60

    # 4. Channels and currency
    fmt_counts = long.groupby(["account_id", "fmt"]).size().unstack(fill_value=0)
    fmt_counts = fmt_counts.reindex(columns=range(len(PAYMENT_FORMATS)), fill_value=0)
    fmt_share = fmt_counts.div((n_in + n_out).replace(0, np.nan), axis=0)
    for code, name in enumerate(PAYMENT_FORMATS):
        f[f"ch_share_{_slug(name)}"] = fmt_share[code]
    f["ch_n_currencies"] = _distinct(long, "cur")
    f["ch_cross_currency_share"] = g["xcur"].mean()

    # 5. Cross-bank
    f["xb_share"] = g["cross_bank"].mean()
    f["xb_n_counterparty_banks"] = _distinct(long, "cp_bank")

    return f


def _graph(span: pd.DataFrame, tier1: pd.DataFrame) -> pd.DataFrame:
    """Group 6: position in the span's transfer graph and the habits of trading partners."""
    import igraph as ig

    n = span.loc[~span["is_self"]]
    nodes = tier1.index.to_numpy()
    pos = pd.Series(np.arange(len(nodes)), index=nodes)
    src, dst = pos.reindex(n["src_id"]).to_numpy(), pos.reindex(n["dst_id"]).to_numpy()
    g = ig.Graph(n=len(nodes), edges=np.column_stack([src, dst]).tolist(), directed=True)
    g_rev = ig.Graph(n=len(nodes), edges=np.column_stack([dst, src]).tolist(), directed=True)
    undirected = g.as_undirected(mode="collapse")

    out = pd.DataFrame(index=tier1.index)
    # PageRank sums to 1 over the graph, so scale by size to compare windows of different sizes.
    out["gr_log_pagerank"] = np.log(np.asarray(g.pagerank(directed=True)) * len(nodes))
    out["gr_log_pagerank_rev"] = np.log(np.asarray(g_rev.pagerank(directed=True)) * len(nodes))
    out["gr_coreness"] = np.asarray(undirected.coreness(), dtype=float)
    out["gr_clustering"] = np.asarray(undirected.transitivity_local_undirected(mode="zero"))

    pairs = pd.DataFrame({"src": n["src_id"].to_numpy(), "dst": n["dst_id"].to_numpy()})
    pairs = pairs.drop_duplicates()
    for side, me, other in (("in", "dst", "src"), ("out", "src", "dst")):
        for feat, name in (("pt_fwd_72h_share", "fwd72"), ("ch_share_ach", "ach")):
            values = tier1[feat].reindex(pairs[other].to_numpy()).to_numpy()
            out[f"gr_{side}_nbr_{name}_mean"] = (
                pd.Series(values).groupby(pairs[me].to_numpy()).mean()
            )
    return out


ENTITY_TYPES = (
    "Partnership",
    "Corporation",
    "Sole Proprietorship",
    "Country",
    "Individual",
    "Direct",
)


def entity_table(meta: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Group 7, static per account: owner's number of accounts and owner type (one-hot)."""
    m = meta.dropna(subset=["bank"]).astype({"bank": "int32"})
    joined = index.merge(
        m[["bank", "account", "entity_id", "entity_name"]], on=["bank", "account"], how="left"
    ).set_index("account_id")
    per_entity = m.groupby("entity_id").size()
    out = pd.DataFrame(index=joined.index)
    out["en_log_accounts_per_entity"] = np.log(joined["entity_id"].map(per_entity).astype(float))
    kind = joined["entity_name"].str.replace(r"\s*#\d+\s*$", "", regex=True)
    for t in ENTITY_TYPES:
        out[f"en_type_{_slug(t)}"] = (kind == t).astype(float).where(kind.notna())
    out["en_type_other"] = (~kind.isin(ENTITY_TYPES)).astype(float).where(kind.notna())
    return out


def account_features(
    tx: pd.DataFrame,
    start,
    end,
    lookback_hours: int = 72,
    forward_hours: tuple[int, ...] | list[int] = (24, 72),
    entity: pd.DataFrame | None = None,
    pair_cap: int = 400,
) -> pd.DataFrame:
    """Tier 1 + Tier 2 features for every account active in [start, end), indexed by account_id.

    `entity` is entity_table(...) output; without it the 8 entity columns are NaN.
    """
    window = window_slice(tx, start, end)
    active = np.union1d(window["src_id"].to_numpy(), window["dst_id"].to_numpy())
    span_start = None if start is None else pd.Timestamp(start) - pd.Timedelta(hours=lookback_hours)
    span = window_slice(tx, span_start, end)
    # Tier 1 for every account in the span (not only active ones) so neighbour averages see
    # partners that traded only during the lookback.
    span_accounts = np.union1d(span["src_id"].to_numpy(), span["dst_id"].to_numpy())
    tier1 = _tier1(span, span_accounts, forward_hours)
    graph = _graph(span, tier1)
    patterns, _, _ = pattern_tables(span, pair_cap)
    f = pd.concat([tier1, graph, patterns], axis=1).reindex(active)
    entity_cols = [
        "en_log_accounts_per_entity",
        *[f"en_type_{_slug(t)}" for t in ENTITY_TYPES],
        "en_type_other",
    ]
    ent = entity.reindex(active) if entity is not None else pd.DataFrame(index=active)
    for col in entity_cols:
        f[col] = ent[col].to_numpy() if col in ent else np.nan

    bad = [c for c in f.columns if any(word in c for word in FORBIDDEN)]
    if bad:
        raise AssertionError(f"label-like feature names: {bad}")
    counts = {p: len(cols) for p, cols in feature_groups(f.columns).items()}
    if counts != N_FEATURES:
        raise AssertionError(f"feature schema drifted: {counts}, expected {N_FEATURES}")
    return f.astype("float32")


def feature_groups(columns) -> dict[str, list[str]]:
    return {p: [c for c in columns if c.startswith(p + "_")] for p in GROUPS}
