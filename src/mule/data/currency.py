"""Convert every amount to USD with exchange rates derived from the transactions themselves.

A cross-currency transaction pays X units of currency a and delivers Y units of currency b,
so usd(b) / usd(a) = X / Y. The median of that ratio per currency pair is a robust rate
estimate, and rates propagate outward from USD through the graph of observed pairs, so a
currency that never trades directly against USD still gets a rate via a third currency.
The rates use no labels, so computing them on the full dataset leaks nothing.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "US Dollar"


def _cross_currency_mask(tx: pd.DataFrame) -> pd.Series:
    paid = tx["currency_paid"].astype(str)
    received = tx["currency_received"].astype(str)
    return (paid != received) & (tx["amount_paid"] > 0) & (tx["amount_received"] > 0)


def estimate_usd_rates(tx: pd.DataFrame, base: str = BASE) -> pd.Series:
    """USD value of one unit of each currency (NaN if no chain of trades links it to USD)."""
    currencies = sorted(
        set(tx["currency_paid"].astype(str).unique())
        | set(tx["currency_received"].astype(str).unique())
    )
    if base not in currencies:
        raise ValueError(f"base currency {base!r} not found in the data")

    mask = _cross_currency_mask(tx)
    pairs = pd.DataFrame(
        {
            "paid": tx.loc[mask, "currency_paid"].astype(str),
            "received": tx.loc[mask, "currency_received"].astype(str),
            "ratio": tx.loc[mask, "amount_paid"] / tx.loc[mask, "amount_received"],
        }
    )
    stats = pairs.groupby(["paid", "received"])["ratio"].agg(["median", "size"]).reset_index()

    neighbours: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    for paid, received, median, size in stats.itertuples(index=False):
        neighbours[paid].append((int(size), received, float(median)))  # usd(rec) = usd(paid) * m
        neighbours[received].append((int(size), paid, 1.0 / float(median)))

    rates = {base: 1.0}
    queue = deque([base])
    while queue:
        current = queue.popleft()
        # Best-supported pairs first, so each rate comes from the most data available.
        for _, nxt, factor in sorted(neighbours[current], reverse=True):
            if nxt not in rates:
                rates[nxt] = rates[current] * factor
                queue.append(nxt)

    return pd.Series({c: rates.get(c, np.nan) for c in currencies}, name="usd_per_unit")


def to_usd(amount: pd.Series, currency: pd.Series, rates: pd.Series) -> np.ndarray:
    """amount * usd_per_unit[currency], vectorised through the categorical codes."""
    currency = currency.astype("category")
    lookup = rates.reindex(currency.cat.categories).to_numpy(dtype=float)
    codes = currency.cat.codes.to_numpy()
    per_unit = np.where(codes >= 0, lookup[np.clip(codes, 0, None)], np.nan)
    return amount.to_numpy(dtype=float) * per_unit


def conversion_check(tx: pd.DataFrame, rates: pd.Series) -> pd.Series:
    """USD paid / USD received on cross-currency transactions; close to 1 means consistent rates."""
    mask = _cross_currency_mask(tx)
    sub = tx.loc[mask]
    paid = to_usd(sub["amount_paid"], sub["currency_paid"], rates)
    received = to_usd(sub["amount_received"], sub["currency_received"], rates)
    return pd.Series(paid / received).describe(percentiles=[0.01, 0.5, 0.99])


def save_rates(rates: pd.Series, path: Path) -> None:
    clean = {k: (None if pd.isna(v) else float(v)) for k, v in rates.items()}
    path.write_text(json.dumps(clean, indent=2))


def load_rates(path: Path) -> pd.Series:
    raw = json.loads(path.read_text())
    return pd.Series({k: (np.nan if v is None else v) for k, v in raw.items()}, name="usd_per_unit")
