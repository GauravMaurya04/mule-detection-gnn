"""Pattern features on hand-built graphs where every count is known."""

import numpy as np
import pandas as pd
import pytest

from mule.features.patterns import ACCOUNT_COLUMNS, EDGE_COLUMNS, edge_patterns, pattern_tables

T0 = pd.Timestamp("2022-09-03")


def _tx(rows, self_rows=()):
    """rows: (minutes after T0, src, dst)."""
    data = [(m, s, d) for m, s, d in rows] + [(m, a, a) for m, a in self_rows]
    df = pd.DataFrame(data, columns=["minute", "src_id", "dst_id"])
    df["ts"] = T0 + pd.to_timedelta(df["minute"], unit="min")
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["tx_id"] = np.arange(len(df))
    df["is_self"] = df["src_id"] == df["dst_id"]
    return df


def _count(feats, account, col):
    return float(np.expm1(feats.loc[account, col]))


def test_scatter_gather_is_found_for_middle_source_and_target():
    # 1 -> {2, 3} -> 9 in time order: a scatter-gather with two intermediaries.
    tx = _tx([(0, 1, 2), (5, 1, 3), (60, 2, 9), (70, 3, 9)])
    f, per_pair, _ = pattern_tables(tx)
    assert _count(f, 2, "pa_log_sg_middle") == pytest.approx(1)
    assert _count(f, 3, "pa_log_sg_middle") == pytest.approx(1)
    assert _count(f, 1, "pa_log_sg_source") == pytest.approx(1)
    assert _count(f, 9, "pa_log_sg_target") == pytest.approx(1)
    first_hop = per_pair.set_index(["src_id", "dst_id"])
    assert first_hop.loc[(1, 2), "sg"] == 1


def test_paths_must_respect_time():
    # 2 sends to 9 before it receives from 1, so 1 -> 2 -> 9 is not a path.
    tx = _tx([(60, 1, 2), (0, 2, 9)])
    f, _, _ = pattern_tables(tx)
    assert _count(f, 2, "pa_log_paths2") == 0


def test_three_cycle_counts_once_per_account():
    tx = _tx([(0, 1, 2), (10, 2, 3), (20, 3, 1), (30, 1, 2)])
    f, _, _ = pattern_tables(tx)
    for account in (1, 2, 3):
        assert _count(f, account, "pa_log_cycle3") == pytest.approx(1), account


def test_fan_in_burst_counts_the_busiest_24_hours():
    rows = [(m, s, 7) for m, s in zip(range(0, 50, 10), range(10, 15), strict=True)]
    rows.append((3 * 1440, 20, 7))  # three days later, outside the burst
    f, _, _ = pattern_tables(_tx(rows))
    assert _count(f, 7, "pa_log_fan_in_24h_max") == pytest.approx(5)


def test_hubs_get_unknown_middle_counts():
    rows = [(i, 100 + i, 5) for i in range(30)] + [(100 + i, 5, 200 + i) for i in range(30)]
    f, _, hubs = pattern_tables(_tx(rows), pair_cap=400)  # 30 x 30 = 900 > 400
    assert 5 in hubs and np.isnan(f.loc[5, "pa_log_paths2"])
    assert not np.isnan(f.loc[5, "pa_log_fan_in_24h_max"])


def test_edge_features_forward_delay_and_previous_receipt():
    # 1 -> 2 at 0 min; 2 -> 3 at 120 min; 2 received from 4 at -60 min (in the lookback).
    tx = _tx([(-60, 4, 2), (0, 1, 2), (120, 2, 3)], self_rows=[(30, 8)])
    e = edge_patterns(tx, T0, T0 + pd.Timedelta(days=1), lookback_hours=72).set_index("tx_id")
    first = tx.index[(tx["src_id"] == 1)][0]
    second = tx.index[(tx["src_id"] == 2)][0]
    assert np.expm1(e.loc[first, "e_pa_log_fwd_delay_h"]) == pytest.approx(2.0)
    assert np.expm1(e.loc[second, "e_pa_log_src_prev_in_h"]) == pytest.approx(2.0)
    self_row = tx.index[tx["is_self"]][0]
    assert e.loc[self_row, EDGE_COLUMNS].isna().all()
    assert len(e) == 3  # only transactions inside the window, not the lookback


def test_edge_patterns_ignore_everything_after_the_window():
    rng = np.random.default_rng(0)
    rows = [
        (int(m), int(s), int(d))
        for m, s, d in zip(
            rng.integers(0, 4 * 1440, 600),
            rng.integers(0, 40, 600),
            rng.integers(0, 40, 600),
            strict=True,
        )
    ]
    tx = _tx(rows)
    end = T0 + pd.Timedelta(days=2)
    base = edge_patterns(tx, T0 + pd.Timedelta(days=1), end, 72)
    later = tx[tx["ts"] >= end].assign(dst_id=lambda d: (d["dst_id"] + 1) % 40)
    tampered = pd.concat([tx[tx["ts"] < end], later], ignore_index=True)
    tampered["is_self"] = tampered["src_id"] == tampered["dst_id"]
    pd.testing.assert_frame_equal(base, edge_patterns(tampered, T0 + pd.Timedelta(days=1), end, 72))


def test_columns_are_complete():
    f, _, _ = pattern_tables(_tx([(0, 1, 2)]))
    assert list(f.columns) == ACCOUNT_COLUMNS
