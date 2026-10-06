"""Leakage tests for the account features: nothing after the window, and no labels, ever."""

import numpy as np
import pandas as pd
import pytest

from mule.features.account import (
    FORBIDDEN,
    N_FEATURES,
    account_features,
    entity_table,
    feature_groups,
)

START, END = pd.Timestamp("2022-09-03"), pd.Timestamp("2022-09-05")


def _random_tx(n=3000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.Timestamp("2022-09-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 6 * 24 * 60, n)), "min"
    )
    src, dst = rng.integers(0, 120, n), rng.integers(0, 120, n)
    cur = pd.Categorical(rng.choice(["US Dollar", "Euro", "Yen"], n))
    tx = pd.DataFrame(
        {
            "tx_id": np.arange(n),
            "ts": ts,
            "src_id": src,
            "dst_id": dst,
            "src_bank": src % 7,
            "dst_bank": dst % 7,
            "amount_usd": rng.lognormal(5, 2, n),
            "currency_paid": cur,
            "currency_received": pd.Categorical(
                np.where(rng.random(n) < 0.8, cur.astype(str), "Euro")
            ),
            "payment_format": pd.Categorical(rng.choice(["ACH", "Cheque", "Wire", "Cash"], n)),
            "is_laundering": (rng.random(n) < 0.05).astype("int8"),
        }
    )
    tx["is_self"] = tx["src_id"] == tx["dst_id"]
    tx["cross_bank"] = tx["src_bank"] != tx["dst_bank"]
    return tx


def test_features_ignore_everything_at_or_after_the_window_end():
    tx = _random_tx()
    base = account_features(tx, START, END)
    tampered = tx.copy()
    later = tampered["ts"] >= END
    tampered.loc[later, "amount_usd"] *= 1000
    tampered.loc[later, "dst_id"] = tampered.loc[later, "src_id"]  # turn them into self-transfers
    extra = tx.loc[later].assign(ts=tx.loc[later, "ts"] + pd.Timedelta(hours=1))
    tampered = pd.concat([tampered, extra]).sort_values("ts", kind="stable").reset_index(drop=True)
    tampered["is_self"] = tampered["src_id"] == tampered["dst_id"]
    pd.testing.assert_frame_equal(base, account_features(tampered, START, END))


def test_features_do_not_read_the_laundering_flag():
    tx = _random_tx()
    flipped = tx.assign(is_laundering=(1 - tx["is_laundering"]).astype("int8"))
    pd.testing.assert_frame_equal(
        account_features(tx, START, END), account_features(flipped, START, END)
    )


def test_feature_names_carry_no_label_words_and_cover_every_active_account():
    tx = _random_tx()
    feats = account_features(tx, START, END)
    assert not [c for c in feats.columns if any(w in c for w in FORBIDDEN)]
    window = tx[(tx["ts"] >= START) & (tx["ts"] < END)]
    active = np.union1d(window["src_id"], window["dst_id"])
    assert np.array_equal(feats.index.to_numpy(), active)


@pytest.mark.parametrize("col", ["pt_fwd_ever_share", "pt_fwd_72h_share", "cp_reciprocity"])
def test_shares_stay_between_zero_and_one(col):
    values = account_features(_random_tx(), START, END)[col].dropna()
    assert values.between(0, 1).all()


def test_known_pass_through_values():
    # Account 1 receives 100 at 00:00 and 10:00, forwards 90 at 01:00 (once).
    t0 = pd.Timestamp("2022-09-03")
    rows = [
        (t0, 0, 1, 100.0),
        (t0 + pd.Timedelta(hours=1), 1, 2, 90.0),
        (t0 + pd.Timedelta(hours=10), 3, 1, 100.0),
    ]
    tx = pd.DataFrame(rows, columns=["ts", "src_id", "dst_id", "amount_usd"])
    tx = tx.assign(
        tx_id=range(3),
        src_bank=1,
        dst_bank=2,
        is_laundering=0,
        is_self=False,
        cross_bank=True,
        currency_paid=pd.Categorical(["US Dollar"] * 3),
        currency_received=pd.Categorical(["US Dollar"] * 3),
        payment_format=pd.Categorical(["ACH"] * 3),
    )
    f = account_features(tx, START, END).loc[1]
    assert f["pt_fwd_ever_share"] == pytest.approx(0.5)  # the 10:00 receipt is never forwarded
    assert f["pt_median_delay_h"] == pytest.approx(1.0)
    assert f["pt_out_in_ratio"] == pytest.approx(90 / 200)
    assert f["cp_log_unique_senders"] == pytest.approx(np.log1p(2))


def test_schema_is_exactly_55_columns_whatever_formats_appear():
    feats = account_features(_random_tx(), START, END)  # only 4 of the 7 formats present
    assert {p: len(c) for p, c in feature_groups(feats.columns).items()} == N_FEATURES
    assert feats.shape[1] == 55  # Tier 1: 32; graph 8, entity 8, patterns 7
    assert (feats["ch_share_bitcoin"].fillna(0) == 0).all()  # absent format -> zero share


def test_per_day_rates_use_active_days_over_the_whole_span():
    # Account 1 sends once in the lookback (Sep 2) and once in the window (Sep 3).
    rows = [(pd.Timestamp("2022-09-02 12:00"), 1, 2), (pd.Timestamp("2022-09-03 12:00"), 1, 3)]
    tx = pd.DataFrame(rows, columns=["ts", "src_id", "dst_id"]).assign(
        tx_id=[0, 1],
        src_bank=1,
        dst_bank=2,
        amount_usd=50.0,
        is_laundering=0,
        is_self=False,
        cross_bank=True,
        currency_paid=pd.Categorical(["US Dollar"] * 2),
        currency_received=pd.Categorical(["US Dollar"] * 2),
        payment_format=pd.Categorical(["ACH"] * 2),
    )
    f = account_features(tx, START, END, lookback_hours=72).loc[1]
    assert f["vol_active_days"] == 2  # lookback day + window day
    assert f["vol_out_per_day"] == pytest.approx(1.0)  # 2 sends over 2 active days, not 2 over 1


def _chain_tx():
    """0 -> 1 -> 2 -> 0 (a triangle) plus 3 -> 1; account 1 forwards its receipt within an hour."""
    t0 = pd.Timestamp("2022-09-03")
    rows = [
        (t0, 0, 1),
        (t0 + pd.Timedelta(minutes=30), 1, 2),
        (t0 + pd.Timedelta(hours=2), 2, 0),
        (t0 + pd.Timedelta(hours=3), 3, 1),
    ]
    tx = pd.DataFrame(rows, columns=["ts", "src_id", "dst_id"])
    n = len(tx)
    return tx.assign(
        tx_id=range(n),
        src_bank=1,
        dst_bank=2,
        amount_usd=100.0,
        is_laundering=0,
        is_self=False,
        cross_bank=True,
        currency_paid=pd.Categorical(["US Dollar"] * n),
        currency_received=pd.Categorical(["US Dollar"] * n),
        payment_format=pd.Categorical(["ACH", "ACH", "Wire", "Wire"]),
    )


def test_graph_features_on_a_known_graph():
    f = account_features(_chain_tx(), START, END)
    assert f.loc[0, "gr_clustering"] == pytest.approx(1.0)  # 0's two partners (1, 2) trade
    assert f.loc[3, "gr_clustering"] == 0.0
    assert f.loc[1, "gr_coreness"] == 2.0 and f.loc[3, "gr_coreness"] == 1.0
    # Account 2's only sender is 1, which forwarded its one receipt within an hour.
    assert f.loc[2, "gr_in_nbr_fwd72_mean"] == pytest.approx(f.loc[1, "pt_fwd_72h_share"])
    # Account 1's senders are 0 (half ACH) and 3 (all Wire).
    expected = (f.loc[0, "ch_share_ach"] + f.loc[3, "ch_share_ach"]) / 2
    assert f.loc[1, "gr_in_nbr_ach_mean"] == pytest.approx(expected)
    assert np.isfinite(f["gr_log_pagerank"]).all()


def test_entity_table_counts_accounts_and_parses_owner_type():
    meta = pd.DataFrame(
        {
            "bank": [1, 1, 2, 3],
            "account": ["a", "b", "c", "d"],
            "entity_id": ["E1", "E1", "E2", "E3"],
            "entity_name": ["Corporation #7", "Corporation #7", "Individual #9", "Something #1"],
        }
    )
    index = pd.DataFrame(
        {
            "account_id": [0, 1, 2, 3, 4],
            "bank": [1, 1, 2, 3, 9],
            "account": ["a", "b", "c", "d", "zz"],
        }
    )
    e = entity_table(meta, index)
    assert e.loc[0, "en_log_accounts_per_entity"] == pytest.approx(np.log(2))
    assert e.loc[0, "en_type_corporation"] == 1 and e.loc[2, "en_type_individual"] == 1
    assert e.loc[3, "en_type_other"] == 1
    assert np.isnan(e.loc[4, "en_type_other"])  # not in the accounts file
