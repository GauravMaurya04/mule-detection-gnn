import pandas as pd

from mule.labels.mule_rule import label_accounts


def _tx(rows):
    """rows: (minute, src, dst, is_laundering)"""
    df = pd.DataFrame(rows, columns=["minute", "src_id", "dst_id", "is_laundering"])
    df["ts"] = pd.Timestamp("2022-09-01") + pd.to_timedelta(df["minute"], unit="min")
    df["is_self"] = df["src_id"] == df["dst_id"]
    return df.sort_values("ts").reset_index(drop=True)


def _status(labels, account):
    row = labels.set_index("account_id").loc[account]
    return "mule" if row["is_mule"] else "adjacent" if row["adjacent"] else "clean"


def test_chain_middle_accounts_are_mules_and_ends_are_adjacent():
    labels = label_accounts(_tx([(1, 0, 1, 1), (2, 1, 2, 1), (3, 2, 3, 1)]))
    assert [_status(labels, a) for a in (0, 1, 2, 3)] == ["adjacent", "mule", "mule", "adjacent"]


def test_cycle_origin_is_not_a_mule():
    labels = label_accounts(_tx([(1, 10, 11, 1), (2, 11, 12, 1), (3, 12, 10, 1)]))
    assert _status(labels, 10) == "adjacent"
    assert _status(labels, 11) == "mule" and _status(labels, 12) == "mule"


def test_receiving_only_after_sending_does_not_count():
    labels = label_accounts(_tx([(1, 20, 21, 1), (5, 22, 20, 1)]))
    assert _status(labels, 20) == "adjacent"


def test_same_minute_receive_and_send_counts():
    labels = label_accounts(_tx([(3, 30, 31, 1), (3, 31, 32, 1)]))
    assert _status(labels, 31) == "mule"


def test_self_transfers_are_ignored_but_count_as_activity():
    labels = label_accounts(_tx([(1, 40, 40, 1), (2, 40, 40, 1)]))
    assert _status(labels, 40) == "clean"


def test_pass_through_across_a_cutoff_counts_where_the_money_leaves():
    tx = _tx([(1, 50, 51, 1), (100, 51, 52, 1)])
    cutoff = pd.Timestamp("2022-09-01") + pd.Timedelta(minutes=50)
    before = label_accounts(tx, end=cutoff, rule="completed")
    after = label_accounts(tx, start=cutoff, rule="completed")
    assert _status(before, 51) == "adjacent"  # received, not yet forwarded
    assert _status(after, 51) == "mule"  # forwards money received in the earlier window


def test_rule_a_loses_the_same_pass_through():
    tx = _tx([(1, 50, 51, 1), (100, 51, 52, 1)])
    cutoff = pd.Timestamp("2022-09-01") + pd.Timedelta(minutes=50)
    before = label_accounts(tx, end=cutoff, rule="within")
    after = label_accounts(tx, start=cutoff, rule="within")
    assert _status(before, 51) == "adjacent" and _status(after, 51) == "adjacent"


def test_rule_b_never_looks_forward_in_time():
    # Receipt after the window ends must not make an in-window send count.
    tx = _tx([(1, 70, 71, 1), (100, 72, 70, 1)])
    cutoff = pd.Timestamp("2022-09-01") + pd.Timedelta(minutes=50)
    assert _status(label_accounts(tx, end=cutoff, rule="completed"), 70) == "adjacent"


def test_cycle_origin_is_not_a_mule_under_rule_b_either():
    tx = _tx([(1, 10, 11, 1), (2, 11, 12, 1), (3, 12, 10, 1)])
    cutoff = pd.Timestamp("2022-09-01") + pd.Timedelta(minutes=2, seconds=30)
    assert _status(label_accounts(tx, start=cutoff, rule="completed"), 10) == "adjacent"


def test_non_laundering_accounts_are_clean():
    labels = label_accounts(_tx([(1, 60, 61, 0), (2, 61, 62, 0)]))
    assert {_status(labels, a) for a in (60, 61, 62)} == {"clean"}
