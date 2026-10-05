import numpy as np
import pandas as pd

from mule.labels.split import assign_window, build_manifest, choose_cut_days

# Daily transaction counts from the Phase 0 report on the real HI-Small data.
HI_SMALL_DAILY = np.array(
    [
        1114921,
        754449,
        207382,
        207430,
        482650,
        482089,
        482751,
        482773,
        654467,
        208325,
        396,
        281,
        184,
        121,
        46,
        46,
        23,
        11,
    ]
)


def test_ibm_search_reproduces_the_hi_small_split():
    # Train Sep 1-6 (days 0-5), validation Sep 7-8 (6-7), test Sep 9-18 (8-17).
    assert choose_cut_days(HI_SMALL_DAILY) == (6, 8)


def test_manifest_windows_are_whole_days_and_no_tail_is_a_subset_of_test():
    ts = pd.Series(pd.date_range("2022-09-01", periods=10 * 24, freq="h"))
    m = build_manifest(ts, no_tail_end="2022-09-09")
    train, val, test, no_tail = (m["windows"][k] for k in ("train", "val", "test", "test_no_tail"))
    assert train[1] == val[0] and val[1] == test[0] == no_tail[0]
    assert all(t.hour == 0 and t.minute == 0 for t in (val[0], test[0], no_tail[1]))
    assert no_tail[1] <= test[1]


def test_every_transaction_gets_exactly_one_main_window():
    ts = pd.Series(pd.date_range("2022-09-01", periods=7 * 24, freq="h"))
    windows = assign_window(ts, build_manifest(ts))
    assert windows.notna().all()
    assert set(windows.cat.categories) == {"train", "val", "test"}
