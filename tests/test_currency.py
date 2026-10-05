import numpy as np
import pandas as pd

from mule.data.currency import estimate_usd_rates, to_usd


def _tx(rows):
    df = pd.DataFrame(
        rows, columns=["amount_paid", "currency_paid", "amount_received", "currency_received"]
    )
    for col in ("currency_paid", "currency_received"):
        df[col] = df[col].astype("category")
    return df


def test_rates_propagate_through_a_third_currency():
    tx = _tx(
        [(110.0, "US Dollar", 100.0, "Euro")] * 3  # 1 EUR = 1.1 USD
        + [(1.0, "Euro", 160.0, "Yen")] * 3  # Yen never trades against USD directly
        + [(5.0, "US Dollar", 5.0, "US Dollar")]
    )
    rates = estimate_usd_rates(tx)
    assert np.isclose(rates["US Dollar"], 1.0)
    assert np.isclose(rates["Euro"], 1.1)
    assert np.isclose(rates["Yen"], 1.1 / 160)


def test_median_ignores_an_outlier_trade():
    tx = _tx([(110.0, "US Dollar", 100.0, "Euro")] * 5 + [(999.0, "US Dollar", 1.0, "Euro")])
    assert np.isclose(estimate_usd_rates(tx)["Euro"], 1.1)


def test_unlinked_currency_gets_nan_and_conversion_uses_codes():
    tx = _tx([(110.0, "US Dollar", 100.0, "Euro"), (3.0, "Rupee", 3.0, "Rupee")])
    rates = estimate_usd_rates(tx)
    assert np.isnan(rates["Rupee"])
    usd = to_usd(tx["amount_paid"], tx["currency_paid"], rates)
    assert np.isclose(usd[0], 110.0) and np.isnan(usd[1])
