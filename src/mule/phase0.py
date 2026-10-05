"""Phase 0 report: every number the plan needs before any modelling starts.

Writes reports/phase0.md, reports/phase0_numbers.json and reports/phase0_delays.png.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mule.config import Config
from mule.data.currency import conversion_check, load_rates
from mule.data.patterns import match_patterns, read_patterns
from mule.features.passthrough import forwarding_delays
from mule.io import ACCOUNT_INDEX_SCHEMA, TX_SCHEMA, read_parquet
from mule.labels.mule_rule import label_accounts
from mule.labels.split import build_manifest
from mule.labels.stage import k_values, labels_by_window, window_summary
from mule.report import Report, _one_per

DELAY_THRESHOLDS_H = (1, 6, 24, 72)


def _dataset_facts(tx: pd.DataFrame, index: pd.DataFrame, rep: Report, nums: dict) -> None:
    n, n_l = len(tx), int(tx["is_laundering"].sum())
    days = tx["ts"].dt.normalize()
    rep.section("1. Dataset facts")
    rep.item("Transactions", n)
    rep.item("Time range", f"{tx['ts'].min()} to {tx['ts'].max()}")
    rep.item("Calendar days with transactions", int(days.nunique()))
    rep.item("Accounts (bank, account pairs)", len(index))
    rep.item("Laundering transactions", f"{n_l:,} ({_one_per(n, n_l)})")
    rep.item("Self-transfers", int(tx["is_self"].sum()))
    rep.item("Cross-bank share", f"{tx['cross_bank'].mean():.1%}")
    daily = (
        tx.groupby(days)
        .agg(transactions=("tx_id", "size"), laundering=("is_laundering", "sum"))
        .reset_index()
    )
    daily["ts"] = daily["ts"].dt.date
    rep.note("Transactions per day:")
    rep.table(daily.rename(columns={"ts": "date"}))
    nums.update(
        transactions=n,
        laundering=n_l,
        accounts=len(index),
        ts_min=str(tx["ts"].min()),
        ts_max=str(tx["ts"].max()),
        days=int(days.nunique()),
    )


def _payment_formats(tx: pd.DataFrame, rep: Report) -> None:
    rep.section("2. Payment formats")
    g = tx.groupby("payment_format", observed=True)["is_laundering"].agg(["size", "sum"])
    g = g.reset_index().rename(columns={"size": "transactions", "sum": "laundering"})
    g["share"] = (g["transactions"] / len(tx)).map(lambda x: f"{x:.1%}")
    pairs = zip(g["transactions"], g["laundering"], strict=True)
    g["laundering_rate"] = [_one_per(t, k) for t, k in pairs]
    rep.table(g.sort_values("transactions", ascending=False))


def _currencies(tx: pd.DataFrame, rates: pd.Series, rep: Report, nums: dict) -> None:
    rep.section("3. Currencies and derived USD rates")
    counts = tx["currency_paid"].astype(str).value_counts()
    table = pd.DataFrame({"currency": rates.index, "usd_per_unit": rates.to_numpy()})
    table["paid_transactions"] = table["currency"].map(counts).fillna(0).astype(int)
    rep.table(table.sort_values("paid_transactions", ascending=False))
    check = conversion_check(tx, rates)
    rep.item("Cross-currency transactions checked", int(check["count"]))
    rep.item("USD paid / USD received, median", float(check["50%"]))
    rep.item(
        "USD paid / USD received, 1st-99th percentile", f"{check['1%']:.4g} to {check['99%']:.4g}"
    )
    rep.note("A median near 1 with a narrow 1-99% range means the derived rates are consistent.")
    nums["usd_rates"] = {k: (None if pd.isna(v) else float(v)) for k, v in rates.items()}
    nums["conversion_ratio_median"] = float(check["50%"])


def _accounts_file(cfg: Config, index: pd.DataFrame, rep: Report, nums: dict) -> None:
    rep.section("4. Accounts file (entities)")
    path = cfg.paths.interim_dir / "accounts_meta.parquet"
    if not path.exists():
        rep.note("Skipped: accounts_meta.parquet missing (see the ingest warning).")
        return
    meta = pd.read_parquet(path)
    rep.item("Rows", len(meta))
    joined = index.merge(
        meta[["bank", "account"]].dropna().astype({"bank": "int32"}).drop_duplicates(),
        on=["bank", "account"],
        how="left",
        indicator=True,
    )
    coverage = float((joined["_merge"] == "both").mean())
    rep.item("Transaction accounts found in the accounts file", f"{coverage:.1%}")
    per_entity = meta.groupby("entity_id").size()
    rep.item("Entities", len(per_entity))
    rep.item("Accounts per entity: mean / max", f"{per_entity.mean():.2f} / {per_entity.max()}")
    rep.item("Entities with more than one account", int((per_entity > 1).sum()))
    types = meta["entity_name"].str.replace(r"\s*#\d+\s*$", "", regex=True).value_counts().head(10)
    rep.note("Entity types parsed from entity names (top 10):")
    rep.table(types.rename_axis("entity_type").reset_index(name="accounts"))
    nums["accounts_file_coverage"] = coverage


def _patterns(cfg: Config, tx: pd.DataFrame, index: pd.DataFrame, rep: Report, nums: dict) -> None:
    rep.section("5. Laundering patterns file")
    path = cfg.paths.raw_dir / f"{cfg.dataset}_Patterns.txt"
    if not path.exists():
        rep.note(f"Skipped: {path} missing.")
        return
    patterns = read_patterns(path)
    matched = match_patterns(patterns, tx, index)
    by_type = matched.groupby("pattern").agg(
        attempts=("attempt_id", "nunique"), transactions=("pattern_row", "size")
    )
    rep.table(by_type.reset_index().sort_values("attempts", ascending=False))
    found = matched["tx_id"].notna()
    n_l = int(tx["is_laundering"].sum())
    covered = matched.loc[found & (matched["is_laundering"] == 1), "tx_id"].nunique()
    rep.item("Pattern rows matched to a transaction", f"{found.mean():.1%}")
    rep.item(
        "Pattern rows with more than one candidate match", int((matched["n_matches"] > 1).sum())
    )
    rep.item(
        "Matched pattern rows labelled laundering",
        f"{(matched.loc[found, 'is_laundering'] == 1).mean():.1%}",
    )
    rep.item("Laundering transactions covered by the patterns file", f"{covered / max(n_l, 1):.1%}")
    nums["pattern_coverage"] = covered / max(n_l, 1)
    nums["pattern_match_rate"] = float(found.mean())


def _splits(tx: pd.DataFrame, cfg: Config, rep: Report, nums: dict) -> None:
    rep.section("6. Time split (IBM day-based) and mule counts per window")
    manifest = build_manifest(
        tx["ts"], cfg.split.shares, cfg.split.no_tail_end, cfg.split.train_snapshot_days
    )
    labels = labels_by_window(tx, manifest, cfg.labels.rule)
    summary = window_summary(tx, manifest, labels)
    rep.note(
        f"Cut days {manifest['cut_days']}, actual shares {manifest['actual_shares']}; "
        f"label rule: {cfg.labels.rule}. mules_rule_A uses the stricter both-inside rule."
    )
    rep.table(summary)
    nums["windows"] = summary.to_dict(orient="records")
    nums["cutoffs"] = {k: [str(a), str(b)] for k, (a, b) in manifest["windows"].items()}
    k_rows = [
        {"window": r["window"], **k_values(r["active_accounts"], r["mules"])}
        for r in nums["windows"]
        if r["window"] not in manifest["train_snapshots"]
    ]
    rep.note("K values per evaluation window:")
    rep.table(pd.DataFrame(k_rows))


def _whole_period(labels: pd.DataFrame, rep: Report, nums: dict) -> None:
    rep.section("7. Whole-period labels (reference only, never used for training)")
    mules, adjacent = int(labels["is_mule"].sum()), int(labels["adjacent"].sum())
    rep.item("Active accounts", len(labels))
    rep.item("Mules", f"{mules:,} ({_one_per(len(labels), mules)})")
    rep.item("Laundering-adjacent", adjacent)
    nums["mules_whole_period"], nums["adjacent_whole_period"] = mules, adjacent


def _delays(tx: pd.DataFrame, labels: pd.DataFrame, cfg: Config, rep: Report, nums: dict) -> None:
    rep.section("8. Forwarding delays (sets X hours for the pass-through features)")
    mule_ids = labels.loc[labels["is_mule"], "account_id"].to_numpy()
    q = [0.1, 0.25, 0.5, 0.75, 0.9]

    every = forwarding_delays(tx)
    every["group"] = np.where(every["account_id"].isin(mule_ids), "mule", "non-mule")
    rows = []
    for group, d in every.groupby("group"):
        quant = d["delay_min"].quantile(q)
        rows.append(
            {
                "incoming transfers into": group,
                "count": len(d),
                "never forwarded": f"{d['delay_min'].isna().mean():.1%}",
                **{f"p{int(p * 100)} min": quant[p] for p in q},
            }
        )
    rep.note("Any incoming transfer to the account's next outgoing transfer, in minutes:")
    rep.table(pd.DataFrame(rows))

    laund = forwarding_delays(tx, laundering_only=True)
    laund = laund.loc[laund["account_id"].isin(mule_ids)]
    shares = {f"<= {h} h": float((laund["delay_min"] <= h * 60).mean()) for h in DELAY_THRESHOLDS_H}
    rep.note("Mules only: laundering receipt to the next laundering send:")
    rep.table(pd.DataFrame([{k: f"{v:.1%}" for k, v in shares.items()}]))
    rep.note("Choose X where mules clearly separate from non-mules but before the curve flattens.")
    nums["mule_laundering_forwarded_share"] = shares

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bins = np.logspace(0, np.log10(max(every["delay_min"].max(), 10) + 1), 60)
        for group, d in every.groupby("group"):
            ax.hist(
                d["delay_min"].dropna() + 1,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                label=group,
            )
        ax.set_xscale("log")
        ax.set_xlabel("minutes from receiving to next send (+1, log scale)")
        ax.set_ylabel("density")
        ax.set_title("Forwarding delay: mules vs non-mules")
        ax.legend()
        fig.tight_layout()
        fig.savefig(cfg.paths.reports_dir / "phase0_delays.png", dpi=120)
        plt.close(fig)
    except Exception as err:  # the plot is a convenience; the numbers above are the record
        rep.note(f"(Delay plot skipped: {err})")


def run_phase0(cfg: Config) -> dict:
    interim, reports = cfg.paths.interim_dir, cfg.paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    tx = read_parquet(interim / "transactions.parquet", TX_SCHEMA)
    index = read_parquet(interim / "account_index.parquet", ACCOUNT_INDEX_SCHEMA)
    rates = load_rates(interim / "usd_rates.json")

    rep, nums = Report("Phase 0 report"), {}
    _dataset_facts(tx, index, rep, nums)
    _payment_formats(tx, rep)
    _currencies(tx, rates, rep, nums)
    # These two read files we have not inspected yet; a format surprise should be reported,
    # not crash the whole report.
    optional_checks = (
        ("accounts file", lambda: _accounts_file(cfg, index, rep, nums)),
        ("patterns file", lambda: _patterns(cfg, tx, index, rep, nums)),
    )
    for name, check in optional_checks:
        try:
            check()
        except Exception as err:
            rep.note(f"**{name} check failed:** {type(err).__name__}: {err}")
    _splits(tx, cfg, rep, nums)
    labels = label_accounts(tx, rule=cfg.labels.rule)
    _whole_period(labels, rep, nums)
    _delays(tx, labels, cfg, rep, nums)

    (reports / "phase0.md").write_text(rep.text())
    (reports / "phase0_numbers.json").write_text(json.dumps(nums, indent=2, default=str))
    return nums
