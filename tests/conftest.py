"""A tiny raw dataset in the exact AMLworld file formats, used by the smoke test.

Laundering chains (all US Dollar ACH):
  stack:  A1 -> B1 -> C1 -> D1        mules B1, C1; adjacent A1 (source), D1 (sink)
  cycle:  E1 -> F1 -> G1 -> E1        mules F1, G1; adjacent E1 (sends before it receives)
  self:   H1 -> H1 twice              neither (self-transfers do not count)
Background transfers spread over five days, including Euro and Yen conversions.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering"
)

LAUNDERING = [
    "2022/09/01 00:10,10,A1,20,B1,100.00,US Dollar,100.00,US Dollar,ACH,1",
    "2022/09/01 00:40,20,B1,30,C1,95.00,US Dollar,95.00,US Dollar,ACH,1",
    "2022/09/01 01:00,30,C1,40,D1,90.00,US Dollar,90.00,US Dollar,ACH,1",
    "2022/09/01 02:00,10,E1,20,F1,500.00,US Dollar,500.00,US Dollar,ACH,1",
    "2022/09/01 02:30,20,F1,30,G1,480.00,US Dollar,480.00,US Dollar,ACH,1",
    "2022/09/01 03:00,30,G1,10,E1,460.00,US Dollar,460.00,US Dollar,ACH,1",
    "2022/09/01 04:00,10,H1,10,H1,50.00,US Dollar,50.00,US Dollar,Reinvestment,1",
    "2022/09/01 04:30,10,H1,10,H1,50.00,US Dollar,50.00,US Dollar,Reinvestment,1",
]


def _background() -> list[str]:
    rows = []
    for day in range(1, 6):
        for hour in range(0, 24, 3):
            ts = f"2022/09/0{day} {hour:02d}:15"
            rows.append(f"{ts},50,X1,60,Y1,110.00,US Dollar,100.00,Euro,Wire,0")  # 1 EUR = 1.1 USD
            rows.append(f"{ts},60,Y1,70,Z1,16000.00,Yen,100.00,Euro,Wire,0")  # 1 EUR = 160 JPY
            rows.append(f"{ts},70,Z1,50,X1,25.00,US Dollar,25.00,US Dollar,Cheque,0")
    return rows


PATTERNS = textwrap.dedent(
    """\
    BEGIN LAUNDERING ATTEMPT - STACK
    2022/09/01 00:10,010,A1,020,B1,100.00,US Dollar,100.00,US Dollar,ACH,1
    2022/09/01 00:40,020,B1,030,C1,95.00,US Dollar,95.00,US Dollar,ACH,1
    2022/09/01 01:00,030,C1,040,D1,90.00,US Dollar,90.00,US Dollar,ACH,1
    END LAUNDERING ATTEMPT - STACK

    BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 3 hops
    2022/09/01 02:00,010,E1,020,F1,500.00,US Dollar,500.00,US Dollar,ACH,1
    2022/09/01 02:30,020,F1,030,G1,480.00,US Dollar,480.00,US Dollar,ACH,1
    2022/09/01 03:00,030,G1,010,E1,460.00,US Dollar,460.00,US Dollar,ACH,1
    END LAUNDERING ATTEMPT - CYCLE
    """
)

ACCOUNTS = textwrap.dedent(
    """\
    Bank Name,Bank ID,Account Number,Entity ID,Entity Name
    Bank #10,010,A1,E1,Corporation #1
    Bank #20,020,B1,E2,Sole Proprietorship #2
    Bank #30,030,C1,E2,Sole Proprietorship #2
    Bank #40,040,D1,E3,Partnership #3
    """
)


@pytest.fixture()
def tiny_project(tmp_path: Path) -> Path:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "HI-Small_Trans.csv").write_text("\n".join([HEADER, *LAUNDERING, *_background()]) + "\n")
    (raw / "HI-Small_Patterns.txt").write_text(PATTERNS)
    (raw / "HI-Small_accounts.csv").write_text(ACCOUNTS)
    (tmp_path / "base.yaml").write_text(
        textwrap.dedent(
            f"""\
            dataset: HI-Small
            paths:
              raw_dir: {raw}
              interim_dir: {tmp_path / "data" / "interim"}
              processed_dir: {tmp_path / "data" / "processed"}
              reports_dir: {tmp_path / "reports"}
            split:
              shares: [0.6, 0.2, 0.2]
              no_tail_end: "2022-09-05"
            """
        )
    )
    return tmp_path
