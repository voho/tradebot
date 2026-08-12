"""Offline tests for the data-fetcher parsing helpers."""

import io
import zipfile
from datetime import datetime, timezone

from tradebot.fetch import _months, _norm_ms, _parse_archive_csv


def test_norm_ms_handles_all_resolutions():
    assert _norm_ms(1_700_000_000) == 1_700_000_000_000  # seconds
    assert _norm_ms(1_700_000_000_000) == 1_700_000_000_000  # ms
    assert _norm_ms(1_700_000_000_000_000) == 1_700_000_000_000  # us


def test_months_spans_year_boundary():
    months = _months(
        datetime(2024, 11, 15, tzinfo=timezone.utc),
        datetime(2025, 2, 1, tzinfo=timezone.utc),
    )
    assert months == ["2024-11", "2024-12", "2025-01", "2025-02"]


def _zip_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("klines.csv", text)
    return buf.getvalue()


def test_parse_archive_csv_plain():
    payload = _zip_bytes(
        "1700000000000,100.0,101.0,99.0,100.5,12.3,1700000299999,0,0,0,0,0\n"
        "1700000300000,100.5,102.0,100.0,101.5,15.0,1700000599999,0,0,0,0,0\n"
    )
    rows = _parse_archive_csv(payload)
    assert len(rows) == 2
    assert rows[0] == (1700000000000, 100.0, 101.0, 99.0, 100.5, 12.3)


def test_parse_archive_csv_skips_header_and_normalizes_us():
    payload = _zip_bytes(
        "open_time,open,high,low,close,volume,close_time,q,c,t,tq,i\n"
        "1700000000000000,100.0,101.0,99.0,100.5,12.3,1,0,0,0,0,0\n"
    )
    rows = _parse_archive_csv(payload)
    assert len(rows) == 1
    assert rows[0][0] == 1700000000000
