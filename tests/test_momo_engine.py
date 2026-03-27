from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from engines.momo_engine import MomoEngine


def _make_engine() -> MomoEngine:
    connection = MagicMock()
    connection.cash.is_connected.return_value = False
    connection.cash.get_ib.return_value = None

    brain = MagicMock()
    reto = MagicMock()
    risk = MagicMock()

    return MomoEngine(
        connection_manager=connection,
        brain=brain,
        reto_tracker=reto,
        risk_manager=risk,
        telegram=None,
    )


@pytest.mark.asyncio
async def test_scan_for_setups_runs_fallback_scan_when_started_late(monkeypatch):
    engine = _make_engine()
    engine._scanner_done_today = False
    engine._scanner_candidates = []
    engine._is_execution_window = MagicMock(return_value=True)

    candidate = SimpleNamespace(
        ticker="ABCD",
        score=80,
        gap_pct=12.0,
        price=5.0,
        news_headline="Contract award",
    )
    engine._scanner.scan_premarket = AsyncMock(return_value=[candidate])
    engine._fetch_intraday_bars = AsyncMock(return_value=MagicMock(empty=True))

    setups = await engine.scan_for_setups()

    engine._scanner.scan_premarket.assert_awaited_once()
    assert engine._scanner_done_today is True
    assert engine._scanner_candidates == [candidate]
    assert len(setups) == 1
    assert setups[0].signal.ticker == "ABCD"


@pytest.mark.asyncio
async def test_scan_for_setups_skips_fallback_outside_execution_window():
    engine = _make_engine()
    engine._scanner_done_today = False
    engine._scanner_candidates = []
    engine._is_execution_window = MagicMock(return_value=False)
    engine._scanner.scan_premarket = AsyncMock(return_value=[])

    setups = await engine.scan_for_setups()

    engine._scanner.scan_premarket.assert_not_called()
    assert setups == []
