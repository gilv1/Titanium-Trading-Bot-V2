"""
Tests for core/reto_tracker.py — phase transitions, auto-scaling, drawdown protection.
"""

from __future__ import annotations

from datetime import date
from datetime import datetime
from datetime import timezone
from unittest.mock import patch

import pytest

import core.reto_tracker as reto_tracker_module
from core.reto_tracker import DailyPnL, RetoTracker, TradeResult


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def tracker() -> RetoTracker:
    """Return a RetoTracker starting at $3,000 (Phase 1)."""
    return RetoTracker(initial_capital=3000.0)


@pytest.fixture(autouse=True)
def isolated_tracker_state(tmp_path, monkeypatch):
    """Isolate tracker persistence files per-test to avoid cross-run contamination."""
    monkeypatch.setattr(
        reto_tracker_module,
        "_RETO_STATE_PATH",
        str(tmp_path / "reto_tracker_state.json"),
    )
    monkeypatch.setattr(
        reto_tracker_module,
        "_RECONCILED_EXEC_IDS_PATH",
        str(tmp_path / "reto_reconciled_exec_ids.json"),
    )


# ──────────────────────────────────────────────────────────────
# Phase detection
# ──────────────────────────────────────────────────────────────


class TestPhaseDetection:
    def test_phase_1_at_3000(self, tracker):
        assert tracker.get_phase() == 1

    def test_phase_1_at_7499(self):
        t = RetoTracker(initial_capital=7499.0)
        assert t.get_phase() == 1

    def test_phase_2_at_7500(self):
        t = RetoTracker(initial_capital=7500.0)
        assert t.get_phase() == 2

    def test_phase_2_at_12000(self):
        t = RetoTracker(initial_capital=12000.0)
        assert t.get_phase() == 2

    def test_phase_3_at_13000(self):
        t = RetoTracker(initial_capital=13000.0)
        assert t.get_phase() == 3

    def test_phase_4_at_19000(self):
        t = RetoTracker(initial_capital=19000.0)
        assert t.get_phase() == 4

    def test_phase_4_at_25000(self):
        t = RetoTracker(initial_capital=25000.0)
        assert t.get_phase() == 4


# ──────────────────────────────────────────────────────────────
# Phase transitions via update_capital
# ──────────────────────────────────────────────────────────────


class TestPhaseTransitions:
    def test_phase_1_to_2(self, tracker):
        """Gain $4,500 → cross Phase 2 threshold."""
        result = TradeResult(engine="futures", pnl=4500.0)
        tracker.update_capital(result)
        assert tracker.get_phase() == 2

    def test_phase_2_to_3(self):
        t = RetoTracker(initial_capital=7500.0)
        result = TradeResult(engine="futures", pnl=5500.0)
        t.update_capital(result)
        assert t.get_phase() == 3

    def test_phase_3_to_4(self):
        t = RetoTracker(initial_capital=13000.0)
        result = TradeResult(engine="futures", pnl=6000.0)
        t.update_capital(result)
        assert t.get_phase() == 4

    def test_capital_floored_at_zero(self, tracker):
        result = TradeResult(engine="futures", pnl=-10000.0)
        tracker.update_capital(result)
        assert tracker.capital == 0.0


# ──────────────────────────────────────────────────────────────
# Auto-scaling: position sizes per phase
# ──────────────────────────────────────────────────────────────


class TestPositionSizing:
    def test_phase_1_futures_contracts(self, tracker):
        assert tracker.get_contracts("futures") == 1

    def test_phase_1_futures_instrument(self, tracker):
        assert tracker.get_futures_instrument() == "MNQ"

    def test_phase_2_futures_contracts(self):
        t = RetoTracker(initial_capital=7500.0)
        assert t.get_contracts("futures") == 2

    def test_phase_3_futures_instrument(self):
        t = RetoTracker(initial_capital=13000.0)
        assert t.get_futures_instrument() == "MNQ"

    def test_phase_4_contracts(self):
        t = RetoTracker(initial_capital=19000.0)
        assert t.get_contracts("futures") == 4

    def test_phase_1_momo_max(self, tracker):
        assert tracker.get_position_size("momo") == 350.0

    def test_phase_2_momo_max(self):
        t = RetoTracker(initial_capital=7500.0)
        assert t.get_position_size("momo") == 600.0

    def test_phase_1_options_max(self, tracker):
        assert tracker.get_position_size("options") == 0.0

    def test_crypto_position_proportional(self, tracker):
        """Crypto allocation should be 30 % of capital."""
        size = tracker.get_position_size("crypto")
        assert abs(size - 3000.0 * 0.30) < 0.01


# ──────────────────────────────────────────────────────────────
# Drawdown protection
# ──────────────────────────────────────────────────────────────


class TestDrawdownProtection:
    def test_drawdown_above_8pct_activates_override(self):
        t = RetoTracker(initial_capital=1000.0)
        # Force today's starting capital to 1000
        t._today_start_capital = 1000.0
        # Lose $90 = 9%
        result = TradeResult(engine="futures", pnl=-90.0)
        t.update_capital(result)
        assert t._drawdown_override is True

    def test_drawdown_override_uses_previous_phase(self):
        t = RetoTracker(initial_capital=3100.0)  # Phase 2
        t._today_start_capital = 3100.0
        result = TradeResult(engine="futures", pnl=-300.0)  # ~9.7% loss
        t.update_capital(result)
        # Should operate at Phase 1 sizing when override active
        assert t._effective_phase() == 1

    def test_no_override_below_8pct(self):
        t = RetoTracker(initial_capital=1000.0)
        t._today_start_capital = 1000.0
        result = TradeResult(engine="futures", pnl=-50.0)  # 5%
        t.update_capital(result)
        assert t._drawdown_override is False


# ──────────────────────────────────────────────────────────────
# Capital milestones
# ──────────────────────────────────────────────────────────────


class TestMilestones:
    def test_milestone_10000_triggered(self):
        t = RetoTracker(initial_capital=9900.0)
        result = TradeResult(engine="futures", pnl=200.0)
        alerts = t.update_capital(result)
        assert any("10,000" in a or "$10,000" in a for a in alerts)

    def test_milestone_only_triggers_once(self):
        t = RetoTracker(initial_capital=9900.0)
        t.update_capital(TradeResult(engine="futures", pnl=200.0))
        alerts2 = t.update_capital(TradeResult(engine="futures", pnl=1.0))
        # Second update should not re-fire the same milestone
        assert not any("10,000" in a for a in alerts2)

    def test_no_milestone_when_capital_low(self, tracker):
        result = TradeResult(engine="futures", pnl=100.0)
        alerts = tracker.update_capital(result)
        assert alerts == []


# ──────────────────────────────────────────────────────────────
# Daily P&L
# ──────────────────────────────────────────────────────────────


class TestDailyPnL:
    def test_daily_pnl_starts_at_zero(self, tracker):
        daily = tracker.get_daily_pnl()
        assert daily.pnl == 0.0

    def test_daily_pnl_reflects_trade(self, tracker):
        tracker.update_capital(TradeResult(engine="futures", pnl=50.0))
        daily = tracker.get_daily_pnl()
        assert daily.pnl == 50.0


class _FakeExecution:
    def __init__(self, exec_id: str, when: datetime) -> None:
        self.execId = exec_id
        self.time = when


class _FakeCommissionReport:
    def __init__(self, realized_pnl: float) -> None:
        self.realizedPNL = realized_pnl


class _FakeFill:
    def __init__(self, exec_id: str, realized_pnl: float, when: datetime) -> None:
        self.execution = _FakeExecution(exec_id, when)
        self.commissionReport = _FakeCommissionReport(realized_pnl)


class _FakeIB:
    def __init__(self, fills: list[_FakeFill]) -> None:
        self._fills = fills

    def fills(self) -> list[_FakeFill]:
        return self._fills


class TestIbkrReconciliation:
    def test_reconcile_applies_today_realized_pnl_once(self):
        t = RetoTracker(initial_capital=3000.0)
        today = datetime.now(tz=timezone.utc)
        ib = _FakeIB(
            [
                _FakeFill("A1", 70.76, today),
                _FakeFill("A2", -3.74, today),
                _FakeFill("A3", 0.00, today),  # opening leg should be ignored
            ]
        )

        applied = t.reconcile_ibkr_realized_pnl(ib, as_of=today.date())
        assert applied == pytest.approx(67.02)
        assert t.capital == pytest.approx(3067.02)

        # Same fills on next sync must not double count
        applied_again = t.reconcile_ibkr_realized_pnl(ib, as_of=today.date())
        assert applied_again == 0.0
        assert t.capital == pytest.approx(3067.02)

    def test_reconcile_ignores_non_matching_day(self):
        t = RetoTracker(initial_capital=3000.0)
        old_day = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
        ib = _FakeIB([_FakeFill("B1", 25.0, old_day)])

        applied = t.reconcile_ibkr_realized_pnl(ib, as_of=date(2026, 3, 24))
        assert applied == 0.0
        assert t.capital == pytest.approx(3000.0)
