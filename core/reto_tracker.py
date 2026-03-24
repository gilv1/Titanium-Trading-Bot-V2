"""
Reto Tracker — Compound Interest & Auto-Scale for Titanium Warrior v3.

Tracks total capital across all engines, determines trading phase (1–4),
calculates appropriate position sizes, and triggers milestone alerts.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from config import settings
from config.settings import MILESTONE_ALERTS, PHASES, PhaseConfig

logger = logging.getLogger(__name__)

_RETO_STATE_PATH = os.path.join("data", "reto_tracker_state.json")
_RECONCILED_EXEC_IDS_PATH = os.path.join("data", "reto_reconciled_exec_ids.json")


@dataclass
class TradeResult:
    """Summary of a single closed trade passed to the Reto Tracker."""

    engine: str
    pnl: float          # realised P&L in dollars (positive = profit)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DailyPnL:
    """Intraday P&L snapshot."""

    date: date
    starting_capital: float
    current_capital: float

    @property
    def pnl(self) -> float:
        return self.current_capital - self.starting_capital

    @property
    def pnl_pct(self) -> float:
        if self.starting_capital == 0:
            return 0.0
        return (self.pnl / self.starting_capital) * 100


class RetoTracker:
    """
    Compound interest tracker that auto-scales position sizes as capital grows.

    Responsibilities:
      - Track total capital.
      - Determine current phase based on capital thresholds.
      - Calculate contracts / max position size per engine per phase.
      - Apply daily drawdown protection (drop >8% in a day → previous-phase sizing).
      - Fire capital-milestone alerts.
    """

    def __init__(self, initial_capital: float | None = None) -> None:
        self._capital: float = initial_capital if initial_capital is not None else settings.INITIAL_CAPITAL
        self._today_start_capital: float = self._capital
        self._today_date: date = date.today()
        self._drawdown_override: bool = False  # True when daily DD protection is active
        self._triggered_milestones: set[float] = set()
        self._capital_source: str = "settings_seed"
        self._reconciled_exec_ids: set[str] = set()
        self._load_reconciled_exec_ids()
        self._load_state()

    # ──────────────────────────────────────────────────────────
    # Capital management
    # ──────────────────────────────────────────────────────────

    @property
    def capital(self) -> float:
        return self._capital

    @property
    def capital_source(self) -> str:
        """Return where current capital was loaded from: state file or settings seed."""
        return self._capital_source

    def update_capital(self, trade_result: TradeResult) -> list[str]:
        """
        Apply trade P&L and recalculate buying power.

        Returns a list of milestone alert messages that were just triggered
        (empty list if none).
        """
        self._capital += trade_result.pnl
        self._capital = max(self._capital, 0.0)  # floor at zero
        self._refresh_daily_reset()

        # Check drawdown protection
        daily_dd_pct = abs(self.get_daily_pnl().pnl_pct) if self.get_daily_pnl().pnl < 0 else 0.0
        if daily_dd_pct > 8.0:
            if not self._drawdown_override:
                logger.warning(
                    "Daily drawdown %.1f%% exceeded 8%%; activating previous-phase sizing protection.",
                    daily_dd_pct,
                )
            self._drawdown_override = True
        else:
            self._drawdown_override = False

        logger.info(
            "Capital updated: pnl=%.2f total=%.2f phase=%d",
            trade_result.pnl,
            self._capital,
            self.get_phase(),
        )
        alerts = self.check_milestones()
        self._save_state()
        return alerts

    def _refresh_daily_reset(self) -> None:
        """Reset daily tracking counter at the start of each new trading day."""
        today = date.today()
        if today != self._today_date:
            self._today_date = today
            self._today_start_capital = self._capital
            self._drawdown_override = False
            self._save_state()

    def _load_state(self) -> None:
        """Load persisted capital state so compounding survives restarts."""
        try:
            if not os.path.exists(_RETO_STATE_PATH):
                self._capital_source = "settings_seed"
                return
            with open(_RETO_STATE_PATH, encoding="utf-8") as fh:
                state = json.load(fh)

            self._capital = float(state.get("capital", self._capital))
            today_start = state.get("today_start_capital", self._today_start_capital)
            self._today_start_capital = float(today_start)

            saved_date = state.get("today_date")
            if saved_date:
                self._today_date = date.fromisoformat(saved_date)

            self._drawdown_override = bool(state.get("drawdown_override", False))
            self._triggered_milestones = {
                float(v) for v in state.get("triggered_milestones", [])
            }
            self._capital_source = "state_file"

            logger.info(
                "Reto tracker state loaded from %s (capital=%.2f, start=%.2f, date=%s).",
                _RETO_STATE_PATH,
                self._capital,
                self._today_start_capital,
                self._today_date.isoformat(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load reto tracker state from %s: %s", _RETO_STATE_PATH, exc)
            self._capital_source = "settings_seed"

    def _save_state(self) -> None:
        """Persist capital state so the bot resumes from the last realised balance."""
        try:
            os.makedirs(os.path.dirname(_RETO_STATE_PATH), exist_ok=True)
            state = {
                "capital": self._capital,
                "today_start_capital": self._today_start_capital,
                "today_date": self._today_date.isoformat(),
                "drawdown_override": self._drawdown_override,
                "triggered_milestones": sorted(self._triggered_milestones),
            }
            with open(_RETO_STATE_PATH, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save reto tracker state to %s: %s", _RETO_STATE_PATH, exc)

    def _load_reconciled_exec_ids(self) -> None:
        """Load IBKR execution ids already applied to tracker capital."""
        try:
            if not os.path.exists(_RECONCILED_EXEC_IDS_PATH):
                return
            with open(_RECONCILED_EXEC_IDS_PATH, encoding="utf-8") as fh:
                payload = json.load(fh)
            ids = payload.get("exec_ids", [])
            self._reconciled_exec_ids = {str(exec_id) for exec_id in ids if exec_id}
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load reconciled execution ids from %s: %s",
                _RECONCILED_EXEC_IDS_PATH,
                exc,
            )

    def _save_reconciled_exec_ids(self) -> None:
        """Persist the set of execution ids already reconciled into challenge capital."""
        try:
            os.makedirs(os.path.dirname(_RECONCILED_EXEC_IDS_PATH), exist_ok=True)
            payload = {"exec_ids": sorted(self._reconciled_exec_ids)}
            with open(_RECONCILED_EXEC_IDS_PATH, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not save reconciled execution ids to %s: %s",
                _RECONCILED_EXEC_IDS_PATH,
                exc,
            )

    def reconcile_ibkr_realized_pnl(self, ib: Any, as_of: date | None = None) -> float:
        """
        Reconcile missed realised P&L from IBKR fills after reconnect/restart.

        Uses commissionReport.realizedPNL from fills and deduplicates via execution id
        so repeated synchronization events from IBKR do not double-count capital.
        Returns the net P&L applied to tracker capital.
        """
        if ib is None:
            return 0.0

        self._refresh_daily_reset()
        reconcile_date = as_of or date.today()
        fills = ib.fills()
        applied_pnl = 0.0
        new_exec_ids: list[str] = []

        for fill in fills:
            execution = getattr(fill, "execution", None)
            report = getattr(fill, "commissionReport", None)
            exec_id = str(getattr(execution, "execId", "")).strip()
            if not exec_id or exec_id in self._reconciled_exec_ids:
                continue

            exec_time = getattr(execution, "time", None)
            if isinstance(exec_time, datetime):
                exec_date = exec_time.date()
                if exec_date != reconcile_date:
                    continue

            realized_pnl_raw = getattr(report, "realizedPNL", 0.0) if report is not None else 0.0
            try:
                realized_pnl = float(realized_pnl_raw)
            except (TypeError, ValueError):
                realized_pnl = 0.0

            # Opening legs typically report zero realized P&L.
            if abs(realized_pnl) < 1e-9:
                continue

            applied_pnl += realized_pnl
            new_exec_ids.append(exec_id)

        if new_exec_ids:
            self._reconciled_exec_ids.update(new_exec_ids)
            self._save_reconciled_exec_ids()

        if abs(applied_pnl) < 1e-9:
            return 0.0

        self._capital += applied_pnl
        self._capital = max(self._capital, 0.0)

        daily_dd_pct = abs(self.get_daily_pnl().pnl_pct) if self.get_daily_pnl().pnl < 0 else 0.0
        self._drawdown_override = daily_dd_pct > 8.0

        self.check_milestones()
        self._save_state()
        logger.info(
            "IBKR reconciliation applied %+.2f from %d fill(s). New capital=%.2f.",
            applied_pnl,
            len(new_exec_ids),
            self._capital,
        )
        return applied_pnl

    # ──────────────────────────────────────────────────────────
    # Phase logic
    # ──────────────────────────────────────────────────────────

    def get_phase(self) -> int:
        """Determine current phase (1–4) based on capital."""
        for phase_num in sorted(PHASES.keys(), reverse=True):
            if self._capital >= PHASES[phase_num].min_capital:
                return phase_num
        return 1

    def _effective_phase(self) -> int:
        """
        Return the phase used for position sizing.

        If drawdown protection is active, use one phase lower to reduce risk.
        """
        actual = self.get_phase()
        if self._drawdown_override and actual > 1:
            return actual - 1
        return actual

    def _phase_config(self) -> PhaseConfig:
        return PHASES[self._effective_phase()]

    # ──────────────────────────────────────────────────────────
    # Position sizing
    # ──────────────────────────────────────────────────────────

    def get_contracts(self, engine: str) -> int:
        """
        Return the number of futures contracts for the given engine.

        Only meaningful for the futures engine.
        """
        if engine == "futures":
            return self._phase_config().futures_contracts
        return 1

    def get_position_size(self, engine: str) -> float:
        """
        Return the maximum dollar amount to risk for a given engine.

        For futures this is expressed as max contracts; for others as $ amount.
        """
        cfg = self._phase_config()
        match engine:
            case "futures":
                return self._capital * (settings.FUTURES_ALLOCATION / 100)
            case "options":
                return cfg.options_max_capital
            case "momo":
                bullets = max(1, settings.MOMO_BULLETS_PER_WEEK)
                return (self._capital * (settings.MOMO_ALLOCATION / 100)) / bullets
            case "crypto":
                # 30 % of capital allocated to crypto (configurable via CRYPTO_ALLOCATION env var)
                return self._capital * (settings.CRYPTO_ALLOCATION / 100)
            case _:
                return 0.0

    def get_futures_instrument(self) -> str:
        """Return "MNQ" or "NQ" based on current phase."""
        return self._phase_config().futures_instrument

    # ──────────────────────────────────────────────────────────
    # Daily P&L
    # ──────────────────────────────────────────────────────────

    def get_daily_pnl(self) -> DailyPnL:
        """Return a DailyPnL object for the current trading day."""
        self._refresh_daily_reset()
        return DailyPnL(
            date=self._today_date,
            starting_capital=self._today_start_capital,
            current_capital=self._capital,
        )

    # ──────────────────────────────────────────────────────────
    # Milestones
    # ──────────────────────────────────────────────────────────

    def check_milestones(self) -> list[str]:
        """
        Check if any capital milestone has been crossed.

        Returns a list of alert messages for milestones newly triggered.
        """
        alerts: list[str] = []
        for milestone in MILESTONE_ALERTS:
            threshold = milestone.capital_threshold
            if self._capital >= threshold and threshold not in self._triggered_milestones:
                self._triggered_milestones.add(threshold)
                alerts.append(milestone.message)
                logger.info("Capital milestone reached: $%.0f — %s", threshold, milestone.message)
        return alerts

    # ──────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """Return a dict snapshot of the tracker state for logging/Telegram."""
        daily = self.get_daily_pnl()
        cfg = self._phase_config()
        return {
            "capital": round(self._capital, 2),
            "phase": self.get_phase(),
            "effective_phase": self._effective_phase(),
            "drawdown_override": self._drawdown_override,
            "daily_pnl": round(daily.pnl, 2),
            "daily_pnl_pct": round(daily.pnl_pct, 2),
            "futures_instrument": cfg.futures_instrument,
            "futures_contracts": cfg.futures_contracts,
            "sessions": cfg.sessions,
        }
