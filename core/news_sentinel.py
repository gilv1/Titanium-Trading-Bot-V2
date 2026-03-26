"""
News Sentinel — Macro Event & Market Context Layer (Layer 3).

Monitors:
- Economic calendar (FOMC, CPI, NFP, PPI, GDP, Retail Sales, etc.)
- VIX level via IBKR
- Market regime detection

Provides context to the AI Evaluator so the bot avoids trading into
high-impact macro events and sizes down during elevated volatility.

Graceful degradation: if VIX data or calendar is unavailable,
defaults to normal operation (size_modifier=1.0, should_pause=False).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.connection import ConnectionManager

logger = logging.getLogger(__name__)

# Path to the static 2026 economic calendar
_CALENDAR_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "economic_calendar_2026.json")

# How long (seconds) to cache the VIX reading to avoid excessive IBKR API calls
_VIX_CACHE_TTL = 300  # 5 minutes


@dataclass
class MarketContext:
    """Snapshot of current macro market conditions."""

    risk_level: str  # "low", "medium", "high", "extreme"
    upcoming_events: list[dict[str, Any]]  # events in next 24 hours
    minutes_to_next_event: int  # minutes to nearest upcoming high/medium event; -1 if none
    vix_level: float  # -1.0 if unavailable
    vix_regime: str  # "calm", "elevated", "fear", "panic"
    size_modifier: float  # 0.0–1.0 multiplier to apply to position size
    should_pause: bool  # True = skip trading entirely this cycle
    reasoning: str  # human-readable explanation

    # Convenience: empty/neutral context for graceful degradation
    @classmethod
    def neutral(cls) -> "MarketContext":
        return cls(
            risk_level="low",
            upcoming_events=[],
            minutes_to_next_event=-1,
            vix_level=-1.0,
            vix_regime="calm",
            size_modifier=1.0,
            should_pause=False,
            reasoning="No macro data available — normal operation",
        )


class NewsSentinel:
    """
    Macro event & market context layer.

    Monitors:
    - Economic calendar (FOMC, CPI, NFP, PPI, GDP, Jobless Claims, etc.)
    - VIX level via IBKR (cached for 5 minutes)
    - Market regime detection

    Provides context to the AI Evaluator for smarter trade decisions.
    """

    # VIX regime thresholds
    VIX_EXTREME: float = 35.0   # pause all trading
    VIX_FEAR: float = 25.0      # 25% size
    VIX_ELEVATED: float = 18.0  # 50% size

    # Economic events rated by market impact
    HIGH_IMPACT_EVENTS: frozenset[str] = frozenset({
        "FOMC", "Federal Funds Rate", "Fed Interest Rate Decision",
        "CPI", "Consumer Price Index",
        "Non-Farm Payrolls", "NFP", "Nonfarm Payrolls",
        "PPI", "Producer Price Index",
        "GDP", "Gross Domestic Product",
        "PCE", "Core PCE",
        "Retail Sales",
        "ISM Manufacturing",
        "Jackson Hole",
        "Powell", "Fed Chair",
        "FOMC Rate Decision",
        "GDP Q1 Advance", "GDP Q2 Advance", "GDP Q3 Advance", "GDP Q4 Advance",
        "GDP Q1 Final", "GDP Q2 Final", "GDP Q3 Final", "GDP Q4 Final",
    })

    MEDIUM_IMPACT_EVENTS: frozenset[str] = frozenset({
        "Jobless Claims", "Initial Claims",
        "Durable Goods",
        "Housing Starts",
        "Consumer Confidence",
        "Michigan Sentiment",
        "Trade Balance",
        "Industrial Production",
    })

    def __init__(self) -> None:
        self._calendar: list[dict[str, Any]] = []
        self._calendar_tz: str = "US/Eastern"
        self._vix_cache: float = -1.0
        self._vix_cache_time: datetime | None = None
        self._last_vix_source: str = "none"
        self._load_calendar()

    # ──────────────────────────────────────────────────────────
    # Calendar loading
    # ──────────────────────────────────────────────────────────

    def _load_calendar(self) -> None:
        """Load the static economic calendar from the JSON file."""
        path = os.path.abspath(_CALENDAR_PATH)
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self._calendar = data.get("events", [])
            self._calendar_tz = data.get("timezone", "US/Eastern")
            logger.info("[news-sentinel] Loaded %d calendar events from %s", len(self._calendar), path)
        except FileNotFoundError:
            logger.warning("[news-sentinel] Calendar file not found at %s — using empty calendar.", path)
            self._calendar = []
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("[news-sentinel] Failed to load calendar: %s", exc)
            self._calendar = []

    # ──────────────────────────────────────────────────────────
    # Event helpers
    # ──────────────────────────────────────────────────────────

    def _event_impact(self, event_name: str) -> str:
        """Return 'high', 'medium', or 'low' based on event name matching."""
        name_upper = event_name.upper()
        for keyword in self.HIGH_IMPACT_EVENTS:
            if keyword.upper() in name_upper:
                return "high"
        for keyword in self.MEDIUM_IMPACT_EVENTS:
            if keyword.upper() in name_upper:
                return "medium"
        return "low"

    def _event_datetime_eastern(self, event: dict[str, Any]) -> datetime | None:
        """Parse an event's date+time string into an Eastern-timezone-aware datetime."""
        try:
            from zoneinfo import ZoneInfo  # Python 3.9+
            tz = ZoneInfo("America/New_York")
            dt_str = f"{event['date']} {event['time']}"
            dt_naive = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
            return dt_naive.replace(tzinfo=tz)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[news-sentinel] Could not parse event datetime: %s", exc)
            return None

    def _now_eastern(self) -> datetime:
        """Current time in America/New_York timezone."""
        from zoneinfo import ZoneInfo
        return datetime.now(tz=ZoneInfo("America/New_York"))

    # ──────────────────────────────────────────────────────────
    # Calendar queries
    # ──────────────────────────────────────────────────────────

    def fetch_economic_calendar(self) -> list[dict[str, Any]]:
        """
        Return today's and tomorrow's economic events with impact ratings.

        Returns a list of dicts: {name, date, time, impact, minutes_away}.
        """
        now = self._now_eastern()
        tomorrow = now + timedelta(days=1)
        today_str = now.strftime("%Y-%m-%d")
        tomorrow_str = tomorrow.strftime("%Y-%m-%d")

        relevant: list[dict[str, Any]] = []
        for event in self._calendar:
            event_date = event.get("date", "")
            if event_date not in (today_str, tomorrow_str):
                continue

            event_dt = self._event_datetime_eastern(event)
            impact = event.get("impact") or self._event_impact(event.get("name", ""))
            minutes_away: int = -1
            if event_dt is not None:
                diff = (event_dt - now).total_seconds() / 60
                minutes_away = int(diff)

            relevant.append({
                "name": event.get("name", ""),
                "date": event_date,
                "time": event.get("time", ""),
                "impact": impact,
                "minutes_away": minutes_away,
            })

        return relevant

    def _get_nearest_impactful_event(
        self,
        events: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, int]:
        """
        Find the nearest high/medium impact event relative to now.

        Returns (event_dict, minutes_away).  minutes_away is negative if the
        event has already passed (within the last 60 minutes).
        """
        nearest: dict[str, Any] | None = None
        nearest_abs = 9999

        for ev in events:
            if ev["impact"] not in ("high", "medium"):
                continue
            mins = ev["minutes_away"]
            # Consider events up to 60 minutes in the past and any time in the future
            if mins < -60:
                continue
            if abs(mins) < nearest_abs:
                nearest_abs = abs(mins)
                nearest = ev

        return nearest, (nearest["minutes_away"] if nearest else -1)

    # ──────────────────────────────────────────────────────────
    # VIX
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _coerce_positive_float(value: Any) -> float | None:
        """Return a positive float if possible, otherwise None."""
        try:
            num = float(value)
            if num > 0:
                return num
        except (TypeError, ValueError):
            return None
        return None

    def _store_vix_cache(self, vix: float, source: str) -> float:
        """Save VIX value + timestamp + source metadata."""
        self._vix_cache = float(vix)
        self._vix_cache_time = datetime.utcnow()
        self._last_vix_source = source
        return self._vix_cache

    async def _fetch_vix_ibkr(self, connection_manager: "ConnectionManager | None") -> float:
        """Primary source: VIX snapshot from IBKR."""
        if connection_manager is None:
            return -1.0
        try:
            from ib_insync import Index  # type: ignore

            ib = connection_manager.margin.get_ib()
            if ib is None or not connection_manager.margin.is_connected():
                return -1.0

            vix_contract = Index("VIX", "CBOE")
            if hasattr(ib, "qualifyContractsAsync"):
                await ib.qualifyContractsAsync(vix_contract)
            else:
                await asyncio.to_thread(ib.qualifyContracts, vix_contract)

            ticker = ib.reqMktData(vix_contract, genericTickList="", snapshot=True)
            await asyncio.sleep(2)
            vix = self._coerce_positive_float(getattr(ticker, "last", None))
            if vix is None:
                vix = self._coerce_positive_float(getattr(ticker, "close", None))
            ib.cancelMktData(vix_contract)
            return vix if vix is not None else -1.0
        except ImportError:
            logger.debug("[news-sentinel] ib_insync not available — IBKR VIX fetch skipped.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[news-sentinel] VIX fetch failed on IBKR: %s", exc)
        return -1.0

    async def _fetch_vix_yahoo_or_stooq(self) -> tuple[float, str]:
        """Secondary source: Yahoo first, then Stooq."""

        def _fetch_yahoo() -> float:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1m&range=1d"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode("utf-8"))

            result = payload.get("chart", {}).get("result", [])
            if not result:
                return -1.0
            meta = result[0].get("meta", {})
            price = self._coerce_positive_float(meta.get("regularMarketPrice"))
            if price is not None:
                return price

            closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
            for value in reversed(closes):
                price = self._coerce_positive_float(value)
                if price is not None:
                    return price
            return -1.0

        def _fetch_stooq() -> float:
            # CSV format: Symbol,Date,Time,Open,High,Low,Close,Volume
            url = "https://stooq.com/q/l/?s=%5Evix&i=d"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:  # noqa: S310
                text = resp.read().decode("utf-8", errors="ignore").strip()

            lines = [ln for ln in text.splitlines() if ln.strip()]
            if len(lines) < 2:
                return -1.0
            cols = [c.strip() for c in lines[1].split(",")]
            if len(cols) < 7:
                return -1.0

            close_val = self._coerce_positive_float(cols[6])
            return close_val if close_val is not None else -1.0

        try:
            vix = await asyncio.to_thread(_fetch_yahoo)
            if vix > 0:
                return vix, "yahoo"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError) as exc:
            logger.debug("[news-sentinel] Yahoo VIX backup failed: %s", exc)

        try:
            vix = await asyncio.to_thread(_fetch_stooq)
            if vix > 0:
                return vix, "stooq"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            logger.debug("[news-sentinel] Stooq VIX backup failed: %s", exc)

        return -1.0, "none"

    async def _fetch_vix_alpha_vantage(self) -> float:
        """Tertiary source: Alpha Vantage GLOBAL_QUOTE for VIX."""
        try:
            from config import settings

            api_key = (settings.ALPHA_VANTAGE_API_KEY or "").strip()
            if not api_key:
                return -1.0

            params = urllib.parse.urlencode(
                {"function": "GLOBAL_QUOTE", "symbol": "VIX", "apikey": api_key}
            )
            url = f"https://www.alphavantage.co/query?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode("utf-8"))

            quote = payload.get("Global Quote", {})
            for key in ("05. price", "02. open", "03. high", "04. low"):
                price = self._coerce_positive_float(quote.get(key))
                if price is not None:
                    return price
        except (ImportError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError) as exc:
            logger.debug("[news-sentinel] Alpha Vantage VIX backup failed: %s", exc)
        return -1.0

    async def get_vix_level(self, connection_manager: "ConnectionManager | None" = None) -> float:
        """
        Fetch current VIX level with fallback chain.

        Returns the last VIX price, or -1.0 if unavailable.
        The value is cached for 5 minutes to avoid excessive IBKR API calls.
        """
        # Serve from cache if fresh
        if (
            self._vix_cache_time is not None
            and self._vix_cache > 0
            and (datetime.utcnow() - self._vix_cache_time).total_seconds() < _VIX_CACHE_TTL
        ):
            self._last_vix_source = "cache"
            return self._vix_cache

        # 1) IBKR (primary)
        vix = await self._fetch_vix_ibkr(connection_manager)
        if vix > 0:
            return self._store_vix_cache(vix, "ibkr")

        # 2) Yahoo/Stooq (secondary)
        vix, source = await self._fetch_vix_yahoo_or_stooq()
        if vix > 0:
            return self._store_vix_cache(vix, source)

        # 3) Alpha Vantage (tertiary)
        vix = await self._fetch_vix_alpha_vantage()
        if vix > 0:
            return self._store_vix_cache(vix, "alpha_vantage")

        self._last_vix_source = "none"
        return -1.0

    @staticmethod
    def _vix_regime(vix: float) -> str:
        """Classify VIX into a human-readable regime label."""
        if vix < 0:
            return "unknown"
        if vix < 15:
            return "calm"
        if vix < 20:
            return "elevated"
        if vix < 30:
            return "fear"
        return "panic"

    # ──────────────────────────────────────────────────────────
    # Risk calculation
    # ──────────────────────────────────────────────────────────

    def _calculate_risk(
        self,
        minutes_to_event: int,
        event_impact: str,
        vix: float,
    ) -> tuple[str, float, bool, list[str]]:
        """
        Core risk calculation logic.

        Returns (risk_level, size_modifier, should_pause, reasons).
        """
        should_pause = False
        size_modifier = 1.0
        risk_level = "low"
        reasons: list[str] = []

        # ── VIX-based sizing ──────────────────────────────────
        if vix > self.VIX_EXTREME:
            risk_level = "extreme"
            should_pause = True
            size_modifier = 0.0
            reasons.append(f"VIX={vix:.1f} EXTREME — trading paused")
        elif vix > self.VIX_FEAR:
            risk_level = "high"
            size_modifier = 0.25
            reasons.append(f"VIX={vix:.1f} FEAR — 25% size")
        elif vix > self.VIX_ELEVATED:
            risk_level = "medium"
            size_modifier = 0.5
            reasons.append(f"VIX={vix:.1f} ELEVATED — 50% size")
        else:
            if vix > 0:
                reasons.append(f"VIX={vix:.1f} CALM — full size")
            else:
                reasons.append("VIX unavailable — full size")

        # ── Event-based pausing ───────────────────────────────
        if event_impact == "high":
            if 0 <= minutes_to_event <= 30:
                should_pause = True
                size_modifier = 0.0
                if risk_level not in ("extreme",):
                    risk_level = "extreme"
                reasons.append(f"HIGH impact event in {minutes_to_event}min — PAUSED")
            elif 30 < minutes_to_event <= 60:
                size_modifier = min(size_modifier, 0.25)
                if risk_level not in ("extreme", "high"):
                    risk_level = "high"
                reasons.append(f"HIGH impact event in {minutes_to_event}min — 25% max")
            elif -30 <= minutes_to_event < 0:
                # Event just happened; wait for volatility to settle
                size_modifier = min(size_modifier, 0.25)
                if risk_level not in ("extreme", "high"):
                    risk_level = "high"
                reasons.append(f"HIGH impact event {abs(minutes_to_event)}min ago — settling")
        elif event_impact == "medium":
            if 0 <= minutes_to_event <= 15:
                size_modifier = min(size_modifier, 0.5)
                if risk_level == "low":
                    risk_level = "medium"
                reasons.append(f"MEDIUM impact event in {minutes_to_event}min — 50% max")

        return risk_level, size_modifier, should_pause, reasons

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_event_patterns() -> "dict[str, Any]":
        """
        Load historical event impact patterns from ``data/event_patterns.json``.

        Returns an empty dict if the file does not exist (graceful degradation).
        """
        try:
            from core.event_analyzer import EventAnalyzer
            return EventAnalyzer.load_patterns()
        except Exception:  # noqa: BLE001
            return {}

    async def get_market_context(
        self,
        connection_manager: "ConnectionManager | None" = None,
    ) -> MarketContext:
        """
        Build a complete market context snapshot.

        Returns risk level, upcoming events, VIX status, and sizing recommendation.
        Gracefully degrades to a neutral context if data is unavailable.
        """
        try:
            # Fetch economic events
            events = self.fetch_economic_calendar()

            # Fetch VIX (with cache)
            vix = await self.get_vix_level(connection_manager)

            # Find nearest relevant event
            nearest_event, minutes_to_next = self._get_nearest_impactful_event(events)
            event_impact = nearest_event["impact"] if nearest_event else "low"

            # Calculate risk
            risk_level, size_modifier, should_pause, reasons = self._calculate_risk(
                minutes_to_event=minutes_to_next,
                event_impact=event_impact,
                vix=vix,
            )
            reasons.append(f"VIX source: {self._last_vix_source}")

            # Enrich with historical event pattern context (if available)
            if nearest_event is not None:
                event_patterns = self._load_event_patterns()
                event_name = nearest_event.get("name", "")
                if event_name in event_patterns:
                    pattern = event_patterns[event_name]
                    historical_context = (
                        f" | Historical: avg {pattern['avg_move_15min_pts']:.0f}pts move, "
                        f"{pattern['reversal_pct']:.0f}% reversal rate, "
                        f"best entry {pattern['best_entry_delay_min']}min after"
                    )
                    reasons.append(historical_context)

            # Collect upcoming events in the next 24 hours (positive minutes_away only)
            upcoming = [e for e in events if 0 <= e["minutes_away"] <= 1440]
            upcoming.sort(key=lambda e: e["minutes_away"])

            return MarketContext(
                risk_level=risk_level,
                upcoming_events=upcoming,
                minutes_to_next_event=minutes_to_next,
                vix_level=vix,
                vix_regime=self._vix_regime(vix),
                size_modifier=size_modifier,
                should_pause=should_pause,
                reasoning="; ".join(reasons) if reasons else "Normal conditions",
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("[news-sentinel] get_market_context failed: %s — returning neutral context.", exc)
            return MarketContext.neutral()
