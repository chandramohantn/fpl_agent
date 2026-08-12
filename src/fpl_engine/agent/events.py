"""Change detection and event system.

Monitors the FPL API and other data sources for significant changes
that should trigger re-planning:

- Injury updates (player status changes)
- Transfer news (new signings available)
- Fixture changes (postponements, rescheduled, DGW announcements)
- Price changes (price rises/falls)
- Deadline approaching (time-sensitive decisions)
- Lineup leaks (unofficial team news)

Events are emitted when a change crosses a significance threshold.
The agent decides whether to re-plan based on event severity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EventSeverity(str, Enum):
    """How significant an event is for decision-making."""

    LOW = "low"          # Informational, no action needed
    MEDIUM = "medium"    # May affect plan, review recommended
    HIGH = "high"        # Likely affects plan, re-optimization needed
    CRITICAL = "critical"  # Definitely affects plan, immediate re-plan


class EventType(str, Enum):
    """Categories of detectable changes."""

    INJURY_UPDATE = "injury_update"
    SUSPENSION = "suspension"
    TRANSFER_NEWS = "transfer_news"
    FIXTURE_CHANGE = "fixture_change"
    PRICE_CHANGE = "price_change"
    DEADLINE_APPROACHING = "deadline_approaching"
    LINEUP_LEAK = "lineup_leak"
    GAMEWEEK_COMPLETED = "gameweek_completed"
    NEW_DATA_AVAILABLE = "new_data_available"


@dataclass
class Event:
    """A detected change that may affect FPL decisions."""

    event_type: EventType
    severity: EventSeverity
    timestamp: datetime
    description: str
    affected_players: list[int] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Event({self.event_type.value}, {self.severity.value}, "
            f"'{self.description[:50]}')"
        )


class ChangeDetector:
    """Detects significant changes by comparing current vs previous state.

    Usage:
        detector = ChangeDetector()
        events = await detector.check_for_changes(previous_state, current_api_data)
    """

    def __init__(self) -> None:
        self._previous_statuses: dict[int, str] = {}
        self._previous_news: dict[int, str] = {}
        self._previous_prices: dict[int, int] = {}
        self._last_check: datetime | None = None

    def detect_changes(
        self,
        current_players: list[dict],
        current_fixtures: list[dict] | None = None,
        deadline: datetime | None = None,
    ) -> list[Event]:
        """Compare current data against previous snapshot and emit events.

        Args:
            current_players: FPL API bootstrap elements (all players).
            current_fixtures: Fixture list (for detecting postponements/DGWs).
            deadline: Next GW deadline (for deadline-approaching events).

        Returns:
            List of Events detected since last check.
        """
        events = []
        now = datetime.now()

        for player in current_players:
            pid = player["id"]
            name = player.get("web_name", str(pid))
            status = player.get("status", "a")
            news = player.get("news", "")
            price = player.get("now_cost", 0)
            chance = player.get("chance_of_playing_next_round")

            # ─── Injury / Status changes ─────────────────────────────
            prev_status = self._previous_statuses.get(pid)
            if prev_status and prev_status != status:
                severity = self._injury_severity(prev_status, status, chance)
                events.append(Event(
                    event_type=EventType.INJURY_UPDATE,
                    severity=severity,
                    timestamp=now,
                    description=f"{name}: status {prev_status}→{status}. {news}",
                    affected_players=[pid],
                    details={
                        "player_name": name,
                        "old_status": prev_status,
                        "new_status": status,
                        "news": news,
                        "chance_of_playing": chance,
                    },
                ))

            # ─── News changes (even without status change) ───────────
            prev_news = self._previous_news.get(pid, "")
            if news and news != prev_news and prev_status:
                if "injury" in news.lower() or "doubt" in news.lower():
                    events.append(Event(
                        event_type=EventType.INJURY_UPDATE,
                        severity=EventSeverity.MEDIUM,
                        timestamp=now,
                        description=f"{name}: {news}",
                        affected_players=[pid],
                        details={"player_name": name, "news": news},
                    ))

            # ─── Price changes ───────────────────────────────────────
            prev_price = self._previous_prices.get(pid)
            if prev_price and prev_price != price:
                direction = "rose" if price > prev_price else "fell"
                events.append(Event(
                    event_type=EventType.PRICE_CHANGE,
                    severity=EventSeverity.LOW,
                    timestamp=now,
                    description=(
                        f"{name}: price {direction} "
                        f"£{prev_price/10:.1f}m → £{price/10:.1f}m"
                    ),
                    affected_players=[pid],
                    details={
                        "player_name": name,
                        "old_price": prev_price,
                        "new_price": price,
                        "direction": direction,
                    },
                ))

            # Update snapshots
            self._previous_statuses[pid] = status
            self._previous_news[pid] = news
            self._previous_prices[pid] = price

        # ─── Deadline approaching ────────────────────────────────────
        if deadline:
            time_to_deadline = (deadline - now).total_seconds() / 3600
            if time_to_deadline <= 2:
                events.append(Event(
                    event_type=EventType.DEADLINE_APPROACHING,
                    severity=EventSeverity.CRITICAL,
                    timestamp=now,
                    description=(
                        f"Deadline in {time_to_deadline:.1f} hours! "
                        "Finalize transfers and captain."
                    ),
                ))
            elif time_to_deadline <= 12:
                events.append(Event(
                    event_type=EventType.DEADLINE_APPROACHING,
                    severity=EventSeverity.HIGH,
                    timestamp=now,
                    description=f"Deadline in {time_to_deadline:.0f} hours.",
                ))
            elif time_to_deadline <= 24:
                events.append(Event(
                    event_type=EventType.DEADLINE_APPROACHING,
                    severity=EventSeverity.MEDIUM,
                    timestamp=now,
                    description=f"Deadline tomorrow ({time_to_deadline:.0f}h).",
                ))

        self._last_check = now
        return events

    def _injury_severity(
        self, old_status: str, new_status: str, chance: int | None
    ) -> EventSeverity:
        """Determine severity of a status change."""
        # Available → injured/suspended = HIGH
        if old_status == "a" and new_status in ("i", "s"):
            return EventSeverity.HIGH
        # Injured → available = MEDIUM (good news, but may not start immediately)
        if old_status in ("i", "s") and new_status == "a":
            return EventSeverity.MEDIUM
        # Available → doubtful = MEDIUM
        if old_status == "a" and new_status == "d":
            if chance is not None and chance <= 25:
                return EventSeverity.HIGH
            return EventSeverity.MEDIUM
        # Doubtful → injured = MEDIUM (was already flagged)
        if old_status == "d" and new_status == "i":
            return EventSeverity.MEDIUM
        return EventSeverity.LOW

    def should_replan(self, events: list[Event], squad: list[int]) -> bool:
        """Determine if detected events warrant re-planning.

        Re-plan if:
        - Any HIGH/CRITICAL event affects a squad player
        - Multiple MEDIUM events accumulate
        """
        squad_set = set(squad)

        critical_or_high = [
            e for e in events
            if e.severity in (EventSeverity.HIGH, EventSeverity.CRITICAL)
            and (not e.affected_players or any(p in squad_set for p in e.affected_players))
        ]

        medium_squad = [
            e for e in events
            if e.severity == EventSeverity.MEDIUM
            and any(p in squad_set for p in e.affected_players)
        ]

        return len(critical_or_high) > 0 or len(medium_squad) >= 3
