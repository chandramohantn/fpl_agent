"""Recommendation explainer.

Generates natural language explanations for FPL recommendations.
Takes model outputs, optimization results, and detected events,
and produces human-readable reasoning.

This is the "voice" of the agent — translating numbers into advice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fpl_engine.agent.events import Event, EventSeverity
from fpl_engine.optimization.gameweek_optimizer import (
    CaptainChoice,
    TransferOption,
)
from fpl_engine.planning.planner import ChipRecommendation, Plan
from fpl_engine.simulation.engine import PlayerSimResult

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """A structured explanation of a recommendation."""

    headline: str
    reasoning: list[str]
    confidence: str  # "high", "medium", "low"
    alternatives: list[str] = None

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []

    def to_text(self) -> str:
        """Format as readable text."""
        lines = [f"**{self.headline}**", ""]
        for reason in self.reasoning:
            lines.append(f"• {reason}")
        if self.alternatives:
            lines.append("")
            lines.append("Alternatives considered:")
            for alt in self.alternatives:
                lines.append(f"  - {alt}")
        lines.append(f"\nConfidence: {self.confidence}")
        return "\n".join(lines)


class Explainer:
    """Generates natural language explanations for recommendations.

    Usage:
        explainer = Explainer(player_names={1: "Haaland", 2: "Salah", ...})
        explanation = explainer.explain_captain(captain_choices, sim_results)
        print(explanation.to_text())
    """

    def __init__(self, player_names: dict[int, str] | None = None) -> None:
        self.names = player_names or {}

    def _name(self, element: int) -> str:
        return self.names.get(element, f"Player#{element}")

    # ─── Captain Explanation ─────────────────────────────────────────────

    def explain_captain(
        self,
        choices: list[CaptainChoice],
        sim_results: dict[int, PlayerSimResult] | None = None,
    ) -> Explanation:
        """Explain captain recommendation."""
        if not choices:
            return Explanation(
                headline="No captain data available",
                reasoning=["Simulation results needed for captain selection"],
                confidence="low",
            )

        best = choices[0]
        name = self._name(best.element)
        reasoning = [
            f"{name} has the highest expected doubled points: "
            f"{best.mean_doubled:.1f} pts",
        ]

        if best.p_haul_doubled > 0.15:
            reasoning.append(
                f"Strong haul potential: {best.p_haul_doubled:.0%} chance of 20+ points"
            )
        if best.p_blank_doubled < 0.20:
            reasoning.append(
                f"Low blank risk: only {best.p_blank_doubled:.0%} chance of ≤4 points"
            )

        # Compare with runner-up
        alternatives = []
        if len(choices) >= 2:
            runner = choices[1]
            runner_name = self._name(runner.element)
            diff = best.mean_doubled - runner.mean_doubled
            alternatives.append(
                f"{runner_name}: {runner.mean_doubled:.1f} pts "
                f"({diff:.1f} pts behind)"
            )

            # Note if runner-up has higher upside
            if runner.upside_90 > best.upside_90:
                alternatives.append(
                    f"  ↳ {runner_name} has higher ceiling "
                    f"(P90={runner.upside_90:.0f} vs {best.upside_90:.0f}) "
                    "— better as a differential"
                )

        runner_diff = (
            choices[1].mean_doubled if len(choices) > 1 else 0
        )
        confidence = "high" if best.mean_doubled - runner_diff > 2 else "medium"

        return Explanation(
            headline=f"Captain {name}",
            reasoning=reasoning,
            confidence=confidence,
            alternatives=alternatives,
        )

    # ─── Transfer Explanation ────────────────────────────────────────────

    def explain_transfer(
        self,
        transfer: TransferOption,
        horizon_gws: int = 1,
    ) -> Explanation:
        """Explain a transfer recommendation."""
        sell_name = self.names.get(transfer.sell, transfer.sell_name)
        buy_name = self.names.get(transfer.buy, transfer.buy_name)

        reasoning = [
            f"Sell {sell_name} (£{transfer.sell_price/10:.1f}m, "
            f"{transfer.sell_xpts:.1f} xPts) → "
            f"Buy {buy_name} (£{transfer.buy_price/10:.1f}m, "
            f"{transfer.buy_xpts:.1f} xPts)",
            f"Net gain: +{transfer.gain:.1f} expected points "
            f"over {horizon_gws} GW{'s' if horizon_gws > 1 else ''}",
        ]

        if transfer.cost > 0:
            reasoning.append(
                f"Requires a -{transfer.cost} hit. "
                f"Still worthwhile: net value = +{transfer.net_value:.1f}"
            )
        else:
            reasoning.append("Free transfer — no cost to make this move")

        if transfer.net_value > 3:
            confidence = "high"
        elif transfer.net_value > 1:
            confidence = "medium"
        else:
            confidence = "low"

        return Explanation(
            headline=f"Transfer: {sell_name} → {buy_name}",
            reasoning=reasoning,
            confidence=confidence,
        )

    # ─── Plan Explanation ────────────────────────────────────────────────

    def explain_plan(self, plan: Plan) -> Explanation:
        """Explain a multi-GW plan."""
        reasoning = [
            f"Planning horizon: GW{plan.starting_gw} to "
            f"GW{plan.starting_gw + plan.horizon - 1}",
            f"Total expected points over plan: {plan.expected_points:.1f}",
        ]

        for i, (action, pts) in enumerate(zip(plan.actions, plan.gw_points)):
            gw = plan.starting_gw + i
            if action.is_roll:
                reasoning.append(f"GW{gw}: Roll transfer ({pts:.1f} pts)")
            elif action.chip:
                reasoning.append(
                    f"GW{gw}: Play {action.chip.value} ({pts:.1f} pts)"
                )
            else:
                transfers_desc = ", ".join(
                    f"{self._name(t.sell)}→{self._name(t.buy)}"
                    for t in action.transfers
                )
                reasoning.append(f"GW{gw}: Transfer {transfers_desc} ({pts:.1f} pts)")

        return Explanation(
            headline=f"Plan for next {plan.horizon} gameweeks",
            reasoning=reasoning,
            confidence="medium",
        )

    # ─── Chip Explanation ────────────────────────────────────────────────

    def explain_chip(self, rec: ChipRecommendation) -> Explanation:
        """Explain a chip timing recommendation."""
        reasoning = [
            f"Best gameweek to play {rec.chip.value}: GW{rec.recommended_gw}",
            f"Expected gain over not playing it: +{rec.expected_gain:.1f} pts",
            rec.reason,
        ]

        return Explanation(
            headline=f"Chip: {rec.chip.value} → GW{rec.recommended_gw}",
            reasoning=reasoning,
            confidence="medium",
        )

    # ─── Event Summary ───────────────────────────────────────────────────

    def explain_events(
        self, events: list[Event], squad: list[int]
    ) -> Explanation:
        """Summarize detected events and their impact."""
        squad_set = set(squad)
        squad_events = [
            e for e in events
            if any(p in squad_set for p in e.affected_players)
        ]
        other_events = [e for e in events if e not in squad_events]

        reasoning = []
        if squad_events:
            reasoning.append(
                f"⚠️ {len(squad_events)} event(s) affecting YOUR squad:"
            )
            for e in squad_events:
                reasoning.append(f"  • [{e.severity.value}] {e.description}")
        if other_events:
            high_others = [
                e for e in other_events
                if e.severity in (EventSeverity.HIGH, EventSeverity.CRITICAL)
            ]
            if high_others:
                reasoning.append(
                    f"\n{len(high_others)} significant event(s) elsewhere:"
                )
                for e in high_others[:5]:
                    reasoning.append(f"  • {e.description}")

        needs_replan = any(
            e.severity in (EventSeverity.HIGH, EventSeverity.CRITICAL)
            for e in squad_events
        )
        if needs_replan:
            reasoning.append("\n🔄 Recommendation: Re-run optimization with updated data")

        severity = "critical" if needs_replan else "medium" if squad_events else "low"
        return Explanation(
            headline=f"{len(events)} changes detected",
            reasoning=reasoning,
            confidence=severity,
        )
