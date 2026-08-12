"""FPL Agent — the conversational AI orchestrator.

Wraps the full FPL decision system (data → models → simulation →
optimization → planning) in an interactive agent that:

1. Maintains game state (squad, bank, transfers, chips)
2. Monitors for changes (injuries, fixtures, prices)
3. Re-plans when significant events occur
4. Answers user questions with reasoned explanations
5. Provides proactive recommendations before deadlines

The agent is the entry point for all user interaction with the system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from fpl_engine.agent.events import (
    ChangeDetector,
    Event,
)
from fpl_engine.agent.explainer import Explainer, Explanation
from fpl_engine.planning.planner import ChipRecommendation, Plan
from fpl_engine.planning.state import PlanningState

logger = logging.getLogger(__name__)


class AgentMode(str, Enum):
    """Agent operating modes."""

    ACTIVE = "active"      # Monitoring and providing proactive advice
    PASSIVE = "passive"    # Only responds when asked
    PLANNING = "planning"  # Currently computing a plan


@dataclass
class AgentMemory:
    """Agent's persistent memory across interactions.

    Tracks decisions made, events seen, and plan history.
    """

    # Current plan
    current_plan: Plan | None = None
    plan_generated_at: datetime | None = None

    # Event history
    events_seen: list[Event] = field(default_factory=list)
    last_check_time: datetime | None = None

    # Decision history
    decisions_made: list[dict] = field(default_factory=list)

    # User preferences
    risk_appetite: str = "balanced"  # "aggressive", "balanced", "conservative"
    favorite_players: list[int] = field(default_factory=list)
    avoid_players: list[int] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Agent's response to a user query or event."""

    message: str
    explanation: Explanation | None = None
    actions_suggested: list[str] = field(default_factory=list)
    needs_user_decision: bool = False
    replan_triggered: bool = False


class FPLAgent:
    """The FPL decision-support agent.

    Usage:
        agent = FPLAgent(state=my_state, player_names=names)

        # Ask a question
        response = agent.ask("Who should I captain this week?")

        # Check for changes
        events = agent.check_for_changes(api_data)

        # Get proactive recommendation
        response = agent.get_recommendation()

        # Update after user makes a decision
        agent.confirm_action("captained Haaland")
    """

    def __init__(
        self,
        state: PlanningState,
        player_names: dict[int, str] | None = None,
        positions: dict[int, str] | None = None,
        mode: AgentMode = AgentMode.ACTIVE,
    ) -> None:
        self.state = state
        self.positions = positions or {}
        self.mode = mode
        self.memory = AgentMemory()
        self.detector = ChangeDetector()
        self.explainer = Explainer(player_names=player_names)
        self._player_names = player_names or {}

    # ─── User Interaction ────────────────────────────────────────────────

    def ask(self, question: str) -> AgentResponse:
        """Handle a user question.

        Interprets the intent and routes to the appropriate handler.
        """
        q = question.lower().strip()

        if any(kw in q for kw in ["captain", "cap", "armband"]):
            return self._handle_captain_query()
        elif any(kw in q for kw in ["transfer", "buy", "sell", "replace"]):
            return self._handle_transfer_query()
        elif any(kw in q for kw in ["starting", "lineup", "start", "bench"]):
            return self._handle_lineup_query()
        elif any(kw in q for kw in ["chip", "wildcard", "free hit", "bench boost", "triple"]):
            return self._handle_chip_query()
        elif any(kw in q for kw in ["plan", "ahead", "next few", "strategy"]):
            return self._handle_plan_query()
        elif any(kw in q for kw in ["news", "injury", "update", "changes"]):
            return self._handle_news_query()
        elif any(kw in q for kw in ["squad", "team", "my players"]):
            return self._handle_squad_query()
        else:
            return AgentResponse(
                message=(
                    "I can help with:\n"
                    "• Captain selection\n"
                    "• Transfer recommendations\n"
                    "• Starting XI / bench order\n"
                    "• Chip timing strategy\n"
                    "• Multi-week planning\n"
                    "• Injury / news updates\n"
                    "\nWhat would you like to know?"
                )
            )

    def check_for_changes(
        self,
        current_players: list[dict],
        current_fixtures: list[dict] | None = None,
        deadline: datetime | None = None,
    ) -> AgentResponse:
        """Check for changes and trigger re-planning if needed.

        Should be called periodically (e.g., every few hours) or
        before making final GW decisions.
        """
        events = self.detector.detect_changes(
            current_players, current_fixtures, deadline
        )

        if not events:
            return AgentResponse(message="No significant changes detected.")

        # Store events
        self.memory.events_seen.extend(events)
        self.memory.last_check_time = datetime.now()

        # Check if we need to re-plan
        needs_replan = self.detector.should_replan(events, self.state.squad)

        explanation = self.explainer.explain_events(events, self.state.squad)

        response = AgentResponse(
            message=explanation.to_text(),
            explanation=explanation,
            replan_triggered=needs_replan,
        )

        if needs_replan:
            response.actions_suggested.append(
                "Re-run optimization with updated player data"
            )
            response.message += (
                "\n\n🔄 I recommend re-running the optimization to account "
                "for these changes."
            )

        return response

    def get_recommendation(self) -> AgentResponse:
        """Get proactive recommendation based on current state.

        Called when the user wants a general "what should I do?" answer.
        """
        gw = self.state.current_gw
        ft = self.state.free_transfers
        chips = [c.value for c in self.state.chips_available]

        lines = [
            f"**GW{gw} Recommendation**",
            "",
            f"Free transfers: {ft}",
            f"Chips available: {', '.join(chips) if chips else 'None'}",
            "",
        ]

        # If we have a plan, summarize it
        if self.memory.current_plan:
            plan = self.memory.current_plan
            if plan.actions:
                action = plan.actions[0]
                if action.is_roll:
                    lines.append("📋 Recommended action: **Roll transfer**")
                    lines.append(
                        "  Banking a free transfer for next week is "
                        "optimal given current options."
                    )
                elif action.chip:
                    lines.append(
                        f"📋 Recommended action: **Play {action.chip.value}**"
                    )
                else:
                    transfers = ", ".join(
                        f"{self._name(t.sell)} → {self._name(t.buy)}"
                        for t in action.transfers
                    )
                    lines.append(f"📋 Recommended transfer: **{transfers}**")

        lines.append("")
        lines.append("Ask me about captain, transfers, or chip timing for details.")

        return AgentResponse(message="\n".join(lines))

    def confirm_action(self, description: str) -> AgentResponse:
        """Record that the user has made a decision."""
        self.memory.decisions_made.append({
            "gw": self.state.current_gw,
            "description": description,
            "timestamp": datetime.now().isoformat(),
        })
        return AgentResponse(
            message=f"✓ Noted: {description} for GW{self.state.current_gw}"
        )

    def set_plan(self, plan: Plan) -> None:
        """Update the agent's current plan."""
        self.memory.current_plan = plan
        self.memory.plan_generated_at = datetime.now()

    def set_chip_recommendations(
        self, recommendations: list[ChipRecommendation]
    ) -> None:
        """Store chip recommendations for future queries."""
        self.memory.chip_recommendations = recommendations

    # ─── Query Handlers ──────────────────────────────────────────────────

    def _handle_captain_query(self) -> AgentResponse:
        """Handle captain-related questions."""
        return AgentResponse(
            message=(
                "To recommend a captain, I need simulation results for "
                "this GW.\n\n"
                "Run the simulation and pass results to me, or I can "
                "recommend based on the current plan.\n\n"
                "Generally: pick the highest-xPts player with a good "
                "fixture. Premium attackers at home against weak teams "
                "are ideal."
            ),
            needs_user_decision=True,
        )

    def _handle_transfer_query(self) -> AgentResponse:
        """Handle transfer-related questions."""
        ft = self.state.free_transfers
        msg = (
            f"You have **{ft} free transfer{'s' if ft > 1 else ''}**.\n\n"
        )
        if self.memory.current_plan and self.memory.current_plan.actions:
            action = self.memory.current_plan.actions[0]
            if action.transfers:
                transfers = ", ".join(
                    f"{self._name(t.sell)} → {self._name(t.buy)}"
                    for t in action.transfers
                )
                msg += f"Plan recommends: {transfers}\n\n"
            elif action.is_roll:
                msg += (
                    "Plan recommends: **Roll** (save transfer for next week)\n\n"
                    "No available transfer offers sufficient improvement "
                    "over your current squad this week."
                )
        else:
            msg += (
                "Run the planner to get specific transfer recommendations "
                "based on upcoming fixtures."
            )
        return AgentResponse(message=msg)

    def _handle_lineup_query(self) -> AgentResponse:
        """Handle starting XI questions."""
        return AgentResponse(
            message=(
                "Starting XI optimization selects the best 11 from your "
                "15-man squad in a valid formation.\n\n"
                "Run `optimize_starting_xi()` with your simulation results "
                "for specific lineup advice."
            )
        )

    def _handle_chip_query(self) -> AgentResponse:
        """Handle chip timing questions."""
        chips = self.state.chips_available
        if not chips:
            return AgentResponse(message="You have no chips remaining this season.")

        lines = ["**Chips available:**"]
        for chip in chips:
            lines.append(f"  • {chip.value}")

        if hasattr(self.memory, "chip_recommendations"):
            lines.append("\n**Recommendations:**")
            for rec in self.memory.chip_recommendations:
                lines.append(f"  • {rec.chip.value}: GW{rec.recommended_gw} "
                           f"(+{rec.expected_gain:.1f} pts)")
                lines.append(f"    {rec.reason}")
        else:
            lines.append(
                "\nRun `plan_chip_strategy()` for specific timing recommendations."
            )

        return AgentResponse(message="\n".join(lines))

    def _handle_plan_query(self) -> AgentResponse:
        """Handle planning questions."""
        if self.memory.current_plan:
            explanation = self.explainer.explain_plan(self.memory.current_plan)
            return AgentResponse(
                message=explanation.to_text(),
                explanation=explanation,
            )
        return AgentResponse(
            message=(
                "No plan currently generated. Run the MCTS planner with "
                "your current state for a multi-GW strategy."
            )
        )

    def _handle_news_query(self) -> AgentResponse:
        """Handle news/update questions."""
        recent = self.memory.events_seen[-10:]  # Last 10 events
        if not recent:
            return AgentResponse(
                message="No recent events. Run `check_for_changes()` to scan for updates."
            )

        lines = ["**Recent events:**"]
        for e in recent:
            icon = {"critical": "🚨", "high": "⚠️", "medium": "ℹ️", "low": "·"}.get(
                e.severity.value, "•"
            )
            lines.append(f"{icon} {e.description}")

        return AgentResponse(message="\n".join(lines))

    def _handle_squad_query(self) -> AgentResponse:
        """Handle squad overview questions."""
        squad = self.state.squad
        by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for eid in squad:
            pos = self.positions.get(eid, "UNK")
            by_pos.setdefault(pos, []).append(self._name(eid))

        lines = [f"**Your squad ({len(squad)} players):**"]
        for pos in ["GK", "DEF", "MID", "FWD"]:
            players = by_pos.get(pos, [])
            if players:
                lines.append(f"  {pos}: {', '.join(players)}")

        lines.append(f"\n  Bank: £{self.state.bank/10:.1f}m")
        lines.append(f"  Free transfers: {self.state.free_transfers}")
        return AgentResponse(message="\n".join(lines))

    def _name(self, element: int) -> str:
        return self._player_names.get(element, f"#{element}")
