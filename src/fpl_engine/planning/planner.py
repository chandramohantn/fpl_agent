"""Multi-gameweek planner using Monte Carlo Tree Search.

Searches the space of possible action sequences across multiple GWs
to find the plan that maximizes total expected points over the planning
horizon (typically 3-8 GWs).

Key insight: A greedy GW-by-GW approach misses opportunities like:
- Rolling a transfer this week to make 2 free transfers next week
- Saving a wildcard for a fixture swing in 3 GWs
- Taking a -4 hit now because the player gains +12 over the next 4 GWs
- Playing Bench Boost on a DGW in 2 weeks

The MCTS planner explores these trade-offs by simulating many possible
futures and backpropagating their values.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fpl_engine.planning.actions import (
    XPtsProvider,
    evaluate_squad_gw,
    generate_actions,
)
from fpl_engine.planning.state import (
    Action,
    Chip,
    PlanningState,
    apply_action,
    compute_action_cost,
)

logger = logging.getLogger(__name__)


# ─── MCTS Node ──────────────────────────────────────────────────────────────


@dataclass
class MCTSNode:
    """A node in the MCTS search tree."""

    state: PlanningState
    action: Action | None = None  # Action that led to this node
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)

    # MCTS statistics
    visits: int = 0
    total_value: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.total_value / max(self.visits, 1)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ucb1(self, exploration: float = 1.41) -> float:
        """Upper Confidence Bound for tree policy."""
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.mean_value

        exploit = self.mean_value
        explore = exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploit + explore


# ─── MCTS Planner ────────────────────────────────────────────────────────────


@dataclass
class Plan:
    """A planned sequence of actions across multiple GWs."""

    actions: list[Action]  # One action per GW in the horizon
    expected_points: float  # Total expected points over the horizon
    gw_points: list[float]  # Per-GW breakdown
    starting_gw: int
    horizon: int

    def summary(self) -> str:
        lines = [
            f"Plan: GW{self.starting_gw}..GW{self.starting_gw + self.horizon - 1}",
            f"Total expected: {self.expected_points:.1f} pts",
            "",
        ]
        for i, (action, pts) in enumerate(zip(self.actions, self.gw_points)):
            gw = self.starting_gw + i
            lines.append(f"  GW{gw}: {action!r} → {pts:.1f} pts")
        return "\n".join(lines)


class MCTSPlanner:
    """Monte Carlo Tree Search planner for multi-GW FPL decisions.

    Usage:
        planner = MCTSPlanner(
            xpts_provider=my_xpts_function,
            player_pool=pool_df,
            positions=pos_dict,
            prices=price_dict,
            teams=team_dict,
        )
        plan = planner.search(initial_state, horizon=5, iterations=1000)
        print(plan.summary())
    """

    def __init__(
        self,
        xpts_provider: XPtsProvider,
        player_pool: pd.DataFrame,
        positions: dict[int, str],
        prices: dict[int, int],
        teams: dict[int, int],
        exploration: float = 1.41,
    ):
        self.xpts_provider = xpts_provider
        self.player_pool = player_pool
        self.positions = positions
        self.prices = prices
        self.teams = teams
        self.exploration = exploration

    def search(
        self,
        initial_state: PlanningState,
        horizon: int = 5,
        iterations: int = 1000,
        seed: int | None = None,
    ) -> Plan:
        """Run MCTS to find the best action sequence.

        Args:
            initial_state: Current game state.
            horizon: Number of GWs to plan ahead.
            iterations: Number of MCTS iterations.
            seed: Random seed.

        Returns:
            Plan with the recommended action sequence.
        """
        logger.info(
            "MCTS: planning GW%d..GW%d, %d iterations",
            initial_state.current_gw,
            initial_state.current_gw + horizon - 1,
            iterations,
        )

        rng = np.random.default_rng(seed)
        root = MCTSNode(state=initial_state)

        for i in range(iterations):
            # 1. Selection: walk tree using UCB1
            node = self._select(root)

            # 2. Expansion: add children if not at horizon
            if node.state.current_gw < initial_state.current_gw + horizon:
                node = self._expand(node)

            # 3. Rollout: simulate from this node to end of horizon
            value = self._rollout(
                node.state, initial_state.current_gw + horizon, rng
            )

            # 4. Backpropagation
            self._backpropagate(node, value)

        # Extract best plan
        return self._extract_plan(root, horizon)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walk tree using UCB1 to find a leaf."""
        while not node.is_leaf:
            node = max(node.children, key=lambda c: c.ucb1(self.exploration))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Generate children (candidate actions) for a leaf node."""
        actions = generate_actions(
            node.state,
            self.player_pool,
            self.xpts_provider,
            self.positions,
            self.prices,
            self.teams,
            max_transfers=2,
            top_k_transfers=3,
            consider_chips=True,
        )

        for action in actions:
            new_state = apply_action(node.state, action, self.prices)
            child = MCTSNode(state=new_state, action=action, parent=node)
            node.children.append(child)

        # Return a random unexplored child
        unexplored = [c for c in node.children if c.visits == 0]
        if unexplored:
            return unexplored[0]
        return node.children[0] if node.children else node

    def _rollout(
        self,
        state: PlanningState,
        end_gw: int,
        rng: np.random.Generator,
    ) -> float:
        """Simulate from state to end of horizon using greedy policy.

        Returns total expected points over remaining GWs.
        """
        total = 0.0
        current = state.copy()

        while current.current_gw < end_gw:
            # Evaluate current GW
            gw_pts = evaluate_squad_gw(
                current, self.xpts_provider, self.positions
            )
            total += gw_pts

            # Greedy action: best single transfer or roll
            actions = generate_actions(
                current, self.player_pool, self.xpts_provider,
                self.positions, self.prices, self.teams,
                max_transfers=1, top_k_transfers=1, consider_chips=False,
            )

            # Pick best action (evaluate each)
            best_action = actions[0]  # Default: roll
            best_value = gw_pts

            for action in actions[1:]:
                cost = compute_action_cost(current, action)
                # Quick estimate: current GW points minus cost
                new_state = apply_action(current, action, self.prices)
                new_gw_pts = evaluate_squad_gw(
                    new_state, self.xpts_provider, self.positions
                )
                if new_gw_pts - cost > best_value:
                    best_value = new_gw_pts - cost
                    best_action = action

            current = apply_action(current, best_action, self.prices)

        return total

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Propagate value up the tree."""
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    def _extract_plan(self, root: MCTSNode, horizon: int) -> Plan:
        """Extract the best action sequence from the tree."""
        actions = []
        gw_points = []
        node = root

        for _ in range(horizon):
            if not node.children:
                break
            # Pick most-visited child (MCTS best-practice)
            best_child = max(node.children, key=lambda c: c.visits)
            actions.append(best_child.action)

            # Evaluate this GW
            gw_pts = evaluate_squad_gw(
                node.state, self.xpts_provider, self.positions
            )
            gw_points.append(gw_pts)
            node = best_child

        total_pts = sum(gw_points) if gw_points else 0.0

        return Plan(
            actions=actions,
            expected_points=total_pts,
            gw_points=gw_points,
            starting_gw=root.state.current_gw,
            horizon=len(actions),
        )


# ─── Chip Strategy Planner ───────────────────────────────────────────────────


@dataclass
class ChipRecommendation:
    """Recommendation for when to play a chip."""

    chip: Chip
    recommended_gw: int
    expected_gain: float  # Extra points from playing chip in this GW vs not
    reason: str


def plan_chip_strategy(
    state: PlanningState,
    xpts_provider: XPtsProvider,
    positions: dict[int, str],
    horizon: int = 10,
) -> list[ChipRecommendation]:
    """Evaluate when to play each remaining chip.

    For each available chip, evaluates the expected gain across
    the planning horizon and recommends the best GW.

    Heuristics:
    - Bench Boost: best on DGWs (most bench player points)
    - Triple Captain: best when top player has easiest fixture
    - Free Hit: best on BGWs (blank gameweeks with many postponements)
    - Wildcard: best around major fixture swings

    Args:
        state: Current state (which chips are available).
        xpts_provider: Expected points function.
        positions: Element → position.
        horizon: How many GWs ahead to evaluate.

    Returns:
        List of ChipRecommendations for each available chip.
    """
    recommendations = []
    gw_start = state.current_gw

    for chip in state.chips_available:
        best_gw = gw_start
        best_gain = 0.0
        best_reason = ""

        for gw_offset in range(horizon):
            gw = gw_start + gw_offset
            test_state = state.copy()
            test_state.current_gw = gw

            # Evaluate without chip
            base_pts = evaluate_squad_gw(test_state, xpts_provider, positions)

            # Evaluate with chip
            chip_pts = evaluate_squad_gw(
                test_state, xpts_provider, positions, chip=chip
            )

            gain = chip_pts - base_pts

            if gain > best_gain:
                best_gain = gain
                best_gw = gw
                best_reason = _chip_reason(chip, gw, gain)

        if best_gain > 0:
            recommendations.append(ChipRecommendation(
                chip=chip,
                recommended_gw=best_gw,
                expected_gain=best_gain,
                reason=best_reason,
            ))

    recommendations.sort(key=lambda r: r.expected_gain, reverse=True)
    return recommendations


def _chip_reason(chip: Chip, gw: int, gain: float) -> str:
    """Generate human-readable reason for chip recommendation."""
    reasons = {
        Chip.BENCH_BOOST: f"GW{gw}: Bench contribute +{gain:.1f} pts (DGW or strong bench)",
        Chip.TRIPLE_CAPTAIN: f"GW{gw}: Captain gains +{gain:.1f} extra pts (easy fixture)",
        Chip.FREE_HIT: f"GW{gw}: Temporary squad gains +{gain:.1f} pts (blank/difficult GW)",
        Chip.WILDCARD: f"GW{gw}: Squad rebuild gains +{gain:.1f} pts (fixture swing)",
    }
    return reasons.get(chip, f"GW{gw}: +{gain:.1f} pts")
