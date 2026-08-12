"""Squad selection optimizer using Integer Linear Programming.

Solves the FPL squad selection as a constrained optimization problem:

Maximize: Σ (expected_points_i × x_i)

Subject to:
    - Budget: Σ (price_i × x_i) ≤ budget
    - Squad size: Σ x_i = 15
    - Position constraints: exactly 2 GK, 5 DEF, 5 MID, 3 FWD
    - Club limit: at most 3 players from any single club
    - x_i ∈ {0, 1} (binary — player is selected or not)

This is a classic 0-1 knapsack variant, solvable exactly with ILP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pulp
from pulp import (
    LpBinary,
    LpMaximize,
    LpProblem,
    LpVariable,
    lpSum,
    value,
)

logger = logging.getLogger(__name__)

# FPL constraints
DEFAULT_BUDGET = 1000  # £100.0m (FPL uses price × 10)
POSITION_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD_SIZE = 15
MAX_PER_CLUB = 3


@dataclass
class PlayerOption:
    """A player available for selection."""

    element: int
    name: str
    position: str  # GK, DEF, MID, FWD
    team: str
    team_id: int
    price: int  # Cost × 10 (e.g., 130 = £13.0m)
    expected_points: float  # Expected points (from simulation mean)
    # Optional extras for display
    p_haul: float = 0.0
    p_blank: float = 0.0
    std: float = 0.0


@dataclass
class SquadSelection:
    """Result of squad optimization."""

    players: list[PlayerOption]
    total_cost: int
    total_expected_points: float
    budget_remaining: int

    def by_position(self) -> dict[str, list[PlayerOption]]:
        """Group selected players by position."""
        result: dict[str, list[PlayerOption]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for p in sorted(self.players, key=lambda x: -x.expected_points):
            result[p.position].append(p)
        return result

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Squad ({len(self.players)} players, £{self.total_cost/10:.1f}m, "
            f"£{self.budget_remaining/10:.1f}m ITB)",
            f"Expected points: {self.total_expected_points:.1f}",
            "",
        ]
        for pos, players in self.by_position().items():
            lines.append(f"  {pos}:")
            for p in players:
                lines.append(
                    f"    {p.name:20s} ({p.team:12s}) £{p.price/10:.1f}m  "
                    f"xPts={p.expected_points:.2f}"
                )
        return "\n".join(lines)


def select_squad(
    player_pool: list[PlayerOption],
    budget: int = DEFAULT_BUDGET,
    existing_squad: list[int] | None = None,
    locked_in: list[int] | None = None,
    excluded: list[int] | None = None,
) -> SquadSelection | None:
    """Select optimal 15-man squad using Integer Linear Programming.

    Args:
        player_pool: All available players with expected points.
        budget: Total budget (default £100m = 1000 in FPL units).
        existing_squad: If set, used for transfer optimization (not initial selection).
        locked_in: Player IDs that MUST be in the squad.
        excluded: Player IDs that CANNOT be in the squad.

    Returns:
        SquadSelection with the optimal 15 players, or None if infeasible.
    """
    logger.info("Optimizing squad from %d players, budget=%d", len(player_pool), budget)

    # Create the ILP problem
    prob = LpProblem("FPL_Squad_Selection", LpMaximize)

    # Decision variables: x_i = 1 if player i is selected
    player_vars = {}
    for p in player_pool:
        player_vars[p.element] = LpVariable(f"x_{p.element}", cat=LpBinary)

    # Objective: maximize expected points
    prob += lpSum(
        player_vars[p.element] * p.expected_points for p in player_pool
    ), "Total_Expected_Points"

    # Constraint 1: Budget
    prob += (
        lpSum(player_vars[p.element] * p.price for p in player_pool) <= budget,
        "Budget",
    )

    # Constraint 2: Squad size = 15
    prob += (
        lpSum(player_vars[p.element] for p in player_pool) == SQUAD_SIZE,
        "Squad_Size",
    )

    # Constraint 3: Position limits
    for pos, count in POSITION_LIMITS.items():
        pos_players = [p for p in player_pool if p.position == pos]
        prob += (
            lpSum(player_vars[p.element] for p in pos_players) == count,
            f"Position_{pos}",
        )

    # Constraint 4: Max 3 per club
    teams = set(p.team_id for p in player_pool)
    for team_id in teams:
        team_players = [p for p in player_pool if p.team_id == team_id]
        prob += (
            lpSum(player_vars[p.element] for p in team_players) <= MAX_PER_CLUB,
            f"Club_{team_id}",
        )

    # Constraint 5: Locked-in players
    if locked_in:
        for eid in locked_in:
            if eid in player_vars:
                prob += player_vars[eid] == 1, f"Lock_{eid}"

    # Constraint 6: Excluded players
    if excluded:
        for eid in excluded:
            if eid in player_vars:
                prob += player_vars[eid] == 0, f"Exclude_{eid}"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != 1:
        logger.warning("Optimization infeasible (status=%d)", prob.status)
        return None

    # Extract solution
    selected = []
    for p in player_pool:
        if value(player_vars[p.element]) == 1:
            selected.append(p)

    total_cost = sum(p.price for p in selected)
    total_xpts = sum(p.expected_points for p in selected)

    result = SquadSelection(
        players=selected,
        total_cost=total_cost,
        total_expected_points=total_xpts,
        budget_remaining=budget - total_cost,
    )

    logger.info(
        "Squad selected: %d players, cost=%d, xPts=%.1f",
        len(selected), total_cost, total_xpts,
    )
    return result
