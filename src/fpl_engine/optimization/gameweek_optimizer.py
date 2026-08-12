"""Gameweek optimization: Starting XI, bench order, captaincy, transfers.

Given a 15-man squad and per-player simulation results, optimizes:
1. Starting XI (11 players in valid formation)
2. Bench order (4 players ranked by expected contribution as auto-sub)
3. Captain selection (maximize expected doubled points)
4. Vice-captain (fallback if captain doesn't play)
5. Transfer recommendations (who to sell/buy for maximum gain)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp
from pulp import LpBinary, LpMaximize, LpProblem, LpVariable, lpSum, value

from fpl_engine.simulation.engine import GameweekSimResult

logger = logging.getLogger(__name__)

MAX_PER_CLUB = 3

# Valid formations: (DEF, MID, FWD) — GK always 1
VALID_FORMATIONS = [
    (3, 5, 2), (3, 4, 3),
    (4, 5, 1), (4, 4, 2), (4, 3, 3),
    (5, 4, 1), (5, 3, 2), (5, 2, 3),
]


# ─── Starting XI Optimizer ───────────────────────────────────────────────────


@dataclass
class StartingXIResult:
    """Optimized starting XI with bench order."""

    starting: list[int]  # 11 element IDs
    bench: list[int]  # 4 element IDs in priority order
    formation: tuple[int, int, int]  # (DEF, MID, FWD)
    captain: int
    vice_captain: int
    expected_points: float  # Starting XI expected (with captain doubled)

    def summary(self, player_names: dict[int, str] | None = None) -> str:
        """Human-readable summary."""
        def name(eid):
            return player_names.get(eid, str(eid)) if player_names else str(eid)

        lines = [
            f"Formation: {self.formation[0]}-{self.formation[1]}-{self.formation[2]}",
            f"Expected points: {self.expected_points:.1f}",
            f"Captain: {name(self.captain)} (C)",
            f"Vice-captain: {name(self.vice_captain)} (V)",
            "",
            f"Starting XI: {[name(e) for e in self.starting]}",
            f"Bench: {[name(e) for e in self.bench]}",
        ]
        return "\n".join(lines)


def optimize_starting_xi(
    squad: list[int],
    sim_result: GameweekSimResult,
    positions: dict[int, str],
) -> StartingXIResult:
    """Select optimal starting XI from a 15-man squad.

    Uses ILP to maximize expected points subject to formation constraints.

    Args:
        squad: List of 15 player element IDs.
        sim_result: Simulation results for this GW.
        positions: Dict mapping element_id → position (GK/DEF/MID/FWD).

    Returns:
        StartingXIResult with starting 11, bench order, and captain.
    """
    # Get expected points for each squad player
    player_xpts = {}
    for eid in squad:
        pr = sim_result.get_player(eid)
        player_xpts[eid] = pr.mean if pr else 0.0

    # ILP: maximize total starting XI expected points
    prob = LpProblem("Starting_XI", LpMaximize)

    # Binary: x_i = 1 if player starts
    x = {eid: LpVariable(f"start_{eid}", cat=LpBinary) for eid in squad}

    # Objective
    prob += lpSum(x[eid] * player_xpts[eid] for eid in squad)

    # Exactly 11 starters
    prob += lpSum(x[eid] for eid in squad) == 11

    # Exactly 1 GK starts
    gks = [eid for eid in squad if positions.get(eid) == "GK"]
    prob += lpSum(x[eid] for eid in gks) == 1

    # Formation constraints (3-5 DEF, 2-5 MID, 1-3 FWD)
    defs = [eid for eid in squad if positions.get(eid) == "DEF"]
    mids = [eid for eid in squad if positions.get(eid) == "MID"]
    fwds = [eid for eid in squad if positions.get(eid) == "FWD"]

    prob += lpSum(x[eid] for eid in defs) >= 3
    prob += lpSum(x[eid] for eid in defs) <= 5
    prob += lpSum(x[eid] for eid in mids) >= 2
    prob += lpSum(x[eid] for eid in mids) <= 5
    prob += lpSum(x[eid] for eid in fwds) >= 1
    prob += lpSum(x[eid] for eid in fwds) <= 3

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract starting XI
    starting = [eid for eid in squad if value(x[eid]) == 1]
    bench = [eid for eid in squad if eid not in starting]

    # Determine formation
    n_def = sum(1 for eid in starting if positions.get(eid) == "DEF")
    n_mid = sum(1 for eid in starting if positions.get(eid) == "MID")
    n_fwd = sum(1 for eid in starting if positions.get(eid) == "FWD")
    formation = (n_def, n_mid, n_fwd)

    # Bench order: sort by expected points (highest first for auto-sub priority)
    # But first bench slot must be a GK (FPL rule: GK only replaces GK)
    bench_gk = [eid for eid in bench if positions.get(eid) == "GK"]
    bench_outfield = [eid for eid in bench if positions.get(eid) != "GK"]
    bench_outfield.sort(key=lambda eid: player_xpts.get(eid, 0), reverse=True)
    bench_ordered = bench_gk + bench_outfield

    # Captain and vice-captain
    captain, vice_captain = select_captain(starting, sim_result)

    # Expected points with captain doubled
    expected = sum(player_xpts[eid] for eid in starting)
    expected += player_xpts.get(captain, 0)  # Captain gets double (add once more)

    return StartingXIResult(
        starting=starting,
        bench=bench_ordered,
        formation=formation,
        captain=captain,
        vice_captain=vice_captain,
        expected_points=expected,
    )


# ─── Captain Selection ───────────────────────────────────────────────────────


@dataclass
class CaptainChoice:
    """Evaluation of a captain option."""

    element: int
    mean_doubled: float  # E[points × 2]
    std_doubled: float
    p_haul_doubled: float  # P(doubled >= 20)
    p_blank_doubled: float  # P(doubled <= 4)
    upside_90: float  # 90th percentile doubled


def select_captain(
    starting_xi: list[int],
    sim_result: GameweekSimResult,
) -> tuple[int, int]:
    """Select captain and vice-captain from starting XI.

    Strategy: maximize expected doubled points (mean × 2).
    In future: consider EO (effective ownership) for differential.

    Returns:
        (captain_id, vice_captain_id)
    """
    candidates = []
    for eid in starting_xi:
        pr = sim_result.get_player(eid)
        if pr is None:
            continue
        candidates.append((eid, pr.mean))

    # Sort by expected points descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    if len(candidates) >= 2:
        return candidates[0][0], candidates[1][0]
    elif len(candidates) == 1:
        return candidates[0][0], candidates[0][0]
    else:
        return starting_xi[0], starting_xi[1] if len(starting_xi) > 1 else starting_xi[0]


def evaluate_captain_options(
    starting_xi: list[int],
    sim_result: GameweekSimResult,
    top_n: int = 5,
) -> list[CaptainChoice]:
    """Evaluate captain options with detailed statistics.

    Returns top N candidates ranked by expected doubled points.
    """
    choices = []

    for eid in starting_xi:
        pr = sim_result.get_player(eid)
        if pr is None:
            continue

        doubled = pr.points_array * 2
        choices.append(CaptainChoice(
            element=eid,
            mean_doubled=float(doubled.mean()),
            std_doubled=float(doubled.std()),
            p_haul_doubled=float((doubled >= 20).mean()),
            p_blank_doubled=float((doubled <= 4).mean()),
            upside_90=float(np.percentile(doubled, 90)),
        ))

    choices.sort(key=lambda c: c.mean_doubled, reverse=True)
    return choices[:top_n]


# ─── Transfer Recommender ────────────────────────────────────────────────────


@dataclass
class TransferOption:
    """A recommended transfer."""

    sell: int  # Element to sell
    sell_name: str
    buy: int  # Element to buy
    buy_name: str
    sell_xpts: float  # Expected points of player being sold
    buy_xpts: float  # Expected points of player being bought
    gain: float  # Net gain = buy_xpts - sell_xpts
    cost: int  # Transfer cost (-4 for a hit, 0 for free)
    net_value: float  # gain - cost
    sell_price: int
    buy_price: int


def recommend_transfers(
    current_squad: list[int],
    sim_result: GameweekSimResult,
    player_pool: pd.DataFrame,
    positions: dict[int, str],
    prices: dict[int, int],
    teams: dict[int, int],
    free_transfers: int = 1,
    budget: int = 0,
    horizon_gws: int = 1,
) -> list[TransferOption]:
    """Recommend transfers based on simulation results.

    For each player in the squad, finds the best replacement at the
    same position within budget, and ranks by net expected gain.

    Args:
        current_squad: Current 15 player element IDs.
        sim_result: Simulation results for upcoming GW(s).
        player_pool: DataFrame of all available players (element, name, position, team_id, price).
        positions: Dict element_id → position.
        prices: Dict element_id → current price.
        teams: Dict element_id → team_id.
        free_transfers: Number of free transfers available (0, 1, or 2).
        budget: Available bank (in FPL price units).
        horizon_gws: Number of GWs to consider (multiply xPts by this).

    Returns:
        List of TransferOption sorted by net_value descending.
    """
    options = []

    # Current squad team counts
    squad_team_counts: dict[int, int] = {}
    for eid in current_squad:
        tid = teams.get(eid, 0)
        squad_team_counts[tid] = squad_team_counts.get(tid, 0) + 1

    for i, sell_id in enumerate(current_squad):
        sell_pos = positions.get(sell_id, "")
        sell_price = prices.get(sell_id, 0)
        sell_pr = sim_result.get_player(sell_id)
        sell_xpts = sell_pr.mean * horizon_gws if sell_pr else 0.0
        sell_team = teams.get(sell_id, 0)

        # Available budget if we sell this player
        available_budget = budget + sell_price

        # Find replacements at same position
        candidates = player_pool[
            (player_pool["position"] == sell_pos)
            & (~player_pool["element"].isin(current_squad))
            & (player_pool["price"] <= available_budget)
        ]

        for _, row in candidates.iterrows():
            buy_id = row["element"]
            buy_price = row["price"]
            buy_team = row.get("team_id", 0)

            # Check club limit (can't have >3 from same team)
            if buy_team != sell_team:
                if squad_team_counts.get(buy_team, 0) >= MAX_PER_CLUB:
                    continue

            buy_pr = sim_result.get_player(buy_id)
            buy_xpts = buy_pr.mean * horizon_gws if buy_pr else 0.0

            gain = buy_xpts - sell_xpts
            # Transfer cost: 0 for free transfers, -4 for hits
            cost = 0 if (i < free_transfers) else 4
            net_value = gain - cost

            if net_value > 0:  # Only recommend positive-value transfers
                options.append(TransferOption(
                    sell=sell_id,
                    sell_name=str(sell_id),
                    buy=buy_id,
                    buy_name=row.get("name", str(buy_id)),
                    sell_xpts=sell_xpts,
                    buy_xpts=buy_xpts,
                    gain=gain,
                    cost=cost,
                    net_value=net_value,
                    sell_price=sell_price,
                    buy_price=buy_price,
                ))

    # Sort by net value
    options.sort(key=lambda o: o.net_value, reverse=True)
    return options
