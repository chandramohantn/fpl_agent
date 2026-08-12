"""Action generation and gameweek evaluation for planning.

Generates candidate actions for a given state and evaluates the expected
points of a state's squad for a gameweek using the simulation engine.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from fpl_engine.planning.state import Action, Chip, PlanningState, Transfer

logger = logging.getLogger(__name__)


# ─── Expected points provider ────────────────────────────────────────────────

# A function that returns expected points for a player in a given GW.
# Signature: (element_id, gameweek) -> expected_points
XPtsProvider = Callable[[int, int], float]


def evaluate_squad_gw(
    state: PlanningState,
    xpts_provider: XPtsProvider,
    positions: dict[int, str],
    chip: Chip | None = None,
) -> float:
    """Evaluate expected points for a squad in the current gameweek.

    Picks the best starting XI (greedy by xPts within formation) and
    captain (highest xPts). Applies chip effects if active.

    Args:
        state: Current planning state (contains squad + current_gw).
        xpts_provider: Function returning expected points per (element, gw).
        positions: Dict element → position.
        chip: Active chip for this GW (affects scoring).

    Returns:
        Expected points for the gameweek.
    """
    gw = state.current_gw
    squad = state.squad

    # Get expected points for all squad players
    player_xpts = {eid: xpts_provider(eid, gw) for eid in squad}

    # Pick best starting XI (greedy within formation constraints)
    starting, bench = _pick_starting_xi(squad, player_xpts, positions)

    # Base points from starting XI
    total = sum(player_xpts[eid] for eid in starting)

    # Captain: highest xPts starter gets doubled (or tripled with TC)
    if starting:
        captain = max(starting, key=lambda e: player_xpts[e])
        multiplier = 2 if chip != Chip.TRIPLE_CAPTAIN else 3
        total += player_xpts[captain] * (multiplier - 1)  # Add the extra

    # Bench Boost: bench players also score
    if chip == Chip.BENCH_BOOST:
        total += sum(player_xpts[eid] for eid in bench)

    return total


def _pick_starting_xi(
    squad: list[int],
    player_xpts: dict[int, float],
    positions: dict[int, str],
) -> tuple[list[int], list[int]]:
    """Greedy starting XI selection within formation constraints.

    Returns (starting_11, bench_4).
    """
    by_pos: dict[str, list[int]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for eid in squad:
        pos = positions.get(eid, "MID")
        by_pos[pos].append(eid)

    # Sort each position by xPts descending
    for pos in by_pos:
        by_pos[pos].sort(key=lambda e: player_xpts.get(e, 0), reverse=True)

    starting = []
    # 1 GK (best)
    if by_pos["GK"]:
        starting.append(by_pos["GK"][0])
    # Minimum: 3 DEF, 2 MID, 1 FWD
    starting.extend(by_pos["DEF"][:3])
    starting.extend(by_pos["MID"][:2])
    starting.extend(by_pos["FWD"][:1])

    # Fill remaining 4 slots with best available (respecting max 5 DEF, 5 MID, 3 FWD)
    remaining_candidates = []
    remaining_candidates.extend(by_pos["DEF"][3:5])
    remaining_candidates.extend(by_pos["MID"][2:5])
    remaining_candidates.extend(by_pos["FWD"][1:3])
    remaining_candidates.sort(key=lambda e: player_xpts.get(e, 0), reverse=True)

    slots_left = 11 - len(starting)
    starting.extend(remaining_candidates[:slots_left])

    bench = [eid for eid in squad if eid not in starting]
    return starting, bench


# ─── Action Generator ────────────────────────────────────────────────────────


def generate_actions(
    state: PlanningState,
    player_pool: pd.DataFrame,
    xpts_provider: XPtsProvider,
    positions: dict[int, str],
    prices: dict[int, int],
    teams: dict[int, int],
    max_transfers: int = 2,
    top_k_transfers: int = 3,
    consider_chips: bool = True,
) -> list[Action]:
    """Generate candidate actions for the current state.

    To keep the search tractable, we don't enumerate all possible actions.
    Instead we generate a curated set:
    - Roll (0 transfers)
    - Top-K single transfers (best xPts gain)
    - Top few double transfers (if 2 FTs or willing to hit)
    - Chip activations (if strategically sensible)

    Args:
        state: Current planning state.
        player_pool: DataFrame of available players.
        xpts_provider: Expected points function.
        positions: Element → position.
        prices: Element → price.
        teams: Element → team_id.
        max_transfers: Max transfers to consider in one GW (excl. wildcard).
        top_k_transfers: How many candidate transfers per position.
        consider_chips: Whether to generate chip actions.

    Returns:
        List of candidate Actions.
    """
    actions: list[Action] = []

    # 1. Roll (always an option)
    actions.append(Action(transfers=()))

    # 2. Find best single transfers
    best_transfers = _find_best_transfers(
        state, player_pool, xpts_provider, positions, prices, teams,
        top_k=top_k_transfers,
    )

    # Single transfer actions
    for transfer in best_transfers[:top_k_transfers]:
        actions.append(Action(transfers=(transfer,)))

    # 3. Double transfers (only if 2 FTs, to avoid hits in planning)
    if state.free_transfers >= 2 and max_transfers >= 2 and len(best_transfers) >= 2:
        # Combine top 2 transfers if they're for different players
        t1, t2 = best_transfers[0], best_transfers[1]
        if t1.sell != t2.sell and t1.buy != t2.buy:
            actions.append(Action(transfers=(t1, t2)))

    # 4. Chip actions
    if consider_chips:
        # Wildcard: signals a full squad rebuild — represented as a special action
        if state.has_wildcard:
            actions.append(Action(transfers=(), chip=Chip.WILDCARD))
        # Bench Boost: play current squad with bench scoring
        if state.has_bench_boost:
            actions.append(Action(transfers=(), chip=Chip.BENCH_BOOST))
        # Triple Captain: current squad, captain tripled
        if state.has_triple_captain:
            actions.append(Action(transfers=(), chip=Chip.TRIPLE_CAPTAIN))
        # Free Hit: temporary squad — represented as special action
        if state.has_free_hit:
            actions.append(Action(transfers=(), chip=Chip.FREE_HIT))

    return actions


def _find_best_transfers(
    state: PlanningState,
    player_pool: pd.DataFrame,
    xpts_provider: XPtsProvider,
    positions: dict[int, str],
    prices: dict[int, int],
    teams: dict[int, int],
    top_k: int = 3,
) -> list[Transfer]:
    """Find the best single transfers (highest xPts gain) for this GW."""
    gw = state.current_gw
    squad_set = set(state.squad)

    # Squad team counts (for club limit)
    team_counts: dict[int, int] = {}
    for eid in state.squad:
        tid = teams.get(eid, 0)
        team_counts[tid] = team_counts.get(tid, 0) + 1

    candidates: list[tuple[float, Transfer]] = []

    for sell_id in state.squad:
        sell_pos = positions.get(sell_id, "")
        sell_price = prices.get(sell_id, 0)
        sell_xpts = xpts_provider(sell_id, gw)
        sell_team = teams.get(sell_id, 0)
        available_budget = state.bank + sell_price

        # Candidate replacements
        replacements = player_pool[
            (player_pool["position"] == sell_pos)
            & (~player_pool["element"].isin(squad_set))
            & (player_pool["price"] <= available_budget)
        ]

        for _, row in replacements.iterrows():
            buy_id = int(row["element"])
            buy_team = int(row.get("team_id", 0))

            # Club limit check
            if buy_team != sell_team and team_counts.get(buy_team, 0) >= 3:
                continue

            buy_xpts = xpts_provider(buy_id, gw)
            gain = buy_xpts - sell_xpts

            if gain > 0:
                candidates.append((gain, Transfer(sell=sell_id, buy=buy_id)))

    # Sort by gain descending, dedupe by sell player (keep best per sell)
    candidates.sort(key=lambda x: x[0], reverse=True)

    seen_sells = set()
    result = []
    for gain, transfer in candidates:
        if transfer.sell not in seen_sells:
            result.append(transfer)
            seen_sells.add(transfer.sell)
        if len(result) >= top_k * 2:
            break

    return result
