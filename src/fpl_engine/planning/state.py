"""FPL planning state space and action model.

Models the FPL game as a finite-horizon sequential decision process:
- State: squad, bank, free transfers, chips available, current GW
- Actions: transfers, chip activations, roll transfer
- Transitions: deterministic (apply action) + stochastic (match outcomes)
- Reward: expected FPL points from simulation

The planning horizon is typically 3-8 GWs ahead. Beyond that, prediction
uncertainty makes detailed planning unreliable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ─── Chips ───────────────────────────────────────────────────────────────────


class Chip(str, Enum):
    """FPL chips (each usable once per season)."""

    WILDCARD = "wildcard"         # Unlimited free transfers for one GW
    FREE_HIT = "free_hit"        # Temporary squad for one GW (reverts next GW)
    BENCH_BOOST = "bench_boost"  # Bench players score points this GW
    TRIPLE_CAPTAIN = "triple_captain"  # Captain gets 3× instead of 2×


# ─── Actions ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Transfer:
    """A single player transfer (sell → buy)."""

    sell: int  # Element to sell
    buy: int   # Element to buy


@dataclass(frozen=True)
class Action:
    """A complete action for a gameweek.

    An action combines:
    - A set of transfers (0, 1, or multiple)
    - Optional chip activation
    - Captain choice (deferred to optimizer, not part of planning action)
    """

    transfers: tuple[Transfer, ...] = ()
    chip: Chip | None = None

    @property
    def n_transfers(self) -> int:
        return len(self.transfers)

    @property
    def is_roll(self) -> bool:
        """Rolling a transfer = making 0 transfers to bank one for next GW."""
        return self.n_transfers == 0 and self.chip is None

    @property
    def hits(self) -> int:
        """Number of point hits (-4 each) based on transfers beyond free."""
        # Note: actual hits depend on state.free_transfers, computed at apply time
        return 0  # Computed during transition

    def __repr__(self) -> str:
        parts = []
        if self.transfers:
            for t in self.transfers:
                parts.append(f"{t.sell}→{t.buy}")
        else:
            parts.append("ROLL")
        if self.chip:
            parts.append(f"[{self.chip.value}]")
        return f"Action({', '.join(parts)})"


# ─── State ───────────────────────────────────────────────────────────────────


@dataclass
class PlanningState:
    """Complete FPL game state at a point in time.

    Represents everything the planner needs to know to make decisions.
    """

    # Squad composition
    squad: list[int]  # 15 element IDs
    bank: int = 0     # Available bank (FPL price units, e.g., 5 = £0.5m)

    # Transfer state
    free_transfers: int = 1  # Available free transfers (1 or 2)

    # Chips remaining
    chips_available: set[Chip] = field(default_factory=lambda: {
        Chip.WILDCARD, Chip.FREE_HIT, Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN,
    })

    # Time
    current_gw: int = 1
    total_gws: int = 38

    # Tracking
    cumulative_points: float = 0.0
    total_hits: int = 0

    @property
    def gws_remaining(self) -> int:
        return self.total_gws - self.current_gw + 1

    @property
    def has_wildcard(self) -> bool:
        return Chip.WILDCARD in self.chips_available

    @property
    def has_free_hit(self) -> bool:
        return Chip.FREE_HIT in self.chips_available

    @property
    def has_bench_boost(self) -> bool:
        return Chip.BENCH_BOOST in self.chips_available

    @property
    def has_triple_captain(self) -> bool:
        return Chip.TRIPLE_CAPTAIN in self.chips_available

    def copy(self) -> PlanningState:
        """Create a deep copy of this state."""
        return PlanningState(
            squad=self.squad.copy(),
            bank=self.bank,
            free_transfers=self.free_transfers,
            chips_available=self.chips_available.copy(),
            current_gw=self.current_gw,
            total_gws=self.total_gws,
            cumulative_points=self.cumulative_points,
            total_hits=self.total_hits,
        )


# ─── State Transitions ───────────────────────────────────────────────────────


def apply_action(
    state: PlanningState,
    action: Action,
    prices: dict[int, int] | None = None,
) -> PlanningState:
    """Apply an action to a state and return the new state.

    Handles:
    - Transfer execution (sell/buy, update bank)
    - Free transfer accounting (hits for extra transfers)
    - Chip usage (remove from available)
    - Free transfer rollover (0 transfers → bank one, max 2)
    - Advance to next GW

    Args:
        state: Current state.
        action: Action to apply.
        prices: Dict of element → price (needed for bank updates on transfers).

    Returns:
        New state after the action.
    """
    new_state = state.copy()

    # Handle chip: Wildcard means all transfers are free
    is_wildcard = action.chip == Chip.WILDCARD
    is_free_hit = action.chip == Chip.FREE_HIT

    # Execute transfers
    if action.transfers:
        n_transfers = len(action.transfers)

        # Compute hits
        if is_wildcard or is_free_hit:
            hits = 0  # All transfers free on WC/FH
        else:
            hits = max(0, n_transfers - state.free_transfers)

        new_state.total_hits += hits
        new_state.cumulative_points -= hits * 4  # -4 per hit

        # Update squad
        squad_set = set(new_state.squad)
        for transfer in action.transfers:
            squad_set.discard(transfer.sell)
            squad_set.add(transfer.buy)

            # Update bank
            if prices:
                sell_price = prices.get(transfer.sell, 0)
                buy_price = prices.get(transfer.buy, 0)
                new_state.bank += sell_price - buy_price

        new_state.squad = sorted(squad_set)

        # Free transfer update: used transfers, reset to 1
        if not is_wildcard and not is_free_hit:
            new_state.free_transfers = 1
    else:
        # No transfers: roll (bank a free transfer, max 2)
        new_state.free_transfers = min(state.free_transfers + 1, 2)

    # Handle Free Hit: squad reverts next GW (store original)
    # Note: for simplicity, Free Hit planning just evaluates the temporary squad
    # and reverts automatically at GW advance.

    # Use chip
    if action.chip:
        new_state.chips_available.discard(action.chip)

    # Advance GW
    new_state.current_gw += 1

    return new_state


def compute_action_cost(
    state: PlanningState,
    action: Action,
) -> int:
    """Compute the point cost (hits) of an action given the current state."""
    if action.chip in (Chip.WILDCARD, Chip.FREE_HIT):
        return 0
    return max(0, action.n_transfers - state.free_transfers) * 4
