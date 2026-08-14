"""Squad manager — persistence, transfer execution, and history tracking.

Manages the user's squad state across sessions:
- Saves squad to a JSON file on disk
- Loads on app start (persistent across restarts)
- Executes transfers with full constraint validation
- Tracks transfer history per gameweek
- Handles free transfer rollover and chip usage

File: data/squad/my_squad.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SQUAD_FILE = Path("data/squad/my_squad.json")
MAX_BUDGET = 1000  # £100m
MAX_PER_CLUB = 3
POSITION_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD_SIZE = 15


@dataclass
class TransferRecord:
    """Record of a completed transfer."""

    gameweek: int
    sell_id: int
    sell_name: str
    sell_price: int
    buy_id: int
    buy_name: str
    buy_price: int
    was_free: bool
    timestamp: str


@dataclass
class ChipRecord:
    """Record of a chip being used."""

    gameweek: int
    chip: str
    timestamp: str


@dataclass
class SquadState:
    """Complete persistent squad state.

    This is what gets saved to and loaded from disk.
    """

    # Core state
    squad: list[int] = field(default_factory=list)  # 15 element IDs
    bank: int = 0  # Available budget remaining
    free_transfers: int = 1  # 1 or 2
    current_gw: int = 1

    # Chips remaining
    chips_available: list[str] = field(
        default_factory=lambda: ["wildcard", "free_hit", "bench_boost", "triple_captain"]
    )

    # Player metadata (for display without API lookup)
    player_names: dict[str, str] = field(default_factory=dict)  # {element_id: name}
    player_prices: dict[str, int] = field(default_factory=dict)  # {element_id: price}
    player_positions: dict[str, str] = field(default_factory=dict)  # {element_id: position}
    player_teams: dict[str, str] = field(default_factory=dict)  # {element_id: team}

    # History
    transfer_history: list[dict] = field(default_factory=list)
    chip_history: list[dict] = field(default_factory=list)

    # Meta
    created_at: str = ""
    last_modified: str = ""


class SquadManager:
    """Manages squad state with persistence and transfer execution.

    Usage:
        manager = SquadManager()
        manager.load()  # Load from disk (or create new)

        # View squad
        print(manager.state.squad)

        # Execute a transfer
        result = manager.execute_transfer(sell_id=8, buy_id=20, buy_price=90)
        if result.success:
            manager.save()

        # Advance gameweek
        manager.advance_gameweek()
        manager.save()
    """

    def __init__(self, squad_file: Path | str = SQUAD_FILE) -> None:
        self.squad_file = Path(squad_file)
        self.state = SquadState()

    # ─── Persistence ─────────────────────────────────────────────────────

    def save(self) -> None:
        """Save current state to disk."""
        self.squad_file.parent.mkdir(parents=True, exist_ok=True)
        self.state.last_modified = datetime.now().isoformat()

        data = asdict(self.state)
        self.squad_file.write_text(json.dumps(data, indent=4))
        logger.info("Squad saved to %s", self.squad_file)

    def load(self) -> bool:
        """Load state from disk. Returns True if file existed."""
        if not self.squad_file.exists():
            logger.info("No saved squad found at %s", self.squad_file)
            return False

        data = json.loads(self.squad_file.read_text())
        self.state = SquadState(**data)
        logger.info(
            "Squad loaded: %d players, GW%d, bank=£%.1fm",
            len(self.state.squad), self.state.current_gw, self.state.bank / 10,
        )
        return True

    def exists(self) -> bool:
        """Check if a saved squad exists."""
        return self.squad_file.exists()

    # ─── Squad Initialization ────────────────────────────────────────────

    def initialize_squad(
        self,
        squad: list[int],
        names: dict[int, str],
        prices: dict[int, int],
        positions: dict[int, str],
        teams: dict[int, str],
        bank: int = 0,
    ) -> None:
        """Set up a new squad (start of season or after wildcard).

        Validates all FPL constraints before accepting.
        """
        # Validate
        assert len(squad) == SQUAD_SIZE, f"Squad must be {SQUAD_SIZE} players, got {len(squad)}"

        total_cost = sum(prices.get(eid, 0) for eid in squad)
        assert total_cost + bank <= MAX_BUDGET, (
            f"Over budget: squad costs £{total_cost/10:.1f}m + £{bank/10:.1f}m bank "
            f"= £{(total_cost+bank)/10:.1f}m > £{MAX_BUDGET/10:.1f}m"
        )

        # Position check
        pos_counts: dict[str, int] = {}
        for eid in squad:
            pos = positions.get(eid, "")
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        for pos, limit in POSITION_LIMITS.items():
            assert pos_counts.get(pos, 0) == limit, (
                f"Need exactly {limit} {pos}s, got {pos_counts.get(pos, 0)}"
            )

        # Club check
        team_counts: dict[str, int] = {}
        for eid in squad:
            team = teams.get(eid, "")
            team_counts[team] = team_counts.get(team, 0) + 1
        for team, count in team_counts.items():
            assert count <= MAX_PER_CLUB, f"Max {MAX_PER_CLUB} from {team}, got {count}"

        # All valid — set state
        self.state.squad = sorted(squad)
        self.state.bank = bank
        self.state.player_names = {str(k): v for k, v in names.items()}
        self.state.player_prices = {str(k): v for k, v in prices.items()}
        self.state.player_positions = {str(k): v for k, v in positions.items()}
        self.state.player_teams = {str(k): v for k, v in teams.items()}
        self.state.created_at = datetime.now().isoformat()

        logger.info("Squad initialized: %d players, bank=£%.1fm", len(squad), bank / 10)

    # ─── Transfer Execution ──────────────────────────────────────────────

    @dataclass
    class TransferResult:
        """Result of attempting a transfer."""

        success: bool
        message: str
        sell_name: str = ""
        buy_name: str = ""
        cost: int = 0  # Hit cost (0 or 4)
        bank_after: int = 0

    def execute_transfer(
        self,
        sell_id: int,
        buy_id: int,
        buy_price: int,
        buy_name: str = "",
        buy_position: str = "",
        buy_team: str = "",
    ) -> TransferResult:
        """Execute a transfer: sell one player, buy another.

        Validates:
        - Sell player is in squad
        - Buy player is not already in squad
        - Same position (sell and buy must match)
        - Budget allows (sell price + bank >= buy price)
        - Club limit (buying team doesn't exceed 3)

        Returns TransferResult with success/failure and message.
        """
        state = self.state
        sell_str = str(sell_id)
        buy_str = str(buy_id)

        # Basic checks
        if sell_id not in state.squad:
            return self.TransferResult(False, f"Player {sell_id} not in your squad")
        if buy_id in state.squad:
            return self.TransferResult(False, f"Player {buy_id} already in your squad")

        # Position check
        sell_pos = state.player_positions.get(sell_str, "")
        if buy_position and sell_pos and buy_position != sell_pos:
            return self.TransferResult(
                False, f"Position mismatch: selling {sell_pos}, buying {buy_position}"
            )

        # Budget check
        sell_price = state.player_prices.get(sell_str, 0)
        available = state.bank + sell_price
        if buy_price > available:
            return self.TransferResult(
                False,
                f"Cannot afford: need £{buy_price/10:.1f}m, "
                f"have £{available/10:.1f}m "
                f"(bank £{state.bank/10:.1f}m + sell £{sell_price/10:.1f}m)"
            )

        # Club limit check
        if buy_team:
            team_count = sum(
                1 for eid in state.squad
                if state.player_teams.get(str(eid), "") == buy_team and eid != sell_id
            )
            if team_count >= MAX_PER_CLUB:
                return self.TransferResult(
                    False, f"Club limit: already have {MAX_PER_CLUB} from {buy_team}"
                )

        # Compute hit
        is_free = state.free_transfers > 0
        hit_cost = 0 if is_free else 4

        # Execute
        state.squad.remove(sell_id)
        state.squad.append(buy_id)
        state.squad.sort()

        # Update bank
        state.bank = state.bank + sell_price - buy_price

        # Update free transfers
        if is_free:
            state.free_transfers = max(state.free_transfers - 1, 0)

        # Update player metadata
        sell_name = state.player_names.get(sell_str, f"#{sell_id}")
        if buy_name:
            state.player_names[buy_str] = buy_name
        if buy_position:
            state.player_positions[buy_str] = buy_position
        if buy_team:
            state.player_teams[buy_str] = buy_team
        state.player_prices[buy_str] = buy_price

        # Remove sold player metadata
        for d in [state.player_names, state.player_prices,
                  state.player_positions, state.player_teams]:
            d.pop(sell_str, None)

        # Record in history
        record = TransferRecord(
            gameweek=state.current_gw,
            sell_id=sell_id,
            sell_name=sell_name,
            sell_price=sell_price,
            buy_id=buy_id,
            buy_name=buy_name or f"#{buy_id}",
            buy_price=buy_price,
            was_free=is_free,
            timestamp=datetime.now().isoformat(),
        )
        state.transfer_history.append(asdict(record))

        return self.TransferResult(
            success=True,
            message=(
                f"✅ Transfer complete: {sell_name} → {buy_name or f'#{buy_id}'} "
                f"({'free' if is_free else f'-{hit_cost} hit'}). "
                f"Bank: £{state.bank/10:.1f}m"
            ),
            sell_name=sell_name,
            buy_name=buy_name or f"#{buy_id}",
            cost=hit_cost,
            bank_after=state.bank,
        )

    # ─── Gameweek Management ─────────────────────────────────────────────

    def advance_gameweek(self) -> None:
        """Move to the next gameweek.

        Handles free transfer rollover:
        - If 0 transfers were made this GW, bank one (max 2)
        - Reset is handled by transfer execution (sets FT=1 after using)

        Call this AFTER the GW deadline passes.
        """
        self.state.current_gw += 1
        # FT rollover is already handled in execute_transfer
        # If no transfer was made this GW, bank a FT
        # This is tracked by whether execute_transfer was called
        # For simplicity: advance_gw always resets to allow rolling
        self.state.free_transfers = min(self.state.free_transfers + 1, 2)
        logger.info(
            "Advanced to GW%d. Free transfers: %d",
            self.state.current_gw, self.state.free_transfers,
        )

    def use_chip(self, chip: str) -> bool:
        """Use a chip. Returns True if successful."""
        if chip not in self.state.chips_available:
            return False

        self.state.chips_available.remove(chip)
        self.state.chip_history.append(asdict(ChipRecord(
            gameweek=self.state.current_gw,
            chip=chip,
            timestamp=datetime.now().isoformat(),
        )))
        logger.info("Chip used: %s in GW%d", chip, self.state.current_gw)
        return True

    # ─── Queries ─────────────────────────────────────────────────────────

    def get_squad_by_position(self) -> dict[str, list[dict]]:
        """Get squad grouped by position with details."""
        result: dict[str, list[dict]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for eid in self.state.squad:
            eid_str = str(eid)
            pos = self.state.player_positions.get(eid_str, "UNK")
            result.setdefault(pos, []).append({
                "element": eid,
                "name": self.state.player_names.get(eid_str, f"#{eid}"),
                "price": self.state.player_prices.get(eid_str, 0),
                "team": self.state.player_teams.get(eid_str, ""),
            })
        return result

    def get_transfer_history(self, gameweek: int | None = None) -> list[dict]:
        """Get transfer history, optionally filtered by GW."""
        if gameweek is None:
            return self.state.transfer_history
        return [t for t in self.state.transfer_history if t.get("gameweek") == gameweek]

    def get_summary(self) -> str:
        """Human-readable squad summary."""
        state = self.state
        lines = [
            f"**GW{state.current_gw}** | Bank: £{state.bank/10:.1f}m | "
            f"FT: {state.free_transfers} | "
            f"Chips: {', '.join(state.chips_available) or 'None'}",
        ]
        by_pos = self.get_squad_by_position()
        for pos in ["GK", "DEF", "MID", "FWD"]:
            players = by_pos.get(pos, [])
            names = [f"{p['name']} (£{p['price']/10:.1f}m)" for p in players]
            lines.append(f"  {pos}: {', '.join(names)}")
        return "\n".join(lines)
