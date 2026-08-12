"""Gameweek and season-level simulation.

Orchestrates Monte Carlo simulations across multiple players and gameweeks.

Levels:
1. Player-match: simulate_player_match_batch() — single player, single fixture
2. Gameweek: simulate_gameweek() — all players in a squad for one GW
3. Season: simulate_season() — chain GW simulations across 38 weeks

Key outputs:
- Per-player: mean, std, percentiles, P(haul), P(blank)
- Per-squad: total points distribution per GW
- Per-season: cumulative points, rank distribution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fpl_engine.simulation.player_sim import (
    PlayerPrediction,
    simulate_player_match_batch,
)

logger = logging.getLogger(__name__)


# ─── Gameweek Simulation ─────────────────────────────────────────────────────


@dataclass
class PlayerSimResult:
    """Monte Carlo simulation result for one player in one gameweek."""

    element: int
    position: str
    n_simulations: int
    points_array: np.ndarray  # Raw points from all simulations

    @property
    def mean(self) -> float:
        return float(self.points_array.mean())

    @property
    def std(self) -> float:
        return float(self.points_array.std())

    @property
    def median(self) -> float:
        return float(np.median(self.points_array))

    @property
    def p10(self) -> float:
        """10th percentile (downside)."""
        return float(np.percentile(self.points_array, 10))

    @property
    def p90(self) -> float:
        """90th percentile (upside)."""
        return float(np.percentile(self.points_array, 90))

    @property
    def p_blank(self) -> float:
        """P(2 or fewer points) — a 'blank'."""
        return float((self.points_array <= 2).mean())

    @property
    def p_haul(self) -> float:
        """P(10+ points) — a 'haul'."""
        return float((self.points_array >= 10).mean())

    @property
    def p_return(self) -> float:
        """P(5+ points) — a 'return' (goal/assist/CS likely)."""
        return float((self.points_array >= 5).mean())

    def percentile(self, q: float) -> float:
        """Get arbitrary percentile."""
        return float(np.percentile(self.points_array, q))

    def to_dict(self) -> dict:
        """Summary statistics as dict."""
        return {
            "element": self.element,
            "position": self.position,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "p10": self.p10,
            "p90": self.p90,
            "p_blank": self.p_blank,
            "p_return": self.p_return,
            "p_haul": self.p_haul,
        }


@dataclass
class GameweekSimResult:
    """Monte Carlo simulation result for an entire gameweek."""

    gameweek: int
    n_simulations: int
    player_results: dict[int, PlayerSimResult] = field(default_factory=dict)

    def get_player(self, element: int) -> PlayerSimResult | None:
        return self.player_results.get(element)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all player results to a summary DataFrame."""
        records = [r.to_dict() for r in self.player_results.values()]
        return pd.DataFrame(records).sort_values("mean", ascending=False)

    def squad_points(
        self,
        squad: list[int],
        captain: int | None = None,
        vice_captain: int | None = None,
    ) -> np.ndarray:
        """Compute total squad points distribution.

        Args:
            squad: List of 11 starting player element IDs.
            captain: Captain element ID (doubled points).
            vice_captain: Vice-captain ID (doubled if captain doesn't play).

        Returns:
            Array of shape (n_simulations,) with total squad points.
        """
        n = self.n_simulations
        total = np.zeros(n)

        captain_played = np.ones(n, dtype=bool)  # Track if captain plays

        for element in squad:
            result = self.player_results.get(element)
            if result is None:
                continue

            pts = result.points_array.copy()

            if element == captain:
                # Captain gets double points
                captain_played = pts > 0
                total += pts * 2
            elif element == vice_captain:
                # VC gets double only when captain doesn't play
                vc_doubled = ~captain_played
                pts_vc = pts.copy()
                pts_vc[vc_doubled] *= 2
                total += pts_vc
            else:
                total += pts

        return total


def simulate_gameweek(
    predictions: list[PlayerPrediction],
    n_simulations: int = 10000,
    seed: int | None = None,
    gameweek: int = 0,
) -> GameweekSimResult:
    """Simulate a full gameweek for all players.

    Runs N Monte Carlo simulations for each player independently.
    Players in the same team share clean sheet outcomes (correlated
    via the same p_clean_sheet input).

    Args:
        predictions: List of PlayerPrediction for all players in this GW.
        n_simulations: Number of Monte Carlo samples per player.
        seed: Random seed for reproducibility.
        gameweek: Gameweek number (for tracking).

    Returns:
        GameweekSimResult with per-player simulation arrays.
    """
    logger.info(
        "Simulating GW%d: %d players × %d simulations",
        gameweek, len(predictions), n_simulations,
    )

    result = GameweekSimResult(
        gameweek=gameweek,
        n_simulations=n_simulations,
    )

    for i, pred in enumerate(predictions):
        # Each player gets their own seed derived from the base seed
        player_seed = seed + i if seed is not None else None

        points = simulate_player_match_batch(
            pred, n_simulations=n_simulations, seed=player_seed
        )

        result.player_results[pred.element] = PlayerSimResult(
            element=pred.element,
            position=pred.position,
            n_simulations=n_simulations,
            points_array=points,
        )

    logger.info("GW%d simulation complete", gameweek)
    return result


# ─── Season Simulation ───────────────────────────────────────────────────────


@dataclass
class SeasonSimResult:
    """Monte Carlo simulation result for a full season."""

    n_simulations: int
    gameweek_results: dict[int, GameweekSimResult] = field(default_factory=dict)

    def cumulative_points(self, element: int) -> np.ndarray:
        """Get cumulative points array across all GWs for a player.

        Returns array of shape (n_simulations,) with total season points.
        """
        total = np.zeros(self.n_simulations)
        for gw_result in self.gameweek_results.values():
            player = gw_result.get_player(element)
            if player is not None:
                total += player.points_array
        return total

    def season_summary(self, elements: list[int] | None = None) -> pd.DataFrame:
        """Get season-level summary for players.

        Returns DataFrame with: element, total_mean, total_std, p10, p90
        """
        if elements is None:
            # Gather all players across all GWs
            elements = set()
            for gw in self.gameweek_results.values():
                elements.update(gw.player_results.keys())
            elements = sorted(elements)

        records = []
        for element in elements:
            total = self.cumulative_points(element)
            if total.sum() == 0:
                continue
            records.append({
                "element": element,
                "total_mean": total.mean(),
                "total_std": total.std(),
                "total_median": np.median(total),
                "total_p10": np.percentile(total, 10),
                "total_p90": np.percentile(total, 90),
            })

        return pd.DataFrame(records).sort_values("total_mean", ascending=False)


def simulate_season(
    gameweek_predictions: dict[int, list[PlayerPrediction]],
    n_simulations: int = 10000,
    seed: int | None = None,
) -> SeasonSimResult:
    """Simulate a full season (multiple gameweeks).

    Args:
        gameweek_predictions: Dict mapping GW number → list of PlayerPredictions.
        n_simulations: Number of Monte Carlo samples.
        seed: Base random seed.

    Returns:
        SeasonSimResult with all GW results.
    """
    logger.info(
        "Simulating season: %d GWs × %d simulations",
        len(gameweek_predictions), n_simulations,
    )

    result = SeasonSimResult(n_simulations=n_simulations)

    for gw_num in sorted(gameweek_predictions.keys()):
        preds = gameweek_predictions[gw_num]
        gw_seed = seed + gw_num * 1000 if seed is not None else None

        gw_result = simulate_gameweek(
            preds, n_simulations=n_simulations, seed=gw_seed, gameweek=gw_num
        )
        result.gameweek_results[gw_num] = gw_result

    logger.info("Season simulation complete: %d GWs", len(result.gameweek_results))
    return result
