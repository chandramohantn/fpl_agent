"""Single-match player point simulator.

The core sampling unit: given a player's predicted distributions for an
upcoming fixture, sample one possible outcome and compute FPL points.

Sampling order (conditional chain):
    1. Sample minutes category: 0 | 1-59 | 60+
    2. If minutes > 0:
        a. Sample goals from Poisson(λ_goals)
        b. Sample assists from Poisson(λ_assists)
        c. Sample yellow card from Bernoulli(p_yc)
        d. Sample red card from Bernoulli(p_rc)
    3. If minutes >= 60:
        e. Sample clean sheet from Bernoulli(p_cs)
    4. If GK and minutes >= 60:
        f. Sample saves from Poisson(λ_saves)
    5. Sample bonus from the predicted distribution
    6. Compute total FPL points from all sampled outcomes

This produces one draw from the player's points distribution.
Running this N times gives the full distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

# ─── FPL Scoring Rules ───────────────────────────────────────────────────────

POINTS_APPEARANCE_SHORT = 1   # 1-59 minutes
POINTS_APPEARANCE_FULL = 2    # 60+ minutes

POINTS_GOAL = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
POINTS_ASSIST = 3
POINTS_CLEAN_SHEET = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
POINTS_SAVES_PER_3 = 1       # Every 3 saves = 1 point
POINTS_PENALTY_SAVE = 5
POINTS_PENALTY_MISS = -2
POINTS_YELLOW_CARD = -1
POINTS_RED_CARD = -3
POINTS_OWN_GOAL = -2
POINTS_GOALS_CONCEDED_PER_2 = -1  # GK/DEF: every 2 goals conceded = -1


# ─── Player Prediction Input ─────────────────────────────────────────────────


@dataclass
class PlayerPrediction:
    """All predicted distributions for a player in one fixture.

    This is the input to the simulator — one per player per fixture.
    Each field represents the output of a component model.
    """

    element: int                # FPL player ID
    position: str               # GK, DEF, MID, FWD
    team: str                   # Team name or ID
    opponent: str               # Opponent name or ID
    is_home: bool               # Home fixture

    # Minutes probabilities (must sum to 1)
    p_no_play: float            # P(0 minutes)
    p_sub: float                # P(1-59 minutes)
    p_full: float               # P(60+ minutes)

    # Goals (Poisson lambda, conditioned on playing)
    lambda_goals: float = 0.0

    # Assists (Poisson lambda, conditioned on playing)
    lambda_assists: float = 0.0

    # Clean sheet (probability, conditioned on 60+ min)
    p_clean_sheet: float = 0.0

    # Saves (Poisson lambda, GK only, conditioned on 60+ min)
    lambda_saves: float = 0.0

    # Cards (conditioned on playing)
    p_yellow_card: float = 0.0
    p_red_card: float = 0.0

    # Bonus (expected value, 0-3)
    expected_bonus: float = 0.0

    # Goals conceded lambda (for GK/DEF penalty, conditioned on 60+ min)
    lambda_goals_conceded: float = 0.0


# ─── Simulated Outcome ───────────────────────────────────────────────────────


@dataclass
class SimulatedOutcome:
    """One simulated match outcome for a player."""

    element: int
    position: str
    minutes: int               # 0, ~30 (sub), or 90 (full)
    goals: int = 0
    assists: int = 0
    clean_sheet: bool = False
    saves: int = 0
    yellow_card: bool = False
    red_card: bool = False
    bonus: int = 0
    goals_conceded: int = 0
    total_points: int = 0


# ─── Core Simulator ──────────────────────────────────────────────────────────


def simulate_player_match(
    pred: PlayerPrediction,
    rng: Generator | None = None,
) -> SimulatedOutcome:
    """Simulate one possible match outcome for a player.

    Samples from all component distributions in the correct conditional
    order and computes the resulting FPL points.

    Args:
        pred: PlayerPrediction with all model outputs.
        rng: NumPy random generator (for reproducibility).

    Returns:
        SimulatedOutcome with sampled stats and computed points.
    """
    if rng is None:
        rng = np.random.default_rng()

    outcome = SimulatedOutcome(element=pred.element, position=pred.position)

    # ─── 1. Sample minutes ───────────────────────────────────────────────

    probs = np.array([pred.p_no_play, pred.p_sub, pred.p_full])
    probs = probs / probs.sum()  # Normalize in case of floating point drift
    minutes_cat = rng.choice([0, 1, 2], p=probs)

    if minutes_cat == 0:
        outcome.minutes = 0
        outcome.total_points = 0
        return outcome
    elif minutes_cat == 1:
        outcome.minutes = rng.integers(1, 60)  # Random sub minutes
    else:
        outcome.minutes = rng.integers(60, 91)  # Full match (60-90)

    # ─── 2. Sample goals (Poisson) ───────────────────────────────────────

    if pred.lambda_goals > 0:
        outcome.goals = int(rng.poisson(pred.lambda_goals))

    # ─── 3. Sample assists (Poisson) ─────────────────────────────────────

    if pred.lambda_assists > 0:
        outcome.assists = int(rng.poisson(pred.lambda_assists))

    # ─── 4. Sample cards ─────────────────────────────────────────────────

    if pred.p_yellow_card > 0:
        outcome.yellow_card = bool(rng.random() < pred.p_yellow_card)
    if pred.p_red_card > 0:
        outcome.red_card = bool(rng.random() < pred.p_red_card)

    # ─── 5. Sample clean sheet (only if 60+ min) ────────────────────────

    if outcome.minutes >= 60 and pred.p_clean_sheet > 0:
        outcome.clean_sheet = bool(rng.random() < pred.p_clean_sheet)

    # ─── 6. Sample saves (GK only, 60+ min) ─────────────────────────────

    if pred.position == "GK" and outcome.minutes >= 60 and pred.lambda_saves > 0:
        outcome.saves = int(rng.poisson(pred.lambda_saves))

    # ─── 7. Sample goals conceded (GK/DEF, 60+ min) ─────────────────────

    if outcome.minutes >= 60 and pred.position in ("GK", "DEF"):
        if pred.lambda_goals_conceded > 0:
            outcome.goals_conceded = int(rng.poisson(pred.lambda_goals_conceded))
        if outcome.clean_sheet:
            outcome.goals_conceded = 0  # Consistency: CS means 0 conceded

    # ─── 8. Sample bonus ─────────────────────────────────────────────────

    # Bonus is 0-3 integer. We sample from a simplified distribution
    # based on expected bonus.
    outcome.bonus = _sample_bonus(pred.expected_bonus, rng)

    # ─── 9. Compute FPL points ───────────────────────────────────────────

    outcome.total_points = _compute_points(outcome)

    return outcome


def simulate_player_match_batch(
    pred: PlayerPrediction,
    n_simulations: int = 10000,
    seed: int | None = None,
) -> np.ndarray:
    """Run N simulations for one player-fixture and return points array.

    Optimized vectorized version for bulk simulations.

    Args:
        pred: PlayerPrediction.
        n_simulations: Number of Monte Carlo samples.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_simulations,) with simulated FPL points.
    """
    rng = np.random.default_rng(seed)

    # Vectorized minutes sampling
    probs = np.array([pred.p_no_play, pred.p_sub, pred.p_full])
    probs = probs / probs.sum()
    minutes_cat = rng.choice([0, 1, 2], size=n_simulations, p=probs)

    played = minutes_cat > 0
    full_match = minutes_cat == 2

    # Initialize points array
    points = np.zeros(n_simulations)

    # Appearance points
    points[minutes_cat == 1] += POINTS_APPEARANCE_SHORT
    points[minutes_cat == 2] += POINTS_APPEARANCE_FULL

    # Goals (only when playing)
    if pred.lambda_goals > 0:
        goals = np.zeros(n_simulations, dtype=int)
        goals[played] = rng.poisson(pred.lambda_goals, size=played.sum())
        points += goals * POINTS_GOAL.get(pred.position, 4)

    # Assists
    if pred.lambda_assists > 0:
        assists = np.zeros(n_simulations, dtype=int)
        assists[played] = rng.poisson(pred.lambda_assists, size=played.sum())
        points += assists * POINTS_ASSIST

    # Clean sheet (60+ min only)
    if pred.p_clean_sheet > 0 and POINTS_CLEAN_SHEET.get(pred.position, 0) > 0:
        cs = np.zeros(n_simulations, dtype=bool)
        cs[full_match] = rng.random(size=full_match.sum()) < pred.p_clean_sheet
        points[cs] += POINTS_CLEAN_SHEET[pred.position]

    # Saves (GK only, 60+ min)
    if pred.position == "GK" and pred.lambda_saves > 0:
        saves = np.zeros(n_simulations, dtype=int)
        saves[full_match] = rng.poisson(pred.lambda_saves, size=full_match.sum())
        points += (saves // 3) * POINTS_SAVES_PER_3

    # Goals conceded penalty (GK/DEF, 60+ min)
    if pred.position in ("GK", "DEF") and pred.lambda_goals_conceded > 0:
        gc = np.zeros(n_simulations, dtype=int)
        gc[full_match] = rng.poisson(pred.lambda_goals_conceded, size=full_match.sum())
        # No penalty if CS
        if pred.p_clean_sheet > 0:
            cs_mask = np.zeros(n_simulations, dtype=bool)
            cs_mask[full_match] = rng.random(size=full_match.sum()) < pred.p_clean_sheet
            gc[cs_mask] = 0
        points -= (gc // 2) * abs(POINTS_GOALS_CONCEDED_PER_2)

    # Yellow cards
    if pred.p_yellow_card > 0:
        yc = np.zeros(n_simulations, dtype=bool)
        yc[played] = rng.random(size=played.sum()) < pred.p_yellow_card
        points[yc] += POINTS_YELLOW_CARD

    # Red cards
    if pred.p_red_card > 0:
        rc = np.zeros(n_simulations, dtype=bool)
        rc[played] = rng.random(size=played.sum()) < pred.p_red_card
        points[rc] += POINTS_RED_CARD

    # Bonus
    if pred.expected_bonus > 0:
        bonus = np.zeros(n_simulations, dtype=int)
        bonus[played] = _sample_bonus_vectorized(
            pred.expected_bonus, played.sum(), rng
        )
        points += bonus

    return points


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _sample_bonus(expected_bonus: float, rng: Generator) -> int:
    """Sample bonus points (0-3) from expected value.

    Uses a simplified categorical distribution:
    P(3) ≈ expected/3 × P(getting any bonus)
    P(0) = 1 - P(1) - P(2) - P(3)
    """
    if expected_bonus <= 0:
        return 0

    # Approximate: P(any bonus) ≈ min(expected_bonus / 1.5, 0.5)
    p_any = min(expected_bonus / 1.5, 0.5)
    # Distribution among 1, 2, 3 is roughly uniform
    p_each = p_any / 3

    r = rng.random()
    if r < (1 - p_any):
        return 0
    elif r < (1 - 2 * p_each):
        return 1
    elif r < (1 - p_each):
        return 2
    else:
        return 3


def _sample_bonus_vectorized(
    expected_bonus: float, n: int, rng: Generator
) -> np.ndarray:
    """Vectorized bonus sampling."""
    if expected_bonus <= 0 or n == 0:
        return np.zeros(n, dtype=int)

    p_any = min(expected_bonus / 1.5, 0.5)
    p_each = p_any / 3
    p_0 = 1 - p_any

    probs = [p_0, p_each, p_each, p_each]
    # Normalize
    probs = np.array(probs)
    probs = probs / probs.sum()

    return rng.choice([0, 1, 2, 3], size=n, p=probs)


def _compute_points(outcome: SimulatedOutcome) -> int:
    """Compute total FPL points from a simulated outcome."""
    pts = 0
    pos = outcome.position

    # Appearance
    if outcome.minutes >= 60:
        pts += POINTS_APPEARANCE_FULL
    elif outcome.minutes > 0:
        pts += POINTS_APPEARANCE_SHORT

    # Goals
    pts += outcome.goals * POINTS_GOAL.get(pos, 4)

    # Assists
    pts += outcome.assists * POINTS_ASSIST

    # Clean sheet (60+ min only)
    if outcome.minutes >= 60 and outcome.clean_sheet:
        pts += POINTS_CLEAN_SHEET.get(pos, 0)

    # Saves (GK)
    if pos == "GK":
        pts += (outcome.saves // 3) * POINTS_SAVES_PER_3

    # Goals conceded (GK/DEF, 60+ min, -1 per 2 conceded)
    if outcome.minutes >= 60 and pos in ("GK", "DEF"):
        pts -= (outcome.goals_conceded // 2)

    # Cards
    if outcome.yellow_card:
        pts += POINTS_YELLOW_CARD
    if outcome.red_card:
        pts += POINTS_RED_CARD

    # Bonus
    pts += outcome.bonus

    return pts
