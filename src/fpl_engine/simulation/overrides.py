"""Apply manual player context to simulation-ready player predictions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from fpl_engine.models.player_context import AvailabilityStatus, PlayerContext
from fpl_engine.simulation.player_sim import PlayerPrediction


@dataclass(frozen=True)
class OverrideEffect:
    """A visible record of a manual context adjustment."""

    element: int
    baseline_play_probability: float
    adjusted_play_probability: float
    reasons: tuple[str, ...]


def apply_manual_overrides(
    predictions: list[PlayerPrediction], contexts: dict[int, PlayerContext]
) -> tuple[list[PlayerPrediction], list[OverrideEffect]]:
    """Apply saved manual contexts to minutes probabilities before simulation.

    Manual availability is treated as authoritative. Fitness, fatigue, and a
    nearby important match reduce full-match and total appearance probability
    in a transparent fallback layer until model-backed predictions are used.
    """
    adjusted_predictions = []
    effects = []
    for prediction in predictions:
        context = contexts.get(prediction.element)
        if context is None:
            adjusted_predictions.append(prediction)
            continue

        adjusted, effect = apply_manual_override(prediction, context)
        adjusted_predictions.append(adjusted)
        if effect is not None:
            effects.append(effect)
    return adjusted_predictions, effects


def apply_manual_override(
    prediction: PlayerPrediction, context: PlayerContext
) -> tuple[PlayerPrediction, OverrideEffect | None]:
    """Apply one PlayerContext and return an adjusted prediction plus its effect."""
    baseline_play = 1.0 - prediction.p_no_play
    p_no_play, p_sub, p_full = _normalise_probabilities(prediction)
    reasons: list[str] = []

    if context.status in {
        AvailabilityStatus.INJURED,
        AvailabilityStatus.SUSPENDED,
        AvailabilityStatus.UNAVAILABLE,
    }:
        adjusted = replace(prediction, p_no_play=1.0, p_sub=0.0, p_full=0.0)
        reasons.append(context.status.value)
        return adjusted, OverrideEffect(prediction.element, baseline_play, 0.0, tuple(reasons))

    if context.chance_of_playing is not None:
        target_play = max(0.0, min(1.0, context.chance_of_playing / 100.0))
        p_no_play, p_sub, p_full = _set_play_probability(target_play, p_sub, p_full)
        reasons.append(f"manual chance {context.chance_of_playing}%")

    if context.status == AvailabilityStatus.DOUBTFUL:
        current_play = 1.0 - p_no_play
        if current_play > 0.75:
            p_no_play, p_sub, p_full = _set_play_probability(0.75, p_sub, p_full)
            reasons.append("doubtful status")

    full_match_factor = 1.0
    if context.returning_from_injury:
        fitness = context.fitness_level
        if fitness is None:
            fitness = max(0.3, 1.0 - context.injury_duration_weeks * 0.08)
        full_match_factor *= 0.5 + 0.5 * fitness
        reasons.append("returning from injury")

    if (
        context.days_since_last_match is not None
        and context.played_minutes_last_match is not None
        and context.played_minutes_last_match >= 60
    ):
        fatigue_score = context.played_minutes_last_match / (
            max(context.days_since_last_match, 0.5) * 90
        )
        if fatigue_score >= 0.5:
            full_match_factor *= max(0.6, 1.0 - 0.2 * fatigue_score)
            reasons.append("short-rest fatigue")

    if context.important_match_in_days is not None and context.important_match_in_days <= 3:
        full_match_factor *= 0.85
        reasons.append("nearby important match")

    if full_match_factor < 1.0:
        reduced_full = p_full * full_match_factor
        displaced_probability = p_full - reduced_full
        p_full = reduced_full
        p_sub += displaced_probability * 0.5
        p_no_play += displaced_probability * 0.5

    p_no_play, p_sub, p_full = _normalise_values(p_no_play, p_sub, p_full)
    adjusted = replace(prediction, p_no_play=p_no_play, p_sub=p_sub, p_full=p_full)
    adjusted_play = 1.0 - p_no_play
    effect = None
    if reasons or abs(adjusted_play - baseline_play) > 1e-9:
        effect = OverrideEffect(prediction.element, baseline_play, adjusted_play, tuple(reasons))
    return adjusted, effect


def _normalise_probabilities(prediction: PlayerPrediction) -> tuple[float, float, float]:
    return _normalise_values(prediction.p_no_play, prediction.p_sub, prediction.p_full)


def _normalise_values(p_no_play: float, p_sub: float, p_full: float) -> tuple[float, float, float]:
    total = p_no_play + p_sub + p_full
    if total <= 0:
        return 1.0, 0.0, 0.0
    return p_no_play / total, p_sub / total, p_full / total


def _set_play_probability(
    target_play: float, p_sub: float, p_full: float
) -> tuple[float, float, float]:
    appearance_total = p_sub + p_full
    if appearance_total <= 0:
        p_sub, p_full = 0.1, 0.9
        appearance_total = 1.0
    return (
        1.0 - target_play,
        target_play * p_sub / appearance_total,
        target_play * p_full / appearance_total,
    )
