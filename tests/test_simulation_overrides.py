from fpl_engine.models.player_context import AvailabilityStatus, PlayerContext
from fpl_engine.simulation.overrides import apply_manual_override
from fpl_engine.simulation.player_sim import PlayerPrediction, simulate_player_match_batch


def _prediction() -> PlayerPrediction:
    return PlayerPrediction(
        element=1,
        position="MID",
        team="Arsenal",
        opponent="Chelsea",
        is_home=True,
        p_no_play=0.05,
        p_sub=0.10,
        p_full=0.85,
    )


def test_zero_chance_override_prevents_appearance():
    adjusted, effect = apply_manual_override(
        _prediction(),
        PlayerContext(player_id=1, chance_of_playing=0, status=AvailabilityStatus.AVAILABLE),
    )

    assert (adjusted.p_no_play, adjusted.p_sub, adjusted.p_full) == (1.0, 0.0, 0.0)
    assert effect is not None
    assert effect.adjusted_play_probability == 0.0
    assert (simulate_player_match_batch(adjusted, n_simulations=100, seed=42) == 0).all()


def test_returning_from_injury_reduces_play_probability():
    adjusted, effect = apply_manual_override(
        _prediction(),
        PlayerContext(
            player_id=1,
            returning_from_injury=True,
            injury_duration_weeks=8,
            fitness_level=0.5,
        ),
    )

    assert adjusted.p_no_play > _prediction().p_no_play
    assert effect is not None
    assert "returning from injury" in effect.reasons
    assert sum((adjusted.p_no_play, adjusted.p_sub, adjusted.p_full)) == 1.0
