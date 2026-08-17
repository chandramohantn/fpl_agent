from types import SimpleNamespace

from fpl_engine.squad.formation import recommend_formations


def test_recommend_formations_selects_highest_expected_points_xi():
    players = [
        SimpleNamespace(element=1, position="GK"),
        SimpleNamespace(element=2, position="GK"),
        *[SimpleNamespace(element=10 + index, position="DEF") for index in range(5)],
        *[SimpleNamespace(element=20 + index, position="MID") for index in range(5)],
        *[SimpleNamespace(element=30 + index, position="FWD") for index in range(3)],
    ]
    results = {player.element: {"mean": 1.0} for player in players}
    results[20]["mean"] = 12.0
    results[21]["mean"] = 11.0
    results[22]["mean"] = 10.0
    results[23]["mean"] = 9.0
    results[24]["mean"] = 8.0

    recommendations = recommend_formations(players, results)

    assert recommendations[0].formation == "3-5-2"
    assert len(recommendations[0].starting) == 11
    assert recommendations[0].captain == 20
