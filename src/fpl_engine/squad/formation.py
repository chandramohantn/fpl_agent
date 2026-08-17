"""Select the highest expected-points valid FPL formation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

VALID_FORMATIONS = (
    (3, 4, 3),
    (3, 5, 2),
    (4, 3, 3),
    (4, 4, 2),
    (4, 5, 1),
    (5, 2, 3),
    (5, 3, 2),
    (5, 4, 1),
)


class SquadPlayer(Protocol):
    element: int
    position: str


@dataclass(frozen=True)
class FormationRecommendation:
    formation: str
    starting: tuple[int, ...]
    bench: tuple[int, ...]
    captain: int
    expected_points: float


def recommend_formations(
    players: list[SquadPlayer], simulation_results: dict[int, dict]
) -> list[FormationRecommendation]:
    """Rank every valid formation by the expected points of its best XI."""
    by_position: dict[str, list[tuple[int, float]]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for player in players:
        expected_points = float(simulation_results.get(player.element, {}).get("mean", 0.0))
        by_position.setdefault(player.position, []).append((player.element, expected_points))
    for position in by_position:
        by_position[position].sort(key=lambda item: item[1], reverse=True)

    recommendations = []
    for defenders, midfielders, forwards in VALID_FORMATIONS:
        required = {"GK": 1, "DEF": defenders, "MID": midfielders, "FWD": forwards}
        if any(len(by_position[position]) < count for position, count in required.items()):
            continue
        starting = tuple(
            element
            for position, count in required.items()
            for element, _ in by_position[position][:count]
        )
        expected_points = sum(
            simulation_results.get(element, {}).get("mean", 0.0) for element in starting
        )
        captain = max(
            starting, key=lambda element: simulation_results.get(element, {}).get("mean", 0.0)
        )
        bench = tuple(
            element
            for element, _ in sorted(
                (
                    (player.element, simulation_results.get(player.element, {}).get("mean", 0.0))
                    for player in players
                    if player.element not in starting
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        recommendations.append(
            FormationRecommendation(
                formation=f"{defenders}-{midfielders}-{forwards}",
                starting=starting,
                bench=bench,
                captain=captain,
                expected_points=float(expected_points),
            )
        )
    return sorted(
        recommendations, key=lambda recommendation: recommendation.expected_points, reverse=True
    )
