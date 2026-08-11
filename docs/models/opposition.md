# Opposition Detail Features

Location: `src/fpl_engine/features/opposition_features.py`

## 1. Purpose

The original models used a single `opp_attack_strength` or `match_difficulty` number to represent the opposition. This module provides **granular opposition profiling** from existing data sources (Understat, FBref, GW data).

A team might have:
- A leaky defence but an excellent goalkeeper (hard to score despite many chances)
- A high-pressing style that creates turnovers but concedes on the counter
- A settled back 4 that's much harder to break down than their xGA suggests

---

## 2. Components

### A. Opponent Defensive Profile (from Understat team data)

| Feature | Description | Impact on |
|---------|-------------|-----------|
| `opp_xga_per_match` | Opponent's expected goals against per match | Goals, Assists |
| `opp_npxga_per_match` | Non-penalty xGA | Goals |
| `opp_goals_conceded_pm` | Actual goals conceded per match | Goals, Assists |
| `opp_deep_allowed_pm` | Deep completions allowed (passes into final third) | Assists |
| `opp_ppda` | Passes Per Defensive Action (lower = more pressing) | Assists, Goals |
| `opp_shot_quality` | xGA per deep completion | Goals |

**Example: Arsenal (best defence) vs Burnley (worst defence)**

| Metric | Arsenal | Burnley |
|--------|---------|---------|
| xGA/match | 0.87 | 1.94 |
| Deep allowed | 3.7 | 7.3 |
| PPDA | 9.6 | 13.2 |

### B. Opponent GK Save Rate

| Feature | Description | Impact on |
|---------|-------------|-----------|
| `opp_gk_save_rate` | Opponent GK's saves / shots on target faced (0.60-0.72) | Goals |
| `opp_gk_saves_per_match` | How many saves the opposing GK makes per match | Goals |

### C. Penalty / Set Piece Taker Flags

| Feature | Description | Impact on |
|---------|-------------|-----------|
| `is_penalty_taker` | Primary penalty taker for their team | Goals (+0.76 xG per pen) |
| `penalty_goals_season` | Number of penalty goals scored this season | Goals |

### D. Defensive Partnership Stability

| Feature | Description | Impact on |
|---------|-------------|-----------|
| `def_continuity` | % of recent GWs where exact same DEF line started | CS (own team) |
| `def_partnership_score` | Average Jaccard similarity between consecutive GW DEF lineups | CS |
| `opp_def_continuity` | Opponent's defensive continuity | Goals, Assists |
| `opp_def_partnership_score` | Opponent's partnership score | Goals, Assists |

---

## 3. How These Enhance Each Model

| Model | New features used | Expected improvement |
|-------|-------------------|---------------------|
| **Goals** | opp_xga, opp_npxga, opp_gk_save_rate, opp_deep_allowed, is_penalty_taker, opp_def_partnership_score | Better fixture differentiation; penalty taker identification |
| **Assists** | opp_xga, opp_deep_allowed, opp_ppda, opp_def_partnership_score | Better fixture difficulty; pressing teams concede more assists |
| **Clean Sheets** | def_continuity, def_partnership_score, opp_ppda | Settled defences keep more CS; pressing opponents create less |

---

## 4. Data Sources

| Feature group | Source | Availability |
|--------------|--------|-------------|
| Opponent defensive profile | Understat team match-by-match data | ✅ Cached in `data/raw/understat/` |
| Opponent GK save rate | Derived from GW data (saves + goals_conceded) | ✅ In ParquetStore |
| Penalty takers | Understat (goals - npg = penalty goals) | ✅ Cached |
| Defensive stability | DEF appearances in GW data | ✅ In ParquetStore |
| Shots allowed per 90 | FBref opponent shooting stats | ✅ `data/raw/fbref/` |
| GK save % (detailed) | FBref keeper stats | ✅ `data/raw/fbref/` |

---

## 5. What Still Needs Implementation (FBref-based)

These features are **available in our FBref data** but not yet wired into the models:

| Signal | FBref column | Description |
|--------|-------------|-------------|
| Shots on target allowed/90 | `opponent_shooting.Standard_SoT` / 90s | Direct measure of defensive shot prevention |
| Shot conversion rate against | `opponent_shooting.Standard_G/SoT` | How clinical opponents are vs this team |
| Penalties conceded | `opponent_shooting.Standard_PKatt` | Set piece opportunity frequency |
| GK save percentage | `team_keeper.Performance_Save%` | More precise than our derived rate |
| CS percentage | `team_keeper.Performance_CS%` | Direct clean sheet frequency |

---

## 6. Usage

```python
from fpl_engine.features.opposition_features import (
    build_opponent_defensive_profile,
    compute_team_gk_save_rates,
    identify_penalty_takers,
    compute_defensive_stability,
    add_opponent_defensive_detail,
    add_opponent_gk_save_rate,
    add_set_piece_flags,
    add_defensive_stability,
    add_opponent_defensive_stability,
)

# Build all profiles for a season
profile = build_opponent_defensive_profile("data/raw/understat", "2025-26")
gk_rates = compute_team_gk_save_rates(store, "2025-26")
pen_takers = identify_penalty_takers(store, "2025-26", "data/raw/understat")
stability = compute_defensive_stability(store, "2025-26")

# Apply to player-GW data
df = add_opponent_defensive_detail(df, profile, team_name_map)
df = add_opponent_gk_save_rate(df, gk_rates, team_name_map)
df = add_set_piece_flags(df, pen_takers)
df = add_defensive_stability(df, stability)          # Own team CS
df = add_opponent_defensive_stability(df, stability)  # Opponent for goals/assists
```
