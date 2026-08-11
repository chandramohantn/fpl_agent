# Saves Prediction Model

## 1. Problem Definition

**Input:** A goalkeeper + upcoming fixture (conditioned on playing 60+ min)

**Output:** Expected saves (λ) and FPL save points distribution:
- P(0 save points) = P(0-2 saves)
- P(1 save point) = P(3-5 saves)
- P(2 save points) = P(6-8 saves)
- P(3+ save points) = P(9+ saves)

**Scope:** GK-only model. Outfield players make 0 saves in FPL.

**FPL scoring:** Every 3 saves = 1 point. A busy goalkeeper against a strong attack can earn 1-3 extra points from saves alone.

## 2. Data Characteristics

| Metric | Value |
|--------|-------|
| Total GK appearances (60+ min) | 2,262 |
| Mean saves per appearance | 3.04 |
| Standard deviation | 1.96 |
| Variance/Mean ratio | 1.27 (mildly overdispersed) |
| Home GK mean saves | 2.79 |
| Away GK mean saves | 3.29 |
| xGC correlation with saves | 0.475 |

**Key insight:** Away goalkeepers face more shots and make more saves. Additionally, saves are strongly correlated with expected goals conceded (xGC) — facing a stronger attack means more shots to save.

## 3. Feature Set (12 features)

| Feature | Importance | Description |
|---------|-----------|-------------|
| `opp_attack_strength` | 270 | Opponent's overall attacking quality (strongest predictor) |
| `saves_season_avg` | 253 | GK's season-to-date saves per match |
| `expected_goals_conceded_season_avg` | 198 | Team's average xGC (shots faced proxy) |
| `saves_roll5` | 196 | Average saves over last 5 GWs |
| `expected_goals_conceded_roll3` | 188 | Recent xGC (last 3 GWs) |
| `expected_goals_conceded_roll5` | 170 | xGC over last 5 GWs |
| `expected_goals_conceded_roll10` | 144 | Longer-term xGC |
| `saves_roll10` | 140 | Saves over last 10 GWs |
| `is_home` | 93 | Home (fewer saves) vs away (more saves) |
| `saves_roll3` | 82 | Very recent saves form |
| `goals_conceded_roll5` | 75 | Actual goals conceded recently |
| `goals_conceded_roll3` | 56 | Very recent goals conceded |

## 4. Results (Test: 2025-26 season)

| Metric | Value |
|--------|-------|
| Predicted λ (mean) | 2.96 |
| Actual saves (mean) | 2.82 |
| MAE | 1.42 |
| RMSE | 1.79 |

#### FPL Save Points Calibration

| Save points | Saves range | Predicted | Actual |
|-------------|------------|-----------|--------|
| 0 pts | 0-2 saves | 44.7% | 46.6% |
| 1 pt | 3-5 saves | 46.0% | 47.1% |
| 2 pts | 6-8 saves | 8.6% | 5.7% |
| 3+ pts | 9+ saves | 0.7% | 0.6% |

| | Predicted | Actual |
|--|-----------|--------|
| **Expected save points per match** | 0.65 | 0.60 |

Calibration is good in the core range (0-1 pts covers 91% of outcomes). Slight over-prediction for high-save matches (2 pts), which is expected given the mild overdispersion.

## 5. How Saves Combine with Other Models

```
Expected GK save pts = P(GK plays 60+ min) × E[save points | plays]

Example: Raya (Arsenal) vs Man City (away)
  Minutes model: P(60+ min) = 0.95
  Saves model: λ = 4.2 (facing strong attack, away)
    P(0-2) = 0.21, P(3-5) = 0.55, P(6-8) = 0.21, P(9+) = 0.03
    E[save pts] = 0.55×1 + 0.21×2 + 0.03×3 = 1.06

  Combined: 0.95 × 1.06 = 1.01 expected save points
```

## 6. Model Artifacts

| File | Description |
|------|-------------|
| `models/saves_v1/model.pkl` | LGBMRegressor (Poisson objective) |
| `models/saves_v1/feature_columns.pkl` | List of 12 feature column names |

**Usage:**
```
from fpl_engine.models.saves import SavesModel

model = SavesModel.load("models/saves_v1")
predictions = model.predict(featured_gk_df)
# Returns: element, lambda_saves, expected_save_pts, p_0pts, p_1pt, p_2pts, p_3plus_pts
```
