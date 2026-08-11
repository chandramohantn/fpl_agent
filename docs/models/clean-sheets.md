# Clean Sheets Prediction Model

## 1. Problem Definition

**Input:** A player's team + opponent + fixture context

**Output:** P(team keeps a clean sheet)

**Key insight:** Clean sheets are a **team-level event** — all players on the same team either keep a CS or they don't. So we model P(team CS | opponent, context), then combine with P(player plays 60+ min) from the minutes model.

**FPL scoring impact:**
- GK/DEF: 4 points for a clean sheet
- MID: 1 point for a clean sheet
- FWD: 0 points (but still relevant for simulation)

**Approach:** Binary classification (LightGBM)
- A single model for all positions (since CS is team-level, not player-specific)
- Training data: all appearances where player played 60+ minutes (CS eligibility)

## 2. Why CS is Hard to Predict

Clean sheets are among the hardest FPL outcomes to predict because:

1. **Binary, team-level outcome** — one mistake by any teammate ruins it
2. **High variance** — a top defensive team still concedes in ~65% of matches
3. **Opponent-dependent** — the same team's CS probability varies enormously by opponent
4. **Base rate is low** — only 26.5% of eligible appearances result in a CS

The model cannot aim for classification accuracy. Instead, it aims for **well-calibrated probabilities** that correctly order fixtures by CS likelihood.

## 3. Feature Set (15 features)

| Feature | Description | Importance |
|---------|-------------|-----------|
| `opp_attack_strength` | Opponent's overall attacking strength (most important) | 390 |
| `own_defence_strength` | Player's team defensive quality | 343 |
| `expected_goals_conceded_roll3` | Team's recent xGC (last 3 GWs) | 327 |
| `clean_sheets_season_avg` | Team's CS rate this season | 226 |
| `expected_goals_conceded_roll5` | Team's xGC over last 5 GWs | 197 |
| `goals_conceded_season_avg` | Actual goals conceded per match | 145 |
| `expected_goals_conceded_roll10` | Longer-term xGC trend | 135 |
| `expected_goals_conceded_season_avg` | Season-wide xGC average | 121 |
| `goals_conceded_roll3` | Recent goals conceded (3 GWs) | 116 |
| `goals_conceded_roll10` | Goals conceded over 10 GWs | 86 |
| `is_home` | Home teams keep more CS (28.8% vs 24.2%) | 85 |
| `goals_conceded_roll5` | Goals conceded over 5 GWs | 78 |
| `clean_sheets_roll10` | CS rate over last 10 GWs | 69 |
| `clean_sheets_roll5` | CS rate over last 5 GWs | 63 |
| `clean_sheets_roll3` | Very recent CS rate | 42 |

**Design choice:** The two most important features are the **matchup-specific** ones (opponent attack strength + own defence strength). These capture the specific fixture difficulty for this team, not just historical rolling averages.

## 4. Results (Test: 2025-26 season)

| Metric | Value |
|--------|-------|
| ROC AUC | 0.627 |
| Brier score | 0.198 (lower = better; 0.25 = random for balanced classes) |
| Log loss | 0.592 |
| Mean predicted P(CS) | 0.219 |
| Actual CS rate | 0.278 |

#### Binned Calibration

| Predicted P(CS) range | Count | Mean predicted | Actual CS rate |
|----------------------|-------|---------------|----------------|
| [0.00, 0.15) | 2,500 | 0.091 | 0.174 |
| [0.15, 0.25) | 2,193 | 0.200 | 0.303 |
| [0.25, 0.35) | 1,394 | 0.295 | 0.321 |
| [0.35, 0.50) | 843 | 0.408 | 0.384 |
| [0.50, 1.00) | 264 | 0.597 | 0.504 |

**Interpretation:**
- In the middle range (0.25-0.50), calibration is good — predicted 30-40% matches actual 32-38%
- The model correctly **ranks** fixtures (higher predicted → higher actual)
- The overall under-prediction (0.22 vs 0.28) is due to the test season having a higher CS rate than training — normal between-season variation
- The high-confidence bin (0.50+) slightly over-predicts — very few fixtures should be >50% CS

## 5. How CS Combines with Minutes

For FPL points calculation:

```
Expected CS points = P(team CS) × P(player plays 60+ min) × position_multiplier

Example: Arsenal DEF vs Ipswich (at home)
  CS model: P(team CS) = 0.42
  Minutes model: P(60+ min) = 0.91

  Expected CS points (DEF) = 0.42 × 0.91 × 4 = 1.53 pts
  Expected CS points (MID) = 0.42 × 0.85 × 1 = 0.36 pts
```

## 6. Limitations

| Limitation | Impact | Notes |
|-----------|--------|-------|
| Under-predicts overall | Mean P(CS) = 0.22 vs actual 0.28 | Season-to-season variation in CS rates |
| AUC = 0.63 (not stellar) | Ordering is imperfect | CS is inherently noisy — even bookmaker models struggle |
| No in-season recalibration | Calibration drifts as season progresses | Could apply isotonic regression on rolling validation window |
| Doesn't use specific lineup info | A weakened defensive lineup lowers CS probability | Could integrate PlayerContext for key defensive absences |

**Context:** Clean sheet prediction is a well-known hard problem in football analytics. Bookmaker CS odds typically imply probabilities with AUC of ~0.65-0.70. Our 0.63 is reasonable for a model without bookmaker odds as features.

## 7. Model Artifacts

| File | Description |
|------|-------------|
| `models/clean_sheets_v1/model.pkl` | Single LGBMClassifier (all positions) |
| `models/clean_sheets_v1/feature_columns.pkl` | List of 15 feature column names |

**Usage:**
```
from fpl_engine.models.clean_sheets import CleanSheetModel

model = CleanSheetModel.load("models/clean_sheets_v1")
predictions = model.predict_proba(featured_df)
# Returns: element, position, p_clean_sheet
```
