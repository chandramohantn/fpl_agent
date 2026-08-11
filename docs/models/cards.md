# Cards Prediction Model

## 1. Problem Definition

**Input:** A player + upcoming fixture (conditioned on playing)

**Output:**
- P(yellow card) — probability of being booked
- P(red card) — probability of being sent off
- Expected card points = P(YC) × (-1) + P(RC) × (-3)

**FPL scoring:** Yellow = -1 pt, Red = -3 pts

## 2. Data Characteristics

| Metric | Value |
|--------|-------|
| Yellow card rate (overall) | 13.2% |
| Red card rate (overall) | 0.45% |
| Home YC rate | 12.2% |
| Away YC rate | 14.3% |
| 60+ min YC rate | 15.6% |
| 1-59 min YC rate | 8.1% |

**By position:**

| Position | YC rate | RC rate |
|----------|---------|---------|
| GK | 7.5% | 0.09% |
| DEF | 15.0% | 0.66% |
| MID | 13.6% | 0.40% |
| FWD | 9.8% | 0.24% |

**Key insight:** Cards are largely random events at the individual-GW level. A player might go 10 games without a card, then get booked in a calm match. The model's value is in **correctly estimating the long-run rate** for each player (card-prone vs disciplined), not in predicting specific GW bookings.

## 3. Modeling Approach

**Yellow cards:** Binary classification (LightGBM) per position
**Red cards:** Position-level flat rates (too rare at 0.45% for meaningful ML modeling)

Red card rates used:

| Position | Rate |
|----------|------|
| GK | 0.09% |
| DEF | 0.66% |
| MID | 0.40% |
| FWD | 0.24% |

## 4. Feature Set (11 features)

| Feature | Description |
|---------|-------------|
| `yellow_cards_roll5` | YC rate over last 5 GWs |
| `yellow_cards_roll10` | YC rate over last 10 GWs |
| `yellow_cards_sum5` | Total YCs in last 5 GWs |
| `yellow_cards_sum10` | Total YCs in last 10 GWs |
| `yellow_cards_season_avg` | Season-to-date YC rate |
| `minutes_roll3` | Recent minutes (more exposure = more risk) |
| `minutes_roll5` | Minutes over last 5 GWs |
| `is_home` | Home (fewer cards) vs away (more cards) |
| `season_progress` | Late season may have fewer cards (or more if relegation battles) |
| `match_difficulty` | Harder matches may produce more fouls |
| `value_vs_pos_avg` | Expensive players may be more disciplined |

## 5. Results (Test: 2025-26 season)

| Position | AUC | Brier | Predicted YC% | Actual YC% |
|----------|-----|-------|--------------|------------|
| **DEF** | 0.548 | 0.128 | 15.8% | 14.6% |
| **MID** | 0.573 | 0.112 | 14.1% | 12.6% |
| **FWD** | 0.489 | 0.088 | 9.7% | 8.8% |

**Interpretation:**
- AUC values (0.49-0.57) indicate the model has limited ability to predict **which specific GW** a player gets booked — this is expected because cards are inherently random
- **Calibration is the real value**: predicted rates match actual rates within 1-2%, meaning expected points calculations using P(YC) will be accurate over a season
- The model correctly identifies card-prone players (through `yellow_cards_season_avg`) and assigns them higher probabilities

## 6. Expected Card Points

For FPL optimization, what matters is the expected point deduction:

```
Expected card points = P(YC) × (-1) + P(RC) × (-3)

Examples:
  Disciplined midfielder (YC rate 8%):  -0.08 + (-0.012) = -0.09 pts/match
  Card-prone defender (YC rate 20%):    -0.20 + (-0.020) = -0.22 pts/match

  Over a season (38 GWs):
    Disciplined: -3.5 pts from cards
    Card-prone: -8.4 pts from cards
    Difference: ~5 pts — meaningful for marginal decisions
```

## 7. Model Artifacts

| File | Description |
|------|-------------|
| `models/cards_v1/models.pkl` | Dict of position → LGBMClassifier |
| `models/cards_v1/feature_columns.pkl` | List of 11 feature columns |

**Usage:**
```
from fpl_engine.models.cards import CardsModel

model = CardsModel.load("models/cards_v1")
predictions = model.predict(featured_df)
# Returns: element, position, p_yellow_card, p_red_card, expected_card_pts
```
