# FPL Engine — Frontend Guide

## 1. Setup and Installation

### Prerequisites

- Python 3.12+
- The FPL Engine project fully installed (all phases)
- Data pipeline already run at least once

### Installation Steps

```bash
# 1. Navigate to the project
cd /path/to/fpl

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Ensure all dependencies are installed
pip install -e ".[dev]"

# 4. Verify data is loaded (if not, run the pipeline)
python scripts/run_pipeline.py

# 5. Start the frontend
streamlit run app/main.py
```

The app opens in your browser at `http://localhost:8501`.

### First-Time Setup Checklist

| Step | Command | Verify |
|------|---------|--------|
| Install project | `pip install -e ".[dev]"` | `python -c "import fpl_engine"` works |
| Load data | `python scripts/run_pipeline.py` | `data/processed/` has parquet files |
| Train models | See Model Management page | `models/` directory has `.pkl` files |
| Start frontend | `streamlit run app/main.py` | Browser opens at localhost:8501 |

---

## 2. App Structure

The frontend has 4 pages accessible from the left sidebar:

```
┌──────────────────┐    ┌────────────────────────────────────────┐
│                  │    │                                        │
│  ⚽ FPL Engine    │    │   Page content area                    │
│                  │    │                                        │
│  ─────────────── │    │   (changes based on selected page)     │
│                  │    │                                        │
│  🏠 Dashboard    │◄───│                                        │
│  📝 Manual Inputs│    │                                        │
│  📋 Planning     │    │                                        │
│  🔧 Model Mgmt  │    │                                        │
│                  │    │                                        │
│  ─────────────── │    │                                        │
│  Quick Actions   │    │                                        │
│                  │    │                                        │
└──────────────────┘    └────────────────────────────────────────┘
     Sidebar                       Main Area
```

---

## 3. Page 1: 🏠 Dashboard

The Dashboard is your primary view for understanding predictions and making gameweek decisions.

### Tab: Squad Overview

**Purpose:** See your current squad and each player's predicted probabilities.

**How to use:**
1. Click "Load Sample Squad" on first visit (or the system loads your saved squad)
2. View the table showing all 11+ players

**What you see:**

| Column | Meaning |
|--------|---------|
| Player | Player name |
| Pos | GK, DEF, MID, FWD |
| Team | Player's team |
| vs | Opponent this gameweek |
| P(play) | Probability of featuring (1 - P(no play)) |
| λ Goals | Expected goals (Poisson rate) |
| λ Assists | Expected assists (Poisson rate) |
| P(CS) | Probability of team clean sheet |

**When to check:** Every gameweek before making decisions.

---

### Tab: Simulations

**Purpose:** Run Monte Carlo simulations to get full point distributions.

**How to use:**
1. Set the number of simulations (default: 10,000 — increase for more precision, decrease for speed)
2. Click "Run Simulations"
3. Wait for the progress bar to complete

**What you see after running:**

| Column | Meaning | How to use it |
|--------|---------|--------------|
| xPts | Average expected points | Primary ranking metric |
| Std | Standard deviation | High std = unpredictable player |
| P10 | 10th percentile (worst case) | "Floor" — bad GW still gives this |
| Median | 50th percentile | Most likely single outcome |
| P90 | 90th percentile (best case) | "Ceiling" — great GW gives this |
| P(blank) | Probability of ≤2 points | Higher = more risk of nothing |
| P(return) | Probability of ≥5 points | Higher = more likely to return |
| P(haul) | Probability of ≥10 points | Higher = explosive upside |

**Interpreting results:**

- **Safe pick:** Low P(blank), moderate xPts, low Std (e.g., nailed DEF with CS potential)
- **High-upside pick:** High P(haul), high Std (e.g., premium FWD with easy fixture)
- **Avoid:** High P(blank) AND low xPts (e.g., rotation-risk player with hard fixture)

**Example output:**

```
Player     xPts   Std    P10  Median  P90   P(blank)  P(return)  P(haul)
Haaland    6.91   4.47   2.0   6.0    13.0   24%       71%        27%
Salah      4.93   3.81   1.0   4.0    10.0   40%       48%        13%
Gabriel    3.76   2.73   1.0   2.0     6.0   51%       44%         2%
Welbeck    1.16   1.48   0.0   1.0     2.0   91%        4%         0%
```

---

### Tab: Captain Comparison

**Purpose:** Compare captain options using simulation distributions (doubled points).

**Requires:** Simulations must be run first.

**What you see:**

| Column | Meaning |
|--------|---------|
| E[2×pts] | Expected doubled points (primary ranking) |
| Std | Volatility of doubled outcome |
| P(haul≥20) | Chance of a massive captain haul |
| P(blank≤4) | Chance captain scores ≤2 (doubled ≤4) |
| P90 | 90th percentile of doubled points |

**How to decide:**

- **Safe strategy (protecting rank):** Pick highest E[2×pts] with lowest P(blank≤4)
- **Aggressive strategy (chasing):** Pick highest P(haul≥20) even if mean is slightly lower
- **Differential:** Pick a player with high ceiling but low ownership (not shown in basic view)

The system highlights the recommended captain with a green success box.

---

## 4. Page 2: 📝 Manual Inputs

The Manual Inputs page is where you provide information that gives the system an edge — things APIs can't tell it.

### Tab: Individual Player

**Purpose:** Set external context for a specific player.

**How to use:**

1. Select a player from the dropdown
2. Fill in whichever fields you have information for
3. Click "💾 Save Override"

**Available fields:**

#### Availability / Injury (left column)

| Field | What to enter | Example |
|-------|---------------|---------|
| Status | Player's availability | "doubtful" if press conference says "we'll assess" |
| Chance of playing (%) | Your estimate 0-100 | 50% if "touch and go" |
| Returning from injury | Check if just back from absence | ✓ if first week back |
| Weeks out | How long they were injured | 6 weeks for hamstring |
| Fitness level | Your estimate 0.0-1.0 | 0.6 if "building fitness" |

#### External Match Context (right column)

| Field | What to enter | Example |
|-------|---------------|---------|
| Days since last match | Last competitive game (any comp) | 3 if played CL on Tuesday |
| Minutes played in last match | How much they played | 90 if played full CL game |
| Important match in how many days? | Next big game | 4 if CL quarter-final on Wednesday |
| Match type | Description | "Champions League QF 2nd Leg" |

**When to use each field:**

| Scenario | Fields to set |
|----------|--------------|
| Player played midweek CL | days_since=3, minutes_last=90 |
| Big CL game coming up | important_match_in_days=3, match_type="CL SF" |
| Manager said "he'll be assessed" | status=doubtful, chance=50% |
| Just returned from 6-week injury | returning=✓, weeks_out=6, fitness=0.6 |
| Suspended for next match | status=suspended, chance=0% |
| Nothing unusual | Don't set anything (system uses historical patterns) |

**Impact on predictions:** After saving overrides, re-run simulations on the Dashboard. The predictions will adjust — e.g., a player who played 90 mins in CL 3 days ago will have lower P(full match) due to fatigue.

### Tab: View Overrides

**Purpose:** See all active overrides and remove ones that are no longer relevant.

Shows a card for each overridden player with all set values. Click "🗑️ Remove" to delete an override (e.g., after the relevant gameweek passes).

---

## 5. Page 3: 📋 Planning

The Planning page handles multi-week strategy.

### Tab: Transfer Advice

**Purpose:** Find the best transfer to make this gameweek.

**Inputs you provide:**
- Free transfers available (1 or 2)
- Bank balance (£m)
- Evaluation horizon (how many GWs to weight the new player)

**What the system does:**
1. For each player in your squad, finds all valid replacements (same position, within budget, club limit)
2. Computes: gain = E[new player × horizon] - E[current player × horizon]
3. Subtracts hit cost if applicable
4. Ranks by net value

**Output:**

| Column | Meaning |
|--------|---------|
| Sell | Who to sell |
| Buy | Who to buy |
| Gain (pts) | Expected point improvement |
| Cost | "Free" or "-4 hit" |
| Net value | Gain minus cost |
| Verdict | ✅ Do it / ❌ Not worth it |

**Decision framework:**
- Net value > 3: Strong recommendation (✅)
- Net value 1-3: Marginal, consider other factors
- Net value < 1: Not worth it, especially if it's a hit

---

### Tab: Chip Strategy

**Purpose:** Determine when to play each chip for maximum value.

**Inputs you provide:**
- Which chips you still have (check/uncheck)
- Current gameweek
- Look-ahead horizon

**What the system evaluates:**

For each available chip, the system checks every remaining GW:
- **Bench Boost:** How much would your bench score if they all played? Highest on DGWs.
- **Triple Captain:** How much extra does your best player gain from 3× vs 2×? Highest on easy fixtures.
- **Wildcard:** How much better could an entirely new squad be? Highest around fixture swings.

**Output:** A card per chip with recommended GW and expected gain.

---

### Tab: Multi-GW Plan (MCTS)

**Purpose:** Generate a strategic plan for the next 3-8 gameweeks.

**Inputs you provide:**
- Planning horizon (how many GWs ahead)
- Search iterations (more = better plan but slower)

**What the system does:**
1. Explores thousands of possible action sequences using Monte Carlo Tree Search
2. For each sequence: "what if I roll this week, transfer next week, play BB week after?"
3. Evaluates each path by simulating expected points
4. Returns the best sequence found

**Output:**

| Column | Meaning |
|--------|---------|
| GW | Which gameweek |
| Action | What to do (Roll / Transfer / Chip) |
| xPts | Expected points that GW |
| Reasoning | Why this action is best |

**How to interpret:**
- "Roll transfer" = don't transfer this week, bank for next week
- "Transfer: A → B" = make this specific transfer
- "Bench Boost" / "Triple Captain" = play chip this week
- Look at the reasoning column to understand why

**Timing guidance:**
- 100 iterations × 3 GWs: ~0.2 seconds (quick check)
- 1000 iterations × 5 GWs: ~3 seconds (recommended)
- 5000 iterations × 8 GWs: ~30 seconds (full strategic planning)

---

## 6. Page 4: 🔧 Model Management

### Tab: Model Status

**Purpose:** See which models are trained and their performance metrics.

**What you see:** A card for each of the 7 models:
- ✅ / ❌ indicating if the model file exists
- Model type (classification, Poisson, regression)
- Performance metric (accuracy, AUC, calibration)
- File size and artifact names

**When to check:** After retraining, or if predictions seem off.

---

### Tab: Retrain

**Purpose:** Retrain prediction models with updated data.

**When to retrain:**
- After loading a new season's data
- Mid-season (after 10+ GWs of new data has accumulated)
- If model calibration drifts (predictions consistently over/under)

**Inputs you provide:**

| Input | Description | Recommendation |
|-------|-------------|----------------|
| Training seasons | Which seasons to train on | All complete seasons |
| Test season | Which season to evaluate on | Most recent complete season |
| Models to retrain | Which models to update | All of them unless testing specific model |
| Trees (n_estimators) | How many trees in the ensemble | 500 (default, increase for marginal improvement) |
| Max depth | How deep each tree can grow | 6 (default, lower to reduce overfitting) |

**Process:**
1. Select training parameters
2. Click "🚀 Start Training"
3. Watch the progress bar
4. Review results table showing per-model performance

**After retraining:** The Dashboard will automatically use the new models next time you run simulations (models are loaded from the `models/` directory).

---

### Tab: Data Status

**Purpose:** See what data is loaded and available for training/prediction.

**What you see:**
- Which seasons have player data, gameweek data, fixtures, teams
- Which seasons have Understat xG data
- Which seasons have FBref advanced stats
- Total processed data size

**Actions available:**
- "🔄 Refresh Data" — updates current season (runs `scripts/refresh.py`)
- "📥 Full Pipeline" — re-runs full ingestion (runs `scripts/run_pipeline.py`)

---

## 7. How Models Reload When Updated

### Automatic Reload Behavior

Streamlit re-executes the page code on every interaction. When you retrain a model:

1. Retraining saves new `.pkl` files to `models/minutes_v2/`, `models/goals_v1/`, etc.
2. Next time you run simulations on the Dashboard, the predictions use whatever models are currently in `models/`
3. No restart needed — Streamlit picks up the new files on next simulation run

### Manual Reload

If you retrain models outside the app (e.g., via command line), the app will use the new models the next time it loads them. You can force a reload by:
- Clearing the browser cache (Streamlit's "Clear Cache" in the hamburger menu)
- Or simply re-running simulations (they always load fresh from disk)

### Model Versioning

Models are saved with version suffixes (`minutes_v2`, `goals_v1`). To compare versions:
1. Retrain with new parameters → saves as the same path (overwrites)
2. To keep old version, manually rename the folder before retraining
3. The system always uses whatever is in the standard path

---

## 8. Providing Inputs — Complete Reference

### What the system fetches automatically (no action needed):

| Data | Source | Frequency |
|------|--------|-----------|
| Player stats (goals, assists, xG, minutes) | FPL API | After each GW (via refresh.py) |
| Fixture difficulty | FPL API + computed | After each GW |
| Team xG/xGA | Understat | After each GW |
| Player prices and ownership | FPL API | Daily |
| Injury status | FPL API | Every refresh |
| Advanced defensive stats | FBref | Manually triggered |

### What you provide (gives the system an edge):

| Input | Where to enter | Impact |
|-------|---------------|--------|
| Midweek match info | Manual Inputs → days_since, minutes_last | Rotation/fatigue prediction |
| Big upcoming match | Manual Inputs → important_match_in_days | Rest prediction |
| Fitness assessment | Manual Inputs → fitness_level | Managed minutes prediction |
| Press conference quotes | Manual Inputs → status, chance_of_playing | Availability prediction |
| Your risk preference | Implicit in captain/transfer decisions | Future: explicit preference setting |

### When NOT to provide manual inputs:

- If you don't have specific information, leave fields at defaults
- The model already uses historical patterns (rolling averages, availability rates)
- Only override when you have information the API doesn't yet reflect

---

## 9. Viewing Outputs — What Each Number Means

### Expected Points (xPts)

**What it is:** The average across 10,000 simulated gameweek outcomes.

**What it means:** "If this player played this exact fixture 10,000 times, they'd average this many points."

**What it does NOT mean:** "This player will score exactly this many points this GW."

### P(blank) — Blank Probability

**Definition:** P(total_points ≤ 2)

**What causes a blank:**
- Player doesn't start (0 pts)
- Player plays but does nothing (2 pts for appearance only)

**Use case:** Avoiding players who are likely to score ≤2. High P(blank) = risky pick.

### P(return) — Return Probability

**Definition:** P(total_points ≥ 5)

**What produces a return:**
- Goal + appearance = 6-8 pts
- Assist + CS = 7 pts
- Multiple events in one game

**Use case:** Identifying players likely to produce an attacking or defensive return.

### P(haul) — Haul Probability

**Definition:** P(total_points ≥ 10)

**What produces a haul:**
- 2+ goals, or goal + assist + bonus
- Clean sheet + goal + bonus (defenders)

**Use case:** Captain selection. The player with the highest P(haul) gives the best chance of a captaincy jackpot.

### Lambda (λ) values

**What it is:** The Poisson rate parameter for goals or assists.

**How to read it:**
- λ = 0.1 → ~10% chance of scoring
- λ = 0.5 → ~39% chance of scoring (1 - e^{-0.5})
- λ = 1.0 → ~63% chance of scoring
- λ = 1.5 → ~78% chance of scoring

---

## 10. Typical Session Workflow

### Quick check (5 minutes, Friday before deadline)

1. Open app → Dashboard → Simulations tab
2. Click "Run Simulations" (10K)
3. Check Captain Comparison → Follow recommendation
4. Confirm starting XI looks right
5. Done

### Full planning session (20 minutes, Thursday)

1. **Manual Inputs:** Enter any midweek match info for your players
2. **Dashboard → Simulations:** Run sims with updated context
3. **Planning → Transfers:** Check if any transfer offers net positive value
4. **Planning → Multi-GW Plan:** Generate 5-GW plan if major decisions pending
5. **Planning → Chips:** Check if this is a good week for any chip
6. **Dashboard → Captain:** Final captain decision

### After new gameweek data arrives (Monday/Tuesday)

1. Terminal: `python scripts/refresh.py`
2. **Model Management → Data Status:** Verify data updated
3. If mid-season (10+ new GWs since last retrain):
   - **Model Management → Retrain:** Retrain with latest data included

---

## 11. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| "No squad loaded" | First visit, no data | Click "Load Sample Squad" or run pipeline |
| Simulation takes too long | Too many simulations | Reduce to 5,000 (still accurate) |
| "Model not trained" in Model Status | Models not yet trained | Go to Retrain tab and train |
| Predictions seem wrong | Stale data or stale models | Run refresh.py, then retrain |
| App won't start | Missing dependencies | `pip install -e ".[dev]"` and `pip install streamlit` |
| Player not in dropdown | Squad not loaded | Load squad from Dashboard first |

---

## 12. Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | 8501 | Port for the web app |
| `SOCCERDATA_DIR` | `data/raw/fbref` | Where FBref data is cached |

### Custom Settings

Edit `pyproject.toml` to adjust:
- Line length and linting rules
- Test configuration
- Dependency versions

### Data Directories

| Path | Contents |
|------|----------|
| `data/raw/historical/` | Cached CSVs from GitHub |
| `data/raw/understat/` | Cached JSON from Understat |
| `data/raw/fbref/` | Cached Parquet from FBref |
| `data/processed/` | Clean Parquet data lake |
| `models/` | Trained model artifacts |

---

## 13. Starting the App

```bash
# Standard start
streamlit run app/main.py

# With custom port
streamlit run app/main.py --server.port 8080

# In headless mode (no auto-open browser)
streamlit run app/main.py --server.headless true
```

The app is fully local — no data leaves your machine. All computation happens on your CPU.
