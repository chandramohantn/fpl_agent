# FPL Decision System — User Guide

## How to Use This Tool to Play FPL Every Week

This guide explains how the system works from a user's perspective. It covers what you need to provide, what the system does with it, and what advice you get back — week by week throughout the season.

---

## 1. The Big Picture

Think of this system as having four layers that work in sequence:

```
YOU provide:              The SYSTEM does:           YOU get back:
─────────────────         ────────────────────       ──────────────────
Your squad                Predict each player's      "Captain Haaland"
Current gameweek          probability of scoring     "Transfer Salah → Son"
Injury news               goals, assists, CS, etc.   "Play Bench Boost GW22"
External match info       ↓                          "Roll your transfer"
                          Simulate 10,000 futures    
                          for every player           Full reasoning for
                          ↓                          every recommendation
                          Find the optimal           
                          decisions mathematically   Confidence levels
                          ↓                          (high/medium/low)
                          Plan 5 weeks ahead         
                          ↓                          Risk assessment
                          Explain everything         (P(haul), P(blank))
                          in plain English           
```

---

## 2. What Inputs You Need to Provide

### Automatic (fetched by the system)

These are pulled from APIs — you don't need to do anything:

| Input | Source | Fetched by |
|-------|--------|-----------|
| All player stats (goals, assists, xG, form, price) | FPL API | `LiveSeasonRefresher` |
| Fixture list and difficulty | FPL API | `LiveSeasonRefresher` |
| Player status (available/injured/doubtful) | FPL API | `ChangeDetector` |
| Team xG/xGA data | Understat | `UnderstatScraper` |
| Opponent shooting/defensive stats | FBref | `FBrefScraper` |
| Historical performance (rolling averages) | ParquetStore | Computed from stored data |

### Manual (you provide when relevant)

These are things the system can't get from APIs — your knowledge gives it an edge:

| Input | When to provide | Example | How it helps |
|-------|----------------|---------|-------------|
| Midweek match info | Player played Champions League | "Salah played 90 mins 3 days ago" | Higher rotation risk |
| Upcoming important match | Big game in next few days | "Liverpool play CL semi-final in 4 days" | May be rested in PL |
| Returning from injury | Player just came back | "Saka returning after 6 weeks out" | Likely managed minutes |
| Fitness assessment | Your judgment on fitness | "Kane looks 60% fit from press conference" | Model reduces minutes probability |
| Press conference hints | Manager quotes | "Pep said 'we will rotate'" | Increases rotation probability |

You provide this via `PlayerContext`:
```python
contexts[salah_id] = contexts[salah_id].override(
    days_since_last_match=3,
    played_minutes_last_match=90,
    important_match_in_days=4,
    important_match_type="Champions League Semi-Final",
)
```

---

## 3. Week-by-Week Workflow

Here's exactly what happens each gameweek, day by day:

### Monday / Tuesday (After GW ends)

**What you do:** Run the data refresh.

```python
python scripts/refresh.py
```

**What the system does:**
1. Fetches completed GW results from FPL API
2. Updates all player stats (goals, assists, points, price changes)
3. Updates fixture results
4. Refreshes Understat xG data

**What you get:**
- Updated data store with latest GW performance
- Any price changes since last check

---

### Wednesday / Thursday (Mid-week planning)

**What you do:** Check for changes and plan ahead.

```python
response = agent.check_for_changes(api_data, deadline=next_deadline)
```

**What the system does:**
1. Compares current player statuses against last snapshot
2. Detects injuries, suspensions, new signings
3. Evaluates if your plan needs updating

**What you get:**
```
⚠️ 2 events affecting YOUR squad:
  • [high] Salah: status a→i. Hamstring - expected back GW15
  • [medium] Palmer: minor knock, 75% chance of playing
  
🔄 Recommendation: Re-run optimization with updated data
```

**If changes are significant, run the planner:**

```python
plan = planner.search(state, horizon=5, iterations=2000)
agent.set_plan(plan)
```

**What you get:**
```
Plan for next 5 gameweeks:
  GW12: Transfer Salah → Son (Salah injured, Son has great fixtures)
  GW13: Roll transfer (bank for GW14)
  GW14: Double transfer: Welbeck → Isak + Rogers → Mbeumo
  GW15: Roll transfer
  GW16: Play Bench Boost (DGW, strong bench)

Total expected: 312 pts over 5 GWs
```

---

### Friday / Saturday (Final decisions)

**What you do:** Ask for specific GW recommendations.

```python
response = agent.ask("Who should I captain?")
response = agent.ask("Should I make the transfer now?")
response = agent.ask("What's my best starting XI?")
```

**What the system does for captain:**
1. Runs 10,000 simulations for each starting XI player
2. Doubles each player's points in each simulation
3. Compares: who gives the highest expected doubled points?
4. Also shows: haul probability, blank risk, upside ceiling

**What you get:**
```
**Captain Haaland**

• Highest expected doubled points: 15.6 pts
• 28% chance of 20+ point haul
• Only 12% chance of blanking (≤4 doubled pts)
• Facing Burnley at home — weakest defence in the league

Alternatives considered:
  - Son: 12.2 pts (3.4 pts behind)
    ↳ Son has higher ceiling (P90=26 vs 24) — better as a differential
  - Palmer: 11.8 pts

Confidence: high
```

**What the system does for starting XI:**
1. Takes your 15-man squad
2. Solves an optimization problem: which 11 maximize expected points in a valid formation?
3. Orders the bench by auto-sub priority

**What you get:**
```
Formation: 4-4-2
Captain: Haaland (C), Vice: Son (V)

Starting XI:
  GK: Raya
  DEF: Gabriel, Saliba, TAA, Estupinan
  MID: Son, Palmer, Saka, Mbeumo
  FWD: Haaland, Watkins

Bench: Henderson, Welbeck, Dalot, Rogers
Expected points: 62.3 (with captain doubled)
```

---

### Saturday (Deadline approaching)

**What the system does automatically:**
```
🚨 Deadline in 2 hours! Finalize your transfers and captain.

Current recommendation:
  • Transfer: Salah → Son ✓ (confirmed)
  • Captain: Haaland
  • Formation: 4-4-2
  
No further changes needed. You're set.
```

---

## 4. What Outputs You Get (Detailed)

### 4.1 Per-Player Predictions

For every player, the system produces:

| Output | Meaning | Example (Haaland vs Burnley, home) |
|--------|---------|-------------------------------------|
| **Expected points** | Average across 10K simulations | 7.5 pts |
| **Standard deviation** | How much it varies | ±4.3 |
| **P(blank)** | Chance of ≤2 points | 18% |
| **P(return)** | Chance of ≥5 points (goal/assist likely) | 72% |
| **P(haul)** | Chance of ≥10 points (multiple returns) | 28% |
| **10th percentile** | Worst realistic outcome | 2 pts |
| **90th percentile** | Best realistic outcome | 14 pts |

### 4.2 Captain Comparison

| Player | E[2×pts] | P(haul ≥20) | P(blank ≤4) | P90 | Confidence |
|--------|----------|-------------|-------------|-----|-----------|
| Haaland | 15.6 | 28% | 12% | 24 | High |
| Son | 12.2 | 15% | 22% | 20 | Medium |
| Palmer | 11.8 | 18% | 25% | 22 | Medium |

### 4.3 Transfer Recommendations

| Sell | Buy | Gain | Cost | Net value | Reasoning |
|------|-----|------|------|-----------|-----------|
| Salah (injured) | Son | +4.2 pts | Free | +4.2 | Salah out 3 GWs, Son has easy run |
| Rogers | Mbeumo | +2.1 pts | -4 hit | -1.9 | Not worth a hit this week |

### 4.4 Multi-Week Plan

```
GW12: Transfer Salah → Son          [saves 3 GWs of 0-point Salah]
GW13: Roll transfer                  [bank for double transfer GW14]
GW14: Welbeck → Isak + Rogers → Mbeumo  [2 FTs, fixture swing]
GW15: Roll transfer                  [save for flexibility]
GW16: Bench Boost                    [DGW with strong bench]

Total expected gain over "do nothing": +18.4 pts
```

### 4.5 Chip Timing

```
Bench Boost → GW22 (DGW, 15 players all have fixtures, +11.2 pts)
Triple Captain → GW28 (Haaland vs bottom team at home, +8.5 pts)
Free Hit → GW33 (Blank GW, only 6 of your players have fixtures, +14.1 pts)
```

---

## 5. How the Layers Work Together

### Step-by-step: "Who should I captain?"

```
1. PREDICTIVE MODELS produce for each player:
   Haaland: P(60+ min)=0.92, λ_goals=1.0, λ_assists=0.2, P(CS)=0.10
   Son:     P(60+ min)=0.88, λ_goals=0.5, λ_assists=0.35, P(CS)=0.20
   Palmer:  P(60+ min)=0.85, λ_goals=0.6, λ_assists=0.25, P(CS)=0.15

2. SIMULATION ENGINE runs 10,000 samples per player:
   For each sample:
     - Roll dice: does Haaland play? (92% chance → yes in most samples)
     - Roll dice: how many goals? (Poisson λ=1.0 → 0,1,2,3...)
     - Roll dice: assists, CS, cards, bonus
     - Compute FPL points for this sample
   Result: array of 10,000 point outcomes per player

3. OPTIMIZATION compares (doubled):
   Haaland doubled: mean=15.6, P(≥20)=28%
   Son doubled:     mean=12.2, P(≥20)=15%
   Palmer doubled:  mean=11.8, P(≥20)=18%
   
   Winner: Haaland (highest mean AND highest haul probability)

4. AGENT explains:
   "Captain Haaland. He has the highest expected doubled points (15.6)
    with a 28% chance of a 20+ point haul. Facing the league's weakest
    defence at home. Confidence: high."
```

### Step-by-step: "Should I take a -4 hit for Isak?"

```
1. MODELS predict for current player (Welbeck) and target (Isak):
   Welbeck: 3.8 xPts this GW, 3.5 next GW, 3.2 GW after = 10.5 over 3 GWs
   Isak:    5.8 xPts this GW, 6.1 next GW, 5.5 GW after = 17.4 over 3 GWs

2. PLANNER evaluates:
   Gain = 17.4 - 10.5 = 6.9 pts over 3 GWs
   Cost = 4 pts (hit)
   Net value = 6.9 - 4 = +2.9 pts

3. AGENT recommends:
   "Yes, take the hit. Isak gains +6.9 expected points over the next 3 GWs,
    minus the -4 hit = net gain of +2.9 points. His fixtures are excellent
    (Burnley, Ipswich, Southampton) while Welbeck faces Arsenal and Liverpool.
    Confidence: medium (3-GW projections have uncertainty)."
```

### Step-by-step: "When should I play Bench Boost?"

```
1. PLANNER evaluates each remaining GW:
   For each GW, compute:
     - How much do your bench players score if they all play?
     - Compare: 11-player total vs 15-player total

2. SIMULATION runs both scenarios:
   GW22 (DGW): Bench players = Welbeck(4.1) + Dalot(3.8) + Rogers(3.5) + Henderson(3.2)
               Extra points from bench = 14.6 pts
               But DGW means they might have 2 games each!
               Simulated bench contribution: ~18.2 pts
   
   GW25 (normal): Bench = 11.2 pts (less value, single fixtures)

3. AGENT recommends:
   "Play Bench Boost in GW22. It's a Double Gameweek — all 4 bench
    players have 2 fixtures. Expected gain: +18.2 pts (vs +11.2 in GW25).
    Make sure your bench is strong before then."
```

---

## 6. Key Concepts to Understand

### Expected Points vs Actual Points

The system predicts **distributions**, not exact outcomes. When it says "Haaland: 7.5 xPts", it means:
- On average, across many gameweeks like this, Haaland scores ~7.5
- In any specific GW, he might score 2 (blank) or 20 (hat-trick)
- The distribution tells you: 18% chance of 0-2, 54% chance of 3-9, 28% chance of 10+

**Why this matters:** You should NOT evaluate the system by single-GW results. A correct prediction of "70% chance of return" will still see 30% blanks. Evaluate over 10+ gameweeks.

### The Hit Decision Framework

Taking a -4 hit is worth it when:
```
E[new player over holding period] - E[current player over holding period] > 4
```

The system computes this automatically, but the key insight is **holding period matters**. A -4 for 1 GW of gain needs +4.1 gain to be worth it. A -4 for 8 GWs of gain only needs +0.5/GW to be worth it.

### Rolling vs Using Transfers

Rolling (making 0 transfers) gives you 2 FTs next week. This is valuable when:
- No single transfer offers >2 pts gain this week
- You want to make a double transfer next week for a fixture swing
- You're planning a squad restructure in 2 weeks

The planner evaluates this trade-off automatically.

### Chips as Multipliers

Each chip multiplies your base strategy:
- **Bench Boost**: Multiplies the value of having a strong bench (DGWs)
- **Triple Captain**: Multiplies the value of having the best captain pick (easiest fixture for premium)
- **Free Hit**: Lets you temporarily optimize for one specific GW (blanks/doubles)
- **Wildcard**: Resets your squad entirely (fixture swings, price rises)

The system evaluates when each multiplier is highest.

---

## 7. Typical Season Timeline

| Period | What the system helps with |
|--------|---------------------------|
| **Pre-season (GW1)** | Initial squad selection: best 15 within £100m |
| **Early season (GW1-8)** | Learning phase: models adjust to new season patterns |
| **October-December** | Fixture congestion: rotation prediction becomes critical |
| **January** | Transfer window: new signings enter the pool |
| **Feb-March** | DGW/BGW planning: chip timing becomes crucial |
| **April-May** | End-game: aggressive plays if chasing, conservative if leading |

---

## 8. What to Do When the System Is Wrong

The system will sometimes be wrong. Here's how to handle it:

| Situation | What happened | What to do |
|-----------|--------------|-----------|
| Captain blanked | Correct prediction, unlucky outcome | Nothing — 30% blanks happen even for best picks |
| Transfer target scored while benched | Model underpredicted rotation | Update PlayerContext with manager info you notice |
| CS prediction failed | Team conceded from a set piece | This is inherent noise — CS is hardest to predict |
| Plan didn't account for new injury | News broke after last refresh | Run `check_for_changes()` more frequently near deadlines |

**Key principle:** Judge the system by average performance over 10+ GWs, not individual decisions. A system that's right 65% of the time will still be wrong every third GW.

---

## 9. Quick Reference: Commands

| What you want | What to run |
|---------------|-------------|
| Refresh data after GW ends | `python scripts/refresh.py` |
| Check for injury/news updates | `agent.check_for_changes(api_data, deadline)` |
| Get full GW recommendation | `agent.get_recommendation()` |
| Captain advice | `agent.ask("Who should I captain?")` |
| Transfer advice | `agent.ask("Should I make a transfer?")` |
| Chip timing | `agent.ask("When should I play my chips?")` |
| Multi-week plan | `agent.ask("What's the strategy for the next 5 weeks?")` |
| Show my squad | `agent.ask("Show my team")` |
| Override player context | `contexts[id].override(days_since_last_match=3, ...)` |
| Confirm a decision | `agent.confirm_action("Captained Haaland")` |
| Full simulation for GW | `simulate_gameweek(predictions, n_simulations=10000)` |
| Optimize squad from scratch | `select_squad(player_pool, budget=1000)` |
