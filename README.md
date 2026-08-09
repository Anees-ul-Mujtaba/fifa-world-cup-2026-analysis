# FIFA World Cup 2026 — Data Analysis & Goals Prediction

**Author:** Anees ul Mujtaba  
**Framework:** PACE (Plan → Analyze → Construct → Execute)  
**Dataset:** FIFA World Cup 2026 Complete Tournament Statistics (Kaggle — simulated tournament data)

---

## Project Overview

An end-to-end data analytics and machine learning project covering a simulated FIFA World Cup 2026 tournament across 48 teams and 104 matches. The project follows the PACE framework and demonstrates a complete analyst workflow: validating data before trusting it, exploring it for meaningful patterns, and building a regression model to predict goals scored — with results reported honestly based on what the data can actually support.

A deliberate focus of this project is the **data quality audit** — the three source files do not fully reconcile with each other, and rather than ignore that, every inconsistency is documented and explained. Knowing when data can and can't support a conclusion is as important a skill as the modeling itself.

---

## Datasets

Three CSV files describing the same simulated tournament:

| File | Rows | Description |
|---|---|---|
| `matches.csv` | 104 | Match-level stats: teams, stage, venue, weather, attendance, possession, shots, xG, goals, result |
| `players.csv` | 3,328 | Player-match performance: goals, assists, passing, tackles, saves, rating |
| `team_stats.csv` | 48 | Tournament-aggregated team standings: wins, losses, goals, points, average possession and xG |

> **Note:** The original Kaggle dataset contained 6 files — verified to be two identical naming conventions of the same 3 files. This project uses one clean copy of each.

---

## What the Notebook Covers

### PLAN
- Defined the analytical objective: explore tournament patterns and predict goals scored per team per match
- Selected `matches.csv` as the primary source of truth after cross-table validation

### ANALYZE — Data Quality Audit
Five documented data quality checks performed before any analysis:

1. **Structural check** — no nulls, no duplicate keys in any file
2. **Cross-table reconciliation** — `team_stats.csv` goal totals do not match `matches.csv` for the majority of teams; files were generated independently. `matches.csv` treated as source of truth throughout.
3. **Missing team** — Switzerland appears in `team_stats.csv` but has zero player records in `players.csv`
4. **Player ID structure** — `Player_ID` resets per match, not per player; no tournament-long top scorer leaderboard is valid in this data
5. **Zero-variance feature** — `Red_Cards` is 0 in every match across the full tournament; excluded from modeling

### ANALYZE — Exploratory Data Analysis
- Tournament structure: match counts and average attendance by stage
- Team performance: top 15 teams by points, possession vs xG scatter plot
- Match dynamics: correlation heatmap of all match stats vs goals scored — **only xG shows meaningful correlation (~0.55)**; shots, possession, corners, fouls, and rank advantage are all near-zero
- xG vs actual goals regression plot (r = 0.57)
- Goals distribution by weather condition and stage (Group vs Knockout)
- Player analysis: rating by position, age distribution, single-match top scoring performances

### CONSTRUCT — Model Building

**Target variable:** Goals scored per team per match (regression problem)

**Modeling steps — all shown explicitly:**
- Reshaped 104 match rows into 208 team-match rows (one row per team per match)
- Feature encoding using `pd.get_dummies()` for Weather and Stage
- 80/20 train/test split with `train_test_split`
- `StandardScaler` fitted on training data only, applied to both train and test
- Baseline: predicting the training mean for every observation
- **Model 1: Linear Regression** — explicit `.fit()` on scaled training data, `.predict()` on scaled test data
- **Model 2: Random Forest Regressor** — fit directly on unscaled data (tree models don't require scaling)

### EXECUTE — Results

| Model | MAE | RMSE | R² | vs Baseline |
|---|---|---|---|---|
| Baseline (predict mean) | ~0.80 | ~0.97 | ~0.00 | — |
| Linear Regression | ~0.59 | ~0.77 | ~0.35 | −26% MAE |
| Random Forest | ~0.60 | ~0.78 | ~0.37 | −25% MAE |

Both models clearly beat the baseline. `xG` dominates as the top feature in both the Linear Regression coefficients and Random Forest feature importances — three independent analyses (EDA correlation, LR coefficients, RF importances) all pointing to the same conclusion.

Diagnostic plots: predicted vs actual scatter and residual plot included for the best-performing model.

---

## Key Findings

1. **Only xG meaningfully predicts goals scored** (r ≈ 0.57) — shots, possession, corners, fouls, and rank advantage are all weak-to-negligible signals in this dataset
2. **team_stats.csv is not internally consistent** with matches.csv — the files were generated independently and should not be mixed in modeling
3. **FIFA rank has ~0 correlation with match outcomes** — confirming this is a synthetic dataset, not real tournament data. Conclusions are scoped accordingly throughout.
4. **R² ≈ 0.35** is the correct, honest result — a higher score would suggest overfitting noise on a 208-row dataset, not genuine predictive power

---

## Why R² Is Modest — and Why That's Fine

An R² in the 0.3–0.4 range is expected when only one feature carries real signal. A model cannot extract signal that isn't in the data. The value of this project is not "high R²" — it's correctly diagnosing what the data supports, building and evaluating models appropriately, and reporting results honestly rather than engineering them to look better than they are.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, reshaping, EDA |
| `numpy` | Numerical operations |
| `matplotlib` | Custom visualizations |
| `seaborn` | Statistical charts (heatmap, boxplot, regression plot) |
| `scikit-learn` | Train/test split, StandardScaler, LinearRegression, RandomForestRegressor, metrics |

---

## How to Run Locally

1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/fifa-world-cup-2026-analysis.git
cd fifa-world-cup-2026-analysis
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Update the data loading paths in the notebook — change the Kaggle paths to local relative paths:
```python
matches = pd.read_csv('data/matches.csv')
players = pd.read_csv('data/players.csv')
teams   = pd.read_csv('data/team_stats.csv')
```

4. Open the notebook
```bash
jupyter notebook fifa-2026-data-analysis-portfolio.ipynb
```

---

## Repo Structure

```
fifa-world-cup-2026-analysis/
├── fifa-2026-data-analysis-portfolio.ipynb   ← main notebook
├── data/
│   ├── matches.csv
│   ├── players.csv
│   └── team_stats.csv
├── README.md
└── requirements.txt
```

---

## Limitations

- **Synthetic data:** player names are anonymized placeholders and FIFA rank shows no correlation with outcomes — findings are "patterns in this dataset," not real football insight
- **Small sample:** 208 team-match rows after reshaping; Random Forest results would be more stable with more data
- **Explanatory, not predictive:** the model uses in-match stats (xG, shots, possession) known only during or after the match — it explains what relates to goals, not what predicts them before kick-off
- **xG circularity:** xG is itself a model-generated estimate, so "xG predicts goals" is partially an expected relationship

---

*Built as part of a data analytics and AI engineering portfolio. Project follows the PACE framework (Plan → Analyze → Construct → Execute).*
