# Decision log

Every choice that shaped the data or the models, with the reason. When an
interviewer asks "why did you…?", the answer should be here.

| # | Area | Decision | Why | Alternative considered |
|---|---|---|---|---|
| 1 | Seasons | Model on 2015/16 onward; Elo warm-up from 2005/06 | Understat xG starts 2014/15, so 2015/16 is the first season with a full year of prior xG. Elo needs several seasons to converge | Start in 2014 and impute xG (adds noise to the first season) |
| 2 | Splits | Walk-forward. Validation 2020–2023, test 2024 + 2025. Season 2026/27 is live only | Mirrors real use (predict the future from the past). Test seasons are touched once, after every choice is final | Random K-fold (leaks future team strength) |
| 3 | Join key | (season, home, away), not date | Each pairing happens exactly once per season. Dates shift with postponements and Understat uses UTC kickoff | Fuzzy date join ±N days (fragile) |
| 4 | Team names | Explicit Understat→football-data dictionary that **fails loudly** | Silent fuzzy matching can drop a promoted club's matches without anyone noticing | `difflib` auto-match (used only to *suggest* a fix in the error) |
| 5 | Conflicts | Keep football-data's score and date when sources disagree, and log it | football-data is the results/odds source of record; Understat supplies xG only | Drop disagreeing rows |
| 6 | Benchmark | Pinnacle closing odds, then average closing, then Bet365 closing | Pinnacle is a low-margin sharp book, and the closing line holds all public info, so it's the hardest honest benchmark. Pre-match odds are reported too, because a model run days earlier competes with those | Soft-book opening odds (flatters the model) |
| 7 | Margin removal | Shin's method as the main reference, proportional normalisation as a check | Shin accounts for the favourite–longshot bias (more margin on longshots) | Power method (similar results; kept out to reduce clutter) |
| 8 | Elo | Tuned K, home advantage and summer regression by grid search on 2008–2019 only; goal-margin multiplier; promoted teams start at the relegated teams' average | Tuning on pre-validation seasons stops the Elo choice from being fitted to the reported seasons | Fixed K = 20 (FiveThirtyEight-style default) |
| 9 | Elo loss | Tune on MSE of expected score (draw = 0.5) | Elo predicts an expected score, not three outcome probabilities | H/D/A log loss via an extra mapping model |
| 10 | Leakage | Strictly-before-date `merge_asof` + perturbation test on 7 dates + forbidden-column guard | The perturbation test checks behaviour, so it also catches leaks a future edit might add. A deliberate leak is caught in the tests | Code review only |
| 11 | Promoted teams | Rolling stats reset when a team has been out for > 200 days; `promoted` flag; Dixon-Coles prior toward the weakest teams | Form from a Premier League stint years ago is misleading | Carry the old stats over |
| 12 | Rest days | League games only, capped at 14 | Cup/European fixtures aren't in the data (known limitation). Beyond ~2 weeks extra rest stops mattering | Uncapped (summer break dominates) |
| 13 | Dixon-Coles | Weekly refit, 3-year window, time decay xi = 0.0019/day (≈1-year half-life), weak N(0, 0.3²) prior on strengths | Recency matters. The prior stops teams with few games getting extreme strengths and huge simulation variance | No prior (promoted teams got title odds > 5% in October on synthetic data) |
| 14 | Models | Elo-logit → logistic regression → LightGBM | Each step adds complexity only if it beats the simpler model on validation | Neural nets (too little data: ~3,800 training matches) |
| 15 | Tuning | Small grids scored by validation log loss | Log loss is a proper scoring rule and punishes overconfidence | Accuracy (blind to probability quality) |
| 16 | Calibration | Temperature scaling for LightGBM, fitted on a season the model didn't train on | One parameter, so it can't overfit. Fixes overall over/under-confidence | Isotonic per class (needs much more data) |
| 17 | Evaluation set | Every model scored on the same matches (result + both markets present) | Otherwise a model can win by being scored on easier games | Each model on whatever it predicted |
| 18 | Significance | Paired bootstrap of per-match log-loss difference vs the market | Shows whether a gap is real or noise on ~760 test matches | Report point estimates only |
| 19 | Simulator | Dixon-Coles scorelines + parameter uncertainty (200 draws × 50 seasons) | Goal difference breaks ties. Without uncertainty, title odds are overconfident | Use ML H/D/A probabilities (no scorelines) |
| 20 | App | Streamlit reading pre-computed artifacts; no odds for upcoming games | Free hosting, fast start, no modelling dependencies. This is a modelling project, not a betting tool | FastAPI + React (more work, same value) |
| 21 | Data collection | Cache completed seasons; refresh only the current season; 3 s between Understat requests; honest User-Agent; raw data not committed | Be a good citizen to free data sources | Re-download everything weekly |

## To fill in after your first real run
- [ ] Final tuned Elo parameters (from `artifacts/tuned_params.json`) and whether any landed on a grid edge
- [ ] Anything surprising in `reports/cleaning_log.md` (score disagreements, unmapped names you added)
- [ ] Whether temperature scaling helped on validation (compare `LightGBM` vs `LightGBM + temperature`)
