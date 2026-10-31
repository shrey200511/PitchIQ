# Write-up template (blog post / LinkedIn article)

Fill every `[ ]` from `reports/results.md` and `reports/resume_metrics.json`.
Don't round in your favour. If the model lost to the market, say so; that's the interesting part.

---

## Can free data get close to the bookmakers? Building PitchIQ

**Hook (2–3 sentences).** Bookmakers' closing odds are one of the best probability forecasts in sport. I wanted to see how close a model built only from free public data could get to them, and what the gap would teach me.

**What I built.** Elo ratings from scratch, xG-based features with a leakage test, a Dixon-Coles goals model, logistic regression and LightGBM. All of it is evaluated walk-forward over [N] seasons against bookmaker probabilities with the margin removed. There's also a 10,000-run season simulator.

**The result (one table).** Paste the test table. Then one honest sentence:
> On [760] test matches, my best model ([name]) had a log loss of [x] against the closing market's [y]. The gap was [z] (95% CI [lo, hi]).

**Three things I learned**
1. *Leakage is the real enemy.* Explain the perturbation test in plain words, and that a deliberately leaky join was caught.
2. *Calibration over accuracy.* Reliability plot plus one sentence on temperature scaling ([helped / didn't help] on validation: [numbers]).
3. *Draws.* Draws are [x]% of matches but the draw is the most likely outcome in [y]% of predictions. Explain why that is.

**Where the gap comes from.** No lineups/injuries, no cup fatigue, xG is someone else's model. The market prices team news; the model can't see it.

**What's next.** Lineup data, backtesting the simulator, live tracking against the market.

Links: repo · live app.

---

# Resume bullets (write ONLY after the real run)

Pattern: *action + method + scale + measured result*. Every number from `resume_metrics.json`.

- Built an end-to-end EPL forecasting pipeline (Python, pandas, LightGBM) joining [N] matches of results, bookmaker odds and xG across [S] seasons; added an automated **perturbation-based leakage test** across [D] cut-off dates.
- Engineered Elo ratings from scratch plus a Dixon-Coles goals model; walk-forward backtest on [n_test] held-out matches reached log loss [x] vs [y] for margin-free bookmaker closing odds ([gap / CI], [beat/trailed] base rates by [z]).
- Improved calibration with temperature scaling (ECE [a] → [b]); explained predictions with SHAP and analysed why draws ([d]% of matches) are almost never the model's top pick.
- Shipped a Dockerised Streamlit app with 10,000-run Monte Carlo title/top-4/relegation odds, refreshed weekly via GitHub Actions; CI runs the full pipeline on synthetic data.

Consulting/analyst framing (for Big 4, Accenture S&C and similar): lead with *benchmarking against a market standard*, *quantified uncertainty (bootstrap CIs)* and *explainability for non-technical users*.
