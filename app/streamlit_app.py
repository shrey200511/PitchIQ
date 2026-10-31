"""PitchIQ web app: read-only viewer for the pipeline's artifacts.

Deliberately has no modelling code and no heavy dependencies. It reads the
small files in artifacts/ (refreshed weekly by GitHub Actions), so it starts
fast and runs on free hosting.

This is a modelling showcase. It shows no bookmaker prices for upcoming games
and has no betting features.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(os.environ.get("PITCHIQ_ROOT", Path(__file__).resolve().parents[1]))
ART, FIG = ROOT / "artifacts", ROOT / "reports" / "figures"

st.set_page_config(page_title="PitchIQ — EPL match model", page_icon="⚽", layout="wide")


@st.cache_data(ttl=3600)
def load():
    def csv(name, **kw):
        p = ART / name
        return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()

    def js(name, default):
        p = ART / name
        return json.loads(p.read_text()) if p.exists() else default

    return dict(meta=js("meta.json", {}), upcoming=csv("upcoming.csv"),
                expl={e["match_id"]: e for e in js("explanations.json", [])},
                sim=csv("season_sim.csv"), pos=csv("position_probs.csv", index_col=0),
                summary=csv("backtest_summary.csv"), boot=csv("bootstrap_vs_market.csv"),
                bmeta=js("backtest_meta.json", {}), elo=csv("elo_current.csv"))


d = load()
meta = d["meta"]

st.title("⚽ PitchIQ")
st.caption("Premier League match-outcome probabilities, explained, plus Monte Carlo season odds. "
           "A data-science portfolio project — not betting advice.")
if meta.get("synthetic"):
    st.error("Demo mode: this deployment is showing SYNTHETIC test data, not real Premier League results.")
if meta:
    st.markdown(f"**Data through:** {meta.get('data_through')} · **Updated:** {meta.get('generated_at')} UTC "
                f"· **Season:** {meta.get('current_season')}/{str(meta.get('current_season', 0) + 1)[-2:]}")

tab_fix, tab_season, tab_model, tab_about = st.tabs(["Upcoming fixtures", "Season odds", "How good is it?", "About"])


def prob_bar(h, dr, a):
    seg = lambda w, c, t: (f"<div style='width:{w*100:.1f}%;background:{c};color:white;text-align:center;"
                           f"font-size:12px;padding:4px 0;white-space:nowrap;overflow:hidden'>{t}</div>")
    return ("<div style='display:flex;border-radius:6px;overflow:hidden;margin:2px 0 10px'>"
            + seg(h, "#2E6FBA", f"H {h:.0%}") + seg(dr, "#8A8F98", f"D {dr:.0%}") + seg(a, "#C8463D", f"A {a:.0%}")
            + "</div>")


with tab_fix:
    up = d["upcoming"]
    if up.empty:
        st.info("No upcoming fixtures available right now (international break, or the fixture list "
                "couldn't be refreshed).")
    else:
        st.write("Model: LightGBM with temperature scaling (main). Dixon-Coles goals model shown for comparison.")
        for r in up.itertuples():
            c1, c2, c3 = st.columns([3, 4, 3])
            c1.markdown(f"**{r.home}** vs **{r.away}**  \n<span style='color:gray'>{r.date}</span>",
                        unsafe_allow_html=True)
            c2.markdown(prob_bar(r.pH, r.pD, r.pA), unsafe_allow_html=True)
            c3.markdown(f"<span style='color:gray;font-size:13px'>Dixon-Coles: "
                        f"{r.dc_pH:.0%} / {r.dc_pD:.0%} / {r.dc_pA:.0%}</span>", unsafe_allow_html=True)

        st.subheader("Why does the model think that?")
        choice = st.selectbox("Pick a fixture", up.match_id,
                              format_func=lambda m: " vs ".join(up.set_index("match_id").loc[m, ["home", "away"]]))
        e = d["expl"].get(choice)
        if e:
            df = pd.DataFrame(e["drivers"])
            df["shown"] = df.apply(lambda x: f"{x.label} = {x.value:.2f}" if pd.notna(x.value) else f"{x.label} = n/a", axis=1)
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("shap_home:Q", title="Push toward (+) / away from (−) a home win  [SHAP, logit units]"),
                y=alt.Y("shown:N", sort=None, title=None),
                color=alt.condition("datum.shap_home > 0", alt.value("#2E6FBA"), alt.value("#C8463D")))
            st.altair_chart(chart, width="stretch")
            st.caption("Top 5 features by absolute SHAP value for the home-win class. Each bar is how much "
                       "that feature moved this prediction relative to the model's average.")

with tab_season:
    sim = d["sim"]
    if sim.empty:
        st.info("Season simulation not available yet.")
    else:
        st.write(f"{meta.get('n_sims', 10000):,} simulations of the {meta.get('remaining_fixtures', '?')} remaining "
                 "fixtures with a Dixon-Coles goals model, including uncertainty in team strengths.")
        show = sim[["team", "played", "points", "gd", "exp_points", "p_title", "p_top4", "p_relegation"]].copy()
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "exp_points": st.column_config.NumberColumn("Exp. points", format="%.1f"),
            "p_title": st.column_config.ProgressColumn("Title", format="%.1f%%", min_value=0, max_value=1),
            "p_top4": st.column_config.ProgressColumn("Top 4", format="%.1f%%", min_value=0, max_value=1),
            "p_relegation": st.column_config.ProgressColumn("Relegation", format="%.1f%%", min_value=0, max_value=1),
        })
        if not d["pos"].empty:
            pos = d["pos"].reset_index().melt(id_vars=d["pos"].index.name or "index", var_name="position", value_name="p")
            pos.columns = ["team", "position", "p"]
            pos["position"] = pos.position.astype(int)
            heat = alt.Chart(pos).mark_rect().encode(
                x=alt.X("position:O", title="Final position"), y=alt.Y("team:N", sort=list(sim.team), title=None),
                color=alt.Color("p:Q", scale=alt.Scale(scheme="blues"), title="Probability"),
                tooltip=["team", "position", alt.Tooltip("p:Q", format=".1%")])
            st.altair_chart(heat, width="stretch")

with tab_model:
    s, bm = d["summary"], d["bmeta"]
    st.write("Walk-forward backtest: every season is predicted by models trained only on earlier seasons. "
             "The bookmaker closing line (margin removed) is the benchmark — it is very hard to beat.")
    if not s.empty:
        t = s[s.split == "test"][["model", "log_loss", "brier", "rps", "accuracy", "ece"]]
        st.markdown(f"**Test seasons {bm.get('test_seasons')}** — {bm.get('n_test_matches', '?')} matches (lower is better, except accuracy)")
        st.dataframe(t.round(4), hide_index=True, width="stretch")
    if not d["boot"].empty:
        st.markdown("**Gap to the closing market** (log loss, 95% bootstrap CI; negative = better than market)")
        st.dataframe(d["boot"].round(4), hide_index=True, width="stretch")
    for img, cap in [("reliability_test.png", "Calibration on test seasons"),
                     ("draws.png", "Why draws are hard"), ("shap_global.png", "What drives the model")]:
        if (FIG / img).exists():
            st.image(str(FIG / img), caption=cap, width="stretch")

with tab_about:
    st.markdown("""
**Data.** Results, match stats and bookmaker odds: [football-data.co.uk](https://www.football-data.co.uk).
Expected goals (xG): [Understat](https://understat.com). Both used for non-commercial research with attribution;
raw data is not redistributed.

**Method.** Elo ratings built from scratch → leakage-checked rolling features (form, xG, rest, head-to-head) →
baselines (base rates, Dixon-Coles) → multinomial logistic regression and LightGBM with temperature scaling →
walk-forward evaluation against margin-free bookmaker probabilities → SHAP explanations → Monte Carlo season simulator.

**Limitations.** No lineups, injuries, or cup/European fixtures (rest days only count league games).
xG is Understat's own model. Market odds include information this model doesn't have.

**Not betting advice.** This app intentionally shows no odds or betting signals for upcoming games.

Code: [github.com/shrey200511/pitchiq](https://github.com/shrey200511/pitchiq)
""")
