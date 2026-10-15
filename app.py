"""Investigator dashboard for the OLG AML and safer-gambling pipeline.

    streamlit run app.py -- --db synth.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import score as S
from features import FEATURES, chip_dump_pairs

AML_COLOR, HARM_COLOR, BOTH_COLOR, NONE_COLOR = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
ROUTE_COLOR = {"AML": AML_COLOR, "HARM": HARM_COLOR, "BOTH": BOTH_COLOR, "NONE": NONE_COLOR}
TIER_ICON = {"Critical": "🔴 Critical", "High": "🟠 High", "Medium": "🟡 Medium", "Low": "⚪ Low"}
TIER_RANK = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
PAGES = ["Queue", "Player 360", "Network", "Scorecard", "Validation"]
DECISIONS = ["Open", "Escalated", "Confirmed suspicious", "Dismissed", "Referred to safer gambling"]


def md(text: str) -> str:
    """Escape $ so Streamlit doesn't render it as LaTeX."""
    return str(text).replace("$", "\\$")


def db_path() -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="synth.db")
    return ap.parse_known_args(sys.argv[1:])[0].db


DB = db_path()
st.set_page_config(page_title="Player Risk Console", page_icon="🛡️", layout="wide")


# --- data -------------------------------------------------------------------------
def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS case_actions
                   (player_id TEXT, ts TEXT, decision TEXT, note TEXT, analyst TEXT)""")
    return con


@st.cache_data
def load_core(db: str):
    con = sqlite3.connect(db)
    R = pd.read_sql("SELECT * FROM risk_scores", con)
    F = pd.read_sql("SELECT * FROM player_features", con)
    why = pd.read_sql("SELECT * FROM risk_reasons", con)
    regions = pd.read_sql("SELECT * FROM regions", con)
    venues = pd.read_sql("SELECT * FROM venues", con)
    rules = pd.read_sql("SELECT * FROM scorecard_rules", con)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    labels = pd.read_sql("SELECT * FROM labels", con) if "labels" in tables else None
    con.close()
    top = why[(why["source"] == "scorecard") & (why["rank"] == 1)]
    R = R.merge(F[["player_id", "home_region", "income_band", "occupation", "age_band", "channel",
                   "peer_group", "network_component", "network_size"]], on="player_id")
    for lens in ("aml", "harm"):
        R = R.merge(top[top["lens"] == lens][["player_id", "reason"]].rename(columns={"reason": f"{lens}_reason"}),
                    on="player_id", how="left")
    return R.set_index("player_id"), F.set_index("player_id"), why, regions, venues, rules, labels


@st.cache_data
def player_activity(db: str, pid: str):
    con = sqlite3.connect(db)
    s = pd.read_sql("SELECT * FROM sessions WHERE player_id = ?", con, params=(pid,),
                    parse_dates=["start_ts", "end_ts"])
    t = pd.read_sql("SELECT * FROM transactions WHERE player_id = ? ORDER BY ts DESC", con, params=(pid,),
                    parse_dates=["ts"])
    con.close()
    return s, t


def case_history(pid: str) -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql("SELECT ts, decision, note, analyst FROM case_actions WHERE player_id = ? ORDER BY ts DESC",
                           con, params=(pid,))


def latest_decisions() -> pd.Series:
    with connect() as con:
        d = pd.read_sql("SELECT player_id, decision, MAX(ts) AS ts FROM case_actions GROUP BY player_id", con)
    return d.set_index("player_id")["decision"]


@st.cache_data
def component_edges(db: str, members: tuple) -> pd.DataFrame:
    con = sqlite3.connect(db)
    marks = ",".join("?" * len(members))
    x = pd.read_sql(f"""SELECT DISTINCT player_id, device_id AS key, 'shared device' AS kind FROM transactions
                        WHERE device_id IS NOT NULL AND player_id IN ({marks})
                        UNION SELECT DISTINCT player_id, instrument_id, 'shared instrument' FROM transactions
                        WHERE instrument_id IS NOT NULL AND player_id IN ({marks})""", con, params=members * 2)
    s = pd.read_sql(f"SELECT * FROM sessions WHERE player_id IN ({marks})", con, params=members,
                    parse_dates=["start_ts", "end_ts"])
    con.close()
    edges = []
    for (key, kind), grp in x.groupby(["key", "kind"]):
        ids = sorted(grp["player_id"].unique())
        edges += [(a, b, kind) for i, a in enumerate(ids) for b in ids[i + 1:]]
    pairs = chip_dump_pairs(s)
    edges += [(a, b, "chip dumping") for a, b in zip(pairs["player_id_l"], pairs["player_id_w"])]
    return pd.DataFrame(edges, columns=["a", "b", "kind"]).drop_duplicates()


# --- charts -----------------------------------------------------------------------
def style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8), hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridwidth=0.5, zeroline=False)
    return fig


def network_figure(R: pd.DataFrame, members: list, focus: str | None = None) -> go.Figure:
    E = component_edges(DB, tuple(sorted(members)))
    G = nx.Graph()
    G.add_nodes_from(members)
    G.add_edges_from(zip(E["a"], E["b"]))
    pos = nx.spring_layout(G, seed=3)
    fig = go.Figure()
    for kind, dash in (("shared device", "solid"), ("shared instrument", "dot"), ("chip dumping", "dash")):
        sub = E[E["kind"] == kind]
        if sub.empty:
            continue
        xs, ys = [], []
        for a, b in zip(sub["a"], sub["b"]):
            xs += [pos[a][0], pos[b][0], None]
            ys += [pos[a][1], pos[b][1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=kind, hoverinfo="skip",
                                 line=dict(color=NONE_COLOR, width=1.5, dash=dash)))
    for rt, color in ROUTE_COLOR.items():
        ids = [m for m in members if R.at[m, "route"] == rt]
        if not ids:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[i][0] for i in ids], y=[pos[i][1] for i in ids], mode="markers+text", name=f"route: {rt}",
            text=ids, textposition="top center",
            marker=dict(color=color, size=[22 if i == focus else 14 for i in ids],
                        line=dict(color="white", width=2)),
            hovertext=[f"{i}<br>AML {R.at[i, 'aml_score']:.0f} · harm {R.at[i, 'harm_score']:.0f}" for i in ids],
            hoverinfo="text"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, 380)


# --- pages ------------------------------------------------------------------------
def open_player(pid: str):
    st.session_state["player"] = pid
    st.session_state["page"] = "Player 360"


def page_queue(R: pd.DataFrame):
    st.title("Priority queue")
    st.caption(f"Scored as of {R['as_of'].iloc[0]} · {len(R):,} players · flag at {S.FLAG} scorecard points "
               f"or {S.MODEL_FLAG}% model confidence")
    routed = R[R["route"] != "NONE"]
    c = st.columns(5)
    c[0].metric("Players routed", len(routed))
    c[1].metric("AML only", int((R["route"] == "AML").sum()))
    c[2].metric("Safer gambling only", int((R["route"] == "HARM").sum()))
    c[3].metric("Both teams", int((R["route"] == "BOTH").sum()))
    c[4].metric("Critical", int(((R["aml_tier"] == "Critical") | (R["harm_tier"] == "Critical")).sum()))

    f = st.columns([1.2, 1, 1, 1, 1.2])
    route = f[0].multiselect("Route", ["AML", "HARM", "BOTH", "NONE"], default=["AML", "HARM", "BOTH"])
    min_score = f[1].slider("Min priority", 0, 100, 0)
    region = f[2].multiselect("Home region", sorted(R["home_region"].unique()))
    source = f[3].selectbox("Flagged by", ["Any", "Scorecard", "Model only"])
    search = f[4].text_input("Player ID", placeholder="P000371")

    q = R[R["route"].isin(route) & (R["priority"] >= min_score)]
    if region:
        q = q[q["home_region"].isin(region)]
    if source == "Scorecard":
        q = q[q["aml_flag"].isin(["scorecard", "both"]) | q["harm_flag"].isin(["scorecard", "both"])]
    elif source == "Model only":
        q = q[q[["aml_flag", "harm_flag"]].isin(["model"]).any(axis=1)
              & ~q[["aml_flag", "harm_flag"]].isin(["scorecard", "both"]).any(axis=1)]
    if search:
        q = q[q.index.str.contains(search.strip(), case=False)]
    q = q.sort_values("queue_rank")
    status = latest_decisions().reindex(q.index).fillna("New")

    view = pd.DataFrame({
        "Rank": q["queue_rank"], "Route": q["route"],
        "AML": q["aml_score"], "Harm": q["harm_score"],
        "AML model": q.get("aml_model"), "Harm model": q.get("harm_model"),
        "Tier": [TIER_ICON[max(a, h, key=TIER_RANK.get)] for a, h in zip(q["aml_tier"], q["harm_tier"])],
        "Top reason": q["aml_reason"].where(q["aml_score"] >= q["harm_score"], q["harm_reason"]).fillna(""),
        "Recommended action": q["recommended_action"], "Case": status,
    }, index=q.index)
    st.caption(f"{len(view):,} players match · select a row to open the player")
    pct = st.column_config.ProgressColumn
    event = st.dataframe(
        view, height=460, width="stretch", on_select="rerun", selection_mode="single-row", key="queue_table",
        column_config={"AML": pct(min_value=0, max_value=100, format="%.0f"),
                       "Harm": pct(min_value=0, max_value=100, format="%.0f"),
                       "AML model": st.column_config.NumberColumn(format="%.0f%%"),
                       "Harm model": st.column_config.NumberColumn(format="%.0f%%"),
                       "Top reason": st.column_config.TextColumn(width="large"),
                       "Recommended action": st.column_config.TextColumn(width="large")})
    rows = event.selection.rows if event and event.selection else []
    if rows:
        pid = view.index[rows[0]]
        st.button(f"Open {pid} in Player 360 →", on_click=open_player, args=(pid,), type="primary")

    st.subheader("Alerts by typology")
    tags = pd.concat([routed["aml_tags"].str.split(",").explode().rename("tag").to_frame().assign(lens="AML"),
                      routed["harm_tags"].str.split(",").explode().rename("tag").to_frame().assign(lens="Harm")])
    tags = tags[tags["tag"].fillna("") != ""]
    counts = tags.groupby(["lens", "tag"]).size().rename("players").reset_index().sort_values("players")
    fig = go.Figure()
    for lens, color in (("AML", AML_COLOR), ("Harm", HARM_COLOR)):
        sub = counts[counts["lens"] == lens]
        fig.add_bar(y=sub["tag"].str.replace("_", " "), x=sub["players"], orientation="h", name=lens,
                    marker=dict(color=color, cornerradius=4), hovertemplate="%{y}: %{x} players<extra></extra>")
    fig.update_layout(bargap=0.35, xaxis_title="Routed players with this tag")
    st.plotly_chart(style(fig, 40 + 26 * len(counts)), width="stretch")


def score_card(col, R, pid, lens, label, color):
    r = R.loc[pid]
    flag = r[f"{lens}_flag"] or "not flagged"
    col.markdown(f"**{label}**")
    col.metric("Scorecard", f"{r[f'{lens}_score']:.0f} / 100")
    col.markdown(TIER_ICON[r[f"{lens}_tier"]])
    if f"{lens}_model" in r:
        col.caption(f"Model confidence {r[f'{lens}_model']:.0f}% · routed by: {flag}")
    tags = r[f"{lens}_tags"]
    if tags:
        col.caption("Tags: " + ", ".join(t.replace("_", " ") for t in tags.split(",")))


def points_chart(F: pd.DataFrame, pid: str, lens: str, color: str):
    P = S.rule_points(F.loc[[pid]]).iloc[0]
    names = [r.name for r in S.RULES if r.lens == lens]
    P = P[names]
    P = P[P > 0].sort_values()
    if P.empty:
        st.caption("No scorecard rule fired.")
        return
    fig = go.Figure(go.Bar(x=P.values, y=[n.replace("_", " ") for n in P.index], orientation="h",
                           marker=dict(color=color, cornerradius=4),
                           hovertemplate="%{y}: %{x:.1f} pts<extra></extra>"))
    fig.update_layout(bargap=0.35, xaxis_title="Points")
    st.plotly_chart(style(fig, 60 + 28 * len(P)), width="stretch")


def peer_chart(F: pd.DataFrame, pid: str, metric: str):
    me = F.loc[pid]
    peers = F[(F["peer_group"] == me["peer_group"]) & (F["n_sessions"] > 0)]
    rng = np.random.default_rng(0)
    log = metric in ("annual_buy_in", "annual_net_loss", "mean_buy_in")
    x = peers[metric].clip(lower=1) if log else peers[metric]
    fig = go.Figure()
    fig.add_scatter(x=x, y=rng.uniform(-1, 1, len(peers)), mode="markers", name="similar players",
                    marker=dict(color=NONE_COLOR, size=8, opacity=0.5), hovertext=peers.index,
                    hovertemplate="%{hovertext}: %{x:,.2f}<extra></extra>")
    fig.add_scatter(x=[max(me[metric], 1) if log else me[metric]], y=[0], mode="markers+text", name=pid,
                    text=[pid], textposition="top center",
                    marker=dict(color=HARM_COLOR, size=16, line=dict(color="white", width=2)),
                    hovertemplate=f"{pid}: %{{x:,.2f}}<extra></extra>")
    fig.add_vline(x=float(x.median()), line=dict(color=NONE_COLOR, dash="dot", width=1),
                  annotation_text="peer median", annotation_position="top")
    fig.update_yaxes(visible=False, range=[-1.6, 1.6])
    fig.update_xaxes(type="log" if log else "linear", title=FEATURES[metric][2])
    st.plotly_chart(style(fig, 260), width="stretch")
    st.caption(f"Peer group: {S.peer_label(me['peer_group'])} · {len(peers)} players")


def timeline_chart(s: pd.DataFrame, as_of: str):
    if s.empty:
        st.caption("No gaming sessions.")
        return
    wk = s.set_index("start_ts").resample("W-MON", label="left", closed="left").agg(
        {"buy_in": "sum", "net_result": "sum"})
    wk["net_loss"] = (-wk["net_result"]).clip(lower=0)
    recent = pd.Timestamp(as_of) - pd.Timedelta(days=30)
    base = wk[wk.index < recent]["buy_in"]
    fig = go.Figure()
    if len(base) >= 4:
        lo, hi = base.quantile(0.25), base.quantile(0.75)
        fig.add_hrect(y0=lo, y1=hi, fillcolor=NONE_COLOR, opacity=0.18, line_width=0,
                      annotation_text="usual range (baseline weeks, 25th–75th pct)", annotation_position="top left")
    fig.add_vrect(x0=recent, x1=pd.Timestamp(as_of), fillcolor=AML_COLOR, opacity=0.06, line_width=0,
                  annotation_text="last 30 days", annotation_position="top right")
    fig.add_scatter(x=wk.index, y=wk["buy_in"], name="Buy-ins", mode="lines+markers",
                    line=dict(color=AML_COLOR, width=2), marker=dict(size=8),
                    hovertemplate="week of %{x|%b %d}: $%{y:,.0f}<extra>Buy-ins</extra>")
    fig.add_scatter(x=wk.index, y=wk["net_loss"], name="Net loss", mode="lines+markers",
                    line=dict(color=HARM_COLOR, width=2), marker=dict(size=8),
                    hovertemplate="week of %{x|%b %d}: $%{y:,.0f}<extra>Net loss</extra>")
    fig.update_yaxes(title="$ per week", tickprefix="$", separatethousands=True)
    st.plotly_chart(style(fig, 320), width="stretch")


def location_chart(s: pd.DataFrame, t: pd.DataFrame, me: pd.Series, regions, venues):
    """Schematic map from region coordinates (no basemap, so it works offline)."""
    vs = s[s["channel"] == "VENUE"].merge(venues[["venue_id", "region"]], on="venue_id")
    visits = vs.groupby("region").size().rename("sessions").reset_index().merge(
        regions, left_on="region", right_on="region_code")
    home = regions[regions["region_code"] == me["home_region"]].iloc[0]
    fig = go.Figure()
    for v in visits.itertuples():
        if v.region != home["region_code"]:
            fig.add_scatter(x=[home["lon"], v.lon], y=[home["lat"], v.lat], mode="lines", showlegend=False,
                            hoverinfo="skip", line=dict(color=NONE_COLOR, width=1, dash="dot"))
    fig.add_scatter(x=regions["lon"], y=regions["lat"], mode="markers+text", name="regions",
                    text=regions["region_code"], textposition="bottom center", textfont=dict(size=11),
                    marker=dict(color=NONE_COLOR, size=8), hovertext=regions["name"], hoverinfo="text")
    if not visits.empty:
        size = 12 + 28 * np.sqrt(visits["sessions"] / visits["sessions"].max())
        fig.add_scatter(x=visits["lon"], y=visits["lat"], mode="markers", name="venue sessions",
                        marker=dict(color=AML_COLOR, size=size, opacity=0.7, line=dict(color="white", width=1)),
                        hovertext=[f"{n}: {c} sessions" for n, c in zip(visits["name"], visits["sessions"])],
                        hoverinfo="text")
    fig.add_scatter(x=[home["lon"]], y=[home["lat"]], mode="markers", name=f"home ({home['name']})",
                    marker=dict(color=HARM_COLOR, size=16, symbol="star", line=dict(color="white", width=1)),
                    hovertext=[f"Home: {home['name']}"], hoverinfo="text")
    fig.update_xaxes(visible=False, range=[-91, -73.5])
    fig.update_yaxes(visible=False, range=[41.8, 49.2], scaleanchor="x", scaleratio=1.35)
    st.plotly_chart(style(fig, 340), width="stretch")
    if visits.empty:
        st.caption("No venue play: online only.")
    online = t[t["ip_country"].notna()]
    if not online.empty:
        where = online["ip_country"].where(online["ip_country"] != "CA", online["ip_region"].fillna("CA"))
        st.caption("Online activity by location: " + ", ".join(f"{k} ({n})" for k, n in where.value_counts().items()))


def page_player(R, F, why, regions, venues):
    order = list(R.sort_values("queue_rank").index)
    if st.session_state.get("player") not in order:
        st.session_state["player"] = order[0]
    pid = st.selectbox("Player", order, key="player",
                       format_func=lambda p: f"{p} · {R.at[p, 'route']} · priority {R.at[p, 'priority']:.0f}")
    r, me = R.loc[pid], F.loc[pid]
    st.title(pid)
    st.caption(f"Age {me['age_band']} · {me['occupation'].replace('_', ' ')} · income {me['income_band']} · "
               f"home {me['home_region']} · {me['channel'].lower()} · {me['loyalty_tier']} tier · "
               f"member since {me['signup_date']}")

    c = st.columns([1, 1, 1.6])
    score_card(c[0], R, pid, "aml", "AML risk", AML_COLOR)
    score_card(c[1], R, pid, "harm", "Safer-gambling risk", HARM_COLOR)
    with c[2]:
        st.markdown(f"**Route: {r['route']}**")
        if r["recommended_action"]:
            for part in r["recommended_action"].split(" | "):
                st.info(md(part))
        else:
            st.success("No action recommended")

    st.subheader("Why this player was flagged")
    c = st.columns(2)
    for col, lens, label, color in ((c[0], "aml", "AML", AML_COLOR), (c[1], "harm", "Safer gambling", HARM_COLOR)):
        with col:
            st.markdown(f"**{label}: scorecard points**")
            points_chart(F, pid, lens, color)
            mine = why[(why["player_id"] == pid) & (why["lens"] == lens)]
            for w in mine[mine["source"] == "scorecard"].itertuples():
                st.markdown(md(f"- {w.reason}"))
            model = mine[mine["source"] == "model"]
            if not model.empty:
                st.caption(md("Model drivers: " + "; ".join(model["reason"])))

    s, t = player_activity(DB, pid)
    st.subheader("Spending over time vs own baseline")
    timeline_chart(s, r["as_of"])

    c = st.columns(2)
    with c[0]:
        st.subheader("Compared with similar players")
        metric = st.selectbox("Measure", ["annual_buy_in", "annual_net_loss", "sessions_per_week", "mean_buy_in",
                                          "night_share", "affordability_buy_in"],
                              format_func=lambda m: FEATURES[m][2])
        peer_chart(F, pid, metric)
    with c[1]:
        st.subheader("Where they play")
        location_chart(s, t, me, regions, venues)

    if me["network_size"] > 1:
        st.subheader(f"Linked players ({int(me['network_size'])} in group)")
        members = list(F.index[F["network_component"] == me["network_component"]])
        st.plotly_chart(network_figure(R, members, focus=pid), width="stretch")

    st.subheader("Case decision")
    c = st.columns([1, 2, 1])
    with c[0].form("case", clear_on_submit=True):
        decision = st.selectbox("Decision", DECISIONS)
        analyst = st.text_input("Analyst", value=st.session_state.get("analyst", ""))
        note = st.text_area("Note")
        if st.form_submit_button("Save decision", type="primary"):
            st.session_state["analyst"] = analyst
            with connect() as con:
                con.execute("INSERT INTO case_actions VALUES (?,?,?,?,?)",
                            (pid, datetime.now().isoformat(timespec="seconds"), decision, note, analyst))
            st.toast(f"Saved: {decision}")
    c[1].dataframe(case_history(pid), width="stretch", hide_index=True)

    st.subheader("Transactions")
    kinds = st.multiselect("Types", sorted(t["txn_type"].unique()), default=[])
    shown = t[t["txn_type"].isin(kinds)] if kinds else t
    st.dataframe(shown.drop(columns=["player_id"]), width="stretch", hide_index=True, height=320)


def page_network(R, F):
    st.title("Linked-player networks")
    st.caption("Players connected by a shared device, a shared payment instrument, or chip-dumping matches")
    comps = (F[F["network_size"] >= 2].groupby("network_component")
             .agg(players=("network_size", "first")).join(
                 R.groupby("network_component").agg(routed=("route", lambda x: int((x != "NONE").sum())),
                                                    max_aml=("aml_score", "max")))
             .sort_values(["max_aml", "players"], ascending=False))
    if comps.empty:
        st.info("No linked players found.")
        return
    comps.index.name = "group"
    c = st.columns([1, 2])
    with c[0]:
        st.dataframe(comps, width="stretch", height=420,
                     column_config={"max_aml": st.column_config.NumberColumn("Max AML score", format="%.0f")})
        pick = st.selectbox("Group", comps.index, format_func=lambda g: f"{g} · {comps.at[g, 'players']} players")
    with c[1]:
        members = list(F.index[F["network_component"] == pick])
        st.plotly_chart(network_figure(R, members), width="stretch")
        m = R.loc[members, ["route", "aml_score", "harm_score", "aml_tags"]].sort_values("aml_score", ascending=False)
        st.dataframe(m, width="stretch")
        st.button("Open first member in Player 360 →", on_click=open_player, args=(m.index[0],))


def page_rules(rules):
    st.title("Scorecard rules")
    st.caption(f"Every point is traceable. A rule scores 0 at or below its lower threshold and full points at or "
               f"above its upper threshold, linearly in between. A lens routes at {S.FLAG} points; tiers: "
               + ", ".join(f"{t} ≥ {c}" for c, t in S.TIERS[:-1]) + ".")
    for lens, label in (("aml", "AML"), ("harm", "Safer gambling")):
        st.subheader(label)
        sub = rules[rules["lens"] == lens].copy()
        sub["description"] = sub["feature"].map(lambda f: FEATURES.get(f, ("", "", f))[2])
        st.dataframe(sub[["name", "tag", "description", "lo", "hi", "points", "min_sessions", "gate"]],
                     width="stretch", hide_index=True)


def page_validation(R, labels):
    st.title("Validation on synthetic scenarios")
    if labels is None:
        st.info("No ground-truth labels in this database.")
        return
    st.caption("Uses the simulator's ground truth; never visible to scoring.")
    true_lens = labels[labels["lens"] != "NONE"].groupby("player_id")["lens"].agg(
        lambda x: "BOTH" if x.nunique() > 1 else x.iloc[0])
    truth = true_lens.reindex(R.index).fillna("NONE")

    rows = []
    for lens in ("AML", "HARM"):
        y = truth.isin([lens, "BOTH"])
        for name, flag in (("scorecard", R[f"{lens.lower()}_score"] >= S.FLAG),
                           ("model", R.get(f"{lens.lower()}_model", pd.Series(0, index=R.index)) >= S.MODEL_FLAG),
                           ("routed (either)", R["route"].isin([lens, "BOTH"]))):
            tp = int((flag & y).sum())
            rows.append({"lens": lens, "signal": name, "flagged": int(flag.sum()),
                         "precision": tp / max(1, flag.sum()), "recall": tp / max(1, y.sum())})
    m = pd.DataFrame(rows)
    st.dataframe(m, width="stretch", hide_index=True,
                 column_config={"precision": st.column_config.NumberColumn(format="%.2f"),
                                "recall": st.column_config.NumberColumn(format="%.2f")})

    c = st.columns(2)
    with c[0]:
        st.subheader("Recall by scenario")
        rec = []
        for typ, grp in labels[labels["lens"] != "NONE"].groupby("typology"):
            lens = grp["lens"].iloc[0]
            rec.append({"typology": typ.replace("_", " "), "lens": lens, "players": len(grp),
                        "recall": R.loc[grp["player_id"], "route"].isin([lens, "BOTH"]).mean()})
        rec = pd.DataFrame(rec).sort_values("recall")
        fig = go.Figure(go.Bar(x=rec["recall"] * 100, y=rec["typology"], orientation="h",
                               marker=dict(color=[ROUTE_COLOR[l] for l in rec["lens"]], cornerradius=4),
                               text=[f"{v:.0%} of {n}" for v, n in zip(rec["recall"], rec["players"])],
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:.0f}%<extra></extra>"))
        fig.update_layout(bargap=0.35, xaxis=dict(range=[0, 118], title="Routed to the right team (%)"))
        st.plotly_chart(style(fig, 60 + 30 * len(rec)), width="stretch")
        st.caption("Blue = AML scenario, orange = safer-gambling scenario")
    with c[1]:
        st.subheader("Hard negatives (legitimate but unusual)")
        hn = labels[labels["is_hard_negative"] == 1]
        hn = hn.assign(routed=R.loc[hn["player_id"], "route"].to_numpy() != "NONE")
        st.dataframe(hn.groupby("typology").agg(players=("player_id", "size"), wrongly_routed=("routed", "sum")),
                     width="stretch")
        st.subheader("Route vs truth")
        st.dataframe(pd.crosstab(truth.rename("truth"), R["route"].rename("routed")), width="stretch")


def main():
    try:
        R, F, why, regions, venues, rules, labels = load_core(DB)
    except Exception as e:  # missing pipeline tables
        st.error(f"Could not load scored data from {DB}: {e}. Run simulate.py, features.py and score.py first.")
        return
    page = st.sidebar.radio("Page", PAGES, key="page")
    st.sidebar.caption(f"Database: {DB}")
    if st.sidebar.button("Reload data"):
        st.cache_data.clear()
    {"Queue": lambda: page_queue(R),
     "Player 360": lambda: page_player(R, F, why, regions, venues),
     "Network": lambda: page_network(R, F),
     "Scorecard": lambda: page_rules(rules),
     "Validation": lambda: page_validation(R, labels)}[page]()


main()
