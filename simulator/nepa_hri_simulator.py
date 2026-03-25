"""
NEPA-HRI Affective Modulation Simulator v0.1
Part of the NEPA (Neuromorphic Edge Processing Architecture) platform.

Coupled dynamical system:
  E_{t+1} = E_t * exp(-lambda) + alpha * A_t + incident_t + noise
  S_{t+1} = S_t * exp(-mu)    + beta  * regulation_t    + noise
  delta_t  = E_t - E_t_desired

Stability Envelope:
  Omega = { (E_t, S_t, delta_t) | |delta_t| <= kappa * S_t }
  Regime A (inside Omega): interaction self-stabilizing
  Regime B (outside Omega): escalation, cross-frame reactivation

Three agents compared:
  1. Sentiment-Only     no trajectory awareness
  2. Static-Valence     fixed positive reply
  3. NEPA-HRI           stability envelope + TD-style learning

Run: python simulator/nepa_hri_simulator.py
Output: output/ folder with 4 charts + CSV
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os, csv

os.makedirs("output", exist_ok=True)
np.random.seed(2026)

# ── Parameters ────────────────────────────────────────────────
T           = 80
LAMBDA      = 0.12
MU          = 0.08
ALPHA       = 0.55
BETA        = 0.45
KAPPA       = 1.8
GAMMA_DISC  = 0.92
ETA_LEARN   = 0.05
WIN_VI      = 4
N_EPISODES  = 500

INCIDENT_FRAMES = [8, 22, 40, 58, 70]
INCIDENT_MAG    = [0.75, 0.55, 0.85, 0.45, 0.65]

COLORS = {
    "Sentiment-Only": "#EF5350",
    "Static-Valence":  "#FF9800",
    "NEPA-HRI":        "#66BB6A",
}

# ── Desired emotion manifold ───────────────────────────────────
E_desired = np.zeros(T)
for t in range(T):
    E_desired[t] = 0.30 + 0.08*np.sin(t/12) + 0.03*np.random.randn()
E_desired = np.clip(E_desired, 0.10, 0.60)

# ── Agent reply functions ──────────────────────────────────────
def reply_sentiment(E_t, S_t, delta_t, theta):
    if E_t > 0.7:    return -0.25 + 0.05*np.random.randn()
    elif E_t > 0.4:  return -0.10 + 0.05*np.random.randn()
    else:            return  0.05 + 0.05*np.random.randn()

def reply_static_valence(E_t, S_t, delta_t, theta):
    return -0.15 + 0.08*np.random.randn()

def reply_nepa(E_t, S_t, delta_t, theta):
    if abs(delta_t) <= KAPPA * S_t:
        return -theta[0] * delta_t + 0.04*np.random.randn()
    else:
        return -theta[1]*np.sign(delta_t)*min(abs(delta_t),0.8) + 0.04*np.random.randn()

# ── Single episode ─────────────────────────────────────────────
def simulate(agent_fn, theta_init, label):
    E, S, delta = np.zeros(T), np.zeros(T), np.zeros(T)
    VI, F, AS   = np.zeros(T), np.zeros(T), np.zeros(T)
    reply       = np.zeros(T)
    theta       = theta_init.copy()
    E[0], S[0], F[0] = 0.25, 0.85, 0.25

    for t in range(1, T):
        incident = INCIDENT_MAG[INCIDENT_FRAMES.index(t)] if t in INCIDENT_FRAMES else 0.0
        r = agent_fn(E[t-1], S[t-1], delta[t-1], theta)
        reply[t] = r
        E[t] = np.clip(E[t-1]*np.exp(-LAMBDA) + ALPHA*r + incident + 0.03*np.random.randn(), 0, 1)
        regulation = max(0.0, -r)
        S[t] = np.clip(S[t-1]*np.exp(-MU) + BETA*regulation + 0.02*np.random.randn(), 0.02, 1.0)
        delta[t] = E[t] - E_desired[t]
        F[t] = 0.75*F[t-1] + 0.25*E[t]
        AS[t] = F[t] - F[t-1]
        w = E[max(0, t-WIN_VI):t+1]
        VI[t] = np.var(w) if len(w) > 1 else 0.0
        if label == "NEPA-HRI":
            td_err = -abs(delta[t]) + GAMMA_DISC*(-abs(delta[t])) - (-abs(delta[t-1]))
            theta[0] = np.clip(theta[0] + ETA_LEARN*td_err,     0.1, 1.5)
            theta[1] = np.clip(theta[1] + ETA_LEARN*td_err*1.5, 0.2, 2.0)

    escalations = [t for t in range(1,T) if abs(delta[t]) > KAPPA*S[t]]
    p_esc = len(escalations) / (T-1)
    t_recover_list, i = [], 0
    while i < len(escalations):
        esc_t, k = escalations[i], escalations[i]+1
        while k < T and abs(delta[k]) > KAPPA*S[k]:
            k += 1
        if k < T:
            t_recover_list.append(k - esc_t)
        while i < len(escalations) and escalations[i] < k:
            i += 1

    return dict(E=E, S=S, delta=delta, VI=VI, F=F, AS=AS, reply=reply,
                escalations=escalations,
                p_escalation=p_esc,
                mean_t_recover=float(np.mean(t_recover_list)) if t_recover_list else 0.0,
                total_affect_load=float(F.sum()),
                theta_final=theta.copy(), label=label)

# ── Batch simulator ────────────────────────────────────────────
def run_batch(agent_fn, theta_init, label, n=N_EPISODES):
    p_escs, t_recs, f_loads = [], [], []
    for _ in range(n):
        r = simulate(agent_fn, theta_init.copy(), label)
        p_escs.append(r["p_escalation"])
        t_recs.append(r["mean_t_recover"])
        f_loads.append(r["total_affect_load"])
    return dict(label=label,
                p_esc_mean=float(np.mean(p_escs)), p_esc_std=float(np.std(p_escs)),
                t_rec_mean=float(np.mean([x for x in t_recs if x>0] or [0])),
                f_load_mean=float(np.mean(f_loads)),
                p_escs=p_escs, t_recs=t_recs, f_loads=f_loads)

# ── Charts ─────────────────────────────────────────────────────
def save_meta(path, caption, desc):
    with open(path+".meta.json","w") as f:
        json.dump({"caption":caption,"description":desc},f)

def make_all_charts(episodes, batches, frames):
    # Chart A: Emotion trajectory
    figA = go.Figure()
    for ep in episodes:
        figA.add_trace(go.Scatter(x=frames, y=ep["E"], mode="lines",
            name=ep["label"], line=dict(width=2.5, color=COLORS[ep["label"]])))
    figA.add_trace(go.Scatter(x=frames, y=E_desired, mode="lines",
        name="E_desired", line=dict(width=1.5, dash="dot", color="#7E57C2")))
    for inc in INCIDENT_FRAMES:
        figA.add_vline(x=inc, line_dash="dash", line_color="orange", line_width=1.2)
    figA.update_layout(title="E(t) Emotion Trajectory — 3 Agents vs Desired Manifold",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(t=70,b=130,l=70,r=30), height=440)
    figA.update_xaxes(title_text="Frame t", tickvals=list(range(0,T,8)))
    figA.update_yaxes(title_text="E(t)", range=[0,1.05])
    figA.write_image("output/hri_emotion_trajectory.png")
    save_meta("output/hri_emotion_trajectory.png","E(t) Emotion Trajectory","3 agents vs desired manifold")

    # Chart B: Stability index
    figB = go.Figure()
    for ep in episodes:
        figB.add_trace(go.Scatter(x=frames, y=ep["S"], mode="lines",
            name=ep["label"], line=dict(width=2.5, color=COLORS[ep["label"]])))
    figB.add_hline(y=0.4, line_dash="dot", line_color="#888", line_width=1,
                   annotation_text="Low S threshold", annotation_position="right")
    for inc in INCIDENT_FRAMES:
        figB.add_vline(x=inc, line_dash="dash", line_color="orange", line_width=1.2)
    figB.update_layout(title="S(t) Stability Index — 3 Agents Over Time",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(t=70,b=130,l=70,r=30), height=420)
    figB.update_xaxes(title_text="Frame t", tickvals=list(range(0,T,8)))
    figB.update_yaxes(title_text="S(t)", range=[0,1.05])
    figB.write_image("output/hri_stability_index.png")
    save_meta("output/hri_stability_index.png","S(t) Stability Index","3 agents over 80 frames")

    # Chart C: Phase diagram
    figC = go.Figure()
    for ep in episodes:
        figC.add_trace(go.Scatter(x=ep["S"][1:], y=np.abs(ep["delta"][1:]),
            mode="markers", name=ep["label"],
            marker=dict(size=7, color=COLORS[ep["label"]], opacity=0.65)))
    s_line = np.linspace(0, 1.05, 200)
    figC.add_trace(go.Scatter(x=s_line, y=KAPPA*s_line, mode="lines",
        name=f"Envelope Omega (kappa={KAPPA})",
        line=dict(color="#7E57C2", width=2.5, dash="dash")))
    figC.add_annotation(x=0.72, y=0.30, text="Regime A — Stable",
        showarrow=False, font=dict(size=12, color="#2E7D32"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#2E7D32", borderwidth=1)
    figC.add_annotation(x=0.20, y=1.85, text="Regime B — Escalation",
        showarrow=False, font=dict(size=12, color="#C62828"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#C62828", borderwidth=1)
    figC.update_layout(title="Phase Diagram — Stability Envelope Omega",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(t=70,b=130,l=70,r=30), height=510)
    figC.update_xaxes(title_text="S(t) Stability Index",
                      range=[0,1.05], tickvals=[0,0.2,0.4,0.6,0.8,1.0])
    figC.update_yaxes(title_text="|delta(t)|", range=[0,2.6])
    figC.write_image("output/hri_phase_diagram.png")
    save_meta("output/hri_phase_diagram.png","Phase Diagram Omega","S_t vs |delta_t|")

    # Chart D: Batch comparison
    short = ["Sentiment","Static-Val","NEPA-HRI"]
    bar_c = [COLORS[b["label"]] for b in batches]
    figD = make_subplots(rows=1, cols=3,
        subplot_titles=["P(escalation)","T_recover (frames)","F_load (sum)"],
        horizontal_spacing=0.12)
    for col,(vals,errs) in enumerate([
        ([b["p_esc_mean"] for b in batches],[b["p_esc_std"] for b in batches]),
        ([b["t_rec_mean"] for b in batches],[0,0,0]),
        ([b["f_load_mean"] for b in batches],[0,0,0])],1):
        figD.add_trace(go.Bar(x=short, y=vals, marker_color=bar_c,
            error_y=dict(type="data",array=errs,visible=(col==1)),
            showlegend=False, width=0.5), row=1, col=col)
    figD.update_layout(title=f"HRI Batch Results — {N_EPISODES} Episodes per Agent",
        margin=dict(t=80,b=80,l=65,r=30), height=430)
    figD.update_traces(cliponaxis=False)
    figD.update_yaxes(title_text="Probability", row=1, col=1)
    figD.update_yaxes(title_text="Frames",      row=1, col=2)
    figD.update_yaxes(title_text="Load (sum)",  row=1, col=3)
    figD.write_image("output/hri_batch_comparison.png")
    save_meta("output/hri_batch_comparison.png",
              f"Batch Comparison N={N_EPISODES}","P(esc), T_recover, F_load")

# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    frames = np.arange(T)
    print("Running single episodes...")
    ep_sent = simulate(reply_sentiment,      [0.4,0.6], "Sentiment-Only")
    ep_val  = simulate(reply_static_valence, [0.4,0.6], "Static-Valence")
    ep_nepa = simulate(reply_nepa,           [0.4,0.8], "NEPA-HRI")
    episodes = [ep_sent, ep_val, ep_nepa]

    print(f"Running batch (N={N_EPISODES} per agent)...")
    batch_sent = run_batch(reply_sentiment,      [0.4,0.6], "Sentiment-Only")
    batch_val  = run_batch(reply_static_valence, [0.4,0.6], "Static-Valence")
    batch_nepa = run_batch(reply_nepa,           [0.4,0.8], "NEPA-HRI")
    batches = [batch_sent, batch_val, batch_nepa]

    print("\n" + "="*65)
    print(f"  {'Agent':<20} {'P(esc)±std':>14} {'T_rec':>8} {'F_load':>10}")
    print("-"*65)
    for b in batches:
        print(f"  {b['label']:<20} {b['p_esc_mean']:.3f}±{b['p_esc_std']:.3f}"
              f"  {b['t_rec_mean']:>8.2f}  {b['f_load_mean']:>10.2f}")
    print("="*65)

    with open("output/hri_batch_results.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent","p_esc_mean","p_esc_std","t_recover_mean","f_load_mean"])
        for b in batches:
            w.writerow([b["label"],round(b["p_esc_mean"],4),round(b["p_esc_std"],4),
                        round(b["t_rec_mean"],3),round(b["f_load_mean"],3)])

    print("\nGenerating charts...")
    make_all_charts(episodes, batches, frames)
    print("Done. Check output/")
