"""Car-following metrics (model vs expert) from follow_log.py output + two figures.
usage: follow_analyze.py MODEL.jsonl EXPERT.jsonl OUTDIR
"""
import json
import os
import sys

import numpy as np

DT = 0.1
GAP0, B = 4.0, 2.5          # expert: stop 4 m behind the lead, comfort decel 2.5 m/s^2
MODEL_LABEL = os.environ.get("MODEL_LABEL", "KeiPilot v0.7")      # legend / title of the model's curves


def load(p):
    return [json.loads(l) for l in open(p)]


def arr(r):
    R = r["rows"]
    t = np.array([x[0] for x in R]); v = np.array([x[1] for x in R])
    reason = np.array([x[2] for x in R]); gap = np.array([np.nan if x[3] is None else x[3] for x in R])
    lv = np.array([x[4] for x in R]); lo = np.array([x[5] for x in R])
    tm = np.array([np.nan if x[6] is None else x[6] for x in R]); te = np.array([x[7] for x in R])
    act = np.array([x[8] for x in R])
    return dict(t=t, v=v, reason=reason, gap=gap, lv=lv, lo=lo, tm=tm, te=te, act=act)


def metrics(runs):
    out = {"routes": len(runs), "status": {}, "km": 0.0}
    TH, SG, TTC, ACC, JERK, lags, starts, gaps_all, trk, bias = [], [], [], [], [], [], [], [], [], []
    follow_frames = total = 0
    switches = 0
    for r in runs:
        out["status"][r["status"]] = out["status"].get(r["status"], 0) + 1
        d = arr(r)
        v, g, lv, lo = d["v"], d["gap"], d["lv"], d["lo"]
        out["km"] += float(np.sum(v) * DT / 1000)
        a = np.gradient(v, DT)
        ACC.append(a); JERK.append(np.gradient(a, DT))
        veh = (lo >= 0) & np.isfinite(g) & (g < 40) & (lv > -0.5)
        total += len(v); follow_frames += int(veh.sum())
        mv = veh & (v > 2.0)
        TH.append(g[mv] / v[mv])
        st = veh & (v < 0.05) & (lv < 0.05) & (g < 15)
        SG.append(g[st])
        cl = veh & (v - lv > 0.3)
        TTC.append(g[cl] / (v[cl] - lv[cl]))
        gaps_all.append(g[veh])
        if np.isfinite(d["tm"]).any():
            fm = veh & (d["reason"] == "vehicle")
            trk.append(np.abs(d["tm"][fm] - d["te"][fm])); bias.append(d["tm"][fm] - d["te"][fm])
        thr, brk = d["act"][:, 1] > 0.05, d["act"][:, 2] > 0.05
        mode = np.where(brk, -1, np.where(thr, 1, 0))
        nz = mode[mode != 0]
        switches += int(np.sum(nz[1:] != nz[:-1]))
        # reaction to a braking lead: same lead id for 2 s, lead speed drops >1.5 m/s within 1 s while gap < 25 m
        k = 10
        i = 0
        while i < len(v) - 40:
            if veh[i] and lo[i] == lo[i + k] and g[i] < 25 and lv[i] - lv[i + k] > 1.5 and v[i] > 1.5:
                j = next((j for j in range(i, min(i + 40, len(v))) if a[j] < -0.5), None)
                if j is not None:
                    lags.append((j - i) * DT)
                i += 40
            else:
                i += 1
        # start-up: ego stopped behind a stopped lead (gap < 10 m); lead pulls away -> how long until ego moves
        i = 1
        while i < len(v) - 50:
            if veh[i] and v[i] < 0.05 and g[i] < 10 and lv[i - 1] < 0.5 <= lv[i]:
                j = next((j for j in range(i, min(i + 50, len(v))) if v[j] > 0.5), None)
                if j is not None:
                    starts.append((j - i) * DT)
                i += 50
            else:
                i += 1
    cat = lambda L: np.concatenate(L) if L else np.array([])
    TH, SG, TTC, A, J, G = map(cat, (TH, SG, TTC, ACC, JERK, gaps_all))
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")
    out.update({
        "follow_share": follow_frames / max(total, 1),
        "headway_s": {"p5": q(TH, 5), "p25": q(TH, 25), "median": q(TH, 50), "lt_1s": float(np.mean(TH < 1.0)) if len(TH) else 0},
        "standstill_gap_m": {"min": float(SG.min()) if len(SG) else None, "p5": q(SG, 5), "median": q(SG, 50), "p95": q(SG, 95), "n": int(len(SG))},
        "min_gap_m": float(G.min()) if len(G) else None,
        "ttc_s": {"min": float(TTC.min()) if len(TTC) else None, "p1": q(TTC, 1), "lt_2s": float(np.mean(TTC < 2.0)) if len(TTC) else 0},
        "decel_p99_ms2": -q(A, 1), "accel_p99_ms2": q(A, 99), "jerk_rms": float(np.sqrt(np.mean(J ** 2))),
        "pedal_switches_per_km": switches / max(out["km"], 1e-6),
        "reaction_lag_s": {"median": q(np.array(lags), 50), "p90": q(np.array(lags), 90), "n": len(lags)},
        "startup_delay_s": {"median": q(np.array(starts), 50), "p90": q(np.array(starts), 90), "n": len(starts)},
    })
    if trk:
        T, Bi = cat(trk), cat(bias)
        out["target_vs_expert_when_following"] = {"mae_ms": float(T.mean()), "bias_ms": float(Bi.mean()),
                                                  "abs_err_p90": q(T, 90)}
    return out


def gap_by_speed(runs, edges):
    v_all, g_all = [], []
    for r in runs:
        d = arr(r)
        veh = (d["lo"] >= 0) & np.isfinite(d["gap"]) & (d["gap"] < 40) & (d["lv"] > -0.5)
        v_all.append(d["v"][veh]); g_all.append(d["gap"][veh])
    v, g = np.concatenate(v_all), np.concatenate(g_all)
    mids, med, p25, p75 = [], [], [], []
    for lo_, hi in zip(edges[:-1], edges[1:]):
        m = (v >= lo_) & (v < hi)
        if m.sum() >= 30:
            mids.append((lo_ + hi) / 2); med.append(np.median(g[m])); p25.append(np.percentile(g[m], 25))
            p75.append(np.percentile(g[m], 75))
    return np.array(mids), np.array(med), np.array(p25), np.array(p75)


def figures(model, expert, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"model": "#2a78d6", "expert": "#eb6834"}
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                         "axes.spines.top": False, "axes.spines.right": False})

    # 1) gap vs speed while following a vehicle (median + interquartile band), expert rule as reference
    edges = np.arange(0, 11.5, 0.5)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, runs in (("expert", expert), ("model", model)):
        x, m, a, b = gap_by_speed(runs, edges)
        ax.fill_between(x * 3.6, a, b, color=C[name], alpha=0.12, linewidth=0)
        ax.plot(x * 3.6, m, color=C[name], linewidth=2, label=MODEL_LABEL if name == "model" else "Expert")
    vv = np.linspace(0, 10.5, 50)
    ax.plot(vv * 3.6, GAP0 + vv ** 2 / (2 * B), color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("expert rule: stop 4 m behind\n(4 + v²/2b, b = 2.5 m/s²)", (34, GAP0 + (34 / 3.6) ** 2 / (2 * B)),
                xytext=(-150, 8), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xlabel("ego speed (km/h)"); ax.set_ylabel("gap to the vehicle ahead (m)")
    ax.set_title(f"Gap kept behind the vehicle ahead, by speed ({os.environ.get('FOLLOW_LABEL', 'test --dense, 20 routes')})\nline = median, band = 25–75 %",
                 loc="left", color=INK, fontsize=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_xlim(0, 44); ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{outdir}/gap_vs_speed.png", dpi=150); plt.close(fig)

    # 2) one following episode from the model run: speeds (top) and gap (bottom), shared time axis
    best = None
    for r in model:
        d = arr(r)
        veh = (d["lo"] >= 0) & np.isfinite(d["gap"]) & (d["gap"] < 30) & (d["lv"] > -0.5)
        # longest stretch behind the same lead vehicle with some speed variation
        i = 0
        while i < len(veh):
            if veh[i]:
                j = i
                while j + 1 < len(veh) and veh[j + 1] and d["lo"][j + 1] == d["lo"][i]:
                    j += 1
                score = (j - i) * (np.ptp(d["lv"][i:j + 1]) > 3)
                if best is None or score > best[0]:
                    best = (score, r, i, j)
                i = j + 1
            else:
                i += 1
    _, r, i, j = best
    d = arr(r)
    sl = slice(max(i - 30, 0), min(j + 30, len(d["t"])))
    if len(sys.argv) > 5:                      # explicit time window
        t0_, t1_ = float(sys.argv[4]), float(sys.argv[5])
        sl = np.where((d["t"] >= t0_) & (d["t"] <= t1_))[0]
        sl = slice(sl[0], sl[-1] + 1)
    t = d["t"][sl]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(t, d["v"][sl] * 3.6, color=C["model"], linewidth=2)
    lead = np.where(d["lo"][sl] == d["lo"][i], d["lv"][sl], np.nan)
    a1.plot(t, lead * 3.6, color=INK2, linewidth=1.5, linestyle=(0, (4, 3)))
    a1.plot(t, d["tm"][sl] * 3.6, color=C["model"], linewidth=1, alpha=0.45)
    a1.plot(t, d["te"][sl] * 3.6, color=C["expert"], linewidth=1.5)
    for y, lab, col in ((d["v"][sl][-1] * 3.6, "ego speed", INK), (d["te"][sl][-1] * 3.6, "expert target", INK)):
        pass
    a1.set_ylabel("km/h")
    a1.legend(["ego speed (KeiPilot driving)", "speed of the vehicle ahead", "KeiPilot target speed",
               "expert target speed"], frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    a1.grid(axis="y", color=GRID, linewidth=0.8)
    fig.suptitle(f"Following one vehicle: town {r['town']} ep {r['episode']} ({MODEL_LABEL} driving)", x=0.01,
                 ha="left", color=INK, fontsize=12)
    a2.plot(t, d["gap"][sl], color=C["model"], linewidth=2)
    a2.axhline(GAP0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    a2.annotate("4 m (expert standstill gap)", (t[-1], GAP0), xytext=(0, -12), textcoords="offset points",
                color=INK2, fontsize=9, ha="right")
    a2.set_ylabel("gap (m)"); a2.set_xlabel("time (s)"); a2.set_ylim(0, None)
    a2.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout(); fig.savefig(f"{outdir}/follow_example.png", dpi=150); plt.close(fig)
    return r["town"], r["episode"], float(t[0]), float(t[-1])


if __name__ == "__main__":
    model, expert = load(sys.argv[1]), load(sys.argv[2])
    out = {"model": metrics(model), "expert": metrics(expert)}
    out["example"] = figures(model, expert, sys.argv[3])
    json.dump(out, open(f"{sys.argv[3]}/follow_metrics.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
