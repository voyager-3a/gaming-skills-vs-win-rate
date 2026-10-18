"""본 실행 결과로 결과물 5종 + 변형 비교 + 이상치 요약을 만든다 (results/main/)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.model import ROOT, load_config  # noqa: E402

OUT = ROOT / "results" / "main"
RUNS = OUT / "runs"

# 차트 색 (dataviz 기본 팔레트): 범주 1·2번, 파랑 단색 램프, 잉크·격자
WR, KD = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#eeede9"

plt.rcParams.update({
    "font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"], "axes.unicode_minus": False,
    "font.weight": 500,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def load():
    runs = {r["name"]: r for r in json.loads((OUT / "runs.json").read_text())}
    corr = {n: pd.read_csv(RUNS / n / "correlations.csv") for n in runs}
    return runs, corr


def fmt_cross(x):
    return "> 1000" if not np.isfinite(x) else f"{x:.0f}"


def fig_base(corr, seeds):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, n in enumerate(seeds):
        c = corr[n]
        lw, alpha = (2, 1) if i == 0 else (1, 0.45)
        lab = (lambda t: t if i == 0 else None)
        a.plot(c["games"], c["rho_winrate"], color=WR, lw=lw, alpha=alpha, marker="o", ms=5 if i == 0 else 0,
               label=lab("승률"))
        a.plot(c["games"], c["rho_kd"], color=KD, lw=lw, alpha=alpha, marker="o", ms=5 if i == 0 else 0,
               label=lab("K/D"))
        b.errorbar(c["games"] * (1 + 0.04 * (i - 1)), c["diff"],
                   yerr=[c["diff"] - c["diff_lo"], c["diff_hi"] - c["diff"]],
                   color=INK2, lw=1.5, marker="o", ms=5, capsize=3, alpha=1 if i == 0 else 0.55,
                   label=f"시드 {n[-1]}")
    c0 = corr[seeds[0]]
    for col, color in (("rho_winrate", WR), ("rho_kd", KD)):
        last = c0.iloc[-1]
        a.annotate(f"{last[col]:.2f}", (last["games"], last[col]), xytext=(6, 0), textcoords="offset points",
                   va="center", color=INK2, fontsize=9)
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xticks(c0["games"], [str(g) for g in c0["games"]])
        ax.minorticks_off()
        ax.set_xlabel("유저당 판수")
    a.set_ylabel("실력 s와의 스피어만 상관")
    a.set_ylim(0, 1)
    a.set_title("기본 설정: 판수별 상관 (굵은 선 시드 0, 옅은 선 시드 1·2)", fontsize=10, loc="left")
    a.legend(frameon=False, loc="lower right")
    b.axhline(0, color=AXIS, lw=1)
    b.set_ylabel("승률 상관 − K/D 상관")
    b.set_title("차이와 부트스트랩 95% 구간 (0 위면 승률 우세)", fontsize=10, loc="left")
    b.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_base_correlations.png", dpi=150)
    plt.close(fig)


def result_map(runs, corr, g):
    rows = []
    for q in g["q"]:
        for k in g["k"]:
            for s in g["sigma"]:
                base = (q, k, s) == (g["base_q"], g["base_k"], g["base_sigma"])
                n = "base_seed0" if base else f"q{q}_k{k}_s{s}"
                c = corr[n].set_index("games")
                rows.append({"q": q, "k": k, "sigma": s, "run": n, "crossover": runs[n]["crossover"],
                             **{f"diff_{cp}": c.loc[cp, "diff"] for cp in c.index},
                             "diff_1000_lo": c.loc[1000, "diff_lo"], "diff_1000_hi": c.loc[1000, "diff_hi"],
                             "rho_winrate_1000": c.loc[1000, "rho_winrate"],
                             "rho_kd_1000": c.loc[1000, "rho_kd"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "result_map.csv", index=False)

    fig, axes = plt.subplots(1, len(g["sigma"]), figsize=(4 * len(g["sigma"]) + 1, 4.2))
    norm = lambda x: np.clip((np.log10(x) - 1) / 2, 0, 1)  # noqa: E731  10판 → 0, 1000판 → 1
    for ax, s in zip(axes, g["sigma"]):
        ax.grid(False)
        sub = df[df["sigma"] == s]
        for i, q in enumerate(g["q"]):
            for j, k in enumerate(g["k"]):
                r = sub[(sub["q"] == q) & (sub["k"] == k)].iloc[0]
                x = r["crossover"]
                if np.isfinite(x):
                    color = BLUES[int(round(norm(x) * (len(BLUES) - 1)))]
                    tc = SURF if norm(x) > 0.5 else INK
                else:
                    color, tc = NEUTRAL, INK
                ax.add_patch(plt.Rectangle((j, i), 0.96, 0.96, color=color, lw=0))
                ax.text(j + 0.48, i + 0.42, fmt_cross(x), ha="center", va="center", fontsize=12,
                        color=tc, weight="bold")
                ax.text(j + 0.48, i + 0.70, f"Δ1000 {r['diff_1000']:+.2f}", ha="center", va="center",
                        fontsize=8, color=tc)
        ax.set_xlim(0, len(g["k"]))
        ax.set_ylim(len(g["q"]), 0)
        ax.set_xticks(np.arange(len(g["k"])) + 0.48, [f"k = {k}" for k in g["k"]])
        ax.set_yticks(np.arange(len(g["q"])) + 0.48, [f"q = {q}" for q in g["q"]])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"σ = {s}", fontsize=11)
    fig.suptitle("결과 지도: 역전 지점(승률 상관이 K/D 상관을 처음 넘는 판수)\n"
                 "회색 = 1000판 안에 역전 없음, 파랑이 진할수록 늦게 역전. Δ1000 = 1000판에서 승률 상관 − K/D 상관",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "fig2_result_map.png", dpi=150)
    plt.close(fig)
    return df


INT_COLS = {"games", "user", "kills", "deaths", "game", "turns", "max_kills", "max_kills_user", "max_deaths",
            "max_deaths_user", "carry_kills", "max_alive_deficit", "winner_max_deficit", "users", "n",
            "crossover", "reason"}


def md_table(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)) and np.isinf(v):
                cells.append("> 1000")
            elif isinstance(v, (float, np.floating)) and c in INT_COLS:
                cells.append(f"{v:.0f}")
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.{floatfmt}f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    cfg = load_config()
    g = cfg["grid"]
    runs, corr = load()
    seeds = [f"base_seed{s}" for s in g["base_seeds"]]
    fig_base(corr, seeds)
    rmap = result_map(runs, corr, g)

    # 3. 스타일 표 (기본 설정, 시드 평균)
    st = pd.concat([pd.read_csv(RUNS / n / "styles.csv").assign(run=n) for n in seeds])
    st_mean = st.groupby("style", sort=False)[["users", "winrate", "kd_mean", "kd_pooled"]].mean()
    st_mean.to_csv(OUT / "styles_base.csv")
    st_k = []
    for k in g["k"]:
        n = "base_seed0" if k == g["base_k"] else f"q{g['base_q']}_k{k}_s{g['base_sigma']}"
        st_k.append(pd.read_csv(RUNS / n / "styles.csv").assign(k=k))
    st_k = pd.concat(st_k)

    # 4. 보정 보고
    cal = []
    for f in sorted((ROOT / "results" / "calibration").glob("*.json")):
        r = json.loads(f.read_text())
        cal.append({"key": r["key"], "assault_capture": r["capture"]["assault"],
                    "sniper_capture": r["capture"]["sniper"],
                    **{f"wr_{s}": r["after"]["winrate"][s] for s in ("assault", "allround", "sniper")},
                    **{f"kd_{s}": r["after"]["kd_pooled"][s] for s in ("assault", "allround", "sniper")},
                    "ok": r["within_tolerance"]})
    cal = pd.DataFrame(cal)
    cal.to_csv(OUT / "calibration_report.csv", index=False)

    # 5. 시간
    timing = json.loads((OUT / "timing.json").read_text())

    # 변형 비교 (모두 시드 0, 기본 설정)
    labels = {"base_seed0": "기본 (킬 62%)", "kill69": "c = 0.83 (킬 69%)", "kill35": "c = 12.504 (킬 35%)",
              "no_ambush": "기습 없음", "one_life": "목숨 1", "steal30": "킬 스틸 0.3"}
    var = []
    for n, lab in labels.items():
        c = corr[n].set_index("games")
        s = runs[n]["summary"]
        var.append({"변형": lab, "킬 몫": s["ticket_loss_share_kill"], "역전 지점": runs[n]["crossover"],
                    **{f"승률/KD {cp}판": f"{c.loc[cp, 'rho_winrate']:.3f} / {c.loc[cp, 'rho_kd']:.3f}"
                       for cp in (10, 100, 1000)},
                    "Δ1000 [95%]": f"{c.loc[1000, 'diff']:+.3f} [{c.loc[1000, 'diff_lo']:+.3f}, "
                                   f"{c.loc[1000, 'diff_hi']:+.3f}]",
                    "인원 열세 역전승": s["outnumbered_rate"], "티켓 역전승": s["comeback_rate"]})
    var = pd.DataFrame(var)
    var.to_csv(OUT / "variants.csv", index=False)

    # 이상치 요약 (기본 설정 시드 0, 상위 5)
    outl = pd.read_csv(RUNS / "base_seed0" / "outliers.csv")
    on_all = pd.concat([pd.read_csv(RUNS / n / "outnumbered.csv").assign(run=n) for n in runs
                        if (RUNS / n / "outnumbered.csv").stat().st_size > 60])

    rep = ["# 본 실행 결과", "",
           f"유저 {cfg['sim']['n_users']}명 × {cfg['sim']['max_games']}판, 34회 (격자 27 + 기본 시드 2 + 변형 5). "
           f"부트스트랩 {cfg['analysis']['bootstrap']}회.", "",
           "## 1. 기본 설정 판수별 상관", "", "![](fig1_base_correlations.png)", ""]
    for n in seeds:
        rep += [f"**{n}** (역전 지점 {fmt_cross(runs[n]['crossover'])})", "",
                md_table(corr[n].rename(columns={"diff_lo": "lo", "diff_hi": "hi"})), ""]
    rep += ["## 2. 결과 지도", "", "![](fig2_result_map.png)", "",
            md_table(rmap[["q", "k", "sigma", "crossover", "diff_100", "diff_1000", "diff_1000_lo",
                           "diff_1000_hi", "rho_winrate_1000", "rho_kd_1000"]]), "",
            "## 3. 스타일별 평균 승률·K/D", "", "기본 설정, 1000판, 시드 3개 평균:", "",
            md_table(st_mean.reset_index()), "", "k별 (q = 0.05, σ = 1):", "", md_table(st_k), "",
            "## 4. 보정 보고", "", md_table(cal, 4), "",
            "## 5. 실행 시간", "",
            f"코어 {timing['workers']}개 병렬. 보정 {timing.get('calibration_seconds', 0):.0f}초, "
            f"본 실행 {timing['main_seconds']:.0f}초, 합계 {timing['total_seconds']:.0f}초.", "",
            "## 변형 (기본 설정, 시드 0)", "", md_table(var), "",
            "## 이상치 (기본 설정 시드 0, 각 상위 5 — 전체 20건은 runs/base_seed0/outliers.csv)", ""]
    for name, d in outl.groupby("list", sort=False):
        d = d.dropna(axis=1, how="all").drop(columns="list").head(5)
        rep += [f"**{name}**", "", md_table(d), ""]
    rep += ["## 인원 열세 역전승 캐리 (실행별)", "",
            md_table(on_all.groupby("run").agg(n=("s", "size"), s_mean=("s", "mean"), e_mean=("e", "mean"),
                                                kills_mean=("carry_kills", "mean")).reset_index()), ""]
    (OUT / "REPORT.md").write_text("\n".join(rep))
    print((OUT / "REPORT.md").read_text()[:6000])


if __name__ == "__main__":
    main()
