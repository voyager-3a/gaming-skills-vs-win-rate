"""추가 그래프 2장 (results/main/):
fig4_sensitivity.png  민감도 전체의 1000판 차이(승률 상관 − K/D 상관)와 95% 구간
fig5_style_slopes.png 스타일별 실력 → 1000판 승률 회귀선 (기본 설정, 현실 조합)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_outputs import AXIS, GRID, INK, INK2, KD, OUT, WR, plt  # noqa: E402
from run_sensitivity import label  # noqa: E402
from scatter import STYLE_COLORS  # noqa: E402
from sim.model import ROOT, load_config  # noqa: E402
from sim.runner import run  # noqa: E402

GROUP_KO = {"steal": "킬 스틸", "leave_flat": "이탈 (실력 무관)", "leave_skill": "이탈 (실력 연동)",
            "squad": "분대", "squad_bonus": "협동 보너스", "squad_part": "간헐적 분대", "combo": "조합",
            "ambush_team": "기습 팀 선정", "ambush_team_leave": "기습 팀 선정 + 이탈",
            "ambush_team_squad": "기습 팀 선정 + 분대"}


def short(cfg, v):
    if v == "base":
        return "기본"
    return label(cfg, v).replace(" (상관 0.0, p_play 1.0, b 0.0)", "").replace("p_play 1.0, ", "")


def fig_sensitivity(cfg):
    df = pd.read_csv(ROOT / "results" / "sensitivity" / "sensitivity.csv")
    rows, seen = [], set()
    for group, d in df.groupby("group", sort=False):
        rows.append(("header", GROUP_KO.get(group, group), None))
        for _, r in d.iterrows():
            rows.append(("row", short(cfg, r["variant"]), r))
    n = len(rows)
    fig, ax = plt.subplots(figsize=(10, 0.27 * n + 1.2))
    ax.grid(axis="y", visible=False)
    base = df[df["variant"] == "base"].iloc[0]["diff_1000"]
    for y, (kind, text, r) in enumerate(rows):
        if kind == "header":
            ax.text(-0.005, y, text, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=9, weight="bold", color=INK)
            continue
        ax.text(-0.012, y, text, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8, color=INK2)
        color = WR if r["diff_1000"] > 0 else KD
        ax.plot([r["diff_1000_lo"], r["diff_1000_hi"]], [y, y], color=color, lw=2, solid_capstyle="round")
        ax.plot(r["diff_1000"], y, "o", color=color, ms=6, mec="white", mew=1)
    ax.axvline(0, color=INK2, lw=1)
    ax.axvline(base, color=AXIS, lw=1, ls="--")
    ax.text(base, -1.2, f"기본 {base:+.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_ylim(n - 0.4, -1.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("1000판 승률 상관 − K/D 상관 (점: 추정, 선: 부트스트랩 95%)")
    xl = ax.get_xlim()
    ax.text(xl[1], -1.2, "승률 우세 →", ha="right", va="bottom", fontsize=8.5, color=WR)
    ax.text(xl[0], -1.2, "← K/D 우세", ha="left", va="bottom", fontsize=8.5, color=KD)
    ax.set_title("민감도 전체 (기본 설정 q 0.05 · k 0.5 · σ 1, 시드 0, 인원 반영 버전)", fontsize=10, loc="left",
                 pad=18)
    fig.subplots_adjust(left=0.42, right=0.98, top=1 - 0.75 / (0.27 * n + 1.2), bottom=0.6 / (0.27 * n + 1.2))
    fig.savefig(OUT / "fig4_sensitivity.png", dpi=150)
    plt.close(fig)


def fig_style_slopes(cfg):
    g, sm = cfg["grid"], cfg["sim"]
    cases = [(None, "기본 설정"), ("combo_real", "현실 조합")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (variant, title) in zip(axes, cases):
        res = run(cfg, n_users=sm["n_users"], max_games=sm["max_games"], checkpoints=sm["checkpoints"],
                  q=g["base_q"], k=g["base_k"], seed=g["base_seeds"][0], sigma=g["base_sigma"], variant=variant)
        i = len(res.checkpoints) - 1
        s, st, wr = res.users.s, res.users.style, res.winrate(i)
        xs = np.linspace(-3, 3, 50)
        for j, (lab, color) in enumerate(STYLE_COLORS):
            m = st == j
            ax.scatter(s[m], wr[m], s=6, color=color, alpha=0.18, linewidths=0)
            b1, b0 = np.polyfit(s[m], wr[m], 1)
            ax.plot(xs, b0 + b1 * xs, color=color, lw=2.2, label=f"{lab}  기울기 {b1:.4f}")
        ax.axhline(0.5, color=AXIS, lw=1)
        ax.axvline(0, color=GRID, lw=1)
        ax.set_xlim(-3.2, 3.2)
        ax.set_xlabel("실력 s")
        ax.set_title(f"{title}: 실력 → 1000판 승률 (스타일별 최소제곱 직선)", fontsize=10, loc="left")
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    axes[0].set_ylabel("1000판 승률")
    fig.text(0.01, 0.01, "보정은 실력 0에서 스타일별 승률을 50%로 맞춘다. 직선이 0 근처에서 만나고 양 끝에서 벌어지는 것이 기울기 차이다.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig5_style_slopes.png", dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    fig_sensitivity(cfg)
    fig_style_slopes(cfg)
    print(OUT / "fig4_sensitivity.png", OUT / "fig5_style_slopes.png")


if __name__ == "__main__":
    main()
