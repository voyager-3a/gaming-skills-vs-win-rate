"""산점도: 실력 s vs 1000판 승률 / K/D. 기본 설정(시드 0)과 현실 조합(combo_real). results/main/fig3_scatter.png"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_outputs import AXIS, INK2, OUT, plt  # noqa: E402  (같은 차트 스타일)
from sim.model import load_config  # noqa: E402
from sim.runner import run  # noqa: E402

# dataviz 기본 팔레트 범주 1~3번 (산점도에서 모든 쌍 검증된 범위)
STYLE_COLORS = [("돌격", "#2a78d6"), ("올라운드", "#eb6834"), ("저격", "#1baf7a")]
CASES = [(None, "기본 설정"), ("combo_real", "현실 조합 (킬 스틸 0.3 + 이탈 0.2 + 분대 0.3, p_play 0.5, b 0.5)")]


def main():
    cfg = load_config()
    g, s = cfg["grid"], cfg["sim"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for row, (variant, title) in zip(axes, CASES):
        res = run(cfg, n_users=s["n_users"], max_games=s["max_games"], checkpoints=s["checkpoints"],
                  q=g["base_q"], k=g["base_k"], seed=g["base_seeds"][0], sigma=g["base_sigma"], variant=variant)
        u, i = res.users, len(res.checkpoints) - 1
        for ax, (y, name) in zip(row, [(res.winrate(i), "승률"), (res.kd(i), "K/D")]):
            for st, (lab, color) in enumerate(STYLE_COLORS):
                m = u.style == st
                ax.scatter(u.s[m], y[m], s=9, color=color, alpha=0.55, linewidths=0, label=lab)
            # 장비 ±1 유저는 테두리로 표시
            eq = u.e != 0
            ax.scatter(u.s[eq], y[eq], s=22, facecolors="none", edgecolors=INK2, linewidths=0.8,
                       label="장비 ±1")
            rho = spearmanr(u.s, y)[0]
            ax.set_title(f"{title}\n실력 vs {name}  (스피어만 {rho:.3f})", fontsize=9.5, loc="left")
            ax.set_xlabel("실력 s")
            ax.set_ylabel(f"1000판 {name}")
            if name == "K/D":
                ax.set_yscale("log")
                ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20) if y.min() / 1.5 <= t <= y.max() * 1.5]
                ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
                ax.minorticks_off()
                ax.axhline(1, color=AXIS, lw=1)
            else:
                ax.axhline(0.5, color=AXIS, lw=1)
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=9, markerscale=1.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_scatter.png", dpi=150)
    print(OUT / "fig3_scatter.png")


if __name__ == "__main__":
    main()
