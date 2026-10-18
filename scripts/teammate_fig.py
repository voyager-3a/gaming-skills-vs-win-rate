"""팀원 지원 γ 그래프: γ별 승률·K/D 상관 (기본 설정, 분대 30%). results/main/fig6_teammate.png"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_outputs import INK2, KD, OUT, WR, plt  # noqa: E402
from sim.model import ROOT  # noqa: E402


def main():
    df = pd.read_csv(ROOT / "results" / "teammate" / "teammate.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, setting in zip(axes, ["기본", "분대 30%"]):
        d = df[df["setting"] == setting].sort_values("gamma")
        for col, color, name in (("wr", WR, "승률"), ("kd", KD, "K/D")):
            ax.plot(d["gamma"], d[f"{col}_1000"], color=color, lw=2, marker="o", ms=6, label=f"{name} 1000판")
            ax.plot(d["gamma"], d[f"{col}_100"], color=color, lw=1.4, ls="--", marker="o", ms=4, alpha=0.7,
                    label=f"{name} 100판")
            last = d.iloc[-1]
            dy = 0 if setting != "기본" else (7 if col == "wr" else -7)  # 기본은 두 값이 붙어 있어 위아래로 뗀다
            ax.annotate(f"{last[f'{col}_1000']:.3f}", (last["gamma"], last[f"{col}_1000"]), xytext=(6, dy),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK2)
        ax.set_xticks([0, 1, 2, 4])
        ax.set_xlim(-0.2, 4.6)
        ax.set_xlabel("팀원 지원 γ")
        ax.set_title(f"{setting}: γ별 실력과의 스피어만 상관", fontsize=10, loc="left")
    axes[0].set_ylabel("실력 s와의 스피어만 상관")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8.5, ncol=2)
    fig.text(0.01, 0.01, "기본 설정은 γ ≈ 3.4에서 1000판 상관이 비긴다. 분대 30%에서는 γ = 4에서도 K/D가 앞선다.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig6_teammate.png", dpi=150)
    print(OUT / "fig6_teammate.png")


if __name__ == "__main__":
    main()
