"""인원 무관 버전(v1, results/v1_headcount_independent)과 인원 반영 버전(v2, results/) 비교표.
결과는 results/comparison/REPORT.md, comparison.csv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.model import ROOT  # noqa: E402

VERSIONS = {"인원 무관 (v1)": ROOT / "results" / "v1_headcount_independent", "인원 반영 (v2)": ROOT / "results"}


def load(base):
    runs = {r["name"]: r for r in json.loads((base / "main" / "runs.json").read_text())}
    sens = pd.read_csv(base / "sensitivity" / "sensitivity.csv").drop_duplicates("variant").set_index("variant")
    return runs, sens


def corr(base, name):
    return pd.read_csv(base / "main" / "runs" / name / "correlations.csv").set_index("games")


def steal_boundary(sens):
    xs = [0.0, 0.15, 0.3, 0.5]
    ds = [sens.loc[v, "diff_1000"] for v in ("base", "steal15", "steal30", "steal50")]
    for i in range(1, len(xs)):
        if np.sign(ds[i - 1]) != np.sign(ds[i]):
            return xs[i - 1] + (0 - ds[i - 1]) * (xs[i] - xs[i - 1]) / (ds[i] - ds[i - 1])
    return np.nan


def fmt_x(x):
    return "> 1000" if np.isinf(x) else f"{x:.0f}"


def main():
    rows = {}
    for ver, base in VERSIONS.items():
        runs, sens = load(base)
        c = corr(base, "base_seed0").loc[1000]
        s = runs["base_seed0"]["summary"]
        crossed = [n for n, r in runs.items() if n.startswith("q") or n.startswith("base_seed0")
                   if np.isfinite(r["crossover"])]
        d = lambda v: sens.loc[v, "diff_1000"]  # noqa: E731
        w = lambda v: sens.loc[v, "rho_winrate_1000"]  # noqa: E731
        b = steal_boundary(sens)
        r = {
            "기본 1000판 승률 / K/D 상관": f"{c['rho_winrate']:.3f} / {c['rho_kd']:.3f}",
            "기본 1000판 차이 [95%]": f"{c['diff']:+.3f} [{c['diff_lo']:+.3f}, {c['diff_hi']:+.3f}]",
            "앞서는 지표 (기본, 1000판)": "K/D" if c["diff"] < 0 else "승률",
            "기본 역전 지점 (시드 0/1/2)": " / ".join(fmt_x(runs[f"base_seed{i}"]["crossover"]) for i in range(3)),
            "결과 지도에서 1000판 안에 역전한 칸": f"{len(crossed)} / 27 ({', '.join(sorted(crossed)) or '-'})",
            "킬 스틸 경계 (비기는 p_steal)": f"≈ {b:.2f}" if np.isfinite(b) else "없음 (0~0.5에서 부호 안 바뀜)",
            "이탈 실력 무관 0.2: 승률 상관 / 차이": f"{w('leave_flat20'):.3f} / {d('leave_flat20'):+.3f}",
            "이탈 실력 연동 0.2: 승률 상관 / 차이": f"{w('leave_skill20'):.3f} / {d('leave_skill20'):+.3f}",
            "이탈 효과 (연동 0.2 − 기본, 차이)": f"{d('leave_skill20') - d('base'):+.3f}",
            "분대 0.3·상관 0: 승률 상관 / 차이": f"{w('squad30_r0'):.3f} / {d('squad30_r0'):+.3f}",
            "분대 0.6·상관 0: 승률 상관 / 차이": f"{w('squad60_r0'):.3f} / {d('squad60_r0'):+.3f}",
            "분대 효과 (0.3·상관 0 − 기본, 차이)": f"{d('squad30_r0') - d('base'):+.3f}",
            "현실 조합: 승률 / K/D / 차이": f"{w('combo_real'):.3f} / {sens.loc['combo_real', 'rho_kd_1000']:.3f} / "
                                     f"{d('combo_real'):+.3f}",
            "티켓 역전승 비율 (기본)": f"{s['comeback_rate']:.2%}",
            "인원 열세 역전승 비율 (기본)": f"{s['outnumbered_rate']:.3%}",
            "인원 열세 역전승 비율 (목숨 1)": f"{runs['one_life']['summary']['outnumbered_rate']:.2%}",
            "기습 발생 (턴당) / 성공": f"{s['ambush_per_turn']:.3f} / {s['ambush_success']:.3f}",
            "킬 몫 (기본)": f"{s['ticket_loss_share_kill']:.1%}",
            "평균 판 길이 (턴)": f"{s['turns_mean']:.1f}",
        }
        rows[ver] = r
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "comparison.csv")
    lines = ["# 인원 무관 버전 vs 인원 반영 버전", "",
             "v1: 기습 팀 동전, 기습 확률 고정 p_ambush. v2: 기습 팀 = 생존 인원 비율, "
             "기습 확률 = p_ambush × (1 + |A생존 − B생존| / 16), 상한 0.6. 기본 설정 시드 0 기준(역전 지점은 시드 3개).", "",
             "| 항목 | " + " | ".join(df.columns) + " |", "|---|" + "---|" * len(df.columns)]
    for idx, r in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(r.values) + " |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
