"""팀원 지원(γ) 민감도 보고. scripts/run_sensitivity.py로 실행한 결과를 읽고, 판 단위 확인 지표를 위해
같은 시드로 다시 돌린다(결과는 결정적이라 같다).

- 10판·100판·1000판 승률·K/D 상관, 1000판 차이와 95% 구간, 역전 지점, 재보정 여부
- 판 단위 확인: 그 판의 (킬 − 데스) vs 그 판 팀원 15명의 평균 실력 s, 피어슨·스피어만 (모든 판·슬롯)
- 분대 실행 추가 확인: 분대원의 1000판 K/D vs 분대 동료 평균 실력 (실력 영향을 뺀 편상관)
결과는 results/teammate/REPORT.md, teammate.csv.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.model import ROOT, load_config  # noqa: E402
from sim.runner import run  # noqa: E402

RUNS = ROOT / "results" / "main" / "runs"
CASES = [("기본", "base", 0.0), ("기본", "gamma1", 1.0), ("기본", "gamma2", 2.0), ("기본", "gamma4", 4.0),
         ("분대 30%", "squad30_r0", 0.0), ("분대 30%", "squad30_g1", 1.0), ("분대 30%", "squad30_g2", 2.0),
         ("분대 30%", "squad30_g4", 4.0), ("현실 조합", "combo_real", 0.0), ("현실 조합", "combo_real_g2", 2.0)]


def _check(variant):
    cfg = load_config()
    g, sm = cfg["grid"], cfg["sim"]
    res = run(cfg, n_users=sm["n_users"], max_games=sm["max_games"], checkpoints=sm["checkpoints"],
              q=g["base_q"], k=g["base_k"], seed=g["base_seeds"][0], sigma=g["base_sigma"],
              variant=None if variant == "base" else variant, store_slots=True)
    s = res.users.s[res.schedule]
    team = np.repeat([0, 1], 16)
    tsum = np.stack([s[:, :16].sum(1), s[:, 16:].sum(1)], 1)[:, team]
    mates = (tsum - s) / 15
    net = res.slot_net.astype(float)
    out = {"variant": variant,
           "game_pearson": float(np.corrcoef(net.ravel(), mates.ravel())[0, 1]),
           "game_spearman": float(stats.spearmanr(net.ravel(), mates.ravel())[0]),
           "game_pearson_given_s": partial(net.ravel(), mates.ravel(), s.ravel())}
    sq = res.squad
    if (sq >= 0).any():
        i = len(res.checkpoints) - 1
        kd = np.log(res.kd(i) + 1e-3)
        mem = np.flatnonzero(sq >= 0)
        mate_s = np.array([res.users.s[(sq == sq[j]) & (np.arange(len(sq)) != j)].mean() for j in mem])
        out["squad_kd_vs_mates_partial"] = partial(kd[mem], mate_s, res.users.s[mem])
        out["squad_wr_vs_mates_partial"] = partial(res.winrate(i)[mem], mate_s, res.users.s[mem])
    return out


def partial(y, x, z):
    """z(자기 실력)를 통제한 y와 x의 편상관."""
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    return float(np.corrcoef(ry, rx)[0, 1])


def main():
    cfg = load_config()
    checks = json.loads((ROOT / "results" / "sensitivity" / "calibration_check.json").read_text())
    with Pool(4) as pool:
        game = {r["variant"]: r for r in pool.map(_check, [v for _, v, _ in CASES])}
    rows = []
    for setting, v, gamma in CASES:
        name = "base_seed0" if v == "base" else v
        c = pd.read_csv(RUNS / name / "correlations.csv").set_index("games")
        info = json.loads((RUNS / name / "summary.json").read_text())
        rec = checks.get(v, {}).get("recalibrated", "-") if v != "base" else "-"
        rows.append({"setting": setting, "variant": v, "gamma": gamma,
                     **{f"wr_{n}": c.loc[n, "rho_winrate"] for n in (10, 100, 1000)},
                     **{f"kd_{n}": c.loc[n, "rho_kd"] for n in (10, 100, 1000)},
                     "diff_1000": c.loc[1000, "diff"], "lo": c.loc[1000, "diff_lo"], "hi": c.loc[1000, "diff_hi"],
                     "crossover": info["crossover"], "recalibrated": rec, **game[v]})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "teammate"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "teammate.csv", index=False)
    fx = lambda x: "> 1000" if np.isinf(x) else f"{x:.0f}"  # noqa: E731
    rc = lambda x: {True: "예", False: "아니오"}.get(x, "-")  # noqa: E731
    lines = ["# 팀원 지원 γ 민감도 (기본 설정, 시드 0, 인원 반영 버전)", "",
             "전투력 += γ × (본인을 뺀 살아 있는 같은 팀원의 실력+장비 평균). 팀원 15명 평균의 표준편차 ≈ 0.26.", "",
             "## 상관 (승률 / K/D)", "",
             "| 설정 | γ | 10판 | 100판 | 1000판 | 1000판 차이 [95%] | 역전 지점 | 재보정 |", "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['setting']} | {r['gamma']:g} | {r['wr_10']:.3f} / {r['kd_10']:.3f} | "
                     f"{r['wr_100']:.3f} / {r['kd_100']:.3f} | {r['wr_1000']:.3f} / {r['kd_1000']:.3f} | "
                     f"{r['diff_1000']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | {fx(r['crossover'])} | "
                     f"{rc(r['recalibrated'])} |")
    lines += ["", "## 확인: 판 단위 (그 판의 킬 − 데스) vs 그 판 팀원 평균 실력", "",
              "모든 판·슬롯(약 200만 개)을 모은 상관. 편상관은 자기 실력 s를 통제한 값.", "",
              "| 설정 | γ | 피어슨 | 스피어만 | 편상관 (자기 실력 통제) | 분대원 1000판 K/D vs 분대 동료 실력 (편상관) | 분대원 1000판 승률 vs 분대 동료 실력 (편상관) |",
              "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        sk = r.get("squad_kd_vs_mates_partial")
        sw = r.get("squad_wr_vs_mates_partial")
        lines.append(f"| {r['setting']} | {r['gamma']:g} | {r['game_pearson']:+.3f} | {r['game_spearman']:+.3f} | "
                     f"{r['game_pearson_given_s']:+.3f} | {'-' if pd.isna(sk) else f'{sk:+.3f}'} | "
                     f"{'-' if pd.isna(sw) else f'{sw:+.3f}'} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
