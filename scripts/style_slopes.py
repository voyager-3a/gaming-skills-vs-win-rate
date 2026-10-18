"""스타일별로 실력이 승률에 주는 영향: 1000판 시점, 기본 설정(시드 0)과 현실 조합(combo_real).

① 승률 ~ 실력 s 최소제곱 기울기와 95% 구간, 세 스타일 기울기가 같은지 F검정(스타일 × 실력 상호작용)
② 스타일 내부 스피어만 상관(실력 vs 승률, 실력 vs K/D)과 부트스트랩 95% 구간
결과는 results/style_slopes/REPORT.md, style_slopes.csv. 보정은 설정 파일 값 그대로 쓴다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.model import ROOT, STYLES, load_config  # noqa: E402
from sim.runner import run  # noqa: E402

CASES = [(None, "기본 설정"), ("combo_real", "현실 조합")]
STYLE_KO = {"assault": "돌격", "allround": "올라운드", "sniper": "저격"}
N_BOOT = 2000


def ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - 2
    se = np.sqrt(resid @ resid / dof * np.linalg.inv(X.T @ X)[1, 1])
    t = stats.t.ppf(0.975, dof)
    return beta[1], se, beta[1] - t * se, beta[1] + t * se, resid @ resid


def slope_equality(s, y, style):
    """기울기 공통(스타일별 절편만) vs 스타일별 기울기: 중첩 모형 F검정."""
    D = np.column_stack([(style == i).astype(float) for i in range(3)])
    X0 = np.column_stack([D, s])
    X1 = np.column_stack([D, D * s[:, None]])
    rss = lambda X: np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2)  # noqa: E731
    r0, r1 = rss(X0), rss(X1)
    df1, df2 = 2, len(y) - X1.shape[1]
    F = (r0 - r1) / df1 / (r1 / df2)
    return F, stats.f.sf(F, df1, df2)


def boot_spearman(x, y, rng):
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    rx, ry = stats.rankdata(x[idx], axis=1), stats.rankdata(y[idx], axis=1)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    r = (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    return np.percentile(r, [2.5, 97.5])


def main():
    cfg = load_config()
    g, sm = cfg["grid"], cfg["sim"]
    rng = np.random.default_rng(0)
    rows, tests = [], []
    for variant, name in CASES:
        res = run(cfg, n_users=sm["n_users"], max_games=sm["max_games"], checkpoints=sm["checkpoints"],
                  q=g["base_q"], k=g["base_k"], seed=g["base_seeds"][0], sigma=g["base_sigma"], variant=variant)
        i = len(res.checkpoints) - 1
        s, st, wr, kd = res.users.s, res.users.style, res.winrate(i), res.kd(i)
        groups = [(STYLE_KO[STYLES[j]], st == j) for j in range(3)] + [("전체", np.ones_like(s, bool))]
        for lab, m in groups:
            b, se, lo, hi, _ = ols(s[m], wr[m])
            rw, rk = stats.spearmanr(s[m], wr[m])[0], stats.spearmanr(s[m], kd[m])[0]
            (rwl, rwh), (rkl, rkh) = boot_spearman(s[m], wr[m], rng), boot_spearman(s[m], kd[m], rng)
            rows.append({"case": name, "style": lab, "n": int(m.sum()), "slope": b, "slope_se": se,
                         "slope_lo": lo, "slope_hi": hi, "rho_winrate": rw, "rho_winrate_lo": rwl,
                         "rho_winrate_hi": rwh, "rho_kd": rk, "rho_kd_lo": rkl, "rho_kd_hi": rkh,
                         "s_sd": s[m].std()})
        F, p = slope_equality(s, wr, st)
        tests.append({"case": name, "F": F, "p": p})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "style_slopes"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "style_slopes.csv", index=False)

    lines = ["# 스타일별 실력 → 승률 영향 (1000판, 시드 0)", "",
             "기울기 = 실력 s가 1 오를 때 1000판 승률 변화(최소제곱). 상관은 스타일 내부 스피어만, 구간은 부트스트랩 95%.",
             "보정은 설정 파일 값(인원 반영 버전) 그대로.", ""]
    for name, d in df.groupby("case", sort=False):
        t = next(x for x in tests if x["case"] == name)
        lines += [f"## {name}", "",
                  "| 스타일 | 유저 | ① 승률 기울기 [95%] | ② 실력–승률 상관 [95%] | ② 실력–K/D 상관 [95%] |",
                  "|---|---|---|---|---|"]
        for _, r in d.iterrows():
            lines.append(f"| {r['style']} | {r['n']} | {r['slope']:.4f} [{r['slope_lo']:.4f}, {r['slope_hi']:.4f}] | "
                         f"{r['rho_winrate']:.3f} [{r['rho_winrate_lo']:.3f}, {r['rho_winrate_hi']:.3f}] | "
                         f"{r['rho_kd']:.3f} [{r['rho_kd_lo']:.3f}, {r['rho_kd_hi']:.3f}] |")
        lines += ["", f"세 스타일 기울기가 같은지 F검정: F = {t['F']:.2f}, p = {t['p']:.3g}", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
