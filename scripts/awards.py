"""재미용 '시상식': 현실 조합(combo_real) 실행의 1000판 시점 기준으로 상마다 1명.

순위는 '상위 %'(작을수록 위). 결과는 results/awards/AWARDS.md, awards.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.metrics import outnumbered_table  # noqa: E402
from sim.model import ROOT, STYLES, load_config  # noqa: E402
from sim.runner import run  # noqa: E402

VARIANT = "combo_real"
STYLE_KO = {"assault": "돌격", "allround": "올라운드", "sniper": "저격"}


def main():
    cfg = load_config()
    g, s = cfg["grid"], cfg["sim"]
    res = run(cfg, n_users=s["n_users"], max_games=s["max_games"], checkpoints=s["checkpoints"],
              q=g["base_q"], k=g["base_k"], seed=g["base_seeds"][0], sigma=g["base_sigma"], variant=VARIANT)
    u, i = res.users, len(res.checkpoints) - 1
    n = u.n
    top = lambda x: 100 * (1 - (rankdata(x) - 1) / (n - 1))  # noqa: E731  상위 % (최고 = 0%)
    df = pd.DataFrame({"user": np.arange(n), "s": u.s, "style": [STYLE_KO[STYLES[x]] for x in u.style],
                       "e": u.e, "squad": res.squad, "winrate": res.winrate(i), "kd": res.kd(i)})
    df["s_top"], df["wr_top"], df["kd_top"] = top(df["s"]), top(df["winrate"]), top(df["kd"])
    size = pd.Series(res.squad[res.squad >= 0]).value_counts()
    df["squad_info"] = [f"분대 {sq} ({size[sq]}명)" if sq >= 0 else "솔로" for sq in res.squad]

    # 업혀간 정도: 분대 동료 평균 실력 − 내 실력
    mate_gap = np.full(n, np.nan)
    for sq in size.index:
        m = np.flatnonzero(res.squad == sq)
        for j in m:
            mate_gap[j] = u.s[m[m != j]].mean() - u.s[j]
    on = outnumbered_table(res)
    carry_count = on["user"].value_counts()
    df["carry_count"] = df["user"].map(carry_count).fillna(0).astype(int)

    # 캐리왕: 동률이면 실력이 높은 쪽을 대표로, 나머지는 공동 수상으로 적는다
    best = carry_count[carry_count == carry_count.max()].index.to_numpy()
    best = best[np.argsort(-u.s[best])]
    king = int(best[0])
    co = (" (공동: " + ", ".join(f"유저 {b}" for b in best[1:]) + ")") if len(best) > 1 else ""
    picks = [
        ("진짜 최강", df["s"].idxmax(), "실력 최고"),
        ("진짜 최약", df["s"].idxmin(), "실력 최저"),
        ("K/D 뻥튀기상", (df["s_top"] - df["kd_top"]).idxmax(),
         "K/D 순위가 실력 순위보다 {:.1f}%p 높음", lambda r: r["s_top"] - r["kd_top"]),
        ("승률 뻥튀기상", (df["s_top"] - df["wr_top"]).idxmax(),
         "승률 순위가 실력 순위보다 {:.1f}%p 높음", lambda r: r["s_top"] - r["wr_top"]),
        ("저평가 고수상", np.minimum(df["wr_top"] - df["s_top"], df["kd_top"] - df["s_top"]).idxmax(),
         "승률·K/D 순위가 둘 다 실력 순위보다 {:.1f}%p 이상 낮음",
         lambda r: min(r["wr_top"] - r["s_top"], r["kd_top"] - r["s_top"])),
        ("업혀간 상", int(np.nanargmax(mate_gap)), "분대 동료 평균보다 실력이 {:.2f} 낮음",
         lambda r: mate_gap[int(r["user"])]),
        ("캐리왕", king, "인원 열세 역전승 캐리 {}회" + co, lambda r: int(r["carry_count"])),
    ]
    rows = []
    for p in picks:
        name, idx, note = p[0], p[1], p[2]
        r = df.loc[idx]
        if len(p) > 3:
            note = note.format(p[3](r))
        rows.append({"상": name, "유저": int(r["user"]), "실력 s": r["s"], "실력 상위": r["s_top"],
                     "스타일": r["style"], "장비 e": r["e"], "분대": r["squad_info"],
                     "승률": r["winrate"], "승률 상위": r["wr_top"], "K/D": r["kd"], "K/D 상위": r["kd_top"],
                     "선정 이유": note})
    aw = pd.DataFrame(rows)
    out = ROOT / "results" / "awards"
    out.mkdir(parents=True, exist_ok=True)
    aw.to_csv(out / "awards.csv", index=False)

    mates = []
    j = picks[5][1]
    for m in np.flatnonzero(res.squad == res.squad[j]):
        r = df.loc[m]
        mates.append(f"| {int(r['user'])}{' (수상자)' if m == j else ''} | {r['s']:+.2f} | {r['s_top']:.1f}% | "
                     f"{r['style']} | {r['winrate']:.3f} ({r['wr_top']:.1f}%) | {r['kd']:.2f} ({r['kd_top']:.1f}%) |")

    lines = ["# 시상식 (현실 조합, 1000판 시점)", "",
             "설정: 킬 스틸 0.3 + 실력 연동 이탈 0.2 + 분대 0.3(상관 0, p_play 0.5, b 0.5), 기본 설정, 시드 0, 유저 2000명.",
             "순위는 상위 %(0% = 1등). 장비 e: +1 사기 전차, −1 폐급, 0 보통.", "",
             "| 상 | 유저 | 실력 s (상위) | 스타일 | 장비 | 분대 | 승률 (상위) | K/D (상위) | 선정 이유 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in aw.iterrows():
        lines.append(f"| {r['상']} | {r['유저']} | {r['실력 s']:+.2f} ({r['실력 상위']:.1f}%) | {r['스타일']} | "
                     f"{r['장비 e']:+.0f} | {r['분대']} | {r['승률']:.3f} ({r['승률 상위']:.1f}%) | "
                     f"{r['K/D']:.2f} ({r['K/D 상위']:.1f}%) | {r['선정 이유']} |")
    lines += ["", "업혀간 상 수상자의 분대:", "",
              "| 유저 | 실력 s | 실력 상위 | 스타일 | 승률 (상위) | K/D (상위) |", "|---|---|---|---|---|---|", *mates, "",
              f"인원 열세 역전승 {len(on)}판, 캐리로 뽑힌 유저 {on['user'].nunique()}명. "
              f"캐리 횟수 상위: " + ", ".join(f"유저 {k} {v}회" for k, v in carry_count.head(5).items()), ""]
    (out / "AWARDS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
