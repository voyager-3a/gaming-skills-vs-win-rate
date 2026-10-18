"""상관, 역전 지점, 부트스트랩, 스타일 표, 판 요약."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .engine import REASON_NAMES
from .model import STYLES
from .runner import RunResult


def correlations(res: RunResult, n_boot: int = 0, seed: int = 0) -> pd.DataFrame:
    """체크포인트별 spearman(s, 승률), spearman(s, K/D), 차이(승률 − K/D)와 부트스트랩 95% 구간."""
    s = res.users.s
    rng = np.random.default_rng(seed)
    rows = []
    for i, cp in enumerate(res.checkpoints):
        wr, kd = res.winrate(i), res.kd(i)
        r_wr, r_kd = spearmanr(s, wr)[0], spearmanr(s, kd)[0]
        row = {"games": cp, "rho_winrate": r_wr, "rho_kd": r_kd, "diff": r_wr - r_kd}
        if n_boot:
            idx = rng.integers(0, len(s), (n_boot, len(s)))
            rs, rw, rk = (rankdata(x[idx], axis=1) for x in (s, wr, kd))
            d = _row_pearson(rs, rw) - _row_pearson(rs, rk)
            row["diff_lo"], row["diff_hi"] = np.percentile(d, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


def _row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def crossover(corr: pd.DataFrame) -> float:
    """승률 상관이 K/D 상관을 처음 넘는 판수. log(판수) 기준 선형 보간. 못 넘으면 inf."""
    g, d = np.log(corr["games"].to_numpy(float)), corr["diff"].to_numpy()
    if d[0] > 0:
        return float(corr["games"].iloc[0])
    for i in range(1, len(d)):
        if d[i] > 0:
            x = g[i - 1] + (0 - d[i - 1]) * (g[i] - g[i - 1]) / (d[i] - d[i - 1])
            return float(np.exp(x))
    return float("inf")


def style_table(res: RunResult, i: int = -1) -> pd.DataFrame:
    """체크포인트 i에서 스타일별 평균 승률, 평균 K/D, 합산 K/D."""
    df = pd.DataFrame({"style": np.array(STYLES)[res.users.style], "winrate": res.winrate(i),
                       "kd": res.kd(i), "kills": res.kills[i], "deaths": res.deaths[i]})
    t = df.groupby("style").agg(users=("winrate", "size"), winrate=("winrate", "mean"),
                                kd_mean=("kd", "mean"), kills=("kills", "sum"),
                                deaths=("deaths", "sum"))
    t["kd_pooled"] = t["kills"] / t["deaths"]
    return t.reindex(list(STYLES)).drop(columns=["kills", "deaths"])


def last_man_table(res: RunResult) -> pd.DataFrame:
    """라스트 맨 승리 판마다 마지막 유저의 실력·스타일·장비."""
    g = res.games[res.games["last_man"]]
    u = g["last_man_user"].to_numpy()
    return pd.DataFrame({"game": g["game"].to_numpy(), "user": u, "s": res.users.s[u],
                         "style": np.array(STYLES)[res.users.style[u]], "e": res.users.e[u],
                         "turns": g["turns"].to_numpy()})


def game_summary(res: RunResult, turn_limit: int) -> dict:
    g = res.games
    reasons = g["reason"].value_counts(normalize=True).reindex(range(3), fill_value=0)
    total_loss = g["loss_kill"].sum() + g["loss_cap"].sum()
    return {
        "n_games": len(g),
        "end_reason": {REASON_NAMES[i]: float(reasons[i]) for i in range(3)},
        "ended_before_limit": float((g["turns"] < turn_limit).mean()),
        "turns_mean": float(g["turns"].mean()),
        "turns_p10_p50_p90": [float(x) for x in np.percentile(g["turns"], [10, 50, 90])],
        "ticket_loss_share_kill": float(g["loss_kill"].sum() / total_loss),
        "ticket_loss_share_capture": float(g["loss_cap"].sum() / total_loss),
        "kills_per_game_per_team": float((g["kills_a"] + g["kills_b"]).mean() / 2),
        "max_kills_one_game": int(g["max_kills"].max()),
        "max_deaths_one_game": int(g["max_deaths"].max()),
        "comeback_rate": float(g["comeback"].mean()),
        "last_man_rate": float(g["last_man"].mean()),
        "team_a_winrate": float((g["winner"] == 0).mean()),
    }
