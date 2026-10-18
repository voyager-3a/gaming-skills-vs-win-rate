"""한 번의 실행: 유저 생성 → 스케줄 → 판 묶음 진행 → 체크포인트 누적 전적."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import play_games
from scipy.stats import norm

from .model import (Users, correlate_squad_skill, make_schedule, make_schedule_squads, make_squads,
                    make_users, style_params)


@dataclass
class RunResult:
    users: Users
    checkpoints: list[int]
    wins: np.ndarray     # (체크포인트, 유저) 누적 승
    kills: np.ndarray    # (체크포인트, 유저) 누적 킬
    deaths: np.ndarray   # (체크포인트, 유저) 누적 데스
    games: pd.DataFrame  # 판 단위 기록
    seconds: float
    squad: np.ndarray | None = None  # 유저별 분대 번호 (-1 = 솔로)
    schedule: np.ndarray | None = None  # (판, 32) 유저 배치
    in_squad: np.ndarray | None = None  # (판, 32) 분대로 참여 중인지

    def winrate(self, i: int) -> np.ndarray:
        return self.wins[i] / self.checkpoints[i]

    def kd(self, i: int) -> np.ndarray:
        return self.kills[i] / np.maximum(self.deaths[i], 1)


def run(cfg: dict, *, n_users: int, max_games: int, checkpoints: list[int], q: float, k: float,
        seed: int, sigma: float | None = None, variant: str | None = None,
        capture_override: dict | None = None, sim_overrides: dict | None = None,
        zero_power: bool = False, zero_style: bool = False) -> RunResult:
    """variant: 설정 파일 [variants]의 기본 설정 전용 변형 (파라미터와 보정값을 함께 바꾼다).
    zero_power: 판에서 s, e를 0으로 둔다 (s는 상관 계산용으로만 남김).
    zero_style: 전원을 올라운드로 둔다."""
    t0 = time.perf_counter()
    variant_sim = cfg.get("variants", {}).get(variant, {}) if variant else {}
    sim = {**cfg["sim"], **variant_sim, **(sim_overrides or {})}
    if sigma is not None:
        sim["sigma"] = sigma
    rng = np.random.default_rng(seed)
    steal_rng = np.random.default_rng([seed, 1])
    users = make_users(n_users, q, sim["equip_strength"], rng)
    if zero_style:
        users.style[:] = 1
    ts = int(sim["team_size"])

    # 분대 (별도 난수: 분대가 없으면 기본 실행과 같다)
    squad_frac = float(sim.get("squad_frac", 0.0))
    squad = np.full(n_users, -1)
    if squad_frac > 0:
        squad_rng = np.random.default_rng([seed, 3])
        squad = make_squads(n_users, squad_frac, squad_rng)
        users.s = correlate_squad_skill(users.s, squad, float(sim.get("squad_corr", 0.0)), squad_rng)

    sp = style_params(cfg, k, capture_override, variant)
    power_u = np.zeros(n_users) if zero_power else users.s + users.e

    # 이탈 확률: flat = base, skill = 2 × base × Φ(−s)
    p_leave_base = float(sim.get("p_leave", 0.0))
    leave_u, leave_rng = None, None
    if p_leave_base > 0:
        s_eff = np.zeros(n_users) if zero_power else users.s
        mode = sim.get("leave_mode", "flat")
        leave_u = (np.full(n_users, p_leave_base) if mode == "flat"
                   else 2 * p_leave_base * norm.cdf(-s_eff))
        leave_rng = np.random.default_rng([seed, 2])

    # 간헐적 분대(p_play)와 협동 보너스(b): 분대로 참여한 판의 교전에서 전투력 +b
    p_play = float(sim.get("squad_p_play", 1.0))
    bonus = float(sim.get("squad_bonus", 0.0))
    if squad_frac > 0:
        sched, in_squad = make_schedule_squads(n_users, max_games, ts, squad, rng, p_play,
                                               np.random.default_rng([seed, 4]))
    else:
        sched = make_schedule(n_users, max_games, ts, rng)
        in_squad = np.zeros(sched.shape, dtype=bool)
    per_round = n_users // (2 * ts)
    chunk = max(1, int(sim["chunk_games"]) // per_round) * per_round

    cps = list(checkpoints)
    cp_index = np.full(max(cps) + 2, -1)
    cp_index[cps] = np.arange(len(cps))
    tot = np.zeros((4, n_users), dtype=np.int64)  # 판수, 승, 킬, 데스
    snap = np.zeros((3, len(cps), n_users), dtype=np.int64)
    records = []

    for start in range(0, len(sched), chunk):
        su = sched[start:start + chunk]
        power_slot = power_u[su] + bonus * in_squad[start:start + chunk] if bonus else power_u[su]
        res = play_games(power_slot, users.style[su], sp, sim, rng, steal_rng,
                         None if leave_u is None else leave_u[su], leave_rng)
        for r0 in range(0, len(su), per_round):
            ru = su[r0:r0 + per_round].ravel()
            sl = slice(r0, r0 + per_round)
            tot[0, ru] += 1
            tot[1, ru] += res.win[sl].ravel()
            tot[2, ru] += res.kills[sl].ravel()
            tot[3, ru] += res.deaths[sl].ravel()
            g = tot[0, ru]
            ci = cp_index[np.minimum(g, len(cp_index) - 1)]
            hit = ci >= 0
            snap[:, ci[hit], ru[hit]] = tot[1:, ru[hit]]

        rows = np.arange(len(su))
        kmax, dmax = res.kills.argmax(1), res.deaths.argmax(1)
        win_kills = np.where(res.win, res.kills, -1)
        carry = win_kills.argmax(1)  # 이긴 팀 최다 킬 (동률이면 앞 슬롯)
        records.append(pd.DataFrame({
            "game": start + rows,
            "round": (start + rows) // per_round,
            "turns": res.turns,
            "reason": res.reason,
            "winner": res.winner,
            "tickets_a": res.tickets[:, 0],
            "tickets_b": res.tickets[:, 1],
            "loss_kill": res.loss_kill,
            "loss_cap": res.loss_cap,
            "kills_a": res.kills[:, :ts].sum(1),
            "kills_b": res.kills[:, ts:].sum(1),
            "deaths_a": res.deaths[:, :ts].sum(1),
            "deaths_b": res.deaths[:, ts:].sum(1),
            "max_kills": res.kills[rows, kmax],
            "max_kills_user": su[rows, kmax],
            "max_deaths": res.deaths[rows, dmax],
            "max_deaths_user": su[rows, dmax],
            "winner_max_deficit": res.winner_max_deficit,
            "winner_max_alive_deficit": res.winner_max_alive_deficit,
            "carry_user": su[rows, carry],
            "carry_kills": res.kills[rows, carry],
            "ambush": res.ambush,
            "ambush_win": res.ambush_win,
            "ambush_kill": res.ambush_kill,
            "leaves": res.leaves,
        }))

    games = pd.concat(records, ignore_index=True)
    games["comeback"] = games["winner_max_deficit"] >= sim["comeback_deficit"]
    games["outnumbered"] = games["winner_max_alive_deficit"] >= sim["outnumbered_deficit"]
    return RunResult(users=users, checkpoints=cps, wins=snap[0], kills=snap[1], deaths=snap[2],
                     games=games, seconds=time.perf_counter() - t0, squad=squad, schedule=sched,
                     in_squad=in_squad)
