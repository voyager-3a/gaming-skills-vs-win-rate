"""판 진행 엔진. 여러 판을 numpy 배열로 동시에, 턴 단위로 갱신한다."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import StyleParams

REASON_TICKET, REASON_WIPE, REASON_TURNS = 0, 1, 2
REASON_NAMES = ("ticket", "wipe", "turn_limit")


@dataclass
class GameResults:
    """판 B개의 결과. 슬롯 배열은 (B, 2×team_size)."""
    kills: np.ndarray
    deaths: np.ndarray
    win: np.ndarray          # 슬롯별 승리 여부
    winner: np.ndarray       # (B,) 0 = A팀, 1 = B팀
    turns: np.ndarray        # (B,)
    reason: np.ndarray       # (B,) REASON_*
    tickets: np.ndarray      # (B, 2) 종료 시 티켓
    loss_kill: np.ndarray    # (B,) 데스로 깎인 티켓 합 (양 팀)
    loss_cap: np.ndarray     # (B,) 점령으로 깎인 티켓 합 (양 팀)
    winner_max_deficit: np.ndarray  # (B,) 이긴 팀이 한때 뒤졌던 최대 티켓 차


def _pick(weights: np.ndarray, u: np.ndarray) -> np.ndarray:
    """행마다 weights에 비례해 열 하나를 뽑는다."""
    cum = np.cumsum(weights, axis=1)
    target = u * cum[:, -1]
    idx = (cum <= target[:, None]).sum(axis=1)
    return np.minimum(idx, weights.shape[1] - 1)


def play_games(power: np.ndarray, style: np.ndarray, sp: StyleParams, sim: dict,
               rng: np.random.Generator) -> GameResults:
    """power = s + e, style은 슬롯별 값 (B, 2×team_size). 앞 절반이 A팀."""
    B, P = power.shape
    ts = P // 2
    T, d, c = float(sim["tickets"]), float(sim["ticket_per_death"]), float(sim["capture_coef"])
    K, sigma = int(sim["turn_limit"]), float(sim["sigma"])

    ew = sp.engage[style]
    pdie = sp.p_die[style]
    cw = sp.capture[style]

    rows = np.arange(B)
    lives = np.full((B, P), int(sim["lives"]), dtype=np.int16)
    kills = np.zeros((B, P), dtype=np.int32)
    deaths = np.zeros((B, P), dtype=np.int32)
    tickets = np.full((B, 2), T)
    done = np.zeros(B, dtype=bool)
    winner = np.full(B, -1, dtype=np.int8)
    turns = np.full(B, K, dtype=np.int32)
    reason = np.full(B, REASON_TURNS, dtype=np.int8)
    loss_kill = np.zeros(B)
    loss_cap = np.zeros(B)
    max_def = np.zeros((B, 2))  # 팀별 최대 열세 (상대 티켓 − 내 티켓)

    for t in range(K):
        act = ~done
        if not act.any():
            break
        alive = lives > 0

        # 3. 양 팀에서 살아 있는 유저 한 명씩, 교전 가중치에 비례
        w = ew * alive
        ia = _pick(w[:, :ts], rng.random(B))
        ib = ts + _pick(w[:, ts:], rng.random(B))

        # 4. 전투력 = s + e + ε, 높은 쪽이 이김. 진 쪽은 p_die로 사망
        pa = power[rows, ia] + sigma * rng.standard_normal(B)
        pb = power[rows, ib] + sigma * rng.standard_normal(B)
        a_wins = pa > pb
        los = np.where(a_wins, ib, ia)
        win_slot = np.where(a_wins, ia, ib)
        die = (rng.random(B) < pdie[rows, los]) & act

        # 5. 데스 처리
        r = rows[die]
        lives[r, los[die]] -= 1
        deaths[r, los[die]] += 1
        kills[r, win_slot[die]] += 1
        tickets[r, (los[die] >= ts).astype(np.int64)] -= d
        loss_kill[die] += d

        # 6. 점령: 압력이 큰 팀이 상대 티켓을 c × (압력 차)만큼 깎는다
        alive = lives > 0
        cwa = cw * alive
        diff = cwa[:, :ts].sum(axis=1) - cwa[:, ts:].sum(axis=1)
        amt = np.where(act, c * np.abs(diff), 0.0)
        tickets[:, 1] -= np.where(diff > 0, amt, 0.0)
        tickets[:, 0] -= np.where(diff < 0, amt, 0.0)
        loss_cap += np.where(diff != 0, amt, 0.0)

        gap = tickets[:, 1] - tickets[:, 0]
        np.maximum(max_def[:, 0], np.where(act, gap, -np.inf), out=max_def[:, 0])
        np.maximum(max_def[:, 1], np.where(act, -gap, -np.inf), out=max_def[:, 1])

        # 7. 종료: 티켓 ≤ 0 또는 전원 목숨 소진
        wiped_a = ~alive[:, :ts].any(axis=1)
        wiped_b = ~alive[:, ts:].any(axis=1)
        lose_a = (tickets[:, 0] <= 0) | wiped_a
        lose_b = (tickets[:, 1] <= 0) | wiped_b
        end = act & (lose_a | lose_b)
        if end.any():
            coin = rng.random(B) < 0.5
            both = lose_a & lose_b
            by_tickets = np.where(tickets[:, 0] == tickets[:, 1], coin,
                                  tickets[:, 1] > tickets[:, 0]).astype(np.int8)
            w_team = np.where(both, by_tickets, np.where(lose_a, 1, 0)).astype(np.int8)
            winner[end] = w_team[end]
            turns[end] = t + 1
            loser_wiped = np.where(w_team == 1, wiped_a, wiped_b)
            reason[end] = np.where(loser_wiped, REASON_WIPE, REASON_TICKET)[end]
            done |= end

    # 턴 상한: 티켓이 많은 쪽이 이김, 같으면 동전
    left = ~done
    if left.any():
        coin = rng.random(B) < 0.5
        by_tickets = np.where(tickets[:, 0] == tickets[:, 1], coin, tickets[:, 1] > tickets[:, 0])
        winner[left] = by_tickets[left].astype(np.int8)

    team = (np.arange(P) >= ts).astype(np.int8)
    win = team[None, :] == winner[:, None]
    return GameResults(kills=kills, deaths=deaths, win=win, winner=winner, turns=turns,
                       reason=reason, tickets=tickets, loss_kill=loss_kill, loss_cap=loss_cap,
                       winner_max_deficit=max_def[rows, winner])
