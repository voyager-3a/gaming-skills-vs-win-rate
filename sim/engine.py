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
    winner_max_deficit: np.ndarray        # (B,) 이긴 팀이 한때 뒤졌던 최대 티켓 차
    winner_max_alive_deficit: np.ndarray  # (B,) 이긴 팀이 한때 뒤졌던 최대 생존 인원 차
    ambush: np.ndarray       # (B,) 기습 횟수
    ambush_win: np.ndarray   # (B,) 기습한 쪽이 교전에서 이긴 횟수
    ambush_kill: np.ndarray  # (B,) 기습으로 나온 킬
    leaves: np.ndarray       # (B,) 이탈 횟수


def _pick(weights: np.ndarray, u: np.ndarray) -> np.ndarray:
    """행마다 weights에 비례해 열 하나를 뽑는다."""
    cum = np.cumsum(weights, axis=1)
    target = u * cum[:, -1]
    idx = (cum <= target[:, None]).sum(axis=1)
    return np.minimum(idx, weights.shape[1] - 1)


def _pick_ambush(aw: np.ndarray, tw: np.ndarray, u: np.ndarray, rows: np.ndarray):
    """한 팀이 기습할 때 기습조 2명(a1, a2)과, 당할 때 표적 1명(tg)을 뽑는다 (팀 내 슬롯)."""
    a1 = _pick(aw, u[0])
    aw2 = aw.copy()
    aw2[rows, a1] = 0.0
    has2 = aw2.sum(axis=1) > 0
    a2 = _pick(aw2, u[1])
    tg = _pick(tw, u[2])
    return a1, a2, has2, tg


def play_games(power: np.ndarray, style: np.ndarray, sp: StyleParams, sim: dict,
               rng: np.random.Generator, steal_rng: np.random.Generator | None = None,
               p_leave: np.ndarray | None = None, leave_rng: np.random.Generator | None = None) -> GameResults:
    """power = s + e, style은 슬롯별 값 (B, 2×team_size). 앞 절반이 A팀.
    킬 스틸은 steal_rng로만 뽑으므로, 같은 rng면 p_steal과 무관하게 승패·데스가 같다.
    p_leave: 슬롯별 이탈 확률 (B, P). 죽을 때마다 이 확률로 판을 나가고 남은 목숨을 버린다 (leave_rng로 뽑음)."""
    B, P = power.shape
    ts = P // 2
    T, d, c = float(sim["tickets"]), float(sim["ticket_per_death"]), float(sim["capture_coef"])
    K, sigma = int(sim["turn_limit"]), float(sim["sigma"])
    p_ambush = float(sim["p_ambush"])
    # 인원 반영 기습: 기습 확률 = p_ambush × (1 + gap_scale × |A생존 − B생존| / team_size), 상한 cap.
    # 기습 팀은 coin(동전) / linear(생존 인원 비율) / square(생존 인원 제곱 비율).
    # coin + gap_scale 0이면 인원 무관 버전(v1)과 같다.
    amb_team = str(sim.get("ambush_team", "linear"))
    gap_scale = float(sim.get("ambush_gap_scale", 1.0))
    amb_cap = float(sim.get("ambush_cap", 0.6))
    p_steal = float(sim.get("p_steal", 0.0))
    team_of = np.arange(P) >= ts

    ew = sp.engage[style]
    pdie = sp.p_die[style]
    cw = sp.capture[style]
    aw_all = sp.ambush_attack[style]
    tw_all = sp.ambush_target[style]

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
    max_def = np.zeros((B, 2))        # 팀별 최대 티켓 열세 (상대 티켓 − 내 티켓)
    max_alive_def = np.zeros((B, 2))  # 팀별 최대 생존 인원 열세 (상대 − 나)
    n_amb = np.zeros(B, dtype=np.int32)
    n_amb_win = np.zeros(B, dtype=np.int32)
    n_amb_kill = np.zeros(B, dtype=np.int32)
    n_leave = np.zeros(B, dtype=np.int32)

    for t in range(K):
        act = ~done
        if not act.any():
            break
        alive = lives > 0

        # 3. 일반 교전: 양 팀에서 살아 있는 유저 한 명씩, 교전 가중치에 비례
        w = ew * alive
        ia = _pick(w[:, :ts], rng.random(B))
        ib = ts + _pick(w[:, ts:], rng.random(B))
        eps = sigma * rng.standard_normal((3, B))
        pa = power[rows, ia] + eps[0]
        pb = power[rows, ib] + eps[1]

        # 3'. 기습: 확률 p_ambush로 일반 교전 대신. 기습하는 팀은 동전으로 정한다.
        #     기습조 2명 중 전투력이 높은 쪽이 대표로 싸우고, 킬·데스도 대표가 진다.
        na = alive[:, :ts].sum(axis=1).astype(float)
        nb = alive[:, ts:].sum(axis=1).astype(float)
        p_amb = p_ambush if gap_scale == 0 else np.minimum(
            p_ambush * (1 + gap_scale * np.abs(na - nb) / ts), max(amb_cap, p_ambush))
        if amb_team == "coin":
            p_a = 0.5
        else:
            wa, wb = (na, nb) if amb_team == "linear" else (na ** 2, nb ** 2)
            p_a = wa / np.maximum(wa + wb, 1)
        amb = (rng.random(B) < p_amb) & act
        amb_by_a = rng.random(B) < p_a
        u = rng.random((6, B))
        aw, tw = aw_all * alive, tw_all * alive
        a1A, a2A, has2A, tgA = _pick_ambush(aw[:, :ts], tw[:, :ts], u[:3], rows)
        a1B, a2B, has2B, tgB = _pick_ambush(aw[:, ts:], tw[:, ts:], u[3:], rows)
        a1B, a2B, tgB = a1B + ts, a2B + ts, tgB + ts
        att1 = np.where(amb_by_a, a1A, a1B)
        att2 = np.where(amb_by_a, a2A, a2B)
        has2 = np.where(amb_by_a, has2A, has2B)
        tgt = np.where(amb_by_a, tgB, tgA)
        p1 = power[rows, att1] + eps[0]
        p2 = np.where(has2, power[rows, att2] + eps[2], -np.inf)
        rep = np.where(p2 > p1, att2, att1)
        p_att = np.maximum(p1, p2)
        p_tgt = power[rows, tgt] + eps[1]
        ia = np.where(amb, np.where(amb_by_a, rep, tgt), ia)
        ib = np.where(amb, np.where(amb_by_a, tgt, rep), ib)
        pa = np.where(amb, np.where(amb_by_a, p_att, p_tgt), pa)
        pb = np.where(amb, np.where(amb_by_a, p_tgt, p_att), pb)

        # 4. 높은 쪽이 이김. 진 쪽은 p_die로 사망
        a_wins = pa > pb
        los = np.where(a_wins, ib, ia)
        win_slot = np.where(a_wins, ia, ib)
        die = (rng.random(B) < pdie[rows, los]) & act
        att_won = amb & (a_wins == amb_by_a)
        n_amb += amb
        n_amb_win += att_won
        n_amb_kill += att_won & die

        # 5'. 킬 스틸: 확률 p_steal로 킬 기록이 이긴 쪽 팀의 다른 살아 있는 유저에게 간다
        credit = win_slot
        if p_steal > 0:
            steal = die & (steal_rng.random(B) < p_steal)
            mates = alive & (team_of[None, :] == (win_slot >= ts)[:, None])
            mates[rows, win_slot] = False
            pick = _pick(mates.astype(float), steal_rng.random(B))
            credit = np.where(steal & mates.any(axis=1), pick, win_slot)

        # 5. 데스 처리
        r = rows[die]
        lives[r, los[die]] -= 1
        deaths[r, los[die]] += 1
        kills[r, credit[die]] += 1

        # 5''. 이탈: 죽은 유저가 확률 p_leave로 판을 나간다 (티켓 추가 감소 없음)
        if p_leave is not None:
            lv = die & (leave_rng.random(B) < p_leave[rows, los])
            lives[rows[lv], los[lv]] = 0
            n_leave += lv
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
        alive_a = alive[:, :ts].sum(axis=1)
        alive_b = alive[:, ts:].sum(axis=1)
        agap = (alive_b - alive_a).astype(float)
        np.maximum(max_alive_def[:, 0], np.where(act, agap, -np.inf), out=max_alive_def[:, 0])
        np.maximum(max_alive_def[:, 1], np.where(act, -agap, -np.inf), out=max_alive_def[:, 1])

        # 7. 종료: 티켓 ≤ 0 또는 전원 목숨 소진
        wiped_a = alive_a == 0
        wiped_b = alive_b == 0
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
                       winner_max_deficit=max_def[rows, winner],
                       winner_max_alive_deficit=max_alive_def[rows, winner],
                       ambush=n_amb, ambush_win=n_amb_win, ambush_kill=n_amb_kill, leaves=n_leave)
