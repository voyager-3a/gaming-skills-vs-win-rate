"""설정 로드, 유저 생성, 스타일 파라미터, 매칭 스케줄."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STYLES = ("assault", "allround", "sniper")
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "config.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


@dataclass
class StyleParams:
    engage: np.ndarray   # (3,) 교전 가중치
    p_die: np.ndarray    # (3,) 졌을 때 사망 확률
    capture: np.ndarray  # (3,) 점령 가중치
    ambush_attack: np.ndarray  # (3,) 기습하는 쪽으로 뽑힐 가중치
    ambush_target: np.ndarray  # (3,) 기습당하는 쪽으로 뽑힐 가중치


def calibration_key(variant: str | None, k: float) -> str:
    return f"{variant or 'base'}:k={float(k)}"


def style_params(cfg: dict, k: float, capture_override: dict | None = None,
                 variant: str | None = None) -> StyleParams:
    """올라운드와의 차이에 k를 곱한다. capture_override는 보정된 최종 점령 가중치이고,
    주지 않으면 설정 파일의 보정값(변형 실행 variant, k별)을 쓴다."""
    table = cfg["styles"]
    out = {}
    for key in ("engage", "p_die", "capture", "ambush_attack", "ambush_target"):
        base = np.array([table[s][key] for s in STYLES], dtype=float)
        out[key] = base[1] + k * (base - base[1])
    if capture_override is None:
        capture_override = cfg.get("calibration", {}).get("capture", {}).get(calibration_key(variant, k))
    if capture_override:
        out["capture"][0] = capture_override["assault"]
        out["capture"][2] = capture_override["sniper"]
    return StyleParams(**out)


@dataclass
class Users:
    s: np.ndarray      # 진짜 실력
    e: np.ndarray      # 장비 편향
    style: np.ndarray  # 0 돌격, 1 올라운드, 2 저격

    @property
    def n(self) -> int:
        return len(self.s)


def make_users(n: int, q: float, m: float, rng: np.random.Generator) -> Users:
    s = rng.standard_normal(n)
    e = np.zeros(n)
    n_eq = int(round(q * n))
    idx = rng.permutation(n)[:n_eq]
    e[idx[: n_eq // 2]] = m
    e[idx[n_eq // 2:]] = -m
    style = rng.integers(0, 3, n)
    return Users(s=s, e=e, style=style)


def make_schedule(n: int, max_games: int, team_size: int, rng: np.random.Generator) -> np.ndarray:
    """라운드마다 유저를 섞어 2×team_size명씩 묶는다. 남는 인원은 쉰다.

    판수가 적은 유저를 먼저 뽑아(동률은 무작위) 모두가 max_games를 채울 때까지 반복한다.
    반환: (판 수, 2×team_size) 유저 인덱스, 라운드 순서대로. 앞 절반이 A팀.
    """
    per_game = 2 * team_size
    per_round = (n // per_game) * per_game
    if per_round == 0:
        raise ValueError("유저 수가 한 판 인원보다 적다")
    games = np.zeros(n, dtype=np.int64)
    rounds = []
    while games.min() < max_games:
        order = np.argsort(games + rng.random(n), kind="stable")[:per_round]
        games[order] += 1
        rounds.append(rng.permutation(order).reshape(-1, per_game))
    return np.concatenate(rounds)
