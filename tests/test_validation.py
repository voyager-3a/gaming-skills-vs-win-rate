"""DESIGN.md '검증 테스트' 5종. 본 실행 전에 전부 통과해야 한다."""
import numpy as np
import pytest
from scipy.stats import kruskal, ks_2samp, spearmanr

from sim.model import load_config
from sim.runner import run

CFG = load_config()
N, GAMES = 640, 300


def _run(**kw):
    base = dict(n_users=N, max_games=GAMES, checkpoints=[GAMES], q=0.05, k=0.5, seed=123)
    return run(CFG, **{**base, **kw})


@pytest.fixture(scope="module")
def null_run():
    return _run(q=0.0, k=0.0, zero_power=True)


@pytest.fixture(scope="module")
def k0_run():
    return _run(q=0.0, k=0.0)


def test_1_null_model_independent_of_skill(null_run):
    """s = 0, e = 0, k = 0: 승률·K/D가 (표시용) 실력과 무관."""
    s, wr, kd = null_run.users.s, null_run.winrate(-1), null_run.kd(-1)
    bound = 4 / np.sqrt(N)
    assert abs(spearmanr(s, wr)[0]) < bound
    assert abs(spearmanr(s, kd)[0]) < bound
    assert abs(wr.mean() - 0.5) < 0.01
    top = s > np.median(s)
    assert ks_2samp(wr[top], wr[~top]).pvalue > 0.001
    assert ks_2samp(kd[top], kd[~top]).pvalue > 0.001


def test_2_kills_equal_opponent_deaths():
    g = _run().games
    assert (g["kills_a"] == g["deaths_b"]).all()
    assert (g["kills_b"] == g["deaths_a"]).all()
    assert np.allclose(g["loss_kill"], (g["deaths_a"] + g["deaths_b"]) * CFG["sim"]["ticket_per_death"])


def test_3_k0_style_kd_equal(k0_run, null_run):
    """k = 0: 스타일별 K/D 분포가 같다. 실력이 있으면 분포 검정만, 실력이 없으면 합산 K/D까지
    (실력이 있으면 스타일 그룹 간 실력 표본 차이가 합산 K/D를 흔들므로)."""
    kd, st = k0_run.kd(-1), k0_run.users.style
    assert kruskal(*[kd[st == i] for i in range(3)]).pvalue > 0.001
    st0, k, d = null_run.users.style, null_run.kills[-1], null_run.deaths[-1]
    pooled = [k[st0 == i].sum() / d[st0 == i].sum() for i in range(3)]
    assert max(pooled) / min(pooled) < 1.02


def test_4_skill_raises_winrate_and_kd(k0_run):
    s, wr, kd = k0_run.users.s, k0_run.winrate(-1), k0_run.kd(-1)
    assert spearmanr(s, wr)[0] > 0.5
    assert spearmanr(s, kd)[0] > 0.5
    q = np.digitize(s, np.quantile(s, [0.2, 0.4, 0.6, 0.8]))
    assert np.all(np.diff([wr[q == i].mean() for i in range(5)]) > 0)
    assert np.all(np.diff([kd[q == i].mean() for i in range(5)]) > 0)


def test_5_same_seed_same_result():
    a = _run(n_users=320, max_games=50, checkpoints=[50])
    b = _run(n_users=320, max_games=50, checkpoints=[50])
    for x, y in [(a.wins, b.wins), (a.kills, b.kills), (a.deaths, b.deaths), (a.users.s, b.users.s)]:
        assert np.array_equal(x, y)
    assert a.games.equals(b.games)
