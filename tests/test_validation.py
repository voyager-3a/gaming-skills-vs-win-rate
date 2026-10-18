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


def test_ambush_best_of_two():
    """실력 0, k = 0, 전원 기습이면 기습 쪽 승률은 max(ε1, ε2) > ε3 = 2/3 (기습조 2명일 때)."""
    g = _run(q=0.0, k=0.0, zero_power=True, sim_overrides={"p_ambush": 1.0}).games
    assert g["ambush"].sum() == g["turns"].sum()
    rate = g["ambush_win"].sum() / g["ambush"].sum()
    assert abs(rate - 2 / 3) < 0.01
    assert (g["kills_a"] == g["deaths_b"]).all() and (g["kills_b"] == g["deaths_a"]).all()


def test_no_ambush_when_p_zero():
    g = _run(sim_overrides={"p_ambush": 0.0}).games
    assert g["ambush"].sum() == 0


def test_kill_steal_keeps_outcomes():
    """킬 스틸은 킬 기록만 옮긴다: 같은 시드면 승패·데스·판별 킬 합이 그대로."""
    a = _run(n_users=320, max_games=100, checkpoints=[100])
    b = _run(n_users=320, max_games=100, checkpoints=[100], sim_overrides={"p_steal": 0.3})
    assert np.array_equal(a.wins, b.wins) and np.array_equal(a.deaths, b.deaths)
    assert a.kills.sum() == b.kills.sum() and not np.array_equal(a.kills, b.kills)
    for col in ["kills_a", "kills_b", "deaths_a", "deaths_b", "winner", "turns"]:
        assert a.games[col].equals(b.games[col])


def test_squads_same_team_and_skill_corr():
    r = _run(n_users=640, max_games=60, checkpoints=[60],
             sim_overrides={"squad_frac": 0.6, "squad_corr": 0.5})
    sq, sch = r.squad, r.schedule
    assert abs((sq >= 0).mean() - 0.6) < 0.01
    team = np.where(np.arange(32) < 16, 0, 1)
    for g in sch[:500]:
        ids = sq[g]
        for s in set(ids[ids >= 0]):
            assert len(set(team[ids == s])) == 1  # 분대는 쪼개지지 않는다
    games = np.bincount(sch.ravel(), minlength=640)
    assert games.min() >= 60
    # 분대원 쌍의 실력 상관 ≈ 0.5
    pairs = [(a, b) for s in range(sq.max() + 1) for a, b in [tuple(np.flatnonzero(sq == s)[:2])]]
    a, b = np.array(pairs).T
    assert abs(np.corrcoef(r.users.s[a], r.users.s[b])[0, 1] - 0.5) < 0.15


def test_leave_mechanics():
    base = _run(n_users=320, max_games=50, checkpoints=[50])
    same = _run(n_users=320, max_games=50, checkpoints=[50], sim_overrides={"p_leave": 0.0})
    assert np.array_equal(base.wins, same.wins) and np.array_equal(base.kills, same.kills)
    allout = _run(n_users=320, max_games=50, checkpoints=[50], sim_overrides={"p_leave": 1.0})
    # 죽으면 반드시 나가므로 판당 데스는 최대 1, 이탈 수 = 데스 수
    g = allout.games
    assert (g["leaves"] == g["deaths_a"] + g["deaths_b"]).all()
    assert allout.deaths[-1].max() <= 50


def test_intermittent_squads_and_bonus():
    kw = dict(n_users=640, max_games=60, checkpoints=[60])
    base = _run(**kw, sim_overrides={"squad_frac": 0.3})
    same = _run(**kw, sim_overrides={"squad_frac": 0.3, "squad_p_play": 1.0, "squad_bonus": 0.0})
    assert np.array_equal(base.wins, same.wins) and np.array_equal(base.kills, same.kills)
    assert np.array_equal(base.in_squad, base.squad[base.schedule] >= 0)  # 항상 분대로 참여
    half = _run(**kw, sim_overrides={"squad_frac": 0.3, "squad_p_play": 0.5})
    member = half.squad[half.schedule] >= 0
    assert abs(half.in_squad[member].mean() - 0.5) < 0.05
    team = np.where(np.arange(32) < 16, 0, 1)
    for g, f in zip(half.schedule[:300], half.in_squad[:300]):
        ids = half.squad[g]
        for s in set(ids[f]):
            assert len(set(team[(ids == s) & f])) == 1
    # 보너스가 크면 분대원 승률이 오른다
    big = _run(**kw, sim_overrides={"squad_frac": 0.3, "squad_bonus": 1.0})
    sq = big.squad >= 0
    assert big.winrate(-1)[sq].mean() > big.winrate(-1)[~sq].mean() + 0.03


def test_headcount_ambush():
    """인원 반영: 기습 확률은 p_ambush 이상, 상한 이하. coin + scale 0이면 p_ambush 그대로."""
    v1 = _run(sim_overrides={"ambush_team": "coin", "ambush_gap_scale": 0.0}).games
    v2 = _run().games
    r1 = v1["ambush"].sum() / v1["turns"].sum()
    r2 = v2["ambush"].sum() / v2["turns"].sum()
    assert abs(r1 - 0.2) < 0.01
    assert 0.2 < r2 <= 0.6
    for mode in ("linear", "square"):
        g = _run(sim_overrides={"ambush_team": mode}).games
        assert (g["kills_a"] == g["deaths_b"]).all()


def test_teammate_support():
    """γ = 0이면 기본과 같고, 실력 0이면 γ와 무관하다. γ가 크면 판 킬−데스가 팀원 평균 실력을 따라간다."""
    kw = dict(n_users=640, max_games=60, checkpoints=[60])
    a = _run(**kw)
    b = _run(**kw, sim_overrides={"teammate_gamma": 0.0})
    assert np.array_equal(a.wins, b.wins) and np.array_equal(a.kills, b.kills)
    z0 = _run(**kw, zero_power=True)
    z4 = _run(**kw, zero_power=True, sim_overrides={"teammate_gamma": 4.0})
    assert np.array_equal(z0.wins, z4.wins)
    r = _run(**kw, sim_overrides={"teammate_gamma": 4.0}, store_slots=True)
    s = r.users.s[r.schedule]
    team = np.repeat([[0, 1]], 16, axis=1)[0]
    tsum = np.stack([s[:, :16].sum(1), s[:, 16:].sum(1)], 1)[:, team]
    mates = (tsum - s) / 15
    assert np.corrcoef(r.slot_net.ravel(), mates.ravel())[0, 1] > 0.1
