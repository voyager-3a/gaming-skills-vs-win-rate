"""점령 계수 c를 이분법으로 정한다: 기본 설정(q, k, σ = 기본값)에서 티켓 감소 킬:점령 = 50:50."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.metrics import game_summary  # noqa: E402
from sim.model import ROOT, load_config  # noqa: E402
from sim.runner import run  # noqa: E402

N_USERS, GAMES, SEED, TOL = 2000, 100, 11, 0.002


def kill_share(cfg, c, sigma):
    res = run(cfg, n_users=N_USERS, max_games=GAMES, checkpoints=[GAMES], q=cfg["grid"]["base_q"],
              k=cfg["grid"]["base_k"], seed=SEED, sigma=sigma, sim_overrides={"capture_coef": c})
    return game_summary(res, cfg["sim"]["turn_limit"])


def main():
    cfg = load_config()
    g = cfg["grid"]
    lo, hi = 0.1, 5.0
    steps = []
    while True:
        c = (lo + hi) / 2
        share = kill_share(cfg, c, g["base_sigma"])["ticket_loss_share_kill"]
        steps.append({"c": c, "kill_share": share})
        print(f"c={c:.4f} kill_share={share:.4f}")
        if abs(share - 0.5) < TOL or hi - lo < 1e-3:
            break
        lo, hi = (c, hi) if share > 0.5 else (lo, c)
    c = round(c, 3)
    check = {}
    for k in g["k"]:
        for s in g["sigma"]:
            res = run(cfg, n_users=N_USERS, max_games=GAMES, checkpoints=[GAMES], q=g["base_q"], k=k,
                      seed=SEED, sigma=s, sim_overrides={"capture_coef": c})
            check[f"k={k},sigma={s}"] = game_summary(res, cfg["sim"]["turn_limit"])["ticket_loss_share_kill"]
    out = ROOT / "results" / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    (out / "capture_coef.json").write_text(json.dumps(
        {"capture_coef": c, "steps": steps, "kill_share_by_setting": check}, indent=2))
    print("c =", c)
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
