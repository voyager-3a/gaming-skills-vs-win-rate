"""점령 계수 c를 이분법으로 정한다: 기본 설정(q, k, σ = 기본값)에서 티켓 감소 킬 몫이 목표값이 되도록.

사용: python scripts/tune_capture.py --target 0.7 --variant c_kill70
점령 가중치는 그 변형의 보정값(없으면 base 보정값)을 쓰고, 찾은 c를 config.toml [variants.변형]에 쓴다.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.metrics import game_summary  # noqa: E402
from sim.model import ROOT, calibration_key, load_config  # noqa: E402
from sim.runner import run  # noqa: E402

N_USERS, GAMES, SEED, TOL = 2000, 100, 11, 0.002


def kill_share(cfg, c, capture):
    g = cfg["grid"]
    res = run(cfg, n_users=N_USERS, max_games=GAMES, checkpoints=[GAMES], q=g["base_q"], k=g["base_k"],
              seed=SEED, sigma=g["base_sigma"], capture_override=capture,
              sim_overrides={"capture_coef": c})
    return game_summary(res, cfg["sim"]["turn_limit"])["ticket_loss_share_kill"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--variant", required=True)
    a = ap.parse_args()
    cfg = load_config()
    caps = cfg["calibration"]["capture"]
    k = cfg["grid"]["base_k"]
    capture = caps.get(calibration_key(a.variant, k)) or caps[calibration_key(None, k)]
    lo, hi, steps = 0.01, 20.0, []
    while True:
        c = (lo + hi) / 2
        share = kill_share(cfg, c, capture)
        steps.append({"c": c, "kill_share": share})
        print(f"c={c:.4f} kill_share={share:.4f}")
        if abs(share - a.target) < TOL or hi - lo < 1e-3:
            break
        lo, hi = (c, hi) if share > a.target else (lo, c)
    c = round(c, 3)
    out = ROOT / "results" / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"capture_coef_{a.variant}.json").write_text(json.dumps(
        {"variant": a.variant, "target_kill_share": a.target, "capture_used": capture,
         "capture_coef": c, "steps": steps}, indent=2))
    path = ROOT / "config.toml"
    text = path.read_text()
    pat = rf"(\[variants\.{a.variant}\]\ncapture_coef = )[0-9.]+"
    assert re.search(pat, text), a.variant
    path.write_text(re.sub(pat, rf"\g<1>{c}", text))
    print("c =", c)


if __name__ == "__main__":
    main()
