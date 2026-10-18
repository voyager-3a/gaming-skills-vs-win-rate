"""스타일 보정: 전원 s = 0, e = 0에서 스타일별 승률이 50% ± 0.5%p가 되도록
돌격·저격 점령 가중치만 조정한다 (교전 가중치, p_die, 기습 가중치는 그대로).

사용: python scripts/calibrate.py --k 0.5 1.0 [--variant one_life]
결과는 results/calibration/에 저장하고 config.toml [calibration.capture]에 기록한다.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.metrics import game_summary, style_table  # noqa: E402
from sim.model import ROOT, calibration_key, load_config, style_params  # noqa: E402
from sim.runner import run  # noqa: E402

FAST_GAMES, H, MAX_STEP, MAX_ITER = 400, 0.25, 0.25, 12


def measure(cfg, k, variant, x, games):
    c = cfg["calibration"]
    res = run(cfg, n_users=c["n_users"], max_games=games, checkpoints=[games], q=0.0, k=k,
              seed=c["seed"], variant=variant, zero_power=True,
              capture_override={"assault": x[0], "sniper": x[1]})
    return res, style_table(res)


def residual(t):
    return np.array([t.loc["assault", "winrate"] - 0.5, t.loc["sniper", "winrate"] - 0.5])


def calibrate(cfg, k, variant):
    tol, full = cfg["calibration"]["tolerance"], cfg["calibration"]["max_games"]
    sp = style_params(cfg, k, capture_override={})  # 보정 전 값 (표 + k 규칙)
    x = np.array([sp.capture[0], sp.capture[2]])
    start = x.copy()
    log = []
    _, t0 = measure(cfg, k, variant, x, full)
    log.append({"x": x.tolist(), "games": full, "winrate": t0["winrate"].tolist()})
    games = FAST_GAMES
    # 준뉴턴(Broyden): 야코비안은 처음에 넓은 차분으로 한 번 재고, 이후 갱신. 한 걸음 폭은 MAX_STEP 이하.
    J, f_prev, x_prev = None, None, None
    for it in range(MAX_ITER):
        _, t = measure(cfg, k, variant, x, games)
        f = residual(t)
        log.append({"x": x.tolist(), "games": games, "winrate": t["winrate"].tolist()})
        print(f"  iter {it} games={games} x={np.round(x, 4)} winrate={np.round(t['winrate'].to_numpy(), 4)}")
        if np.all(np.abs(t["winrate"] - 0.5) <= tol * 0.6):
            if games == full:
                break
            games = full
            continue
        if J is None:
            J = np.empty((2, 2))
            for j in range(2):
                xh = x.copy()
                xh[j] += H
                J[:, j] = (residual(measure(cfg, k, variant, xh, games)[1]) - f) / H
        elif f_prev is not None:
            dx, df = x - x_prev, f - f_prev
            if dx @ dx > 1e-8:
                J = J + np.outer(df - J @ dx, dx) / (dx @ dx)
        step = -np.linalg.solve(J, f)
        step *= min(1.0, MAX_STEP / np.abs(step).max())
        x_prev, f_prev = x.copy(), f
        x = np.maximum(x + step, 0.0)
    res, t = measure(cfg, k, variant, x, full)
    ok = bool(np.all(np.abs(t["winrate"] - 0.5) <= tol))
    summ = game_summary(res, cfg["sim"]["turn_limit"])
    return {"key": calibration_key(variant, k), "start": start.tolist(),
            "capture": {"assault": round(float(x[0]), 4), "sniper": round(float(x[1]), 4)},
            "within_tolerance": ok,
            "before": {"winrate": dict(zip(t0.index, t0["winrate"].round(4)))},
            "after": {"winrate": dict(zip(t.index, t["winrate"].round(4))),
                      "kd_pooled": dict(zip(t.index, t["kd_pooled"].round(4))),
                      "kd_mean": dict(zip(t.index, t["kd_mean"].round(4)))},
            "kill_share": summ["ticket_loss_share_kill"],
            "ambush_success": summ["ambush_success"], "log": log}


def write_config(entries):
    path = ROOT / "config.toml"
    text = path.read_text()
    head, rest = text.split("[calibration.capture]\n", 1)
    comment, tail = rest.split("\n[grid]", 1)
    lines = [ln for ln in comment.splitlines() if ln.startswith("#")]
    current = dict(re.findall(r'^"([^"]+)" = (\{.*\})$', comment, flags=re.M))
    for key, cap in entries.items():
        current[key] = f"{{ assault = {cap['assault']}, sniper = {cap['sniper']} }}"
    body = "\n".join(lines + [f'"{k}" = {v}' for k, v in sorted(current.items())])
    path.write_text(head + "[calibration.capture]\n" + body + "\n\n[grid]" + tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, nargs="+", required=True)
    ap.add_argument("--variant", default=None)
    a = ap.parse_args()
    cfg = load_config()
    out = ROOT / "results" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for k in a.k:
        print(f"calibrate {calibration_key(a.variant, k)}")
        r = calibrate(cfg, k, a.variant)
        entries[r["key"]] = r["capture"]
        (out / f"{r['key'].replace(':', '_')}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print(json.dumps({x: r[x] for x in r if x != "log"}, indent=2, ensure_ascii=False))
    write_config(entries)


if __name__ == "__main__":
    main()
