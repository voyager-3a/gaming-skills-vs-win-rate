"""전체 실행: 스타일 보정(병렬) → 본 실행 34회(병렬) → 실행별 분석 저장.

27개 격자(q × k × σ) + 기본 설정 시드 1, 2 + 기본 설정 전용 변형 5개.
결과는 results/main/runs/<이름>/, 시간 기록은 results/main/timing.json.
사용: python scripts/run_all.py [--workers 4] [--skip-calibration]
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate import calibrate, write_config  # noqa: E402
from sim.metrics import correlations, crossover, game_summary, outnumbered_table, style_table  # noqa: E402
from sim.model import ROOT, STYLES, calibration_key, load_config  # noqa: E402
from sim.runner import run  # noqa: E402

OUT = ROOT / "results" / "main"
# 민감도 변형은 scripts/run_sensitivity.py가 점검 후 필요할 때만 보정한다
SAME_CALIBRATION_AS_BASE = ()


def jobs(cfg):
    g = cfg["grid"]
    out = []
    for q in g["q"]:
        for k in g["k"]:
            for s in g["sigma"]:
                base = (q, k, s) == (g["base_q"], g["base_k"], g["base_sigma"])
                name = "base_seed0" if base else f"q{q}_k{k}_s{s}"
                out.append(dict(name=name, q=q, k=k, sigma=s, variant=None,
                                seed=g["base_seeds"][0] if base else g["other_seed"]))
    for seed in g["base_seeds"][1:]:
        out.append(dict(name=f"base_seed{seed}", q=g["base_q"], k=g["base_k"], sigma=g["base_sigma"],
                        variant=None, seed=seed))
    sens = {v for group in cfg.get("sensitivity", {}).values() for v in group}
    for v in cfg.get("variants", {}):
        if v in sens:
            continue
        out.append(dict(name=v, q=g["base_q"], k=g["base_k"], sigma=g["base_sigma"], variant=v,
                        seed=g["base_seeds"][0]))
    return out


def _calibrate_job(args):
    k, variant = args
    t0 = time.perf_counter()
    r = calibrate(load_config(), k, variant)
    r["seconds"] = time.perf_counter() - t0
    return r


def outliers(res, top):
    i = len(res.checkpoints) - 1
    u = res.users
    df = pd.DataFrame({"user": np.arange(u.n), "s": u.s, "e": u.e, "style": np.array(STYLES)[u.style],
                       "winrate": res.winrate(i), "kd": res.kd(i), "kills": res.kills[i],
                       "deaths": res.deaths[i]})
    pct = lambda x: rankdata(x) / len(x)  # noqa: E731
    df["rank_gap_kd"] = pct(df["kd"]) - pct(df["s"])
    df["rank_gap_winrate"] = pct(df["winrate"]) - pct(df["s"])
    lists = [
        ("kd_bubble", df.nlargest(top, "rank_gap_kd")),
        ("kd_hidden_master", df.nsmallest(top, "rank_gap_kd")),
        ("winrate_bubble", df.nlargest(top, "rank_gap_winrate")),
        ("winrate_hidden_master", df.nsmallest(top, "rank_gap_winrate")),
    ]
    g = res.games
    who = lambda col: df.set_index("user").loc[g[col]].reset_index(drop=True)[["s", "e", "style"]]  # noqa: E731
    gk = pd.concat([g[["game", "max_kills", "max_kills_user", "turns"]], who("max_kills_user")], axis=1)
    lists += [
        ("game_max_kills", gk.nlargest(top, "max_kills")),
        ("game_longest", g.nlargest(top, "turns")[["game", "turns", "reason", "tickets_a", "tickets_b"]]),
        ("game_ticket_comeback", g[g["comeback"]].nlargest(top, "winner_max_deficit")[
            ["game", "winner_max_deficit", "turns", "tickets_a", "tickets_b"]]),
    ]
    on = outnumbered_table(res)
    lists.append(("game_outnumbered", on.nlargest(top, "max_alive_deficit")))
    return pd.concat([d.assign(list=n) for n, d in lists], ignore_index=True), on


def _run_job(job):
    cfg = load_config()
    s, a = cfg["sim"], cfg["analysis"]
    res = run(cfg, n_users=s["n_users"], max_games=s["max_games"], checkpoints=s["checkpoints"],
              q=job["q"], k=job["k"], seed=job["seed"], sigma=job["sigma"], variant=job["variant"])
    corr = correlations(res, n_boot=a["bootstrap"], seed=job["seed"])
    styles = style_table(res)
    summ = game_summary(res, s["turn_limit"])
    outl, outn = outliers(res, a["outliers_top"])
    d = OUT / "runs" / job["name"]
    d.mkdir(parents=True, exist_ok=True)
    corr.to_csv(d / "correlations.csv", index=False)
    styles.to_csv(d / "styles.csv")
    outl.to_csv(d / "outliers.csv", index=False)
    outn.to_csv(d / "outnumbered.csv", index=False)
    info = {**job, "seconds": res.seconds, "crossover": crossover(corr), "summary": summ,
            "outnumbered_carry": {"n": len(outn), "s_mean": float(outn["s"].mean()) if len(outn) else None,
                                  "style_counts": outn["style"].value_counts().to_dict()}}
    (d / "summary.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-calibration", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    timing = {"workers": a.workers}
    t_all = time.perf_counter()

    with Pool(a.workers) as pool:
        if not a.skip_calibration:
            t0 = time.perf_counter()
            cal_jobs = [(k, None) for k in cfg["grid"]["k"] if k != 0]
            sens = {v for group in cfg.get("sensitivity", {}).values() for v in group}
            cal_jobs += [(cfg["grid"]["base_k"], v) for v in cfg.get("variants", {})
                         if v not in SAME_CALIBRATION_AS_BASE and v not in sens]
            cal = pool.map(_calibrate_job, cal_jobs)
            entries = {r["key"]: r["capture"] for r in cal}
            base_key = calibration_key(None, cfg["grid"]["base_k"])
            for v in SAME_CALIBRATION_AS_BASE:
                entries[calibration_key(v, cfg["grid"]["base_k"])] = entries[base_key]
            old = cfg["calibration"]["capture"]
            changed = {k: (old.get(k), v) for k, v in entries.items() if old.get(k) != v}
            write_config(entries)
            cdir = ROOT / "results" / "calibration"
            cdir.mkdir(parents=True, exist_ok=True)
            for r in cal:
                (cdir / f"{r['key'].replace(':', '_')}.json").write_text(
                    json.dumps(r, indent=2, ensure_ascii=False))
            timing["calibration_seconds"] = time.perf_counter() - t0
            timing["calibration_per_job"] = {r["key"]: r["seconds"] for r in cal}
            timing["calibration_changed"] = changed
            print(f"calibration {timing['calibration_seconds']:.0f}s, changed={changed}", flush=True)

        t0 = time.perf_counter()
        js = jobs(cfg)
        infos = []
        for info in pool.imap_unordered(_run_job, js):
            infos.append(info)
            print(f"  {info['name']}: {info['seconds']:.1f}s crossover={info['crossover']}", flush=True)
        timing["main_seconds"] = time.perf_counter() - t0

    timing["total_seconds"] = time.perf_counter() - t_all
    timing["runs"] = {i["name"]: i["seconds"] for i in sorted(infos, key=lambda i: i["name"])}
    (OUT / "timing.json").write_text(json.dumps(timing, indent=2, ensure_ascii=False))
    (OUT / "runs.json").write_text(json.dumps(sorted(infos, key=lambda i: i["name"]), indent=2,
                                              ensure_ascii=False, default=float))
    print(f"total {timing['total_seconds']:.0f}s (main {timing['main_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
