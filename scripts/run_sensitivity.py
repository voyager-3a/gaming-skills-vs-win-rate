"""기본 설정 전용 민감도: 보정 점검(필요할 때만 재보정) → 실행(병렬) → 보고.

[sensitivity]의 변형마다
  1. 전원 s = 0, e = 0에서 base 점령 가중치로 스타일별 승률을 잰다.
  2. 셋 다 50% ± 허용 오차 안이면 base 보정값을 쓰고, 벗어나면 그 변형을 재보정한다.
  3. 유저 2000 × 1000판으로 돌려 1000판 상관과 역전 지점을 기록한다.
결과는 results/sensitivity/, 실행별 상세는 results/main/runs/<변형>/.
사용: python scripts/run_sensitivity.py [--workers 4]
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate import calibrate, measure, write_config  # noqa: E402
from run_all import _run_job  # noqa: E402
from sim.model import ROOT, calibration_key, load_config  # noqa: E402

OUT = ROOT / "results" / "sensitivity"
RUNS = ROOT / "results" / "main" / "runs"


def _check(variant):
    """base 보정값으로 스타일 승률 점검, 벗어나면 재보정."""
    cfg = load_config()
    k = cfg["grid"]["base_k"]
    tol, full = cfg["calibration"]["tolerance"], cfg["calibration"]["max_games"]
    base_cap = cfg["calibration"]["capture"][calibration_key(None, k)]
    t0 = time.perf_counter()
    _, t = measure(cfg, k, variant, np.array([base_cap["assault"], base_cap["sniper"]]), full)
    check = {s: float(t.loc[s, "winrate"]) for s in t.index}
    ok = all(abs(v - 0.5) <= tol for v in check.values())
    out = {"variant": variant, "check_winrate": check, "within_tolerance": ok, "recalibrated": not ok,
           "capture": dict(base_cap)}
    if not ok:
        r = calibrate(cfg, k, variant)
        out["capture"] = r["capture"]
        out["after_winrate"] = r["after"]["winrate"]
        out["after_within_tolerance"] = r["within_tolerance"]
        (ROOT / "results" / "calibration" / f"{r['key'].replace(':', '_')}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False))
    out["seconds"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true",
                    help="이미 점검·실행한 변형은 기존 결과를 쓴다 (코드가 같으면 결과도 같다)")
    a = ap.parse_args()
    cfg = load_config()
    g = cfg["grid"]
    groups = cfg["sensitivity"]
    variants = sorted({v for vs in groups.values() for v in vs if v != "base"})
    OUT.mkdir(parents=True, exist_ok=True)
    timing = {"workers": a.workers}
    t_all = time.perf_counter()

    prev = {}
    if a.skip_existing and (OUT / "calibration_check.json").exists():
        prev = {v: c for v, c in json.loads((OUT / "calibration_check.json").read_text()).items()
                if v in variants and (RUNS / v / "summary.json").exists()}
    todo = [v for v in variants if v not in prev]
    timing["skipped"] = sorted(prev)
    with Pool(a.workers) as pool:
        t0 = time.perf_counter()
        checks = {**prev, **{c["variant"]: c for c in pool.map(_check, todo)}}
        write_config({calibration_key(v, g["base_k"]): c["capture"] for v, c in checks.items()})
        timing["check_seconds"] = time.perf_counter() - t0
        for v, c in sorted(checks.items()):
            print(f"  check {v}: {c['check_winrate']} recal={c['recalibrated']}", flush=True)
        (OUT / "calibration_check.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False))

        t0 = time.perf_counter()
        jobs = [dict(name=v, q=g["base_q"], k=g["base_k"], sigma=g["base_sigma"], variant=v,
                     seed=g["base_seeds"][0]) for v in todo]
        infos = {}
        for info in pool.imap_unordered(_run_job, jobs):
            infos[info["name"]] = info
            print(f"  {info['name']}: {info['seconds']:.1f}s crossover={info['crossover']}", flush=True)
        timing["run_seconds"] = time.perf_counter() - t0
    timing["total_seconds"] = time.perf_counter() - t_all
    timing["runs"] = {n: i["seconds"] for n, i in sorted(infos.items())}
    (OUT / "timing.json").write_text(json.dumps(timing, indent=2))
    report(cfg, checks)


LABELS = {"p_steal": "킬 스틸", "p_leave": "이탈", "leave_mode": "", "squad_frac": "분대",
          "squad_corr": "상관", "squad_p_play": "p_play", "squad_bonus": "b"}


def label(cfg, v):
    if v == "base":
        return "기본 (킬 스틸·이탈·분대 없음)"
    p = cfg["variants"][v]
    parts = []
    if "p_steal" in p:
        parts.append(f"킬 스틸 {p['p_steal']}")
    if "p_leave" in p:
        parts.append(f"이탈 {'실력 연동' if p.get('leave_mode') == 'skill' else '실력 무관'} {p['p_leave']}")
    if "squad_frac" in p:
        parts.append(f"분대 {p['squad_frac']} (상관 {p.get('squad_corr', 0.0)}, p_play {p.get('squad_p_play', 1.0)}, "
                     f"b {p.get('squad_bonus', 0.0)})")
    return " + ".join(parts)


def summary_table(cfg, df):
    """민감도 전체를 한 표로, 그리고 묶음 안에서 앞서는 지표가 바뀌는 경계."""
    d = df.drop_duplicates("variant")
    fmt = lambda x: "> 1000" if np.isinf(x) else f"{x:.0f}"  # noqa: E731
    out = ["## 전체 요약 (1000판)", "",
           "| 설정 | 승률 상관 | K/D 상관 | 차이 (승률 − K/D) [95%] | 역전 지점 | 재보정 |", "|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        rec = {True: "예", False: "아니오"}.get(r["recalibrated"], "-")
        out.append(f"| {label(cfg, r['variant'])} | {r['rho_winrate_1000']:.3f} | {r['rho_kd_1000']:.3f} | "
                   f"{r['diff_1000']:+.3f} [{r['diff_1000_lo']:+.3f}, {r['diff_1000_hi']:+.3f}] | "
                   f"{fmt(r['crossover'])} | {rec} |")
    out += ["", "## 앞서는 지표가 바뀌는 경계 (1000판 차이의 부호가 묶음 안 이웃 설정 사이에서 바뀌는 곳)", ""]
    found = False
    for group, gd in df.groupby("group", sort=False):
        gd = gd.reset_index(drop=True)
        for i in range(1, len(gd)):
            a, b = gd.iloc[i - 1], gd.iloc[i]
            if np.sign(a["diff_1000"]) != np.sign(b["diff_1000"]):
                found = True
                line = f"- **{group}**: {label(cfg, a['variant'])} ({a['diff_1000']:+.3f}) → {label(cfg, b['variant'])} ({b['diff_1000']:+.3f})"
                va, vb = cfg["variants"].get(a["variant"], {}), cfg["variants"].get(b["variant"], {})
                keys = [k for k in set(va) | set(vb) if va.get(k, 0) != vb.get(k, 0)
                        and isinstance(vb.get(k, 0), (int, float))]
                if len(keys) == 1:
                    k = keys[0]
                    x0, x1 = float(va.get(k, 0)), float(vb.get(k, 0))
                    x = x0 + (0 - a["diff_1000"]) * (x1 - x0) / (b["diff_1000"] - a["diff_1000"])
                    line += f". 선형 보간으로 {k} ≈ {x:.2f}에서 비김"
                out.append(line)
    if not found:
        out.append("- 없음")
    out.append("")
    return out


def report(cfg, checks=None):
    if checks is None:
        checks = json.loads((OUT / "calibration_check.json").read_text())
    groups = cfg["sensitivity"]
    rows = []
    for group, vs in groups.items():
        for v in vs:
            name = "base_seed0" if v == "base" else v
            info = json.loads((RUNS / name / "summary.json").read_text())
            c = pd.read_csv(RUNS / name / "correlations.csv").set_index("games")
            p = {kk: cfg["variants"][v][kk] for kk in cfg["variants"][v]} if v != "base" else {}
            ch = checks.get(v, {})
            s = info["summary"]
            rows.append({
                "group": group, "variant": v,
                "params": ", ".join(f"{kk}={vv}" for kk, vv in p.items()) or "-",
                "rho_winrate_1000": c.loc[1000, "rho_winrate"], "rho_kd_1000": c.loc[1000, "rho_kd"],
                "diff_1000": c.loc[1000, "diff"], "diff_1000_lo": c.loc[1000, "diff_lo"],
                "diff_1000_hi": c.loc[1000, "diff_hi"],
                "rho_winrate_100": c.loc[100, "rho_winrate"], "rho_kd_100": c.loc[100, "rho_kd"],
                "crossover": info["crossover"],
                "recalibrated": ch.get("recalibrated", False) if v != "base" else "-",
                "check_wr_assault": ch.get("check_winrate", {}).get("assault"),
                "check_wr_allround": ch.get("check_winrate", {}).get("allround"),
                "check_wr_sniper": ch.get("check_winrate", {}).get("sniper"),
                "kill_share": s["ticket_loss_share_kill"],
                "leaves_per_game": s.get("leaves_per_game", 0.0),
                "turns_mean": s["turns_mean"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sensitivity.csv", index=False)
    fmt = lambda x: "> 1000" if isinstance(x, float) and np.isinf(x) else (  # noqa: E731
        f"{x:.0f}" if isinstance(x, float) else str(x))
    lines = ["# 민감도 (기본 설정 q = 0.05, k = 0.5, σ = 1, 시드 0)", "",
             "값은 근거 없는 임의값이므로 범위 전체를 보고한다. 점검 승률은 전원 s = 0, e = 0, base 점령 가중치.", ""]
    for group, d in df.groupby("group", sort=False):
        lines += [f"## {group}", "",
                  "| 변형 | 파라미터 | 1000판 승률 / K/D | Δ1000 [95%] | 100판 승률 / K/D | 역전 지점 | 재보정 | 점검 승률 (돌격/올라운드/저격) | 판당 이탈 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            chk = ("-" if r["check_wr_assault"] is None or pd.isna(r["check_wr_assault"]) else
                   f"{r['check_wr_assault']:.3f} / {r['check_wr_allround']:.3f} / {r['check_wr_sniper']:.3f}")
            lines.append(
                f"| {r['variant']} | {r['params']} | {r['rho_winrate_1000']:.3f} / {r['rho_kd_1000']:.3f} | "
                f"{r['diff_1000']:+.3f} [{r['diff_1000_lo']:+.3f}, {r['diff_1000_hi']:+.3f}] | "
                f"{r['rho_winrate_100']:.3f} / {r['rho_kd_100']:.3f} | {fmt(r['crossover'])} | "
                f"{'예' if r['recalibrated'] is True else ('아니오' if r['recalibrated'] is False else '-')} | "
                f"{chk} | {r['leaves_per_game']:.1f} |")
        lines.append("")
    rec = [(v, c) for v, c in checks.items() if c["recalibrated"]]
    if rec:
        lines += ["## 재보정한 변형", "", "| 변형 | 돌격 / 저격 점령 가중치 | 재보정 후 승률 (돌격/올라운드/저격) | 허용 오차 안 |",
                  "|---|---|---|---|"]
        for v, c in rec:
            w = c["after_winrate"]
            lines.append(f"| {v} | {c['capture']['assault']} / {c['capture']['sniper']} | "
                         f"{w['assault']:.4f} / {w['allround']:.4f} / {w['sniper']:.4f} | "
                         f"{'예' if c['after_within_tolerance'] else '아니오'} |")
        lines.append("")
    lines += summary_table(cfg, df)
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
