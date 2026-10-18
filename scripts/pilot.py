"""파일럿: 유저 500 × 200판, 기본 설정 1회. 결과를 results/pilot/에 저장한다."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.metrics import correlations, game_summary, style_table  # noqa: E402
from sim.model import ROOT, load_config  # noqa: E402
from sim.runner import run  # noqa: E402


def main():
    cfg = load_config()
    p = cfg["pilot"]
    out = ROOT / "results" / "pilot"
    out.mkdir(parents=True, exist_ok=True)

    res = run(cfg, n_users=p["n_users"], max_games=p["max_games"], checkpoints=p["checkpoints"],
              q=p["q"], k=p["k"], seed=p["seed"])
    summary = {"seconds": res.seconds, "params": p,
               "sim": {k: cfg["sim"][k] for k in ("capture_coef", "turn_limit", "tickets",
                                                   "ticket_per_death", "lives")},
               **game_summary(res, cfg["sim"]["turn_limit"])}
    corr = correlations(res)
    styles = style_table(res)

    # 전체 규모 1회 실측으로 본 실행 시간 추정
    s = cfg["sim"]
    full = run(cfg, n_users=s["n_users"], max_games=s["max_games"], checkpoints=s["checkpoints"],
               q=cfg["grid"]["base_q"], k=cfg["grid"]["base_k"], seed=0)
    user_games = p["n_users"] * p["max_games"]
    summary["time_estimate"] = {
        "pilot_seconds_per_1M_user_games": res.seconds / user_games * 1e6,
        "full_run_seconds_measured": full.seconds,
        "main_11_runs_seconds": 11 * full.seconds,
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    corr.to_csv(out / "correlations.csv", index=False)
    styles.to_csv(out / "styles.csv")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(corr.round(3).to_string(index=False))
    print(styles.round(3).to_string())


if __name__ == "__main__":
    main()
