"""파일럿: 유저 500 × 200판. 기본 설정(q, k)에서 σ별 1회, 그리고 기본 설정 전용 변형([variants])별 1회.
결과를 results/pilot/에 저장한다."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.metrics import correlations, game_summary, outnumbered_table, style_table  # noqa: E402
from sim.model import ROOT, load_config  # noqa: E402
from sim.runner import run  # noqa: E402


def main():
    cfg = load_config()
    p = cfg["pilot"]
    out = ROOT / "results" / "pilot"
    out.mkdir(parents=True, exist_ok=True)

    cases = [(f"sigma={s}", s, None) for s in p["sigma"]]
    cases += [(v, cfg["grid"]["base_sigma"], v) for v in cfg.get("variants", {})]
    summaries, corrs, styles, outn = {}, [], [], []
    for name, sigma, variant in cases:
        res = run(cfg, n_users=p["n_users"], max_games=p["max_games"], checkpoints=p["checkpoints"],
                  q=p["q"], k=p["k"], seed=p["seed"], sigma=sigma, variant=variant)
        summaries[name] = {"seconds": res.seconds, **game_summary(res, cfg["sim"]["turn_limit"])}
        corrs.append(correlations(res).assign(case=name))
        styles.append(style_table(res).assign(case=name))
        outn.append(outnumbered_table(res).assign(case=name))

    summary = {"params": {k: v for k, v in p.items() if k != "sigma"},
               "sim": {k: cfg["sim"][k] for k in ("capture_coef", "p_ambush", "lives")},
               "variants": cfg.get("variants", {}), "cases": summaries}
    corr, styles, outn = pd.concat(corrs), pd.concat(styles), pd.concat(outn)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    corr.to_csv(out / "correlations.csv", index=False)
    styles.to_csv(out / "styles.csv")
    outn.to_csv(out / "outnumbered.csv", index=False)

    keys = ["ticket_loss_share_kill", "turns_mean", "end_reason", "comeback_rate", "outnumbered_rate",
            "ambush_per_turn", "ambush_success", "ambush_kill_rate", "ambush_share_of_kills"]
    print(pd.DataFrame({n: {k: s[k] for k in keys} for n, s in summaries.items()}).to_string())
    print(corr.round(3).to_string(index=False))
    print(styles.round(3).to_string())
    print(outn.groupby("case").agg(n=("s", "size"), s_mean=("s", "mean"), e_mean=("e", "mean"),
                                   kills=("carry_kills", "mean")).round(3).to_string())
    print(outn.groupby(["case", "style"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
