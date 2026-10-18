"""파일럿: 유저 500 × 200판, 기본 설정(q, k)에서 σ별 1회씩. 결과를 results/pilot/에 저장한다."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.metrics import correlations, game_summary, last_man_table, style_table  # noqa: E402
from sim.model import ROOT, load_config  # noqa: E402
from sim.runner import run  # noqa: E402


def main():
    cfg = load_config()
    p = cfg["pilot"]
    out = ROOT / "results" / "pilot"
    out.mkdir(parents=True, exist_ok=True)

    summaries, corrs, styles, last_men = {}, [], [], []
    for sigma in p["sigma"]:
        res = run(cfg, n_users=p["n_users"], max_games=p["max_games"], checkpoints=p["checkpoints"],
                  q=p["q"], k=p["k"], seed=p["seed"], sigma=sigma)
        summaries[f"sigma={sigma}"] = {"seconds": res.seconds,
                                       **game_summary(res, cfg["sim"]["turn_limit"])}
        corrs.append(correlations(res).assign(sigma=sigma))
        styles.append(style_table(res).assign(sigma=sigma))
        last_men.append(last_man_table(res).assign(sigma=sigma))

    summary = {"params": {k: v for k, v in p.items() if k != "sigma"},
               "capture_coef": cfg["sim"]["capture_coef"], "by_sigma": summaries}
    corr, styles, last_men = pd.concat(corrs), pd.concat(styles), pd.concat(last_men)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    corr.to_csv(out / "correlations.csv", index=False)
    styles.to_csv(out / "styles.csv")
    last_men.to_csv(out / "last_man.csv", index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(corr.round(3).to_string(index=False))
    print(styles.round(3).to_string())
    print(last_men.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
