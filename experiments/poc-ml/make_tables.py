"""Print the PoC results JSON as Markdown tables ."""
import json
import sys
from pathlib import Path

r = json.loads((Path(__file__).parent / "results" / "poc-results.json").read_text())
h = r["hardware"]
print(f"Hardware: {h['cpu']}, {h['logical_cores']} logical cores, {h['ram_gb']} GB RAM, {h['os']}\n")
print(f"Train wells: {', '.join(r['split']['train_wells'])} ({r['split']['train_rows']} rows). "
      f"Held-out wells: {', '.join(r['split']['test_wells'])} ({r['split']['test_rows']} rows).\n")


def t(v, nd=1):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))


L = r.get("lithology")
if L:
    print(f"### Lithology ({L['n_test']} scored samples)\n")
    print("| Model | Training samples | Accuracy | Penalty score | Train / fine-tune (s) | Predict per 1,000 (s) | Predict per well (s, est.) |")
    print("|---|---|---|---|---|---|---|")
    names = {"always_shale": "Always 'shale' (floor)", "xgb_same_sample": "XGBoost, same samples",
             "xgb_all_rows": "XGBoost, all rows of training wells"}
    for k, n in names.items():
        if k in L:
            v = L[k]
            print(f"| {n} | {v.get('n_train', '–')} | {v['accuracy']:.3f} | {v['penalty_score']:.3f} | {t(v.get('train_s'))} | "
                  f"{t(v.get('pred_ms_per_1000', None) and v['pred_ms_per_1000'] / 1000, 3)} | {t(v.get('pred_s_per_well_est'), 2)} |")
    print()
D = r.get("density")
if D:
    print(f"### Density, RHOB ({D['n_test']} scored samples; test RHOB mean {D['test_rhob_mean']}, std {D['test_rhob_std']} g/cc)\n")
    print("| Model | MAE | RMSE | R² | mean σ | ±1σ cover (ideal 0.68) | ±2σ cover (ideal 0.95) | Train / fine-tune (s) | Predict per 1,000 (s) | Predict per well (s, est.) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    order = ["predict_train_mean", "xgb_reg_same_sample", "xgb_reg_all_rows", "xgb_quantile_same_sample"]
    for tag in ["10bins_uniform", "15bins_uniform", "10bins_quantile"]:
        order += [f"bin_floor_{tag}", f"xgb_binclass_{tag}"]
    for k in order:
        if k not in D:
            continue
        v = D[k]
        pm = v.get("pred_ms_per_1000")
        print(f"| {k} | {v['MAE']:.3f} | {v['RMSE']:.3f} | {v['R2']:.2f} | {t(v.get('mean_std'), 3)} | {t(v.get('cover_1sigma'), 2)} | "
              f"{t(v.get('cover_2sigma'), 2)} | {t(v.get('train_s'))} | {t(pm and pm / 1000, 3)} | {t(v.get('pred_s_per_well_est'), 2)} |")
