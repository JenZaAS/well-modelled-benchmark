"""FH2026 smoke test: imports, one tiny batch per model on CPU, and a small XGBoost fit.

Runs fully offline (HF_HUB_OFFLINE=1), so a PASS means the laptop has everything it needs.

    python fh2026_smoke_test.py --cache-root C:\\Users\\me\\fh2026-cache

Exit code 0 = all steps passed, 1 = something failed (see the summary at the end).
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

RESULTS: list[tuple[str, bool, float, str]] = []

LITHO = {30000: 0, 65030: 1, 65000: 2, 80000: 3, 74000: 4, 70000: 5,
         70032: 6, 88000: 7, 86000: 8, 99000: 9, 90000: 10, 93000: 11}


def step(name):
    def wrap(fn):
        def run(*a, **k):
            t0 = time.perf_counter()
            try:
                note = fn(*a, **k) or ""
                ok = True
            except Exception as e:  # noqa: BLE001
                note = f"{type(e).__name__}: {e}"[:300]
                ok = False
                traceback.print_exc()
            dt = time.perf_counter() - t0
            RESULTS.append((name, ok, dt, note))
            print(f"[{'PASS' if ok else 'FAIL'}] {name} ({dt:.1f} s) {note}", flush=True)
        return run
    return wrap


def hardware() -> str:
    import platform
    import subprocess
    cpu = platform.processor() or platform.machine()
    try:
        if platform.system() == "Windows":
            cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.strip() or cpu
        elif platform.system() == "Linux":
            cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:  # noqa: BLE001
        pass
    ram = "?"
    try:
        if platform.system() == "Windows":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("t", ctypes.c_ulonglong),
                            ("a", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
            ms = MS(); ms.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            ram = f"{ms.t / 2**30:.1f}"
        else:
            ram = f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30:.1f}"
    except Exception:  # noqa: BLE001
        pass
    return f"{cpu} | {os.cpu_count()} logical cores | {ram} GB RAM | {platform.platform()}"


IMPORTS = ["numpy", "pandas", "pyarrow", "lasio", "duckdb", "pdfplumber", "requests",
           "matplotlib", "openpyxl", "sklearn", "scipy", "xgboost", "lightgbm", "torch",
           "transformers", "huggingface_hub", "momentfm", "chronos", "tsfm_public"]


@step("imports")
def check_imports():
    vers = []
    missing = []
    for m in IMPORTS:
        try:
            mod = importlib.import_module(m)
            vers.append(f"{m} {getattr(mod, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            missing.append(f"{m} ({type(e).__name__}: {e})")
    print("   " + ", ".join(vers))
    if missing:
        raise ImportError("; ".join(missing))
    import torch
    if torch.cuda.is_available():
        return "note: CUDA visible, but everything here runs on CPU"
    return f"{len(vers)} packages, torch CPU threads={torch.get_num_threads()}"


@step("MOMENT-1-small reconstruction (1 x 1 x 512)")
def check_moment():
    import torch
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-small", model_kwargs={"task_name": "reconstruction"})
    model.init()
    model.eval()
    x = torch.randn(1, 1, 512)
    mask = torch.ones(1, 512, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x_enc=x, input_mask=mask)
    rec = out.reconstruction
    assert tuple(rec.shape) == (1, 1, 512), rec.shape
    assert torch.isfinite(rec).all()
    return f"inference {time.perf_counter() - t0:.2f} s"


@step("MOMENT-1-small embedding (8 x 3 x 512)")
def check_moment_embed():
    import torch
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-small", model_kwargs={"task_name": "embedding"})
    model.init()
    model.eval()
    x = torch.randn(8, 3, 512)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x_enc=x)
    emb = out.embeddings
    assert emb.shape[0] == 8 and torch.isfinite(emb).all()
    return f"shape {tuple(emb.shape)}, inference {time.perf_counter() - t0:.2f} s"


def _chronos(repo):
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(repo, device_map="cpu", dtype=torch.float32)
    ctx = torch.sin(torch.linspace(0, 20, 256)) + 0.1 * torch.randn(256)
    if type(pipe).__name__.startswith("Chronos2"):
        # Chronos-2 takes (n_series, n_variates, length): 3 curves at once, like GR/NPHI/RHOB
        ctx = torch.stack([ctx, ctx.roll(5), -ctx]).unsqueeze(0)
    else:
        ctx = ctx.unsqueeze(0)  # Chronos-Bolt is univariate: (n_series, length)
    t0 = time.perf_counter()
    q, mean = pipe.predict_quantiles(ctx, prediction_length=32,
                                     quantile_levels=[0.1, 0.5, 0.9])
    q = q[0] if isinstance(q, list) else q
    assert torch.isfinite(torch.as_tensor(q)).all()
    return f"{type(pipe).__name__}, quantiles {tuple(q.shape)}, inference {time.perf_counter() - t0:.2f} s"


@step("Chronos-Bolt-tiny forecast (256 -> 32)")
def check_bolt_tiny():
    return _chronos("amazon/chronos-bolt-tiny")


@step("Chronos-Bolt-small forecast (256 -> 32)")
def check_bolt_small():
    return _chronos("amazon/chronos-bolt-small")


@step("Chronos-2 forecast (3 variates, 256 -> 32)")
def check_chronos2():
    return _chronos("amazon/chronos-2")


@step("TTM r2 forecast (1 x 512 x 3 -> 96)")
def check_ttm():
    import torch
    from tsfm_public import TinyTimeMixerForPrediction
    model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2", revision="main")
    model.eval()
    x = torch.randn(1, 512, 3)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(past_values=x)
    y = out.prediction_outputs
    assert y.shape[1] == 96 and torch.isfinite(y).all(), y.shape
    return f"output {tuple(y.shape)}, inference {time.perf_counter() - t0:.2f} s"


@step("XGBoost on FORCE 2020 rows (lithology + RHOB)")
def check_xgb(data_dir: Path):
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error

    csv = data_dir / "train.csv"
    src = csv if csv.exists() else data_dir / "train.zip"
    feats = ["DEPTH_MD", "GR", "NPHI", "RDEP", "DTC"]
    cols = ["WELL", "RHOB", "FORCE_2020_LITHOFACIES_LITHOLOGY"] + feats
    df = pd.read_csv(src, sep=";", usecols=cols, nrows=60000)
    df = df.dropna(subset=["GR", "RHOB"])
    wells = df["WELL"].unique()
    assert len(wells) >= 2, "need at least two wells in the sample"
    test = df[df["WELL"] == wells[-1]]
    train = df[df["WELL"] != wells[-1]]

    # Lithology classifier, scored with the contest penalty matrix
    A = np.load(data_dir / "penalty_matrix.npy")
    ytr = train["FORCE_2020_LITHOFACIES_LITHOLOGY"].map(LITHO).to_numpy()
    yte = test["FORCE_2020_LITHOFACIES_LITHOLOGY"].map(LITHO).to_numpy()
    classes = np.unique(ytr)
    remap = {c: i for i, c in enumerate(classes)}
    clf = xgb.XGBClassifier(n_estimators=50, max_depth=6, tree_method="hist", n_jobs=-1)
    t0 = time.perf_counter()
    clf.fit(train[feats], np.vectorize(remap.get)(ytr))
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = classes[clf.predict(test[feats])]
    t_pred = (time.perf_counter() - t0) / len(test) * 1000
    score = -np.mean(A[yte, pred])
    acc = float(np.mean(pred == yte))

    # RHOB regression (experiment A baseline shape)
    reg = xgb.XGBRegressor(n_estimators=100, max_depth=6, tree_method="hist", n_jobs=-1)
    reg.fit(train[feats], train["RHOB"])
    mae = mean_absolute_error(test["RHOB"], reg.predict(test[feats]))
    return (f"{len(train)} train / {len(test)} test rows, {len(wells)} wells; "
            f"litho penalty score {score:.3f}, acc {acc:.2f}, fit {t_fit:.2f} s, predict {t_pred * 1000:.1f} ms/1000; "
            f"RHOB MAE {mae:.3f} g/cc")


@step("lasio reads one FORCE LAS file")
def check_las(data_dir: Path):
    import io
    import zipfile
    import lasio
    zpath = data_dir / "LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip"
    with zipfile.ZipFile(zpath) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".las"))
        text = zf.read(name).decode("utf-8", "replace")
    las = lasio.read(io.StringIO(text))
    return f"{Path(name).name}: {len(las.curves)} curves, {len(las.index)} samples"


@step("DuckDB query on train.csv")
def check_duckdb(data_dir: Path):
    import duckdb
    csv = data_dir / "train.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} (run the data step first)")
    con = duckdb.connect()
    n_wells, n_rows = con.execute(
        "SELECT count(DISTINCT WELL), count(*) FROM read_csv(?, delim=';', header=true)",
        [str(csv)]).fetchone()
    return f"{n_wells} wells, {n_rows} rows"


@step("Sodir FactPages snapshot readable")
def check_sodir(sodir_dir: Path):
    import pandas as pd
    tops = pd.read_csv(sodir_dir / "strat_litho_wellbore.csv")
    wb = pd.read_csv(sodir_dir / "wellbore_exploration_all.csv", low_memory=False)
    return f"{len(tops)} tops, {len(wb)} exploration wellbores"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--online", action="store_true", help="allow Hugging Face network access")
    a = ap.parse_args()
    root = Path(a.cache_root).expanduser().resolve()
    os.environ.setdefault("HF_HOME", str(root / "hf"))
    if not a.online:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    print(f"Hardware: {hardware()}")
    print(f"Cache root {root}  (HF_HOME={os.environ['HF_HOME']}, offline={not a.online})\n")

    check_imports()
    if not a.skip_models:
        check_moment()
        check_moment_embed()
        check_bolt_tiny()
        check_bolt_small()
        check_chronos2()
        check_ttm()
    if not a.skip_data:
        data_dir = root / "data" / "force2020"
        check_xgb(data_dir)
        check_las(data_dir)
        check_duckdb(data_dir)
        sodir = root / "data" / "sodir_factpages"
        if (sodir / "strat_litho_wellbore.csv").exists():
            check_sodir(sodir)
        else:
            print("[SKIP] Sodir snapshot not downloaded (optional, use -WithSodir)")

    print("\n================ SMOKE TEST SUMMARY ================")
    for name, ok, dt, note in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {dt:6.1f} s  {name}")
    n_fail = sum(not ok for _, ok, _, _ in RESULTS)
    print("====================================================")
    print("ALL PASSED" if n_fail == 0 else f"{n_fail} STEP(S) FAILED - see messages above")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
