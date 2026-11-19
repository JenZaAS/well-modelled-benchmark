"""FH2026 offline prep: download FORCE 2020 data, Sodir FactPages CSVs, offline docs and model weights.

Safe to re-run: files that already exist with the right size and checksum are skipped.
Called by setup-fh2026.ps1, but works on its own:

    python fh2026_fetch.py --cache-root C:\\Users\\me\\fh2026-cache --data --models --sodir --docs

Everything goes under the cache root, never into a git repo.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import sys
import time
import urllib.parse
from pathlib import Path

import requests

ZENODO = "https://zenodo.org/api/records/4351156/files/{}/content"
GITHUB = ("https://raw.githubusercontent.com/bolgebrygg/"
          "Force-2020-Machine-Learning-competition/master/lithology_competition/data/{}")

# name, url, size in bytes, (algorithm, hex digest). Verified 2026-11-17.
FORCE_FILES = [
    # GitHub: competition CSVs in the shape everyone uses (semicolon-separated)
    ("train.zip", GITHUB.format("train.zip"), 91653972,
     ("sha256", "e996c8466f2d0dc7116daa71485cc90c31fcfac6345a5bb33165026478b642e9")),
    ("leaderboard_test_features.csv", GITHUB.format("leaderboard_test_features.csv"), 31688732,
     ("sha256", "4688d93728feab7e86855b9315ddfc8b498387927954235f7dbbfa2216550781")),
    ("leaderboard_test_target.csv", GITHUB.format("leaderboard_test_target.csv"), 3694123,
     ("sha256", "4fe5e59809d98eb9b8b0a3836776bd52e5c7e63ffefa31476fa17011a27ef382")),
    ("hidden_test.csv", GITHUB.format("hidden_test.csv"), 30292497,
     ("sha256", "c75e817bfefcabd2bacdefd9e1c74dbd829d24543646bf0b85ef8fbf5c06f069")),
    ("penalty_matrix.npy", GITHUB.format("penalty_matrix.npy"), 1280,
     ("sha256", "c312481af94d3557616b7ed79b5d1a12afee380fd471188785ab31874f159d43")),
    # Zenodo record 4351156 (CC BY 4.0): all LAS files plus NPD tables and scoring matrix
    ("LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip",
     ZENODO.format("LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip"), 169785172,
     ("md5", "da0d72d5783a780ac71779453d9fbfbf")),
    ("lithology scoring matrix cost function.xlsx",
     ZENODO.format(urllib.parse.quote("lithology scoring matrix cost function.xlsx")), 14558,
     ("md5", "61bd1a80b00a22b53ce8cd54150a1da0")),
    ("NPD_Lithostratigraphy_member_formations_all_wells.xlsx",
     ZENODO.format("NPD_Lithostratigraphy_member_formations_all_wells.xlsx"), 168501,
     ("md5", "f17eb022983ec728a5e811892aeea41b")),
    ("NPD_Lithostratigraphy_groups_all_wells.xlsx",
     ZENODO.format("NPD_Lithostratigraphy_groups_all_wells.xlsx"), 80797,
     ("md5", "5c42fbc98b446c170fd9924dfd5cf546")),
    ("NPD_Casing_depth_most_wells.xlsx",
     ZENODO.format("NPD_Casing_depth_most_wells.xlsx"), 27080,
     ("md5", "45e291e6e5ff251056ff288978ef4ffb")),
]

FACTPAGES = ("https://factpages.sodir.no/public?/Factpages/external/tableview/{}"
             "&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used"
             "&CultureCode=en&rs:Format=CSV&Top100=false")
SODIR_TABLES = [
    "wellbore_exploration_all",   # name, NPDID, coordinates, KB elevation, water depth, TD, content
    "wellbore_development_all",
    "wellbore_other_all",
    "strat_litho_wellbore",       # lithostratigraphic tops/bases per wellbore (group/formation/member)
    "strat_litho_wellbore_core",
    "field",
    "licence",
]

GH = "https://raw.githubusercontent.com/"
HF = "https://huggingface.co/{}/raw/main/README.md"
# Offline docs: (file name, url). No checksums (they change); skipped if present.
DOCS = [
    ("duckdb-docs.pdf", "https://blobs.duckdb.org/docs/duckdb-docs.pdf"),
    ("duckdb-docs.md", "https://blobs.duckdb.org/docs/duckdb-docs.md"),
    ("lasio-README.md", GH + "kinverarity1/lasio/main/README.md"),
    ("lasio-basic-example.rst", GH + "kinverarity1/lasio/main/docs/source/basic-example.rst"),
    ("lasio-data-section.rst", GH + "kinverarity1/lasio/main/docs/source/data-section.rst"),
    ("pdfplumber-README.md", GH + "jsvine/pdfplumber/stable/README.md"),
    ("moment-README.md", GH + "moment-timeseries-foundation-model/moment/main/README.md"),
    ("moment-tutorial-imputation.ipynb", GH + "moment-timeseries-foundation-model/moment/main/tutorials/imputation.ipynb"),
    ("moment-1-small-modelcard.md", HF.format("AutonLab/MOMENT-1-small")),
    ("chronos-README.md", GH + "amazon-science/chronos-forecasting/main/README.md"),
    ("chronos-2-quickstart.ipynb", GH + "amazon-science/chronos-forecasting/main/notebooks/chronos-2-quickstart.ipynb"),
    ("chronos-2-modelcard.md", HF.format("amazon/chronos-2")),
    ("chronos-bolt-small-modelcard.md", HF.format("amazon/chronos-bolt-small")),
    ("granite-tsfm-README.md", GH + "ibm-granite/granite-tsfm/main/README.md"),
    ("ttm-r2-modelcard.md", HF.format("ibm-granite/granite-timeseries-ttm-r2")),
    ("force2020-README.md", GH + "bolgebrygg/Force-2020-Machine-Learning-competition/master/lithology_competition/readme.md"),
    ("force2020-CITATION.cff", GH + "bolgebrygg/Force-2020-Machine-Learning-competition/master/CITATION.cff"),
    ("force2020-starter_notebook.ipynb", GH + "bolgebrygg/Force-2020-Machine-Learning-competition/master/lithology_competition/data/starter_notebook.ipynb"),
    ("force2020-technical-retrospective.pdf", GH + "bolgebrygg/Force-2020-Machine-Learning-competition/master/lithology_competition/technical_retrospective_force%202020%20lithofacies%20competition.pdf"),
    ("NLOD-2.0.html", "https://data.norge.no/nlod/en/2.0"),
]

# repo id -> licence (checked on the model cards 2026-11-17)
MODELS = {
    "AutonLab/MOMENT-1-small": "MIT",
    "amazon/chronos-bolt-tiny": "Apache-2.0",
    "amazon/chronos-bolt-small": "Apache-2.0",
    "amazon/chronos-2": "Apache-2.0",
    "ibm-granite/granite-timeseries-ttm-r2": "Apache-2.0",  # 'main' = 512 context, 96 horizon
}

def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_good(path: Path, size: int, check: tuple[str, str]) -> bool:
    return path.exists() and path.stat().st_size == size and digest(path, check[0]) == check[1]


def download(url: str, dest: Path, timeout: int = 120) -> None:
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done, last = 0, time.time()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 5:
                    pct = f"{100 * done / total:5.1f}%" if total else ""
                    print(f"      {done / 1e6:8.1f} MB {pct}", flush=True)
                    last = time.time()
    os.replace(tmp, dest)


def fetch_force(root: Path) -> list[tuple[str, bool, str]]:
    out_dir = root / "data" / "force2020"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, url, size, check in FORCE_FILES:
        dest = out_dir / name
        label = f"FORCE {name}"
        if is_good(dest, size, check):
            print(f"  skip  {name} (already there, checksum OK)")
            results.append((label, True, "cached"))
            continue
        print(f"  get   {name} ({size / 1e6:.1f} MB)")
        try:
            download(url, dest)
            ok = is_good(dest, size, check)
            results.append((label, ok, "downloaded" if ok else "size/checksum mismatch"))
            if not ok:
                print(f"  FAIL  {name}: size or checksum does not match")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {name}: {e}")
            results.append((label, False, str(e)[:120]))
    # Unpack train.zip once so pandas/duckdb can read train.csv directly
    z = out_dir / "train.zip"
    if z.exists() and not (out_dir / "train.csv").exists():
        import zipfile
        print("  unzip train.zip -> train.csv (~280 MB)")
        with zipfile.ZipFile(z) as zf:
            zf.extract("train.csv", out_dir)
    return results


def fetch_sodir(root: Path, refresh: bool) -> list[tuple[str, bool, str]]:
    out_dir = root / "data" / "sodir_factpages"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for table in SODIR_TABLES:
        dest = out_dir / f"{table}.csv"
        label = f"Sodir {table}"
        if dest.exists() and dest.stat().st_size > 1000 and not refresh:
            print(f"  skip  {table}.csv (already there; use -RefreshSodir to update)")
            results.append((label, True, "cached"))
            continue
        print(f"  get   {table}.csv")
        try:
            download(FACTPAGES.format(table), dest, timeout=300)
            head = dest.read_bytes()[:200].decode("utf-8-sig", "replace")
            ok = dest.stat().st_size > 1000 and "," in head and "<html" not in head.lower()
            results.append((label, ok, f"{dest.stat().st_size / 1e6:.1f} MB" if ok else "not a CSV"))
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {table}: {e}")
            results.append((label, False, str(e)[:120]))
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    (out_dir / "SOURCE.txt").write_text(
        "Source: Sodir FactPages (https://factpages.sodir.no), CSV table exports.\n"
        "Licence: Norwegian Licence for Open Government Data (NLOD 2.0). Credit: Sodir.\n"
        f"Retrieved (UTC): {stamp}\n", encoding="utf-8")
    return results


def fetch_docs(root: Path) -> list[tuple[str, bool, str]]:
    out_dir = root / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, url in DOCS:
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 500:
            print(f"  skip  {name}")
            results.append((f"doc {name}", True, "cached"))
            continue
        print(f"  get   {name}")
        try:
            download(url, dest)
            results.append((f"doc {name}", True, f"{dest.stat().st_size / 1e3:.0f} kB"))
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {name}: {e}")
            results.append((f"doc {name}", False, str(e)[:120]))
    return results


def fetch_models(root: Path) -> list[tuple[str, bool, str]]:
    os.environ.setdefault("HF_HOME", str(root / "hf"))
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import snapshot_download
    results = []
    for repo, lic in MODELS.items():
        label = f"model {repo}"
        print(f"  get   {repo} ({lic})")
        try:
            path = snapshot_download(repo_id=repo, revision="main")
            results.append((label, True, path))
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {repo}: {e}")
            results.append((label, False, str(e)[:120]))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--data", action="store_true", help="FORCE 2020 files")
    ap.add_argument("--sodir", action="store_true", help="Sodir FactPages CSVs")
    ap.add_argument("--refresh-sodir", action="store_true")
    ap.add_argument("--models", action="store_true", help="Hugging Face model weights")
    ap.add_argument("--docs", action="store_true", help="offline docs and licence texts (~20 MB)")
    a = ap.parse_args()
    root = Path(a.cache_root).expanduser().resolve()
    results = []
    if a.data:
        print("\n== FORCE 2020 data ==")
        results += fetch_force(root)
    if a.sodir:
        print("\n== Sodir FactPages snapshot ==")
        results += fetch_sodir(root, a.refresh_sodir)
    if a.docs:
        print("\n== Offline docs ==")
        results += fetch_docs(root)
    if a.models:
        print("\n== Model weights (HF_HOME = %s) ==" % os.environ.get("HF_HOME", root / "hf"))
        results += fetch_models(root)
    print("\n== Download summary ==")
    for label, ok, note in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}  [{note}]")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
