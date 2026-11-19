"""Re-runnable onboarding: data paths and the datasets a contributor wants.

    python -m wmb.onboarding [--status] [--redo [data_paths|force2020|...]]
                             [--data-root PATH] [--choose NAME] [--decline NAME]
                             [--non-interactive] [--state FILE]

State lives in the gitignored ``<repo>/.wmb/onboarding_state.json`` (override with
``--state`` or ``WMB_ONBOARDING_STATE``). Bump ``ONBOARDING_VERSION`` and add a
``CHANGELOG`` line when a dataset or step is added.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from wmb import paths

ONBOARDING_VERSION = 1
CHANGELOG: dict[int, str] = {
    1: "Initial onboarding: data paths and the FORCE 2020 demo dataset.",
}

OFFICIAL_DEMO = "force2020"
COMING_DATASETS = {
    "nlog_dts": "NLOG-DTS",
    "kansas_kgs_dts": "Kansas-KGS-DTS",
    "volve": "Volve",
    "sodir_factpages": "Sodir FactPages",
}
VERIFIABLE = {OFFICIAL_DEMO}
STEPS = ("data_paths", "datasets")


def state_path(override: str | Path | None = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("WMB_ONBOARDING_STATE")
    return Path(env) if env else paths.repo_root() / ".wmb" / "onboarding_state.json"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_state(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"onboarding_version": 0, "updated": None, "data_root": None, "steps": {}, "datasets": {}}


def save_state(path: Path, state: dict) -> None:
    state["onboarding_version"] = ONBOARDING_VERSION
    state["updated"] = _now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def whats_new(saved_version: int) -> list[tuple[int, str]]:
    return sorted((v, t) for v, t in CHANGELOG.items() if v > saved_version)


class Prompter:
    """input() wrapper that falls back to defaults when there is no usable stdin."""

    def __init__(self, interactive: bool = True):
        self.interactive = interactive

    def ask(self, question: str, default: str = "") -> str:
        if not self.interactive:
            return default
        suffix = f" [{default}]" if default else ""
        try:
            answer = input(f"{question}{suffix}: ").strip()
        except (EOFError, OSError):
            self.interactive = False
            return default
        return answer or default

    def yes_no(self, question: str, default: bool) -> bool:
        ans = self.ask(f"{question} (y/n)", "y" if default else "n").lower()
        return ans.startswith("y")

    def choice(self, name: str, default: str) -> str:
        ans = self.ask(f"{name}: use (y), decline (n) or skip for now (s)", {"chosen": "y", "declined": "n"}.get(default, "s")).lower()
        return {"y": "chosen", "n": "declined"}.get(ans[:1], "skipped")


def verify_dataset(key: str) -> bool:
    """Run the same verification as ``python -m wmb.datasets --check``."""
    from wmb.datasets.force2020 import check_force2020

    if key != OFFICIAL_DEMO:
        return False
    print(f"Verifying {key} ...")
    return check_force2020() == 0


def _wants(redo: str | None, name: str) -> bool:
    return redo in ("all", name)


def _incomplete(entry: dict) -> bool:
    return entry.get("choice") not in {"chosen", "declined"}


def _write_paths_preserving(root: str | Path, dest: Path) -> Path:
    """Update data_root without dropping other keys already in the yaml."""
    root = Path(root).expanduser()
    if dest.is_file():
        try:
            import yaml
        except ImportError:
            return paths.write_local_paths(root, dest)
        data = yaml.safe_load(dest.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return paths.write_local_paths(root, dest)
        data["data_root"] = root.as_posix()
        dest.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        paths._reset_for_tests()
        return dest
    return paths.write_local_paths(root, dest)


def run(args: argparse.Namespace) -> int:
    spath = state_path(args.state)
    state = load_state(spath)
    saved = int(state.get("onboarding_version", 0))
    p = Prompter(interactive=not args.non_interactive)
    redo = args.redo
    choose, decline = set(args.choose or []), set(args.decline or [])
    first_run = saved == 0

    print(f"wmb onboarding (version {ONBOARDING_VERSION}); state: {spath}")
    if not first_run and saved < ONBOARDING_VERSION:
        print("What's new since your last run:")
        for v, text in whats_new(saved):
            print(f"  v{v}: {text}")

    steps = state.setdefault("steps", {})
    datasets = state.setdefault("datasets", {})

    # Step 1: data_paths.local.yaml
    cfg = paths.config_path()
    if steps.get("data_paths", {}).get("done") and cfg.is_file() and not _wants(redo, "data_paths") \
            and not args.data_root:
        print("Step 1 (data paths): done, skipping.")
    else:
        root = args.data_root
        if not root and cfg.is_file() and not _wants(redo, "data_paths"):
            print(f"Step 1 (data paths): using existing {cfg}")
            paths._reset_for_tests()
        else:
            if not root:
                existing = state.get("data_root") or ""
                root = p.ask("Folder that holds (or will hold) your well data", existing)
            if root:
                written = _write_paths_preserving(root, cfg)
                print(f"Step 1 (data paths): wrote {written} (local, not committed).")
                state["data_root"] = str(Path(root).expanduser())
            else:
                print("Step 1 (data paths): no folder given. See docs/SETUP_DATA_PATHS.md and rerun.")
        if cfg.is_file():
            try:
                state["data_root"] = str(paths.data_root())
                steps["data_paths"] = {"done": True, "date": _now()}
            except paths.DataPathsError as exc:
                print(f"Step 1 (data paths): config not usable yet ({exc}).")

    # Step 2: FORCE 2020 demo, then coming datasets
    paths._reset_for_tests()
    entry = datasets.setdefault(OFFICIAL_DEMO, {})
    if OFFICIAL_DEMO in choose:
        entry["choice"] = "chosen"
    elif OFFICIAL_DEMO in decline:
        entry["choice"] = "declined"
    elif _incomplete(entry) or _wants(redo, OFFICIAL_DEMO) or redo == "datasets":
        print("FORCE 2020 is the official demo dataset (118 wells, see datasets/force2020/README.md).")
        entry["choice"] = p.choice(OFFICIAL_DEMO, entry.get("choice") or "chosen")
    else:
        print(f"Step 2 ({OFFICIAL_DEMO}): already {entry['choice']}, skipping.")

    for key, label in COMING_DATASETS.items():
        e = datasets.setdefault(key, {})
        if key in choose:
            e["choice"] = "chosen"
        elif key in decline:
            e["choice"] = "declined"
        elif _incomplete(e) or _wants(redo, key) or redo == "datasets":
            if _incomplete(e):
                print(f"Coming dataset: {label} (not available yet)")
            e["choice"] = p.choice(key, e.get("choice") or "skipped")

    for key, e in datasets.items():
        if e.get("choice") == "chosen" and key in VERIFIABLE:
            ok = False
            try:
                ok = verify_dataset(key)
            except (paths.DataPathsError, OSError, RuntimeError) as exc:
                print(f"  could not verify {key}: {exc}")
            e["verified"], e["verified_date"] = ok, _now()
            if not ok:
                print(f"  {key} not verified. Fetch it with: python datasets/{key}/fetch.py "
                      f"(see datasets/{key}/README.md), then rerun: python -m wmb.onboarding")
        elif e.get("choice") != "chosen":
            e.pop("verified", None)
            e.pop("verified_date", None)
    steps["datasets"] = {"done": True, "date": _now()}
    save_state(spath, state)
    print(f"Saved {spath}")
    return 0


def status(args: argparse.Namespace) -> int:
    spath = state_path(args.state)
    state = load_state(spath)
    saved = int(state.get("onboarding_version", 0))
    print(f"onboarding version: {saved or 'not started'} (current {ONBOARDING_VERSION})")
    print(f"state file: {spath}" + ("" if spath.is_file() else "  (not created yet)"))
    print(f"updated: {state.get('updated') or '-'}")
    print(f"data_root: {state.get('data_root') or '-'}")
    steps = state.get("steps", {})
    print("steps: " + ", ".join(f"{s}={'done' if steps.get(s, {}).get('done') else 'pending'}" for s in STEPS))
    print("datasets:")
    for key in (OFFICIAL_DEMO, *COMING_DATASETS):
        e = state.get("datasets", {}).get(key, {})
        ver = ""
        if e.get("choice") == "chosen" and key in VERIFIABLE:
            ver = ("  verified " if e.get("verified") else "  NOT verified ") + (e.get("verified_date") or "never")
        print(f"  {key:<16} {e.get('choice', 'undecided')}{ver}")
    news = whats_new(saved) if spath.is_file() else []
    if news:
        print("what's new since the saved version:")
        for v, text in news:
            print(f"  v{v}: {text}")
        print("Run: python -m wmb.onboarding")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m wmb.onboarding", description=__doc__.split("\n")[0])
    ap.add_argument("--status", action="store_true", help="print state; no prompts")
    ap.add_argument("--redo", nargs="?", const="all", metavar="NAME",
                    help="redo everything, or one step/dataset (data_paths, force2020, ...)")
    ap.add_argument("--data-root", help="data folder (skips the prompt)")
    ap.add_argument("--choose", action="append", metavar="DATASET", help="use this dataset (repeatable)")
    ap.add_argument("--decline", action="append", metavar="DATASET", help="decline this dataset (repeatable)")
    ap.add_argument("--non-interactive", action="store_true", help="never prompt; use flags and defaults")
    ap.add_argument("--state", help="state file (default <repo>/.wmb/onboarding_state.json)")
    args = ap.parse_args(argv)
    if args.status:
        return status(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
