"""Benchmark runner and scoreboard (owned by the Benchmark workstream)."""


def __getattr__(name: str):
    # lazy, so `python -m benchmark.make_tasks` does not import itself twice
    if name == "load_tasks":
        from benchmark.make_tasks import load_tasks
        return load_tasks
    raise AttributeError(name)
