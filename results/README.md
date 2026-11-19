# Results (gitignored)

`python -m benchmark.run` writes `results/<contender>/<dataset>/<target>/scores.json` here.
Everything in this folder is ignored by git **except** this README and small summary tables in
`results/summary/` (`*.md`, `*.json`), written by `python -m benchmark.scoreboard --write`.
Predictions with curve values are data: keep them out of git.
