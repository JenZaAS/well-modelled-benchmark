# weights/

Local cache for the `red_ttm` and `red_moment` backbones and TTM fine-tunes.
Everything here except this README is git-ignored. Never commit weights.

```
weights/
  hf/                                Hugging Face cache (e.g. AutonLab/MOMENT-1-small)
  local/granite-timeseries-ttm-r2/   optional local snapshot of the TTM base model
  ttm/<TARGET>/                      point-forecast fine-tune (config.json, model.safetensors)
  ttm/<TARGET>-probabilistic/        quantile (q05/q50/q95) fine-tune plus conformal-width metadata
```

Put the cache elsewhere with `WMB_RED_CACHE` (the legacy `WMB_WHITE_CACHE` still works).
Resolution order: `WMB_RED_CACHE`, `WMB_WHITE_CACHE`, this folder, then the old
`workstreams/white_finetuned_tsfm/weights/` while it still holds `hf/` or `ttm/` and
this folder holds neither.
