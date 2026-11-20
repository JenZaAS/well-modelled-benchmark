# red_moment — AutonLab MOMENT-1-small

Zero-shot wrapper around [`AutonLab/MOMENT-1-small`](https://huggingface.co/AutonLab/MOMENT-1-small).
Part of the red foundation-model workstream; enable it with `WMB_RED_MODELS=moment`
(opt-in; legacy `WMB_WHITE_MODELS=moment` also works). Licence: [MIT](LICENSE.md).
The published head is **reconstruction/imputation** (the forecasting head is not pre-trained), so the backend fills a masked gap inside a 512-sample window. p10/p90 use the same training-set interval as `red_ttm`.

`momentfm==0.1.4` *declares* `transformers==4.33.3`. Install with `--no-deps` so those pins are not applied. This was verified on the box with `transformers 4.57.6` and `torch 2.11.0+cpu`.
Full install notes, weights handling and a quick check: **[SETUP.md](SETUP.md)**.

```bash
python -m pip install --no-deps momentfm==0.1.4
export WMB_RED_MODELS=moment HF_HUB_OFFLINE=1
python -m benchmark.check_model red_moment
python -m benchmark.run --model red_moment --test rhob_gap50
```

`MOMENT-1-large` is a heavier option and is not the default.

Earlier results were recorded under the id `white_moment`; that id is a deprecated alias
for `red_moment` (see `src/wmb/aliases.py`), and `results/white_moment/` is still read.
