"""Torch side: load the pinned IBM Granite checkpoints and run batched forwards.

Imported only by the opt-in Granite contenders and the experiment pipeline, so default physics runs never
need torch. Weights are read from the local Hugging Face cache (`<cache_dir>/hf/hub`) with
`local_files_only=True`; `fetch_weights()` is the one explicit download.

Input policy (both models): target only. No calendar, frequency token or covariate is fed. A synthetic array
index stands in for time; one native depth sample is one model step.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from workstreams.black_physics.granite import config as C

TTM_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TTM_HEAD_KEYS = (".head.", ".multi_quantile_head_block.")       # trend + residual prediction heads
TSPULSE_HEAD_KEYS = ("decoder_with_head.time_head.",)           # time-domain reconstruction head only


def _torch():
    import torch

    return torch


def snapshot_path(repo: str, revision: str, *, download: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, cache_dir=str(C.hf_home() / "hub"),
                                  allow_patterns=["config.json", "model.safetensors", "README.md", "model.sig"],
                                  local_files_only=not download))


def fetch_weights() -> dict:
    """Download the two pinned snapshots once (explicit; nothing else downloads)."""
    return {repo: str(snapshot_path(repo, rev, download=True))
            for repo, rev in ((C.TTM_REPO, C.TTM_REVISION), (C.TSPULSE_REPO, C.TSPULSE_REVISION))}


def pick_device(prefer: str | None = None):
    torch = _torch()
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def versions() -> dict:
    import importlib.metadata as md

    torch = _torch()
    out = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    for pkg in ("transformers", "granite-tsfm", "numpy"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


class _Backend:
    head_keys: tuple[str, ...] = ()

    def head_names(self) -> list[str]:
        return [n for n, _ in self.model.named_parameters() if any(k in n for k in self.head_keys)]

    def head_state(self) -> dict:
        names = set(self.head_names())
        return {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if n in names}

    def load_head(self, state: dict) -> None:
        torch = _torch()
        params = dict(self.model.named_parameters())
        names = set(self.head_names())
        if set(state) != names:
            raise ValueError(f"head checkpoint keys do not match the model head ({len(state)} vs {len(names)})")
        with torch.no_grad():
            for n, v in state.items():
                if params[n].shape != v.shape:
                    raise ValueError(f"head tensor {n}: shape {tuple(v.shape)} != {tuple(params[n].shape)}")
                params[n].copy_(v.to(params[n].device))

    def set_trainable(self) -> list[str]:
        """Freeze everything except the head; frozen modules stay in eval mode, the head in train mode."""
        names = set(self.head_names())
        for n, p in self.model.named_parameters():
            p.requires_grad_(n in names)
        self.model.eval()
        for mn, m in self.model.named_modules():
            if mn and any(k in f".{mn}." for k in self.head_keys):
                m.train()
        return sorted(names)

    def n_parameters(self) -> dict:
        names = set(self.head_names())
        tot = sum(p.numel() for p in self.model.parameters())
        head = sum(p.numel() for n, p in self.model.named_parameters() if n in names)
        return {"total": int(tot), "head": int(head)}


class TTMBackend(_Backend):
    """TTM-R3 lite, decomposed (trend + residual), context 512 -> 336 steps, quantiles 0.1..0.9."""

    arm = "ttm"
    head_keys = TTM_HEAD_KEYS

    def __init__(self, device=None, head_state: dict | None = None, path: Path | None = None):
        from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import TinyTimeMixerForDecomposedPrediction

        self.device = pick_device(device) if not hasattr(device, "type") else device
        p = path or snapshot_path(C.TTM_REPO, C.TTM_REVISION)
        # Load this class explicitly: IBM's get_model dispatches on the "-dec-" branch name, not on a SHA.
        self.model = TinyTimeMixerForDecomposedPrediction.from_pretrained(str(p)).to(self.device).eval()
        cfg = self.model.config
        assert cfg.decompose is True, "TTM-R3 checkpoint must be decomposed"
        assert cfg.context_length == C.CONFIG.ttm_context and cfg.prediction_length == C.CONFIG.ttm_horizon
        assert not cfg.resolution_prefix_tuning, "no frequency prefix may be used"
        assert [round(q, 6) for q in cfg.quantile_levels] == list(TTM_QUANTILES), cfg.quantile_levels
        self.context, self.horizon = cfg.context_length, cfg.prediction_length
        if head_state is not None:
            self.load_head(head_state)

    def batch(self, contexts: list[np.ndarray]):
        """Left-pad each finite context to 512 with its own mean (zero after TTM's std scaling), mask False."""
        torch = _torch()
        n, L = len(contexts), self.context
        x = np.zeros((n, L), np.float32)
        m = np.zeros((n, L), bool)
        for i, c in enumerate(contexts):
            c = np.asarray(c, float)[-L:]
            if not len(c) or not np.isfinite(c).all():
                raise ValueError("TTM context must be non-empty and finite")
            x[i] = float(c.mean())
            x[i, L - len(c):] = c
            m[i, L - len(c):] = True
        return (torch.from_numpy(x).unsqueeze(-1).to(self.device), torch.from_numpy(m).unsqueeze(-1).to(self.device))

    def forward(self, past, mask):
        """Quantile forecasts [B, 9, 336] in input units (the model inverts its own scaling)."""
        out = self.model(past_values=past, past_observed_mask=mask, return_loss=False)
        return out.quantile_outputs[..., 0]

    def forecast(self, contexts: list[np.ndarray], batch_size: int = 128) -> np.ndarray:
        torch = _torch()
        res = []
        with torch.no_grad():
            for i in range(0, len(contexts), batch_size):
                past, mask = self.batch(contexts[i:i + batch_size])
                res.append(self.forward(past, mask).float().cpu().numpy())
        return np.concatenate(res) if res else np.zeros((0, len(TTM_QUANTILES), self.horizon))


class TSPulseBackend(_Backend):
    """TSPulse R1 hybrid dual-head, 512-sample masked reconstruction with a user mask."""

    arm = "tspulse"
    head_keys = TSPULSE_HEAD_KEYS

    def __init__(self, device=None, head_state: dict | None = None, path: Path | None = None):
        from tsfm_public.models.tspulse import TSPulseForReconstruction

        self.device = pick_device(device) if not hasattr(device, "type") else device
        p = path or snapshot_path(C.TSPULSE_REPO, C.TSPULSE_REVISION)
        self.model = TSPulseForReconstruction.from_pretrained(str(p), num_input_channels=1,
                                                              mask_type="user").to(self.device).eval()
        cfg = self.model.config
        assert cfg.context_length == C.CONFIG.tspulse_window and cfg.mask_type == "user"
        self.window = cfg.context_length
        if head_state is not None:
            self.load_head(head_state)

    def batch(self, windows: np.ndarray):
        """NaN = unobserved. Placeholders are the row's observed mean (replaced by TSPulse's mask token anyway)."""
        torch = _torch()
        w = np.asarray(windows, float)
        if w.ndim != 2 or w.shape[1] != self.window:
            raise ValueError(f"TSPulse windows must be [n, {self.window}]")
        m = np.isfinite(w)
        if not m.any(axis=1).all():
            raise ValueError("every TSPulse window needs at least one observed value")
        fill = np.nanmean(np.where(m, w, np.nan), axis=1, keepdims=True)
        x = np.where(m, w, fill).astype(np.float32)
        return (torch.from_numpy(x).unsqueeze(-1).to(self.device), torch.from_numpy(m).unsqueeze(-1).to(self.device))

    def forward(self, past, mask):
        """Reconstruction [B, 512] in input units (TSPulse inverts its RevIN scaling for this output)."""
        out = self.model(past_values=past, past_observed_mask=mask, return_loss=False)
        return out.reconstruction_outputs[..., 0]

    def reconstruct(self, windows: np.ndarray, batch_size: int = 128) -> np.ndarray:
        torch = _torch()
        res = []
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                past, mask = self.batch(windows[i:i + batch_size])
                res.append(self.forward(past, mask).float().cpu().numpy())
        return np.concatenate(res) if res else np.zeros((0, self.window))


def make_backend(arm: str, device=None, head_state: dict | None = None):
    if arm == "ttm":
        return TTMBackend(device, head_state)
    if arm == "tspulse":
        return TSPulseBackend(device, head_state)
    raise ValueError(arm)
