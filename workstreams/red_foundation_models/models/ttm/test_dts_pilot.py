"""Checkpoint identity and opt-in contracts for the published DTS comparison."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workstreams.red_foundation_models import contender as registry
from workstreams.red_foundation_models.models.ttm.contender import CombinedPilotDTS, FrozenR2DTS


class TestDTSPilot(unittest.TestCase):
    def test_registration(self) -> None:
        self.assertEqual(registry._classes("ttm_r2_dts_pilot"), (CombinedPilotDTS, FrozenR2DTS))
        for cls in registry._classes("ttm_r2_dts_pilot"):
            self.assertTrue(cls().supports("DTS"))
            self.assertFalse(cls().supports("DTC"))
            self.assertFalse(cls().supports("RHOB"))

    def test_no_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"WMB_TTM_DTS_PILOT": tmp}):
            with self.assertRaisesRegex(FileNotFoundError, "WMB_TTM_DTS_PILOT"):
                CombinedPilotDTS().load_backend()
            (Path(tmp) / "model.safetensors").write_bytes(b"not the pilot")
            with self.assertRaisesRegex(ValueError, "Checkpoint mismatch"):
                CombinedPilotDTS().load_backend()
