"""Deterministic dashboard checks; neither real wells nor weights required."""
from pathlib import Path

import numpy as np
import pytest

from wmb.contender import Prediction
from wmb.io import Well
from .dashboard_data import load_local, predict_tracks, validate


class Stub:
    def fit(self, wells: list[Well], target: str) -> None:
        self.names = {w.name for w in wells}

    def predict(self, well: Well, target: str) -> Prediction:
        assert well.name not in self.names
        assert target not in well.curves or np.isnan(well.curves[target]).any()
        mean = np.ones_like(well.depth)
        return Prediction(mean, mean, {0.1: mean - 1, 0.9: mean + 1})


def test_validation_keeps_truth_out_of_fit_and_prediction() -> None:
    depth = np.arange(500.0)
    train = [Well(str(i), depth, {'RHOB': depth.copy()}) for i in range(10)]
    model = Stub()
    result = validate(model, train, [Well('test', depth)], 'RHOB')
    assert [row['wells'] for row in result.inventory] == [8, 2, 1]
    assert set(result.tracks).isdisjoint(model.names | {'test'})
    total = 0
    for frame in result.tracks.values():
        mask = frame['Hidden truth'].notna()
        total += mask.sum()
        assert frame.loc[mask, 'Observed'].isna().all()
        assert frame.loc[~mask, 'P50'].isna().all()
        np.testing.assert_array_equal(frame.loc[mask, 'Hidden truth'], depth[mask])
    assert total > 0
    assert all(row['n'] == total for row in result.scores.values())
    assert all(np.isfinite(w.curves['RHOB']).all() for w in train)


@pytest.mark.parametrize('curves', [{}, {'RHOB': np.array([3., np.nan, 4.])}])
def test_new_well_tracks_without_truth(curves: dict) -> None:
    well = Well('new', np.arange(3.), curves)
    model = Stub()
    model.fit([], 'RHOB')
    frame = predict_tracks(model, well, 'RHOB')
    np.testing.assert_array_equal(frame.Depth, well.depth)
    for column in ('P10', 'P50', 'P90'):
        np.testing.assert_array_equal(frame[column].notna(), frame.Observed.isna())
    assert 'Hidden truth' not in frame


def test_load_local_csv(tmp_path: Path) -> None:
    path = tmp_path / 'new.csv'
    path.write_text('WELL;DEPTH_MD;RHOB\nA;100;2.4\nA;101;-999.25\n')
    well, = load_local(path)
    assert well.name == 'A' and np.isnan(well.curves['RHOB'][1])
    with pytest.raises(ValueError, match='existing local file'):
        load_local(tmp_path / 'absent.las')