from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import pynascar.driver as driver_module
from pynascar.core.process_data import NASCARDataProcessor
from pynascar.driver import Driver, DriversData
from pynascar.race import RaceDriverData, RaceResults, RaceTelemetry


def make_race(driver_id=1, team='New team', laps=None, pits=None):
    return SimpleNamespace(
        results=RaceResults(results=pd.DataFrame([{
            'driver_id': driver_id, 'driver_name': 'Test Driver', 'team': team,
            'car_number': '01', 'manufacturer': 'Ford', 'finishing_position': 2,
        }])),
        telemetry=RaceTelemetry(
            lap_times=laps if laps is not None else pd.DataFrame(),
            pit_stops=pits if pits is not None else pd.DataFrame(),
        ),
        driver_data=RaceDriverData(),
    )


@pytest.mark.parametrize('stage_number', [1, 2, 3])
def test_stage_points_from_processed_feed(stage_number):
    race = make_race()
    stage = NASCARDataProcessor.process_stage_data({'results': [{
        'driver_id': 1, 'driver_fullname': 'Test Driver',
        'finishing_position': 2, 'stage_points': 9,
    }]}, stage_number)
    setattr(race.results, f'stage_{stage_number}', stage)
    driver = Driver(1)
    driver.add_race_data(race, 100)
    assert driver.race_data[100][f'stage{stage_number}_points'] == 9
    assert driver.race_data[100][f'stage{stage_number}_position'] == 2


def test_legacy_stage_points_column():
    race = make_race()
    race.results.stage_1 = pd.DataFrame([{'driver_id': 1, 'points': 4}])
    driver = Driver(1)
    driver.add_race_data(race, 100)
    assert driver.race_data[100]['stage1_points'] == 4


def test_pit_stops_can_be_filtered_by_race_without_mutating_telemetry():
    pits = pd.DataFrame([
        {'driver_id': 1, 'lap': 10, 'total_duration': 13.5},
        {'driver_id': 2, 'lap': 11, 'total_duration': 14.0},
    ])
    original = pits.copy(deep=True)
    driver = Driver(1)
    driver.add_race_data(make_race(pits=pits), 100)
    driver.add_race_data(make_race(pits=pits), 101)
    season = DriversData(2025, 1, drivers={1: driver})
    assert season.driver_pit_stops(1)['race_id'].tolist() == [100, 101]
    assert season.driver_pit_stops(1, 101)['race_id'].tolist() == [101]
    assert season.driver_pit_stops(1, 999).empty
    assert_frame_equal(pits, original)


def mock_season(monkeypatch, races):
    # Schedule.get_finished_races() returns newest first.
    monkeypatch.setattr(driver_module, 'Schedule', lambda *args: SimpleNamespace(
        get_finished_races=lambda: pd.DataFrame({'race_id': list(races)})))
    monkeypatch.setattr(driver_module, 'load_df', lambda *args, **kwargs:
                        races[kwargs['race_id']].results.results)
    monkeypatch.setattr(driver_module, 'Race', lambda year, series, race_id, **kwargs:
                        races[race_id])


def test_season_keeps_latest_nonblank_driver_info_and_per_race_teams(monkeypatch):
    mock_season(monkeypatch, {
        103: make_race(team=' '),
        102: make_race(team='New team'),
        101: make_race(team='Old team'),
    })
    season = DriversData.build(2025, 1)
    assert season.get_driver(1).team == 'New team'
    assert season.to_dataframe().iloc[0]['team'] == 'New team'
    assert season.race_dataframe(101).iloc[0]['team'] == 'Old team'
    assert season.race_dataframe(102).iloc[0]['team'] == 'New team'
    assert season.race_ids == [103, 102, 101]


def legacy_lap_metrics(laps, driver_id):
    """Reference calculation from the original per-driver implementation."""
    frame = laps.copy()
    rows = frame[frame.driver_id == driver_id]
    if rows.empty:
        return {}
    frame['lap_speed_max'] = frame.groupby('Lap')['lap_speed'].transform('max')
    frame['speed_rank'] = frame.groupby('Lap')['lap_speed'].rank(
        ascending=False, method='min')
    return {
        'avg_lap_speed': rows.lap_speed.mean(),
        'fastest_lap': rows.lap_speed.max(),
        'total_laps': rows.Lap.max(),
        'leader_laps': int((rows.lap_speed == frame.loc[rows.index, 'lap_speed_max']).sum()),
        'avg_speed_rank': frame[frame.driver_id == driver_id].speed_rank.mean(),
    }


@pytest.fixture
def laps():
    # Tied speeds, missing speed/lap/driver IDs, and an unmapped fastest car.
    return pd.DataFrame({
        'driver_id': [1, 2, 3, 1, 2, 3, 1, 2, 3, np.nan, 1],
        'Lap': pd.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 2, None], dtype='Int64'),
        'lap_speed': [100., 100., np.nan, 110., 90., np.nan, 120., np.nan, np.nan, 130., 95.],
    }, index=[4, 8, 9, 12, 16, 18, 20, 24, 27, 32, 40])


@pytest.mark.parametrize('nullable_speed', [False, True])
def test_batch_lap_analysis_matches_existing_metrics_without_mutating(laps, nullable_speed):
    if nullable_speed:
        laps['lap_speed'] = laps['lap_speed'].astype('Float64')
    original = laps.copy(deep=True)
    actual = driver_module._analyze_laps(laps)
    for driver_id in [1, 2, 3]:
        assert_frame_equal(pd.DataFrame([actual[driver_id]]),
                           pd.DataFrame([legacy_lap_metrics(laps, driver_id)]),
                           check_exact=True)
    assert_frame_equal(laps, original)


def test_standalone_driver_still_computes_lap_metrics(laps):
    driver = Driver(1)
    driver.add_race_data(make_race(laps=laps), 100)
    expected = legacy_lap_metrics(laps, 1)
    actual = {key: driver.race_data[100][key] for key in expected}
    assert_frame_equal(pd.DataFrame([actual]), pd.DataFrame([expected]), check_exact=True)


@pytest.mark.parametrize('laps', [pd.DataFrame(), pd.DataFrame({'Lap': [1]})])
def test_absent_laps_have_no_metrics(laps):
    assert driver_module._analyze_laps(laps) == {}


def test_season_analyzes_laps_once_per_race(monkeypatch, laps):
    race = make_race(laps=laps)
    race.results.results = pd.concat([
        make_race(driver_id=i).results.results for i in [1, 2, 3, 4]
    ], ignore_index=True)
    mock_season(monkeypatch, {101: race, 100: race})
    analyze = Mock(wraps=driver_module._analyze_laps)
    monkeypatch.setattr(driver_module, '_analyze_laps', analyze)
    season = DriversData.build(2025, 1)
    assert analyze.call_count == 2
    assert len(season.drivers) == 4
    for race_id in [100, 101]:
        for driver_id in [1, 2, 3]:
            expected = legacy_lap_metrics(laps, driver_id)
            actual = {key: season.drivers[driver_id].race_data[race_id][key]
                      for key in expected}
            assert_frame_equal(pd.DataFrame([actual]), pd.DataFrame([expected]),
                               check_exact=True)
        assert 'avg_lap_speed' not in season.drivers[4].race_data[race_id]


def test_all_missing_lap_numbers_and_single_driver():
    laps = pd.DataFrame({
        'driver_id': [1, 1], 'Lap': pd.array([None, None], dtype='Int64'),
        'lap_speed': [100., 110.],
    })
    assert_frame_equal(pd.DataFrame([driver_module._analyze_laps(laps)[1]]),
                       pd.DataFrame([legacy_lap_metrics(laps, 1)]), check_exact=True)


def test_standalone_analysis_recomputes_after_telemetry_changes(laps):
    race = make_race(laps=laps)
    driver = Driver(1)
    driver.add_race_data(race, 100)
    race.telemetry.lap_times.loc[4, 'lap_speed'] = 200.
    driver.add_race_data(race, 100)
    assert driver.race_data[100]['fastest_lap'] == 200.
