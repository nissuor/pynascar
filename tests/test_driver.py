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


def test_pit_stops_by_race():
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


def test_latest_team(monkeypatch):
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


@pytest.fixture
def laps():
    # Ties, missing speeds, and an unmapped fastest car.
    return pd.DataFrame({
        'driver_id': [1, 2, 3, 1, 2, 3, 1, 2, 3, np.nan],
        'Lap': pd.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 2], dtype='Int64'),
        'lap_speed': [100., 100., np.nan, 110., 90., np.nan, 120., np.nan, np.nan, 130.],
    }, index=[4, 8, 9, 12, 16, 18, 20, 24, 27, 32])


@pytest.mark.parametrize('nullable_speed', [False, True])
def test_lap_metrics(laps, nullable_speed):
    if nullable_speed:
        laps['lap_speed'] = laps['lap_speed'].astype('Float64')
    original = laps.copy(deep=True)
    expected = {
        1: (110., 120., 3, 2, 4 / 3),
        2: (95., 100., 3, 1, 2.),
        3: (None, None, 3, 0, None),
    }
    columns = ['avg_lap_speed', 'fastest_lap', 'total_laps',
               'leader_laps', 'avg_speed_rank']
    batch = driver_module._analyze_laps(laps)
    for driver_id, values in expected.items():
        driver = Driver(driver_id)
        driver.add_race_data(make_race(driver_id=driver_id, laps=laps), 100)
        for metrics in [batch[driver_id], driver.race_data[100]]:
            for column, value in zip(columns, values):
                if value is None:
                    assert pd.isna(metrics[column])
                else:
                    assert metrics[column] == value
    assert_frame_equal(laps, original)


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
        assert season.drivers[1].race_data[race_id]['avg_lap_speed'] == 110.
        assert 'avg_lap_speed' not in season.drivers[4].race_data[race_id]


def test_all_missing_lap_numbers_and_single_driver():
    laps = pd.DataFrame({
        'driver_id': [1, 1], 'Lap': pd.array([None, None], dtype='Int64'),
        'lap_speed': [100., 110.],
    })
    metrics = driver_module._analyze_laps(laps)[1]
    assert metrics['avg_lap_speed'] == 105.
    assert metrics['leader_laps'] == 0
    assert pd.isna(metrics['total_laps'])
    assert pd.isna(metrics['avg_speed_rank'])


def test_lap_metrics_recompute(laps):
    race = make_race(laps=laps)
    driver = Driver(1)
    driver.add_race_data(race, 100)
    race.telemetry.lap_times.loc[4, 'lap_speed'] = 200.
    driver.add_race_data(race, 100)
    assert driver.race_data[100]['fastest_lap'] == 200.


@pytest.mark.parametrize('missing_column', ['driver_id', 'Lap', 'lap_speed'])
def test_partial_laps(monkeypatch, missing_column):
    laps = pd.DataFrame({'driver_id': [1], 'Lap': [1], 'lap_speed': [100.]})
    laps = laps.drop(columns=[missing_column])
    original = laps.copy(deep=True)
    assert driver_module._analyze_laps(laps) == {}

    race = make_race(laps=laps)
    driver = Driver(1)
    driver.add_race_data(race, 100)
    assert driver.race_data[100]['finishing_position'] == 2
    assert 'avg_lap_speed' not in driver.race_data[100]

    mock_season(monkeypatch, {100: race})
    season = DriversData.build(2025, 1)
    assert season.get_driver(1).race_data[100]['finishing_position'] == 2
    assert 'avg_lap_speed' not in season.get_driver(1).race_data[100]
    assert_frame_equal(laps, original)
