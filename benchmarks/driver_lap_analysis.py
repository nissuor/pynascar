"""Compare field lap analysis against the original per-driver calculation.

Run from the repository root:
    PYTHONPATH=src python benchmarks/driver_lap_analysis.py
    PYTHONPATH=src python benchmarks/driver_lap_analysis.py --laps /path/to/laps.parquet

Input parquet uses Race.telemetry.lap_times columns. No network calls or writes.
Timing covers lap analysis for the entire field, excluding I/O and other stats.
"""
import argparse
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from pynascar.driver import _analyze_laps


def original_analysis(laps):
    metrics = {}
    for driver_id in laps['driver_id'].dropna().unique():
        frame = laps.copy()
        rows = frame[frame['driver_id'] == driver_id]
        frame['lap_speed_max'] = frame.groupby('Lap')['lap_speed'].transform('max')
        frame['speed_rank'] = frame.groupby('Lap')['lap_speed'].rank(
            ascending=False, method='min')
        metrics[driver_id] = {
            'avg_lap_speed': rows['lap_speed'].mean(),
            'fastest_lap': rows['lap_speed'].max(),
            'total_laps': rows['Lap'].max(),
            'leader_laps': int((rows['lap_speed'] == frame.loc[rows.index, 'lap_speed_max']).sum()),
            'avg_speed_rank': frame[frame['driver_id'] == driver_id]['speed_rank'].mean(),
        }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--laps', help='Existing lap-times parquet file')
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.laps:
        laps = pd.read_parquet(args.laps)
    else:
        rng = np.random.default_rng(2025)
        laps = pd.DataFrame({
            'driver_id': np.repeat(np.arange(1, 41), 200),
            'Lap': np.tile(np.arange(1, 201), 40),
            'lap_speed': rng.normal(175, 10, 8000),
        })
    assert_frame_equal(
        pd.DataFrame.from_dict(original_analysis(laps), orient='index'),
        pd.DataFrame.from_dict(_analyze_laps(laps), orient='index'),
        check_exact=True,
    )
    before, after = [], []
    for _ in range(args.repeats):
        start = perf_counter()
        original_analysis(laps)
        before.append(perf_counter() - start)
        start = perf_counter()
        _analyze_laps(laps)
        after.append(perf_counter() - start)
    print(f'{len(laps):,} lap rows, {laps.driver_id.nunique()} drivers; exact metric parity')
    print(f'Median of {args.repeats}: {median(before)*1000:.2f} ms -> '
          f'{median(after)*1000:.2f} ms ({median(before)/median(after):.2f}x)')


if __name__ == '__main__':
    main()
