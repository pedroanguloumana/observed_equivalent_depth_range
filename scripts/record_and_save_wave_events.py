"""Identify wave events in parallel and save one CSV per wave profile.

Run from the project root after installing the package with ``pip install -e .``::

    python scripts/record_and_save_wave_events.py
    python scripts/record_and_save_wave_events.py --waves Kelvin MRG --workers 2

Inputs: data/wave_filtered_fields/noaa_olr.<wave>.nc
Outputs: data/wave_event_records/noaa_olr.<wave>.csv
Events meet a 1.5-sigma OLR threshold and span at least one elapsed day.
Existing CSVs are preserved unless --overwrite is supplied.
"""
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from src.loading import load_wave_filtered_olr
from src.paths import DATA_DIR
from src.wave_profiles import BOX_PROFILES as profiles

THRESH_SIGMA = 1.5
MIN_DURATION_DAYS = 1.0

EVENT_COLUMNS = [
    "wave", "start", "stop", "duration_days",
    "start_lon_min", "start_lon_max",
    "stop_lon_min", "stop_lon_max",
    "lat_min", "lat_max",
    "max_olr_sigma", "min_olr_sigma",
    "mean_width_deg", "max_width_deg", "min_width_deg",
    "radon_m_s", "extent_duration_m_s",
]
EVENT_DIR = DATA_DIR / "wave_event_records"


def record_and_save_wave_events(profile, overwrite=False):
    """Identify one profile's events and save their widths (degrees) and speeds (m/s)."""
    print(f"[{profile}] Loading filtered OLR", flush=True)
    event_records = []
    with load_wave_filtered_olr(profile) as wave_data:
        field = wave_data.sel(lat=slice(0, None))
        hov = (
            field.weighted(np.cos(np.deg2rad(field.lat)))
            .mean("lat")
            .transpose("time", "lon")
            .compute()
        )

    olr_std = hov.std().item()
    event_threshold = THRESH_SIGMA * olr_std
    candidate_event_mask = hov <= -event_threshold

    # Eight-neighbour connectivity includes diagonally touching pixels.
    labels, n_candidates = ndimage.label(
        candidate_event_mask,
        structure=np.ones((3, 3), dtype=bool),
    )
    candidate_slices = ndimage.find_objects(labels)
    print(f"[{profile}] Found {n_candidates} candidate events", flush=True)

    for event_id, (time_slice, lon_slice) in enumerate(candidate_slices, start=1):
        event_hov = hov.isel(time=time_slice, lon=lon_slice)
        event_mask = xr.DataArray(
            labels[time_slice, lon_slice] == event_id,
            dims=event_hov.dims,
            coords=event_hov.coords,
        )

        # Use elapsed time: two consecutive daily timestamps span one day.
        duration_days = (
            (event_hov.time[-1] - event_hov.time[0]) / np.timedelta64(1, "D")
        ).item()

        if duration_days < MIN_DURATION_DAYS:
            continue
        print(
            f"[{profile}] Processing candidate {event_id}/{n_candidates} "
            f"({duration_days:g} days)",
            flush=True,
        )
        #
        # Get event widths for each time
        #
        widths = xr.full_like(event_hov.time, float("nan"), dtype=float)
        west_lons = xr.full_like(widths, float("nan"))
        east_lons = xr.full_like(widths, float("nan"))

        for t in event_hov.time:
            event_row = event_hov.sel(time=t).where(event_mask.sel(time=t))
            if not (event_row < 0).any().item():
                continue

            trough_lon = event_row.idxmin("lon").item()
            row = hov.sel(time=t).sortby("lon")

            # Copies on either side allow the search to cross the longitude seam.
            row = xr.concat([
                row.assign_coords(lon=row.lon - 360),
                row,
                row.assign_coords(lon=row.lon + 360),
            ], dim="lon")

            left = row.where(
                (row.lon < trough_lon) & (row >= 0), drop=True
            )
            right = row.where(
                (row.lon > trough_lon) & (row >= 0), drop=True
            )

            if not left.sizes["lon"] or not right.sizes["lon"]:
                continue

            segment = row.sel(lon=slice(
                left.lon.max().item(),
                right.lon.min().item(),
            ))
            if not segment.notnull().all().item():
                continue

            left_outer = segment.isel(lon=0)
            left_inner = segment.isel(lon=1)
            right_inner = segment.isel(lon=-2)
            right_outer = segment.isel(lon=-1)

            left_zero = (
                left_outer.lon
                - left_outer * (left_inner.lon - left_outer.lon)
                / (left_inner - left_outer)
            )
            right_zero = (
                right_inner.lon
                - right_inner * (right_outer.lon - right_inner.lon)
                / (right_outer - right_inner)
            )

            west_lons.loc[{"time": t}] = left_zero.item()
            east_lons.loc[{"time": t}] = right_zero.item()
            widths.loc[{"time": t}] = (right_zero - left_zero).item()

        #
        # Get phase speeds for each wave event
        #
        metres_per_degree = 2 * np.pi * 6_371_000 / 360

        # Preserve all event timestamps.
        amplitude = (
            (-event_hov).where(event_mask, 0)
            .clip(min=0)
            .fillna(0)
        )
        amplitude = amplitude.reindex(
            lon=hov.lon, fill_value=0
        ).sortby("lon")

        elapsed_seconds = (
            (amplitude.time - amplitude.time[0])
            / np.timedelta64(1, "s")
        )

        # Circular amplitude-weighted centre.
        angle = np.deg2rad(amplitude.lon)
        centre_lon = np.rad2deg(np.arctan2(
            (amplitude * np.sin(angle)).sum("lon"),
            (amplitude * np.cos(angle)).sum("lon"),
        ))

        # Unwrap movement across the longitude seam.
        steps = (centre_lon.diff("time") + 180) % 360 - 180
        centre_unwrapped = xr.concat([
            centre_lon.isel(time=slice(0, 1)),
            centre_lon.isel(time=0) + steps.cumsum("time"),
        ], dim="time")
        longitude_shift = 360 * (
            (centre_unwrapped - (west_lons + east_lons) / 2) / 360
        ).round()

        west_lons = west_lons + longitude_shift
        east_lons = east_lons + longitude_shift

        speed_extent = (
            (centre_unwrapped[-1] - centre_unwrapped[0])
            * metres_per_degree / elapsed_seconds[-1]
        ).item()

        # Periodic field for the Radon projections.
        periodic_amplitude = xr.concat([
            amplitude.assign_coords(lon=amplitude.lon - 360),
            amplitude,
            amplitude.assign_coords(lon=amplitude.lon + 360),
        ], dim="lon")

        origins = amplitude.lon.rename({"lon": "origin"})
        lon_min = amplitude.lon.min().item()

        radon_scores = xr.DataArray(
            float("nan"),
            dims=["speed"],
            coords={"speed": np.linspace(-50, 50, 201)},
        )

        for speed in radon_scores.speed:
            path = (
                origins
                + speed.item() * elapsed_seconds / metres_per_degree
            )
            path = (path - lon_min) % 360 + lon_min

            along_paths = periodic_amplitude.interp(lon=path)
            projection = (
                along_paths.assign_coords(time=elapsed_seconds)
                .integrate("time")
            )
            radon_scores.loc[{"speed": speed}] = (
                projection**2
            ).sum("origin").item()

        speed_radon = radon_scores.idxmax("speed").item()

        phase_speeds = xr.DataArray(
            [speed_radon, speed_extent],
            dims=["estimator"],
            coords={"estimator": ["radon", "extent_duration"]},
            name="phase_speed",
            attrs={"units": "m s-1", "positive": "eastward"},
        )

        #
        # Record event information
        #
        event_records.append({
            "wave": profile,
            "start": event_hov.time.values[0],
            "stop": event_hov.time.values[-1],
            "duration_days": duration_days,
            "start_lon_min": west_lons.isel(time=0).item(),
            "start_lon_max": east_lons.isel(time=0).item(),
            "stop_lon_min": west_lons.isel(time=-1).item(),
            "stop_lon_max": east_lons.isel(time=-1).item(),
            "lat_min": field.lat.min().item(),
            "lat_max": field.lat.max().item(),
            "max_olr_sigma": event_hov.where(event_mask).max().item() / olr_std,
            "min_olr_sigma": event_hov.where(event_mask).min().item() / olr_std,
            "mean_width_deg": widths.mean("time").item(),
            "max_width_deg": widths.max("time").item(),
            "min_width_deg": widths.min("time").item(),
            "radon_m_s": phase_speeds.sel(
                estimator="radon"
            ).item(),
            "extent_duration_m_s": phase_speeds.sel(
                estimator="extent_duration"
            ).item(),
        })

    # Keep column headers even when no candidates pass the duration criterion.
    events = pd.DataFrame(event_records, columns=EVENT_COLUMNS)
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVENT_DIR / f"noaa_olr.{profile}.csv"
    events.to_csv(output_path, index=False, mode="w" if overwrite else "x")
    print(f"[{profile}] Saved {len(events)} events to {output_path}", flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--waves", nargs="+", choices=list(profiles), default=list(profiles),
        help="Wave profiles to process (default: all profiles).",
    )
    parser.add_argument(
        "--workers", type=int, default=len(profiles),
        help="Maximum parallel processes (default: one per profile). Use 1 to run serially.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing event CSVs.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    waves = list(dict.fromkeys(args.waves))

    # Check every input and output before starting any expensive processing.
    for wave in waves:
        input_path = DATA_DIR / "wave_filtered_fields" / f"noaa_olr.{wave}.nc"
        if not input_path.is_file():
            parser.error(f"Missing filtered OLR input: {input_path}")
        output_path = EVENT_DIR / f"noaa_olr.{wave}.csv"
        if output_path.exists() and not args.overwrite:
            parser.error(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    workers = min(args.workers, len(waves))
    print(f"Processing {', '.join(waves)} with {workers} worker(s)", flush=True)
    if workers == 1:
        for wave in waves:
            record_and_save_wave_events(wave, args.overwrite)
    else:
        # Spawn gives each worker its own NetCDF/HDF5 state on PCs and HPC systems.
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(record_and_save_wave_events, wave, args.overwrite): wave
                for wave in waves
            }
            for future in as_completed(futures):
                wave = futures[future]
                try:
                    future.result()
                except Exception as error:
                    raise RuntimeError(f"Event identification failed for {wave}") from error


if __name__ == "__main__":
    main()
