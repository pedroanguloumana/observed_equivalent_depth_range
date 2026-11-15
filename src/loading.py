"""Code for loading and handling data
"""
import xarray as xr
import pandas as pd
from src.paths import DATA_DIR

def load_noaa_daily_olr(
        sel_dict: dict={
            'lat': slice(-20, 20),
            'time': slice("1990-01-01", "2025-12-31")
        }
    ):
    file = DATA_DIR / "OLR-Daily_v02r00_s19790101_e20251231.nc"
    data =  xr.open_dataset(file).sel(sel_dict)['olr']
    assert not data.isnull().any().compute().item(), "OLR contains NaN values"
    return data


def load_wave_filtered_olr(wave_name):
    file =  DATA_DIR / "wave_filtered_fields" / f"noaa_olr.{wave_name}.nc"
    return xr.open_dataarray(file)


def load_wave_event_record(wave_name):
    file =  DATA_DIR / "wave_event_records" / f"noaa_olr.{wave_name}.csv"
    return pd.read_csv(file)

def load_noaa_daily_olr_spectral_power(force_overwrite=False):
     file = DATA_DIR / "spectral_power" / "noaa_olr_spectral_power.nc"
     if file.is_file() and not force_overwrite:
         return xr.open_dataset(file)
     elif not file.is_file() or force_overwrite:
         from src.spectral_power import spectral_power_pipeline
         olr = load_noaa_daily_olr()
         print("Spectral power not cached, computing now...")
         spectral_power = {
            'symmetric': spectral_power_pipeline(olr, 'symmetric'),
            'antisymmetric': spectral_power_pipeline(olr, 'antisymmetric'),
        }
         dataset = xr.Dataset(spectral_power)
         file.parent.mkdir(parents=True, exist_ok=True)
         dataset.to_netcdf(file)
         return dataset
