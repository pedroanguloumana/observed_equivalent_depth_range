"""Script that computes and saves wave-filtered OLR data. 
"""
import xarray as xr
import xr_ccew as tw
from src.loading import (
    load_noaa_daily_olr
)
from pathlib import Path
from src.wave_profiles import BOX_PROFILES as profiles
from src.paths import DATA_DIR

# Load OLR data
olr = load_noaa_daily_olr()
deseasonalized_anomalies = tw.remove_mean_and_linear_trend(
    tw.remove_harmonics_of_seasonal_cycle(
        olr, num_harmonics=4
    )
)

# Taper the full record before the FFT.
windowed_anomalies = tw.apply_window(
    deseasonalized_anomalies,
    dim="time",
    window="tukey",
    points_per_end=90,  # Samples per end; approximately days for daily data.
)

save_dir = DATA_DIR / "wave_filtered_fields"
save_dir.mkdir(parents=True, exist_ok=True)

for wave_name, profile in profiles.items():
    print(f"Filtering {wave_name}...")
    filtered_data = tw.filter_field(windowed_anomalies, profile)
    save_path = save_dir / f"noaa_olr.{wave_name}.nc"
    filtered_data.to_netcdf(save_path)