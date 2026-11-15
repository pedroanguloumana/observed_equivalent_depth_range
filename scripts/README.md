# Scripts

This folder contains command-line scripts used to support the project’s workflows. 
These scripts automate tasks such as data downloads, preprocessing, environment setup,
and figure generation. Each script should be runnable from the project root and 
document its usage at the top of the file.

Typical contents:
- Data download or sync scripts
- Data cleaning or transformation steps
- Automation tools for generating figures or reports
- Environment or setup utilities

## Identify and save wave events

Activate the project environment and install the package with `python -m pip install -e .`
before running from the project root:

```sh
# Run all five BOX_PROFILES in separate processes.
python scripts/record_and_save_wave_events.py

# Limit concurrency, for example to match an HPC CPU or memory allocation.
python scripts/record_and_save_wave_events.py --workers 2

# Process selected profiles, or use --workers 1 for serial execution.
python scripts/record_and_save_wave_events.py --waves Kelvin MRG --workers 2
```

Inputs are the existing `data/wave_filtered_fields/noaa_olr.<wave>.nc` files
produced by `create_and_save_wave_filtered_olr.py`. Available wave names are
`Kelvin`, `n1ERW`, `n0EIG`, `MRG`, and `n1WIG`. Each process loads its own input;
choose `--workers` to fit available CPUs and memory. Missing inputs cause an error
before processing starts.

Each profile produces `data/wave_event_records/noaa_olr.<wave>.csv`. Existing
outputs cause an error unless `--overwrite` is supplied. Profiles with no retained
events still produce a CSV with column headers. A successful profile's CSV is
available even if another profile later fails; any worker failure makes the command
exit unsuccessfully.

The calculation uses the full input time range, a cosine-latitude-weighted mean
over available latitudes at or north of the equator, and an OLR threshold of
`-1.5` times the standard deviation across that profile's time–longitude field.
Candidates use eight-neighbour connectivity and are retained when the elapsed
time from first to last timestamp is **strictly greater than 3 days**. These
choices are defined in the Python script. Widths use interpolated zero crossings;
both speed estimates are positive eastward, with Radon speeds searched from
`-50` to `50 m/s` in `0.5 m/s` increments.

CSV columns are `wave`, `start`, `stop`, `max_olr_sigma`, `min_olr_sigma`,
`mean_width_deg`, `max_width_deg`, `min_width_deg`, `radon_m_s`, and
`extent_duration_m_s`. The OLR columns are signed values divided by the profile's
standard deviation; widths are in longitude degrees and speeds in metres per
second. Missing width estimates are saved as empty CSV fields. Read a file later
with:

```python
import pandas as pd
from src.paths import DATA_DIR

events = pd.read_csv(
    DATA_DIR / "wave_event_records" / "noaa_olr.Kelvin.csv",
    parse_dates=["start", "stop"],
)
```
