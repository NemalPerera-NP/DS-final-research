import json
from pathlib import Path


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    md("""# Fixed-run weather features and M1 datasets

This notebook builds weather features for the frozen M0 P95/P90 observations using one archived ECMWF IFS HRES run per target day. For a Sri Lanka target date `D`, it selects the `D-1 00:00 UTC` run. Its hourly timestamps are retained in UTC; household-day aggregation uses `Asia/Colombo`. The run, lead-time window, grid coordinates, completeness, and mapping status are saved for audit.

This is a fixed-initialization historical forecast backtest, not observed weather and not proof of exact historical publication time. ECMWF IFS HRES 9 km runs are available in Open-Meteo's Single Runs archive from 2024-03-14; earlier M0 rows remain in the full M1 exports with explicit missing-weather status and are excluded only from a paired M0/M1 experiment. The forecast fields for a local day are the 24 hourly model timestamps labelled 00:30 through 23:30 Asia/Colombo. Rain and radiation are hourly quantities labelled by UTC time, so the half-hour local-midnight boundary is approximated by the available hourly forecast grid.

Existing stitched Historical Forecast artifacts are left untouched and are not used here. All datasets and audits are CSV.
"""),
    code("""from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent
DATA_DIR = ROOT / "data" / "processed"
WEATHER_DIR = ROOT / "data" / "external" / "weather"
CACHE_DIR = ROOT / ".cache" / "open_meteo_single_runs"
WEATHER_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

P95_PATH = DATA_DIR / "05_model_ready_m0_p95.csv"
P90_PATH = DATA_DIR / "05_model_ready_m0_p90.csv"
SPLIT_PATH = DATA_DIR / "04_master_split.csv"
DAILY_PATH = DATA_DIR / "03_daily_with_target.csv"
COORDINATE_PATH = WEATHER_DIR / "07_depot_coordinate_lookup.csv"
HOURLY_PATH = WEATHER_DIR / "07_hourly_fixed_run_weather_utc.csv"
DAILY_WEATHER_PATH = WEATHER_DIR / "07_daily_fixed_run_weather_by_depot.csv"
M1_P95_PATH = DATA_DIR / "07_model_ready_m1_p95.csv"
M1_P90_PATH = DATA_DIR / "07_model_ready_m1_p90.csv"
PAIR_AUDIT_PATH = DATA_DIR / "07_m0_m1_pairing_audit.csv"
REQUEST_AUDIT_PATH = WEATHER_DIR / "07_single_run_request_audit.csv"
MANIFEST_PATH = WEATHER_DIR / "07_fixed_run_weather_manifest.json"

API_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
WEATHER_MODEL = "ecmwf_ifs"
MODEL_RESOLUTION_KM_NOMINAL = 9
METER_TIMEZONE = "Asia/Colombo"
MIN_TARGET_DATE = pd.Timestamp("2024-03-15")
FORECAST_HOURS = 48
HOURLY_VARIABLES = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "rain", "shortwave_radiation"]
COORDINATE_BATCH_SIZE = 20
REFRESH_API_CACHE = False
REQUEST_PAUSE_SECONDS = 0.15

print("Project:", ROOT)
print("Forecast product: individual ECMWF IFS HRES run; nominal resolution:", MODEL_RESOLUTION_KM_NOMINAL, "km")
print("UTC source timestamps; local daily feature timezone:", METER_TIMEZONE)
"""),
    md("""## 1. Load the frozen M0 rows, split, network assignment, and coordinates

The master split is authoritative. This notebook does not recreate or alter it. Network metadata comes from the household-day table; weather is looked up by the existing branch–depot key. Missing/unknown depot assignments and incomplete weather remain visible in the row-level pairing audit.
"""),
    code("""def read_csv(path, **kwargs):
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return pd.read_csv(path, encoding="utf-8", low_memory=False, **kwargs)

p95 = read_csv(P95_PATH, parse_dates=["measurement_date"])
p90 = read_csv(P90_PATH, parse_dates=["measurement_date"])
split = read_csv(SPLIT_PATH, parse_dates=["target_date"])
daily = read_csv(DAILY_PATH, parse_dates=["measurement_date"], dtype={"household_id": "string"})
coordinates = read_csv(COORDINATE_PATH, dtype={"weather_location_id": "string", "branch": "string", "depot": "string"})

for name, frame in [("P95", p95), ("P90", p90)]:
    if frame["row_id"].duplicated().any():
        raise ValueError(f"{name} M0 row_id is not unique")
if not p95["row_id"].reset_index(drop=True).equals(p90["row_id"].reset_index(drop=True)):
    raise ValueError("P95 and P90 M0 row_id order differs")
if split["row_id"].duplicated().any() or set(split["row_id"]) != set(p95["row_id"]):
    raise ValueError("04_master_split does not exactly cover the frozen M0 row IDs")

daily["branch"] = daily["branch"].astype("string").str.strip()
daily["depot"] = daily["depot"].astype("string").str.strip()
daily["substation_id"] = daily["substation_id"].astype("string").str.strip()
daily_key = daily[["household_id", "measurement_date", "branch", "depot", "substation_id"]].copy()
if daily_key.duplicated(["household_id", "measurement_date"]).any():
    raise ValueError("Household-day network metadata is not unique by household/date")

def assignments_for(frame):
    rows = frame[["row_id", "household_id", "measurement_date"]].copy()
    rows = rows.merge(daily_key, on=["household_id", "measurement_date"], how="left", validate="one_to_one")
    rows = rows.merge(split[["row_id", "outer_split", "seen_in_training", "test_cohort"]], on="row_id", how="left", validate="one_to_one")
    rows["weather_location_id"] = (rows["branch"].astype("string").str.upper().str.strip() + "::" + rows["depot"].astype("string").str.upper().str.strip())
    unknown = rows[["branch", "depot"]].isna().any(axis=1) | rows["branch"].astype("string").str.casefold().isin(["unknown", "none", "nan", "<na>"]) | rows["depot"].astype("string").str.casefold().isin(["unknown", "none", "nan", "<na>"])
    rows.loc[unknown, "weather_location_id"] = pd.NA
    return rows

assignment = assignments_for(p95)
if assignment["outer_split"].isna().any():
    raise ValueError("Some M0 rows did not inherit a frozen split assignment")

coordinates["weather_location_id"] = coordinates["weather_location_id"].astype("string").str.upper().str.strip()
coordinates["latitude"] = pd.to_numeric(coordinates["latitude"], errors="coerce")
coordinates["longitude"] = pd.to_numeric(coordinates["longitude"], errors="coerce")
if coordinates["weather_location_id"].duplicated().any():
    raise ValueError("Coordinate lookup must be unique by weather_location_id")
if (coordinates["latitude"].notna() & ~coordinates["latitude"].between(-90, 90)).any() or (coordinates["longitude"].notna() & ~coordinates["longitude"].between(-180, 180)).any():
    raise ValueError("Coordinate lookup contains invalid latitude/longitude")

assignment = assignment.merge(coordinates[["weather_location_id", "latitude", "longitude", "coordinate_source"]], on="weather_location_id", how="left", validate="many_to_one")
assignment["coordinate_available"] = assignment["latitude"].notna() & assignment["longitude"].notna()
print("Frozen rows:", f"{len(p95):,}", "| P95/P90 exact row alignment: yes")
print("Split counts:")
display(split.groupby(["outer_split", "test_cohort"], dropna=False).size().rename("rows").reset_index())
print("Weather row mapping:")
display(assignment.assign(target_date=assignment.measurement_date.dt.date).groupby(["outer_split", "coordinate_available"], dropna=False).size().rename("rows").reset_index())
print("Coordinate entries:", len(coordinates), "with valid coordinates:", int(coordinates[["latitude", "longitude"]].notna().all(axis=1).sum()))
"""),
    md("""## 2. Request one fixed run per target date and unique coordinate batch

For target date `D`, the selected model initialization is `D-1 00:00 UTC`. The household-day prediction cutoff is local midnight at the start of `D` (`D-1 18:30 UTC`); this run precedes it by 18.5 hours. The target-day weather timestamps should have approximately 19–42 hour leads due to the UTC hourly grid and Sri Lanka's UTC+05:30 offset. Requests are cached by exact run and coordinates, and individual failures are written to an audit rather than hidden.
"""),
    code("""# Use only dates that occur in the frozen data and are supported by the IFS archive.
target_dates = pd.DatetimeIndex(sorted(assignment["measurement_date"].dt.normalize().dropna().unique()))
supported_dates = target_dates[target_dates >= MIN_TARGET_DATE]
known_coordinates = coordinates.dropna(subset=["latitude", "longitude"]).copy()
known_coordinates["coordinate_key"] = known_coordinates.apply(lambda r: f"{r.latitude:.6f},{r.longitude:.6f}", axis=1)
unique_coordinates = known_coordinates[["coordinate_key", "latitude", "longitude"]].drop_duplicates("coordinate_key").sort_values("coordinate_key").reset_index(drop=True)
print("All target dates:", len(target_dates), "| fixed-run-supported dates:", len(supported_dates), "from", supported_dates.min().date(), "to", supported_dates.max().date())
print("Unique input coordinate pairs:", len(unique_coordinates), "| API calls planned:", len(supported_dates) * int(np.ceil(len(unique_coordinates) / COORDINATE_BATCH_SIZE)))

retry = Retry(total=5, connect=5, read=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset(["GET"]), respect_retry_after_header=True)
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retry))
session.headers.update({"User-Agent": "smart-meter-research-fixed-run-weather/1.0"})

hourly_frames, request_records = [], []
for target_date in supported_dates:
    run_time = (pd.Timestamp(target_date).tz_localize("UTC") - pd.Timedelta(days=1)).replace(hour=0)
    run_text = run_time.strftime("%Y-%m-%dT%H:%M")
    for batch_start in range(0, len(unique_coordinates), COORDINATE_BATCH_SIZE):
        batch = unique_coordinates.iloc[batch_start:batch_start + COORDINATE_BATCH_SIZE]
        params = {
            "latitude": ",".join(f"{x:.6f}" for x in batch["latitude"]),
            "longitude": ",".join(f"{x:.6f}" for x in batch["longitude"]),
            "run": run_text, "models": WEATHER_MODEL,
            "hourly": ",".join(HOURLY_VARIABLES), "forecast_hours": FORECAST_HOURS,
            "timezone": "GMT", "timeformat": "unixtime",
            "temperature_unit": "celsius", "wind_speed_unit": "kmh", "precipitation_unit": "mm",
        }
        cache_sig = json.dumps({"url": API_URL, "params": params}, sort_keys=True, separators=(",", ":"))
        cache_path = CACHE_DIR / (sha256(cache_sig.encode("utf-8")).hexdigest() + ".json")
        started = time.time()
        try:
            if cache_path.exists() and not REFRESH_API_CACHE:
                response_json = json.loads(cache_path.read_text(encoding="utf-8"))
                status = "cache_hit"
            else:
                response = session.get(API_URL, params=params, timeout=(20, 180))
                response.raise_for_status()
                response_json = response.json()
                cache_path.write_text(json.dumps(response_json), encoding="utf-8")
                status = "fetched"
            results = response_json if isinstance(response_json, list) else [response_json]
            if len(results) != len(batch):
                raise ValueError(f"Expected {len(batch)} locations; received {len(results)}")
            for (_, loc), result in zip(batch.iterrows(), results):
                block = result.get("hourly") or {}
                if "time" not in block:
                    raise ValueError("API response did not include hourly time")
                frame = pd.DataFrame({"weather_timestamp_utc": pd.to_datetime(block["time"], unit="s", utc=True)})
                for variable in HOURLY_VARIABLES:
                    frame[variable] = block.get(variable)
                frame["coordinate_key"] = loc["coordinate_key"]
                frame["input_latitude"] = loc["latitude"]
                frame["input_longitude"] = loc["longitude"]
                frame["grid_latitude"] = result.get("latitude")
                frame["grid_longitude"] = result.get("longitude")
                frame["grid_elevation_m"] = result.get("elevation")
                frame["weather_model"] = WEATHER_MODEL
                frame["forecast_run_utc"] = run_time
                hourly_frames.append(frame)
            request_records.append({"target_date": target_date.strftime("%Y-%m-%d"), "forecast_run_utc": run_text, "coordinate_count": len(batch), "status": status, "elapsed_seconds": round(time.time() - started, 2), "error": ""})
        except Exception as exc:
            request_records.append({"target_date": target_date.strftime("%Y-%m-%d"), "forecast_run_utc": run_text, "coordinate_count": len(batch), "status": "failed", "elapsed_seconds": round(time.time() - started, 2), "error": str(exc)[:500]})
        time.sleep(REQUEST_PAUSE_SECONDS)
    if len(request_records) % 25 == 0:
        print("Run-date requests processed:", len(request_records), "/", len(supported_dates))

request_audit = pd.DataFrame(request_records)
request_audit.to_csv(REQUEST_AUDIT_PATH, index=False, encoding="utf-8")
if not hourly_frames:
    raise RuntimeError("No fixed-run hourly forecasts were returned. See request audit.")
hourly_coordinates = pd.concat(hourly_frames, ignore_index=True)
if hourly_coordinates.duplicated(["coordinate_key", "forecast_run_utc", "weather_timestamp_utc"]).any():
    raise ValueError("Fixed-run hourly API output contains duplicate coordinate/run/time keys")
print("Hourly coordinate rows:", f"{len(hourly_coordinates):,}", "| request statuses:")
display(request_audit["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="requests"))
"""),
    md("""## 3. Convert source timestamps to depot-days and audit weather completeness

The source CSV remains UTC. Each location/run is converted to Sri Lanka time; target-date rows are aggregated only when all 24 target-day timestamps are present. The output carries the exact run and observed forecast-lead range, and does not impute missing weather.
"""),
    code("""coordinate_to_locations = known_coordinates[["coordinate_key", "weather_location_id", "branch", "depot"]].drop_duplicates()
hourly = hourly_coordinates.merge(coordinate_to_locations, on="coordinate_key", how="inner", validate="many_to_many")
hourly["weather_timestamp_local"] = hourly["weather_timestamp_utc"].dt.tz_convert(METER_TIMEZONE)
hourly["target_date"] = hourly["weather_timestamp_local"].dt.strftime("%Y-%m-%d")
hourly["forecast_run_utc"] = pd.to_datetime(hourly["forecast_run_utc"], utc=True)
hourly["forecast_lead_hours"] = (hourly["weather_timestamp_utc"] - hourly["forecast_run_utc"]).dt.total_seconds() / 3600

# Keep each valid time only for the target date associated with its D-1 00Z run.
hourly["run_target_date"] = (hourly["forecast_run_utc"].dt.tz_convert(None).dt.normalize() + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
target_hourly = hourly.loc[hourly["target_date"].eq(hourly["run_target_date"])].copy()
hourly.to_csv(HOURLY_PATH, index=False, encoding="utf-8")

daily_weather = (
    target_hourly.groupby(["weather_location_id", "branch", "depot", "run_target_date", "forecast_run_utc"], as_index=False, dropna=False)
    .agg(
        weather_hour_count=("weather_timestamp_utc", "nunique"),
        temperature_2m_mean_c=("temperature_2m", "mean"),
        temperature_2m_min_c=("temperature_2m", "min"),
        temperature_2m_max_c=("temperature_2m", "max"),
        relative_humidity_2m_mean_pct=("relative_humidity_2m", "mean"),
        wind_speed_10m_mean_kmh=("wind_speed_10m", "mean"),
        wind_speed_10m_max_kmh=("wind_speed_10m", "max"),
        rain_sum_mm=("rain", "sum"),
        shortwave_radiation_mean_wm2=("shortwave_radiation", "mean"),
        shortwave_radiation_sum_kwh_m2=("shortwave_radiation", lambda s: s.sum(min_count=24) / 1000),
        forecast_lead_min_hours=("forecast_lead_hours", "min"),
        forecast_lead_max_hours=("forecast_lead_hours", "max"),
        grid_latitude=("grid_latitude", "first"),
        grid_longitude=("grid_longitude", "first"),
        grid_elevation_m=("grid_elevation_m", "first"),
    )
)
daily_weather["weather_complete_24h"] = daily_weather["weather_hour_count"].eq(24)
weather_features = ["temperature_2m_mean_c", "temperature_2m_min_c", "temperature_2m_max_c", "relative_humidity_2m_mean_pct", "wind_speed_10m_mean_kmh", "wind_speed_10m_max_kmh", "rain_sum_mm", "shortwave_radiation_mean_wm2", "shortwave_radiation_sum_kwh_m2"]
# Aggregated means can hide isolated null hourly values; inspect the source rows too.
hourly_missing = target_hourly.assign(_any_missing=target_hourly[HOURLY_VARIABLES].isna().any(axis=1)).groupby(["weather_location_id", "run_target_date"], as_index=False)["_any_missing"].sum().rename(columns={"_any_missing": "weather_hourly_missing_value_count"})
daily_weather = daily_weather.merge(hourly_missing, on=["weather_location_id", "run_target_date"], how="left", validate="one_to_one")
daily_weather["weather_complete_values"] = daily_weather[weather_features].notna().all(axis=1) & daily_weather["weather_hourly_missing_value_count"].eq(0)
daily_weather["weather_model"] = WEATHER_MODEL
daily_weather["forecast_issue_time_semantics"] = "D-1 00:00 UTC fixed initialization"
daily_weather["weather_timestamp_timezone"] = "UTC source; target-day group by Asia/Colombo"
if daily_weather.duplicated(["weather_location_id", "run_target_date"]).any():
    raise ValueError("Daily fixed-run weather must be unique by location and local target date")
daily_weather.to_csv(DAILY_WEATHER_PATH, index=False, encoding="utf-8")
print("Hourly UTC rows:", f"{len(hourly):,}", "| daily forecasts:", f"{len(daily_weather):,}")
display(daily_weather[["weather_hour_count", "weather_complete_24h", "weather_complete_values"]].value_counts(dropna=False).rename("location-days").reset_index())
display(daily_weather[weather_features + ["forecast_lead_min_hours", "forecast_lead_max_hours"]].describe().T)
"""),
    md("""## 4. Create full-row M1 exports and the paired-comparison audit

Both M1 CSVs preserve every frozen M0 row, its target, and its split assignment. Weather columns are missing where the archive, location mapping, run, or 24-hour forecast is unavailable. Those values are not filled. `m1_weather_complete` identifies the common-row cohort for the later M0-versus-M1 experiment; use the same row IDs and splits for both models.
"""),
    code("""weather_daily_join = daily_weather.rename(columns={"run_target_date": "target_date"}).copy()
weather_daily_join["target_date"] = pd.to_datetime(weather_daily_join["target_date"])
weather_daily_join = weather_daily_join[["weather_location_id", "target_date", *weather_features, "weather_hour_count", "weather_complete_24h", "weather_complete_values", "forecast_run_utc", "forecast_lead_min_hours", "forecast_lead_max_hours", "grid_latitude", "grid_longitude", "grid_elevation_m"]]

row_weather = assignment.copy()
row_weather["target_date"] = row_weather["measurement_date"].dt.normalize()
row_weather = row_weather.merge(weather_daily_join, on=["weather_location_id", "target_date"], how="left", validate="many_to_one")
row_weather["archive_date_supported"] = row_weather["target_date"].ge(MIN_TARGET_DATE)
row_weather["forecast_run_utc"] = pd.to_datetime(row_weather["forecast_run_utc"], utc=True)
row_weather["forecast_cutoff_utc"] = row_weather["target_date"].dt.tz_localize(METER_TIMEZONE).dt.tz_convert("UTC")
row_weather["run_to_cutoff_hours"] = (row_weather["forecast_cutoff_utc"] - row_weather["forecast_run_utc"]).dt.total_seconds() / 3600
row_weather["assumed_release_utc_at_6h"] = row_weather["forecast_run_utc"] + pd.Timedelta(hours=6)
row_weather["assumed_release_margin_before_cutoff_hours"] = (row_weather["forecast_cutoff_utc"] - row_weather["assumed_release_utc_at_6h"]).dt.total_seconds() / 3600
timing_rows = row_weather["forecast_run_utc"].notna()
if not row_weather.loc[timing_rows, "run_to_cutoff_hours"].eq(18.5).all():
    raise ValueError("A forecast run does not precede local-midnight cutoff by 18.5 hours")
if not row_weather.loc[timing_rows, "assumed_release_margin_before_cutoff_hours"].ge(0).all():
    raise ValueError("The conservative 6-hour release assumption is after the prediction cutoff")
row_weather["m1_weather_complete"] = row_weather["coordinate_available"] & row_weather["weather_complete_24h"].fillna(False) & row_weather["weather_complete_values"].fillna(False)
row_weather["weather_row_status"] = "complete_fixed_run_weather"
row_weather.loc[~row_weather["weather_complete_values"].fillna(False).astype(bool), "weather_row_status"] = "weather_values_missing"
row_weather.loc[~row_weather["weather_complete_24h"].fillna(False).astype(bool), "weather_row_status"] = "not_24_hours"
row_weather.loc[row_weather["forecast_run_utc"].isna(), "weather_row_status"] = "forecast_run_missing"
row_weather.loc[~row_weather["archive_date_supported"].fillna(False).astype(bool), "weather_row_status"] = "before_archive_start"
row_weather.loc[~row_weather["coordinate_available"].fillna(False).astype(bool), "weather_row_status"] = "coordinate_missing"
row_weather.loc[row_weather["weather_location_id"].isna(), "weather_row_status"] = "unknown_depot"

def add_weather(frame, p90_target=False):
    result = frame.merge(split[["row_id", "outer_split", "seen_in_training", "test_cohort"]], on="row_id", how="left", validate="one_to_one")
    result = result.merge(row_weather[["row_id", *weather_features, "m1_weather_complete", "weather_row_status"]], on="row_id", how="left", validate="one_to_one")
    if result["outer_split"].isna().any() or len(result) != len(frame):
        raise ValueError("M1 export changed the frozen M0 rows or lost split assignments")
    return result

m1_p95 = add_weather(p95)
m1_p90 = add_weather(p90)
if not m1_p95["row_id"].reset_index(drop=True).equals(p95["row_id"].reset_index(drop=True)) or not m1_p90["row_id"].reset_index(drop=True).equals(p90["row_id"].reset_index(drop=True)):
    raise ValueError("M1 row order does not match the corresponding frozen M0 dataset")

m1_p95.to_csv(M1_P95_PATH, index=False, encoding="utf-8")
m1_p90.to_csv(M1_P90_PATH, index=False, encoding="utf-8")
pair_audit = row_weather[["row_id", "household_id", "target_date", "outer_split", "seen_in_training", "test_cohort", "weather_location_id", "branch", "depot", "substation_id", "coordinate_available", "archive_date_supported", "forecast_run_utc", "forecast_cutoff_utc", "run_to_cutoff_hours", "assumed_release_utc_at_6h", "assumed_release_margin_before_cutoff_hours", "forecast_lead_min_hours", "forecast_lead_max_hours", "weather_hour_count", "weather_complete_24h", "weather_complete_values", "m1_weather_complete", "weather_row_status"]].copy()
pair_audit.to_csv(PAIR_AUDIT_PATH, index=False, encoding="utf-8")
print("M1 P95/P90 row counts:", len(m1_p95), len(m1_p90), "| common complete-weather rows:", int(pair_audit["m1_weather_complete"].sum()))
display(pair_audit.groupby(["outer_split", "weather_row_status"], dropna=False).size().rename("rows").reset_index())
"""),
    md("""## 5. Save provenance manifest and validation summary

Do not train a weather model unless the paired complete-weather cohort has enough development and test observations, both outcomes, and adequate event counts. If coverage is weak, treat weather as an exploratory extension and preserve M0 as the primary baseline.
"""),
    code("""manifest = {
    "source": "Open-Meteo Single Runs API",
    "api_url": API_URL,
    "api_product": "individual_model_run",
    "weather_model": WEATHER_MODEL,
    "nominal_model_grid_km": MODEL_RESOLUTION_KM_NOMINAL,
    "hourly_variables": HOURLY_VARIABLES,
    "source_timestamp_timezone": "UTC",
    "aggregation_timezone": METER_TIMEZONE,
    "run_policy": "target local date D uses D-1 00:00 UTC run",
    "prediction_cutoff_policy": "00:00 local Asia/Colombo at start of target date",
    "run_to_cutoff_hours": 18.5,
    "conservative_release_assumption_hours_after_run": 6,
    "release_margin_before_cutoff_hours_under_assumption": 12.5,
    "forecast_hour_window": FORECAST_HOURS,
    "target_day_hour_definition": "24 model timestamps grouped by Asia/Colombo calendar date (00:30-23:30 local)",
    "min_supported_target_date": MIN_TARGET_DATE.strftime("%Y-%m-%d"),
    "m0_rows": int(len(p95)),
    "m1_weather_complete_rows": int(pair_audit["m1_weather_complete"].sum()),
    "m1_weather_complete_by_split": {str(k): int(v) for k, v in pair_audit.groupby("outer_split")["m1_weather_complete"].sum().items()},
    "m1_weather_complete_p95_events": int(m1_p95.loc[m1_p95["m1_weather_complete"], "target"].sum()),
    "m1_weather_complete_p90_events": int(m1_p90.loc[m1_p90["m1_weather_complete"], "target"].sum()),
    "request_status_counts": {str(k): int(v) for k, v in request_audit["status"].value_counts(dropna=False).items()},
    "limitations": [
        "Archive availability does not independently prove the forecast was publicly accessible at its nominal historical release time.",
        "Approximate town/depot-area coordinates and model grid weather are geographic proxies, not depot observations.",
        "Sri Lanka local day is represented by hourly forecast timestamps 00:30 through 23:30 local; sub-hour boundary amounts are approximate.",
        "Weather is not imputed; pre-archive and unmapped rows remain in full M1 exports with explicit status.",
    ],
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
}
MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
"""),
    md("""## Handoff

Use `07_model_ready_m1_p95.csv` and `07_model_ready_m1_p90.csv` as full-row weather-augmented datasets. For a fair weather ablation, filter M0 and M1 to the same `row_id` values where `m1_weather_complete == True`; keep the frozen `outer_split` and `test_cohort`. Do not impute missing weather or compare metrics on different test rows. P95 remains primary; P90 remains sensitivity analysis.

Key outputs: UTC hourly forecasts, depot-day features, full-row P95/P90 M1 datasets, request audit, row-level pairing audit, and JSON provenance manifest.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, cell in enumerate(cells):
    cell["id"] = f"fixed-weather-{i:02d}"
Path("notebooks/07_weather_data_acquisition.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print("Wrote notebooks/07_weather_data_acquisition.ipynb with", len(cells), "cells")
