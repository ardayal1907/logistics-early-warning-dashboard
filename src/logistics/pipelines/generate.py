"""
Smart Logistics & Green Supply Chain - Synthetic Data Generator
==================================================================
Generates 1,500 rows of realistic synthetic logistics data.
- Delay (Actual_Delay_Days): produced by a two-stage "zero-inflated" (hurdle)
  model — first it is decided whether the shipment is late at all, then a
  number of days is drawn only for the late ones. See build_delays().
- CO2_Emission_kg: computed from distance, tonnage and vehicle type using the
  emission factors in config.py.

Why zero-inflated? (v2 revision)
--------------------------------
The previous version produced delay from an additive score, and because every
component was positive, **91%** of shipments ended up late. Consequences:

  * Under Storm / Snow / High traffic the delay rate approached 100% — meaning
    those variables lost all DISCRIMINATING POWER (a ceiling effect).
  * Downstream, the Power BI report showed `High Risk Rate %` = 84.3%. An
    "Early Warning Panel" where 84% of shipments glow red warns of nothing; an
    alert only carries information when it fires rarely.

In real logistics, delays are RARE but HEAVY-TAILED. The process was therefore
split in two: delay probability (logistic) and delay severity (truncated
Poisson). Target delay rate ~22%.

⚠️ A NOTE ON RANDOMNESS AND FUNCTION ORDER
   The output is reproducible only because a single seeded Generator is drawn
   from in a fixed order. The functions below take that generator as an argument
   and MUST be called in the order main() calls them. Reordering two calls — even
   without changing any logic — shifts every subsequent draw and produces a
   completely different dataset. The test suite pins the resulting hash.

Importing this module has no side effects: nothing runs until main() is called.
"""

import logging

import numpy as np
import pandas as pd

# The carbon model lives in one place only. Imported from the domain package
# directly now - src/config.py is a shim and is not part of the wheel.
from logistics.domain.carbon import (
    BASE_EMISSION,
    EMISSION_FACTOR,
    MIN_CO2_KG,
    TRAFFIC_CO2_MULT,
    WEATHER_CO2_MULT,
)
from logistics.infrastructure.fingerprint import write_csv_deterministically
from logistics.settings import Settings

logger = logging.getLogger(__name__)

RAW_FILENAME = "smart_logistics_data.csv"

# Fixed seed for reproducibility
RNG_SEED = 42
N_ROWS = 1500

# ---------------------------------------------------------------------------
# 1) Constants / reference lists
# ---------------------------------------------------------------------------

CITIES = [
    "Istanbul", "Ankara", "Izmir", "Bursa", "Antalya", "Gaziantep",
    "Konya", "Adana", "Mersin", "Kayseri", "Samsun", "Kocaeli",
    "Trabzon", "Denizli", "Eskisehir", "Sakarya", "Diyarbakir",
    "Malatya", "Erzurum", "Van"
]

VENDOR_COUNT = 25
VENDOR_IDS = [f"VEND-{str(i).zfill(3)}" for i in range(1, VENDOR_COUNT + 1)]

WEATHER_CONDITIONS = ["Normal", "Rain", "Storm", "Snow"]

# ---------------------------------------------------------------------------
# SEASONALITY - weather is month-dependent (Northern Hemisphere cycle)
# ---------------------------------------------------------------------------
# In an earlier version weather was drawn from a fixed [0.60, 0.20, 0.10, 0.10]
# distribution all year round, so January and July were indistinguishable. That
# made the time dimension meaningless: a chronological train/test split would
# have been no different from a random one.
#
# DESIGN RULE: seasonality only determines WHICH MONTH each weather condition is
# more frequent in. The effect of weather on delay (WEATHER_LOGIT) and its
# interaction with the vendor (WEATHER_VENDOR_AMP) DO NOT CHANGE AT ALL. No new
# signal is introduced; the existing signal is merely given a temporal order:
#
#     month  ->  weather distribution  ->  (weather x vendor)  ->  delay
#
# The monthly distributions were chosen so that the annual MARGINAL distribution
# stays close to the old [0.60, 0.20, 0.10, 0.10] (verified by print_summary).
#
#                     Normal  Rain  Storm  Snow
MONTHLY_WEATHER_PROBS = {
    1:  [0.30, 0.15, 0.17, 0.38],   # January   - peak winter, snow dominates
    2:  [0.32, 0.15, 0.17, 0.36],   # February  - still harsh winter
    3:  [0.50, 0.25, 0.12, 0.13],   # March     - transition, snow receding
    4:  [0.62, 0.28, 0.08, 0.02],   # April     - spring rains
    5:  [0.70, 0.25, 0.05, 0.00],   # May       - rainy but mild
    6:  [0.82, 0.14, 0.04, 0.00],   # June      - summer beginning
    7:  [0.88, 0.09, 0.03, 0.00],   # July      - clearest month of the year
    8:  [0.87, 0.10, 0.03, 0.00],   # August    - clear, occasional downpour
    9:  [0.75, 0.19, 0.06, 0.00],   # September - start of autumn
    10: [0.62, 0.28, 0.09, 0.01],   # October   - autumn rains
    11: [0.45, 0.27, 0.14, 0.14],   # November  - first snow and storms
    12: [0.33, 0.16, 0.16, 0.35],   # December  - winter sets in
}

# Monthly shipment volume (relative weight). Volume rises in the year-end
# holiday/campaign season - a well-known pattern in logistics.
MONTHLY_VOLUME_WEIGHTS = {
    1: 0.85, 2: 0.80, 3: 0.90, 4: 0.95, 5: 1.00, 6: 1.00,
    7: 0.95, 8: 0.90, 9: 1.05, 10: 1.10, 11: 1.25, 12: 1.35,
}

# The data spans the last 12 months.
DATA_END_DATE = pd.Timestamp("2026-07-31")
DATA_START_DATE = DATA_END_DATE - pd.DateOffset(months=12) + pd.Timedelta(days=1)

TRAFFIC_LEVELS = ["Low", "Medium", "High"]
TRAFFIC_PROBS = [0.35, 0.40, 0.25]

VEHICLE_TYPES = ["Diesel Truck", "Electric Semi", "Hybrid Van"]
VEHICLE_PROBS = [0.55, 0.20, 0.25]  # the fleet is still mostly diesel

# --- Delay model ------------------------------------------------------------
MAX_DELAY_DAYS = 5
TARGET_DELAY_RATE = 0.22          # target overall delay rate

WEATHER_LOGIT = {"Normal": 0.00, "Rain": 0.60, "Storm": 2.00, "Snow": 1.40}
TRAFFIC_LOGIT = {"Low": 0.00, "Medium": 0.50, "High": 1.20}

VENDOR_PIVOT = 3.25               # mean vendor rating -> centring point
VENDOR_SLOPE = 0.80
DISTANCE_PIVOT = 625.0
DISTANCE_SLOPE = 0.50

# Weather x Vendor INTERACTION.
# In an additive model, bad weather hits everyone equally and the importance of
# vendor quality is erased by the ceiling effect. In reality, operational
# competence matters MORE under bad conditions: a good vendor manages the storm,
# a poor one does not.
#
# Because the vendor term is CENTRED on VENDOR_PIVOT, increasing this multiplier
# widens the SPREAD rather than shifting the mean:
#   - good vendor (rating > 3.25) -> term negative -> even more negative in bad weather
#   - poor vendor (rating < 3.25) -> term positive -> even more positive in bad weather
# So "bad weather hits everyone, but hits a quality vendor less", and thanks to
# the centring, the MARGINAL effect of weather is largely preserved.
WEATHER_VENDOR_AMP = {"Normal": 0.00, "Rain": 0.20, "Storm": 0.75, "Snow": 0.50}

WEATHER_SEVERITY = {"Normal": 0.00, "Rain": 0.25, "Storm": 1.10, "Snow": 0.75}
TRAFFIC_SEVERITY = {"Low": 0.00, "Medium": 0.15, "High": 0.45}
SEVERITY_BASE = 0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def solve_intercept(linear, target, lo=-8.0, hi=4.0, iters=90):
    """Binary-search the constant that makes E[sigmoid(linear + c)] = target.

    Rather than writing the base delay rate as a hand-picked magic number, we
    derive it, so the rate stays on target when the other coefficients change.
    Consumes no randomness.
    """
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if (1.0 / (1.0 + np.exp(-(linear + mid)))).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _log_factorial(k):
    return np.cumsum(np.concatenate(([0.0], np.log(np.arange(1, k.max() + 1)))))[k]


def truncated_poisson_days(lam_vec, max_days, generator):
    """A PROPERLY truncated Poisson draw over the range 1..max_days.

    Simply writing `np.clip(1 + Poisson(lam), 1, max_days)` piles the tail up at
    max_days and breaks monotonicity (e.g. producing more 5-day than 4-day
    delays). Instead we normalise the pmf over k = 0..max_days-1 and sample by
    inverse CDF. Requires no scipy.

    Draws exactly one uniform vector from `generator`.
    """
    k = np.arange(max_days)
    log_pmf = (-lam_vec[:, None] + k[None, :] * np.log(lam_vec[:, None])
               - _log_factorial(k)[None, :])
    pmf = np.exp(log_pmf)
    pmf /= pmf.sum(axis=1, keepdims=True)
    u = generator.random(len(lam_vec))[:, None]
    return 1 + (u > pmf.cumsum(axis=1)).sum(axis=1)


# ---------------------------------------------------------------------------
# 2) Generating the base columns
# ---------------------------------------------------------------------------
def build_base_columns(rng, n_rows: int = N_ROWS) -> dict:
    """Vendors, cities, distances, weights, dates, weather, traffic, vehicles.

    ⚠️ The order of the draws below is load-bearing for reproducibility.
    """
    # Each vendor is assigned a fixed "quality" rating (between 1.0 and 5.0) so
    # that the same vendor shows a consistent performance profile.
    vendor_base_rating = {vid: round(rng.uniform(1.5, 5.0), 1) for vid in VENDOR_IDS}

    shipment_ids = [f"SHP-{str(i).zfill(5)}" for i in range(1, n_rows + 1)]
    vendor_ids = rng.choice(VENDOR_IDS, size=n_rows)

    # Small noise around the vendor's base rating gives realistic per-shipment variety.
    vendor_ratings = np.array([
        np.clip(vendor_base_rating[vid] + rng.normal(0, 0.15), 1.0, 5.0)
        for vid in vendor_ids
    ]).round(1)

    # Origin / Destination: cannot be the same city
    origins = rng.choice(CITIES, size=n_rows)
    destinations = np.array([
        rng.choice([c for c in CITIES if c != o]) for o in origins
    ])

    distance_km = rng.uniform(50, 1200, size=n_rows).round(1)
    weight_tons = rng.uniform(1, 25, size=n_rows).round(2)

    # Shipment_Date - spread over the last 12 months with seasonal volume
    all_days = pd.date_range(DATA_START_DATE, DATA_END_DATE, freq="D")
    day_weights = np.array([MONTHLY_VOLUME_WEIGHTS[d.month] for d in all_days],
                           dtype=float)
    day_weights /= day_weights.sum()

    shipment_dates = pd.DatetimeIndex(
        rng.choice(all_days, size=n_rows, p=day_weights)
    ).sort_values()          # chronological order: Shipment_ID follows the timeline

    months = shipment_dates.month.to_numpy()

    # Weather is drawn PER MONTH (this is where seasonality enters).
    weather = np.empty(n_rows, dtype=object)
    for m, probs in MONTHLY_WEATHER_PROBS.items():
        mask = months == m
        if mask.any():
            weather[mask] = rng.choice(WEATHER_CONDITIONS, size=int(mask.sum()),
                                       p=probs)
    weather = weather.astype(str)

    # Traffic and vehicle type are NOT seasonal - the scope is kept deliberately narrow.
    traffic = rng.choice(TRAFFIC_LEVELS, size=n_rows, p=TRAFFIC_PROBS)
    vehicle = rng.choice(VEHICLE_TYPES, size=n_rows, p=VEHICLE_PROBS)

    return {
        "shipment_ids": shipment_ids,
        "shipment_dates": shipment_dates,
        "vendor_ids": vendor_ids,
        "vendor_ratings": vendor_ratings,
        "origins": origins,
        "destinations": destinations,
        "distance_km": distance_km,
        "weight_tons": weight_tons,
        "weather": weather,
        "traffic": traffic,
        "vehicle": vehicle,
    }


# ---------------------------------------------------------------------------
# 3) Actual_Delay_Days - two-stage (zero-inflated / hurdle) delay model
# ---------------------------------------------------------------------------
def build_delays(rng, cols: dict) -> tuple:
    """STAGE 1 - Will it be late?  P(delay) = sigmoid(logit), Bernoulli draw
    STAGE 2 - By how many days? A Poisson truncated to 1..MAX_DELAY_DAYS,
              drawn only for the late shipments

    Separating the two stages naturally produces the "most shipments on time; the
    late ones are seriously late" structure. A single-stage additive model cannot,
    because once every component is positive, reaching 0 becomes impossible.

    Returns (actual_delay_days, intercept).
    """
    weather, traffic = cols["weather"], cols["traffic"]
    vendor_ratings, distance_km = cols["vendor_ratings"], cols["distance_km"]
    n_rows = len(vendor_ratings)

    vendor_effect = (VENDOR_PIVOT - vendor_ratings) * VENDOR_SLOPE
    amp = np.array([1.0 + WEATHER_VENDOR_AMP[w] for w in weather])

    logit_wo_intercept = (
        np.array([WEATHER_LOGIT[w] for w in weather])
        + np.array([TRAFFIC_LOGIT[t] for t in traffic])
        + vendor_effect * amp
        + ((distance_km - DISTANCE_PIVOT) / 1200.0) * DISTANCE_SLOPE
    )

    intercept = solve_intercept(logit_wo_intercept, TARGET_DELAY_RATE)
    delay_probability = 1.0 / (1.0 + np.exp(-(logit_wo_intercept + intercept)))
    is_delayed = rng.random(n_rows) < delay_probability

    lam = (
        SEVERITY_BASE
        + np.array([WEATHER_SEVERITY[w] for w in weather])
        + np.array([TRAFFIC_SEVERITY[t] for t in traffic])
        + (VENDOR_PIVOT - vendor_ratings) * 0.18
    ).clip(0.05, None)

    actual_delay_days = np.where(
        is_delayed, truncated_poisson_days(lam, MAX_DELAY_DAYS, rng), 0
    ).astype(int)

    return actual_delay_days, intercept


# ---------------------------------------------------------------------------
# 4) CO2_Emission_kg
# ---------------------------------------------------------------------------
def build_co2(rng, cols: dict) -> np.ndarray:
    """CO2 = (distance * tonnage * emission_factor + fixed base)
             * weather multiplier * traffic multiplier
             * a small random noise term (±5%, for realism)

    The constants come from src/config.py, shared with the Streamlit demo.
    """
    vehicle, weather, traffic = cols["vehicle"], cols["weather"], cols["traffic"]
    distance_km, weight_tons = cols["distance_km"], cols["weight_tons"]

    base_co2 = distance_km * weight_tons * np.array([EMISSION_FACTOR[v] for v in vehicle])
    fixed_co2 = np.array([BASE_EMISSION[v] for v in vehicle])

    weather_mult = np.array([WEATHER_CO2_MULT[w] for w in weather])
    traffic_mult = np.array([TRAFFIC_CO2_MULT[t] for t in traffic])

    co2_before_noise = (base_co2 + fixed_co2) * weather_mult * traffic_mult

    co2_noise = rng.normal(1.0, 0.03, size=len(distance_km))
    return np.clip(co2_before_noise * co2_noise, MIN_CO2_KG, None).round(2)


# ---------------------------------------------------------------------------
# 5) Assembly
# ---------------------------------------------------------------------------
def assemble_dataframe(cols: dict, actual_delay_days, co2_emission_kg) -> pd.DataFrame:
    return pd.DataFrame({
        "Shipment_ID": cols["shipment_ids"],
        "Shipment_Date": cols["shipment_dates"].strftime("%Y-%m-%d"),
        "Vendor_ID": cols["vendor_ids"],
        "Vendor_Rating": cols["vendor_ratings"],
        "Origin": cols["origins"],
        "Destination": cols["destinations"],
        "Distance_km": cols["distance_km"],
        "Weight_tons": cols["weight_tons"],
        "Weather_Condition": cols["weather"],
        "Traffic_Density": cols["traffic"],
        "Vehicle_Type": cols["vehicle"],
        "Actual_Delay_Days": actual_delay_days,
        "CO2_Emission_kg": co2_emission_kg,
    })


def generate_dataset(seed: int = RNG_SEED, n_rows: int = N_ROWS) -> tuple:
    """Build the whole dataset from one seeded generator.

    Returns (dataframe, intercept). Exposed separately from main() so tests can
    generate data without touching the filesystem.
    """
    rng = np.random.default_rng(seed)
    cols = build_base_columns(rng, n_rows)
    actual_delay_days, intercept = build_delays(rng, cols)
    co2_emission_kg = build_co2(rng, cols)
    return assemble_dataframe(cols, actual_delay_days, co2_emission_kg), intercept


# ---------------------------------------------------------------------------
# 6) Composition
# ---------------------------------------------------------------------------
def main(settings: Settings | None = None) -> pd.DataFrame:
    """Generate the raw dataset and write it. Paths come from Settings.

    The 27 print() calls that used to live here moved to
    logistics.pipelines.report.render_generation_report - a scheduled run has
    no terminal, and a function that prints cannot be called by anything that
    does not want its stdout written to.
    """
    settings = settings or Settings.from_env()
    df, intercept = generate_dataset()

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.raw_dir / RAW_FILENAME
    # Deterministic line endings: the byte hash of this file is recorded in the
    # model metadata, so it must not depend on the operating system.
    write_csv_deterministically(df, output_path)

    logger.info(
        "Generated %d rows (%d columns) to %s; intercept %.4f, delay rate %.4f",
        len(df), len(df.columns), output_path, intercept,
        float((df["Actual_Delay_Days"] > 0).mean()),
    )
    return df


if __name__ == "__main__":  # pragma: no cover
    from logistics.pipelines.report import render_generation_report

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    print(render_generation_report(main(), TARGET_DELAY_RATE))
