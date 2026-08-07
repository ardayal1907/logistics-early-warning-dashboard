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

# Approximate city-centre coordinates (WGS84). These exist so that Distance_km
# can be a FUNCTION OF THE CITY PAIR instead of an independent draw. The v1
# generator drew distance from uniform(50, 1200) with no reference to origin or
# destination, so the same pair carried wildly different distances - Istanbul ->
# Denizli ranged from 67.9 km to 1,169.2 km across six shipments. A star schema
# whose Dim_Route cannot answer "how far is this route" is not a route dimension.
CITY_COORDS = {
    "Istanbul":   (41.01, 28.98),
    "Ankara":     (39.93, 32.86),
    "Izmir":      (38.42, 27.14),
    "Bursa":      (40.19, 29.06),
    "Antalya":    (36.90, 30.71),
    "Gaziantep":  (37.07, 37.38),
    "Konya":      (37.87, 32.48),
    "Adana":      (37.00, 35.32),
    "Mersin":     (36.81, 34.64),
    "Kayseri":    (38.73, 35.49),
    "Samsun":     (41.29, 36.33),
    "Kocaeli":    (40.77, 29.94),
    "Trabzon":    (41.00, 39.72),
    "Denizli":    (37.78, 29.09),
    "Eskisehir":  (39.78, 30.52),
    "Sakarya":    (40.76, 30.38),
    "Diyarbakir": (37.91, 40.24),
    "Malatya":    (38.35, 38.31),
    "Erzurum":    (39.90, 41.27),
    "Van":        (38.49, 43.38),
}

# Great-circle distance understates what a lorry drives. 1.25 is the standard
# road-detour factor for a country with this topography; it puts Istanbul -> Van
# at ~1,590 km against a real road distance of roughly 1,650 km.
ROAD_DETOUR_FACTOR = 1.25

# Per-shipment variation around the route's base distance: depot pick-up legs,
# diversions, roadworks. The roadmap specified +/-5%.
DISTANCE_NOISE_PCT = 0.05

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
# Fleet-wide marginal the assignment is calibrated to reproduce. In v1 this was
# fed straight to rng.choice, so Vehicle_Type was independent of every other
# column - the W3 finding. An operator does not dispatch at random, and a
# generator that says they do makes every abatement figure a measure of its own
# noise rather than of fleet inefficiency.
VEHICLE_PROBS = [0.55, 0.20, 0.25]  # the fleet is still mostly diesel

# --- Vehicle assignment (v2) ------------------------------------------------
# Utilities in a softmax, deliberately MILD. Two constraints pull against each
# other here:
#   * too weak and the column is noise again, which is what v1 got wrong;
#   * too strong and Vehicle_Type becomes a proxy for Distance_km and
#     Weight_tons, which are already model features. The optimiser would then
#     read "switch to Electric Semi" as "make the shipment shorter", which is
#     not a decision anyone can take.
# MEASURED with these coefficients on the shipped seed: the diesel share runs
# 48.0% on the shortest distance quartile to 70.9% on the longest, and the
# marginal lands at 56.5/19.1/24.3 against the 55/20/25 target. A real dispatcher
# is at least this predictable; the longest lane is still not 100% diesel, so the
# column carries information without becoming a deterministic restatement of
# Distance_km. Per-vendor diesel share spans 0.25 to 0.75 (sd 0.14).
#
# Vehicle_Type still has NO causal effect on delay - build_delays does not read
# it. The correlation created here is operational (who dispatches what), not
# causal, which is exactly the structure a modal-allocation optimiser needs.
VEHICLE_BASE_UTILITY = {"Diesel Truck": 0.72, "Electric Semi": -0.42, "Hybrid Van": -0.18}

# Long haul favours diesel; battery range still rules the electric semi out of
# the longest lanes, and a van is not a long-distance vehicle.
VEHICLE_DISTANCE_COEF = {"Diesel Truck": 0.45, "Electric Semi": -0.25, "Hybrid Van": -0.70}

# Heavy loads favour the two trucks over the van.
VEHICLE_WEIGHT_COEF = {"Diesel Truck": 0.35, "Electric Semi": 0.30, "Hybrid Van": -1.00}

# Per-vendor fleet character: some operators have invested in electrification,
# others have not. Drawn once per vendor, so a vendor is consistent across its
# shipments the way a real fleet would be.
VENDOR_FLEET_BIAS_SD = 0.45

# --- Delay model ------------------------------------------------------------
MAX_DELAY_DAYS = 5
TARGET_DELAY_RATE = 0.22          # target overall delay rate

WEATHER_LOGIT = {"Normal": 0.00, "Rain": 0.60, "Storm": 2.00, "Snow": 1.40}
TRAFFIC_LOGIT = {"Low": 0.00, "Medium": 0.50, "High": 1.20}

VENDOR_PIVOT = 3.25               # mean vendor rating -> centring point
VENDOR_SLOPE = 0.80
# Centring point and span for the distance term. Both follow from the distance
# matrix rather than the old uniform(50, 1200) draw: the mean over all 380 ordered
# city pairs is 694 km and the widest is Istanbul <-> Van at 1,765 km. Keeping the
# old 625 / 1200 would have left the term off-centre and over-scaled, which
# solve_intercept would have masked at the mean while distorting the spread.
DISTANCE_PIVOT = 694.0
DISTANCE_SCALE_KM = 1800.0
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
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres. Consumes no randomness."""
    radius = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def build_distance_matrix() -> dict[tuple[str, str], float]:
    """Road distance for every ordered city pair, derived once from CITY_COORDS.

    Symmetric by construction and independent of the RNG, so the same pair always
    resolves to the same base distance no matter which shipment asks. This is the
    fix for the distance rule: `test_distance_is_a_function_of_the_city_pair`.
    """
    matrix: dict[tuple[str, str], float] = {}
    for a, (lat1, lon1) in CITY_COORDS.items():
        for b, (lat2, lon2) in CITY_COORDS.items():
            if a == b:
                continue
            matrix[(a, b)] = round(
                haversine_km(lat1, lon1, lat2, lon2) * ROAD_DETOUR_FACTOR, 1
            )
    return matrix


ROUTE_DISTANCE_KM = build_distance_matrix()


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
def assign_vehicles(rng, distance_km, weight_tons, vendor_ids, vendor_fleet_bias):
    """Dispatch a vehicle type per shipment from route, load and vendor fleet.

    A softmax over three utilities. Distance and weight are standardised to
    roughly [-1, 1] over their observed span so the coefficients read as "logit
    swing across the full range" rather than "per kilometre".

    Returns a str array. Consumes exactly one uniform draw per row, after the
    per-vendor bias has been drawn by the caller - the draw order is load-bearing
    for reproducibility (see the module docstring).
    """
    n_rows = len(distance_km)

    # Standardise: 694 km and 13 t are the respective centres, and the divisors
    # are half the span, so a full-range move is about +/-1.
    d_std = (np.asarray(distance_km, dtype=float) - DISTANCE_PIVOT) / 900.0
    w_std = (np.asarray(weight_tons, dtype=float) - 13.0) / 12.0

    base = np.array([VEHICLE_BASE_UTILITY[v] for v in VEHICLE_TYPES])
    d_coef = np.array([VEHICLE_DISTANCE_COEF[v] for v in VEHICLE_TYPES])
    w_coef = np.array([VEHICLE_WEIGHT_COEF[v] for v in VEHICLE_TYPES])

    bias = np.array([vendor_fleet_bias[vid] for vid in vendor_ids])   # (n, 3)

    utility = (
        base[None, :]
        + np.outer(d_std, d_coef)
        + np.outer(w_std, w_coef)
        + bias
    )

    utility -= utility.max(axis=1, keepdims=True)      # softmax overflow guard
    probs = np.exp(utility)
    probs /= probs.sum(axis=1, keepdims=True)

    # One uniform per row, turned into a categorical pick by inverse CDF. Doing
    # it this way rather than looping rng.choice keeps the randomness consumption
    # a single vectorised draw, which is what makes the dataset reproducible.
    draws = rng.random(n_rows)
    picks = (probs.cumsum(axis=1) < draws[:, None]).sum(axis=1).clip(0, len(VEHICLE_TYPES) - 1)
    return np.array(VEHICLE_TYPES, dtype=object)[picks].astype(str)


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

    # Distance is a property of the ROUTE, not of the shipment. The base comes
    # from the city-pair matrix; the per-shipment noise is pick-up legs and
    # diversions, not a fresh draw of "how far apart are these two cities".
    base_distance = np.array([ROUTE_DISTANCE_KM[(o, d)]
                              for o, d in zip(origins, destinations, strict=True)])
    distance_noise = rng.uniform(1.0 - DISTANCE_NOISE_PCT, 1.0 + DISTANCE_NOISE_PCT,
                                 size=n_rows)
    distance_km = (base_distance * distance_noise).round(1)

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

    # Traffic is NOT seasonal - the scope is kept deliberately narrow.
    traffic = rng.choice(TRAFFIC_LEVELS, size=n_rows, p=TRAFFIC_PROBS)

    # Vehicle type is DISPATCHED, not drawn (v2). See the block comment above
    # VEHICLE_BASE_UTILITY for why the coefficients are deliberately mild.
    vendor_fleet_bias = {
        vid: rng.normal(0, VENDOR_FLEET_BIAS_SD, size=len(VEHICLE_TYPES))
        for vid in VENDOR_IDS
    }
    vehicle = assign_vehicles(rng, distance_km, weight_tons, vendor_ids,
                              vendor_fleet_bias)

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
        + ((distance_km - DISTANCE_PIVOT) / DISTANCE_SCALE_KM) * DISTANCE_SLOPE
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
