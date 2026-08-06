"""The CO2 figures in the generated data must come from the constants in config.py.

The generator multiplies the deterministic figure by a N(1.0, 0.03) noise term for
realism, so an exact match is not expected. What must hold is:
  * every row is within a few sigma of the deterministic value, and
  * across 1,500 rows the mean ratio collapses to 1.0.
If someone edits an emission factor in config.py without regenerating the data (or
worse, re-declares one somewhere else), both checks break.
"""

import config
import pytest

NOISE_SIGMA = 0.03


def _expected(row):
    return config.compute_co2_kg(
        row.Distance_km, row.Weight_tons, row.Vehicle_Type,
        row.Weather_Condition, row.Traffic_Density,
    )


def test_sample_rows_match_within_noise_band(raw_data):
    """A handful of specific rows, checked individually."""
    sample = raw_data.head(25)
    for row in sample.itertuples():
        expected = _expected(row)
        ratio = row.CO2_Emission_kg / expected
        assert 1 - 5 * NOISE_SIGMA < ratio < 1 + 5 * NOISE_SIGMA, (
            f"{row.Shipment_ID}: CSV has {row.CO2_Emission_kg:.2f} kg but config.py "
            f"gives {expected:.2f} kg (ratio {ratio:.4f}) - the emission constants "
            "and the generated data have drifted apart."
        )


def test_mean_ratio_collapses_to_one(raw_data):
    """Across the whole dataset the noise averages out."""
    expected = raw_data.apply(_expected, axis=1)
    mean_ratio = (raw_data.CO2_Emission_kg / expected).mean()
    assert mean_ratio == pytest.approx(1.0, abs=0.005), (
        f"Mean CO2 ratio is {mean_ratio:.4f}, expected ~1.0. A systematic offset "
        "means the formula in config.py no longer matches the one used to generate "
        "the data."
    )


def test_vehicle_ordering_follows_emission_factors(raw_data):
    """Diesel > Hybrid > Electric, as the emission factors dictate."""
    mean_by_vehicle = raw_data.groupby("Vehicle_Type").CO2_Emission_kg.mean()
    ordered = mean_by_vehicle.sort_values(ascending=False).index.tolist()
    assert ordered == ["Diesel Truck", "Hybrid Van", "Electric Semi"], (
        f"Unexpected emission ordering: {ordered}. Check EMISSION_FACTOR in config.py."
    )


def test_floor_is_respected(raw_data):
    assert (raw_data.CO2_Emission_kg >= config.MIN_CO2_KG).all(), \
        f"Some rows fall below the MIN_CO2_KG floor of {config.MIN_CO2_KG} kg."
