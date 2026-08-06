"""Vehicle_Type -> Delay_Risk_Probability leakage check.

Vehicle_Type is not in the scored fact table; it comes from Dim_Route
via Route_ID. See check_leakage.py for the column discovery.

Run:  python leakage_test.py
"""

from pathlib import Path

import pandas as pd

PROCESSED = Path(__file__).resolve().parent / "data" / "processed"

fact = pd.read_csv(PROCESSED / "Fact_Shipments_with_ML.csv")
route = pd.read_csv(PROCESSED / "Dim_Route.csv")[["Route_ID", "Vehicle_Type"]]

df = fact.merge(route, on="Route_ID", how="left")

assert df["Vehicle_Type"].notna().all(), "unmatched Route_ID after merge"

by = df.groupby("Vehicle_Type")["Delay_Risk_Probability"].mean()

print("n rows           :", len(df))
print("group means:")
for level, value in by.items():
    print(f"  {level:<16} n={int((df['Vehicle_Type'] == level).sum()):<5} "
          f"mean={value:.6f}")

leak = (by.max() - by.min()).max()
print(f"\nleak = {leak!r}")
print(f"leak = {leak:.6f}")
print("threshold = 0.05")

assert leak < 0.05, f"Vehicle_Type p'ye sızıyor ({leak:.3f})"
print("\nPASS")
