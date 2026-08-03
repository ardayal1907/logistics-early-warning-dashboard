"""
ETL: smart_logistics_data.csv (ham/flat veri) -> Yıldız Şema (Star Schema)
============================================================================
Üretilen tablolar:
  1) Dim_Vendor.csv      - Vendor_ID (PK), Vendor_Rating
  2) Dim_Route.csv        - Route_ID (PK), Origin, Destination, Vehicle_Type,
                             Weather_Condition, Traffic_Density
  3) Dim_Date.csv         - Date_ID (PK, YYYYMMDD), Full_Date, Year, Quarter,
                             Month, Month_Name, Month_Year, Season,
                             Day_Of_Week, Is_Weekend
  4) Fact_Shipments.csv   - Shipment_ID (PK), Date_ID (FK), Vendor_ID (FK),
                             Route_ID (FK), Weight_tons, Distance_km,
                             Actual_Delay_Days, CO2_Emission_kg

Önemli tasarım notu:
  Ham veride Vendor_Rating, aynı Vendor_ID için sevkiyattan sevkiyata küçük
  gürültü nedeniyle hafifçe farklılık gösterebiliyor (gerçekçilik amacıyla
  eklenmişti). Star schema'da bir boyut tablosu satır başına TEK bir gerçeği
  temsil etmelidir; bu yüzden Dim_Vendor'da her Vendor_ID için ortalama
  (mean) Vendor_Rating alınıp tek bir satıra indirgenmiştir. Sevkiyat bazlı
  orijinal puan isteniyorsa Fact_Shipments'a ayrı bir sütun olarak eklenebilir
  (bu script varsayılan olarak eklemiyor, çünkü spesifikasyon Fact tablosunda
  Vendor_Rating istemiyor).
"""

from pathlib import Path

import pandas as pd

# Yollar repo köküne göre çözülür; script'i hangi dizinden çağırdığınız fark etmez.
ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = ROOT / "data" / "raw" / "smart_logistics_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 0) Ham veriyi oku
# ---------------------------------------------------------------------------
df = pd.read_csv(RAW_CSV_PATH)

required_cols = [
    "Shipment_ID", "Shipment_Date", "Vendor_ID", "Vendor_Rating", "Origin",
    "Destination", "Distance_km", "Weight_tons", "Weather_Condition",
    "Traffic_Density", "Vehicle_Type", "Actual_Delay_Days", "CO2_Emission_kg",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Ham CSV'de eksik sütun(lar) var: {missing}")

print(f"Ham veri okundu: {len(df)} satır, {len(df.columns)} sütun\n")

# ---------------------------------------------------------------------------
# 1) Dim_Vendor - Vendor_ID (PK), Vendor_Rating
# ---------------------------------------------------------------------------
# Aynı Vendor_ID için birden fazla (hafifçe farklı) Vendor_Rating değeri
# olabileceğinden, tedarikçi başına ortalama puan alınarak tekilleştirilir.
dim_vendor = (
    df.groupby("Vendor_ID", as_index=False)["Vendor_Rating"]
    .mean()
    .round(2)
    .sort_values("Vendor_ID")
    .reset_index(drop=True)
)

assert dim_vendor["Vendor_ID"].is_unique, "Dim_Vendor.Vendor_ID benzersiz olmalı!"

# ---------------------------------------------------------------------------
# 2) Dim_Route - Route_ID (PK) + Origin, Destination, Vehicle_Type,
#    Weather_Condition, Traffic_Density kombinasyonu
# ---------------------------------------------------------------------------
route_cols = ["Origin", "Destination", "Vehicle_Type", "Weather_Condition", "Traffic_Density"]

dim_route = (
    df[route_cols]
    .drop_duplicates()
    .sort_values(route_cols)
    .reset_index(drop=True)
)
dim_route.insert(0, "Route_ID", [f"RT-{str(i).zfill(5)}" for i in range(1, len(dim_route) + 1)])

assert dim_route["Route_ID"].is_unique, "Dim_Route.Route_ID benzersiz olmalı!"

# ---------------------------------------------------------------------------
# 2b) Dim_Date - Date_ID (PK) + standart takvim öznitelikleri
# ---------------------------------------------------------------------------
# Kimball tarzı tarih boyutu. İki tasarım kararı:
#
#   1) Date_ID = YYYYMMDD tamsayısı ("smart key"). Klasik Kimball
#      konvansiyonudur: doğal olarak sıralanır, join'i ucuzdur ve Power BI
#      tarafında sayısal bir ilişki kolonu olur.
#   2) Tablo, veride sevkiyat OLMAYAN günleri de içerir (kesintisiz takvim).
#      Bu, zaman zekâsı (time intelligence) ölçüleri için şarttır: eksik
#      günler olan bir tarih tablosunda hareketli ortalama, YTD, önceki yılla
#      kıyas gibi hesaplar sessizce yanlış sonuç verir.
shipment_dates = pd.to_datetime(df["Shipment_Date"])
calendar = pd.date_range(shipment_dates.min(), shipment_dates.max(), freq="D")

SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

dim_date = pd.DataFrame({
    "Date_ID": calendar.strftime("%Y%m%d").astype(int),
    "Full_Date": calendar.strftime("%Y-%m-%d"),
    "Year": calendar.year,
    "Quarter": "Q" + calendar.quarter.astype(str),
    "Month": calendar.month,
    "Month_Name": calendar.strftime("%B"),
    "Month_Year": calendar.strftime("%Y-%m"),
    "Season": [SEASON_BY_MONTH[m] for m in calendar.month],
    "Day_Of_Week": calendar.strftime("%A"),
    "Is_Weekend": calendar.dayofweek >= 5,
})

assert dim_date["Date_ID"].is_unique, "Dim_Date.Date_ID benzersiz olmalı!"
assert len(dim_date) == (calendar.max() - calendar.min()).days + 1, \
    "Dim_Date takvimi kesintili! Zaman zekâsı ölçüleri bozulur."

# ---------------------------------------------------------------------------
# 3) Fact_Shipments - Shipment_ID (PK), Vendor_ID (FK), Route_ID (FK),
#    Weight_tons, Distance_km, Actual_Delay_Days, CO2_Emission_kg
# ---------------------------------------------------------------------------
# Route_ID'yi bulmak için ham veriyi Dim_Route ile route_cols üzerinden
# eşleştiriyoruz (bu bir lookup/merge işlemidir, yeni satır üretmez).
fact_shipments = df.merge(dim_route, on=route_cols, how="left")

# Tarih boyutuna FK: Shipment_Date -> Date_ID (YYYYMMDD)
fact_shipments["Date_ID"] = (
    pd.to_datetime(fact_shipments["Shipment_Date"]).dt.strftime("%Y%m%d").astype(int)
)

fact_shipments = fact_shipments[
    [
        "Shipment_ID",
        "Date_ID",
        "Vendor_ID",
        "Route_ID",
        "Weight_tons",
        "Distance_km",
        "Actual_Delay_Days",
        "CO2_Emission_kg",
    ]
].sort_values("Shipment_ID").reset_index(drop=True)

assert fact_shipments["Shipment_ID"].is_unique, "Fact_Shipments.Shipment_ID benzersiz olmalı!"
assert fact_shipments["Route_ID"].notna().all(), "Eşleşmeyen Route_ID bulundu!"
assert fact_shipments["Date_ID"].notna().all(), "Eşleşmeyen Date_ID bulundu!"
assert fact_shipments["Vendor_ID"].isin(dim_vendor["Vendor_ID"]).all(), "Geçersiz Vendor_ID FK bulundu!"
assert fact_shipments["Route_ID"].isin(dim_route["Route_ID"]).all(), "Geçersiz Route_ID FK bulundu!"
assert fact_shipments["Date_ID"].isin(dim_date["Date_ID"]).all(), "Geçersiz Date_ID FK bulundu!"

# ---------------------------------------------------------------------------
# 4) CSV olarak dışa aktar
# ---------------------------------------------------------------------------
dim_vendor.to_csv(PROCESSED_DIR / "Dim_Vendor.csv", index=False)
dim_route.to_csv(PROCESSED_DIR / "Dim_Route.csv", index=False)
dim_date.to_csv(PROCESSED_DIR / "Dim_Date.csv", index=False)
fact_shipments.to_csv(PROCESSED_DIR / "Fact_Shipments.csv", index=False)

# ---------------------------------------------------------------------------
# 5) Özet / doğrulama çıktısı
# ---------------------------------------------------------------------------
print(f"Dim_Vendor.csv      -> {len(dim_vendor)} satır (benzersiz tedarikçi)")
print(f"Dim_Route.csv       -> {len(dim_route)} satır (benzersiz rota kombinasyonu)")
print(f"Dim_Date.csv        -> {len(dim_date)} satır (kesintisiz takvim: "
      f"{dim_date.Full_Date.min()} .. {dim_date.Full_Date.max()})")
print(f"Fact_Shipments.csv  -> {len(fact_shipments)} satır (orijinal sevkiyat sayısıyla aynı: {len(df)})")

print("\n--- Dim_Vendor önizleme ---")
print(dim_vendor.head())

print("\n--- Dim_Route önizleme ---")
print(dim_route.head())

print("\n--- Dim_Date önizleme ---")
print(dim_date.head())
print(f"\nSevkiyatı olmayan gün sayısı: "
      f"{len(dim_date) - fact_shipments.Date_ID.nunique()} "
      f"(takvimde var, fact'te yok - bu NORMAL ve gereklidir)")

print("\n--- Fact_Shipments önizleme ---")
print(fact_shipments.head())

print("\nBütünlük kontrolleri başarılı: tüm PK'lar benzersiz, tüm FK'lar boyut "
      "tablolarında mevcut.")
