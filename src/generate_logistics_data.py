"""
Akıllı Lojistik ve Yeşil Tedarik Zinciri - Sentetik Veri Üretici
==================================================================
1500 satırlık gerçekçi sentetik lojistik verisi üretir.
- Gecikme (Actual_Delay_Days): İki aşamalı "zero-inflated" (hurdle) modelle
  üretilir — önce sevkiyatın gecikip gecikmeyeceğine karar verilir, sonra
  yalnızca gecikenler için gün sayısı çekilir. Ayrıntı için bkz. Bölüm 3.
- CO2_Emission_kg: Mesafe, tonaj ve araç tipine göre gerçekçi emisyon
  faktörleriyle hesaplanır.

Neden zero-inflated? (v2 revizyonu)
-----------------------------------
Önceki sürüm gecikmeyi toplamsal (additive) bir skordan üretiyordu ve tüm
bileşenler pozitif olduğu için sevkiyatların **%91'i** gecikiyordu. Sonuçları:

  * Storm / Snow / High-traffic altında gecikme oranı %100'e dayanıyordu — yani
    bu değişkenlerin hiçbir AYRIM GÜCÜ kalmıyordu (tavan etkisi).
  * Aşağı akıştaki Power BI raporunda `High Risk Rate %` = %84.3 çıkıyordu.
    Sevkiyatların %84'ünün kırmızı yandığı bir "Erken Uyarı Paneli" hiçbir şey
    uyarmaz; bir alarm ancak seyrek ateşlediğinde bilgi taşır.

Gerçek lojistikte gecikmeler SEYREK ama KUYRUKLUDUR. Bu yüzden süreç ikiye
ayrıldı: gecikme olasılığı (lojistik) ve gecikme şiddeti (kesilmiş Poisson).
Hedef gecikme oranı ~%22.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Yollar repo köküne göre çözülür; script'i hangi dizinden çağırdığınız fark etmez.
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Tekrarlanabilirlik için sabit seed
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

N_ROWS = 1500

# ---------------------------------------------------------------------------
# 1) Sabit tanımlar / referans listeleri
# ---------------------------------------------------------------------------

CITIES = [
    "Istanbul", "Ankara", "Izmir", "Bursa", "Antalya", "Gaziantep",
    "Konya", "Adana", "Mersin", "Kayseri", "Samsun", "Kocaeli",
    "Trabzon", "Denizli", "Eskisehir", "Sakarya", "Diyarbakir",
    "Malatya", "Erzurum", "Van"
]

VENDOR_COUNT = 25
VENDOR_IDS = [f"VEND-{str(i).zfill(3)}" for i in range(1, VENDOR_COUNT + 1)]

# Her tedarikçiye sabit bir "kalite" puanı atanır (1.0 - 5.0 arası),
# böylece aynı tedarikçinin siparişlerinde tutarlı bir performans profili olur.
vendor_base_rating = {
    vid: round(rng.uniform(1.5, 5.0), 1) for vid in VENDOR_IDS
}

WEATHER_CONDITIONS = ["Normal", "Rain", "Storm", "Snow"]

# ---------------------------------------------------------------------------
# MEVSİMSELLİK - hava koşulu artık aya bağlı (kuzey yarımküre döngüsü)
# ---------------------------------------------------------------------------
# Önceki sürümde hava tüm yıl boyunca sabit [0.60, 0.20, 0.10, 0.10]
# dağılımından çekiliyordu; yani Ocak ile Temmuz arasında hiçbir fark yoktu.
# Bu, zaman boyutunu anlamsız kılıyordu: kronolojik bir train/test bölmesi
# rastgele bölmeden farksız olurdu.
#
# TASARIM KURALI: Mevsimsellik yalnızca hangi hava koşulunun HANGİ AYDA daha
# sık görüldüğünü belirler. Havanın gecikmeye etkisi (weather_logit) ve
# tedarikçiyle etkileşimi (weather_vendor_amp) HİÇ DEĞİŞMEZ. Yeni bir sinyal
# eklenmiyor; var olan sinyale bir zaman düzeni kazandırılıyor:
#
#     ay  ->  hava dağılımı  ->  (hava x tedarikçi)  ->  gecikme
#
# Aylık dağılımlar, yıllık MARJİNAL dağılım eski [0.60, 0.20, 0.10, 0.10]
# değerine yakın kalacak şekilde seçilmiştir (script sonunda doğrulanır).
# Böylece genel gecikme oranı ve etkileşim yapısı korunur, değişen tek şey
# bunların yıl içine nasıl dağıldığıdır.
#
#                     Normal  Rain  Storm  Snow
MONTHLY_WEATHER_PROBS = {
    1:  [0.30, 0.15, 0.17, 0.38],   # Ocak    - kışın zirvesi, kar baskın
    2:  [0.32, 0.15, 0.17, 0.36],   # Şubat   - hâlâ sert kış
    3:  [0.50, 0.25, 0.12, 0.13],   # Mart    - geçiş, kar azalıyor
    4:  [0.62, 0.28, 0.08, 0.02],   # Nisan   - ilkbahar yağmurları
    5:  [0.70, 0.25, 0.05, 0.00],   # Mayıs   - yağmurlu ama ılıman
    6:  [0.82, 0.14, 0.04, 0.00],   # Haziran - yaz başlıyor
    7:  [0.88, 0.09, 0.03, 0.00],   # Temmuz  - yılın en açık ayı
    8:  [0.87, 0.10, 0.03, 0.00],   # Ağustos - açık, seyrek sağanak
    9:  [0.75, 0.19, 0.06, 0.00],   # Eylül   - sonbahar başlangıcı
    10: [0.62, 0.28, 0.09, 0.01],   # Ekim    - sonbahar yağmurları
    11: [0.45, 0.27, 0.14, 0.14],   # Kasım   - ilk kar ve fırtınalar
    12: [0.33, 0.16, 0.16, 0.35],   # Aralık  - kış bastırıyor
}

# Aylık sevkiyat hacmi (göreli ağırlık). Yıl sonu tatil/kampanya sezonunda
# hacim artar - lojistikte iyi bilinen bir örüntü.
MONTHLY_VOLUME_WEIGHTS = {
    1: 0.85, 2: 0.80, 3: 0.90, 4: 0.95, 5: 1.00, 6: 1.00,
    7: 0.95, 8: 0.90, 9: 1.05, 10: 1.10, 11: 1.25, 12: 1.35,
}

# Veri son 12 ayı kapsar.
DATA_END_DATE = pd.Timestamp("2026-07-31")
DATA_START_DATE = DATA_END_DATE - pd.DateOffset(months=12) + pd.Timedelta(days=1)

TRAFFIC_LEVELS = ["Low", "Medium", "High"]
TRAFFIC_PROBS = [0.35, 0.40, 0.25]

VEHICLE_TYPES = ["Diesel Truck", "Electric Semi", "Hybrid Van"]
VEHICLE_PROBS = [0.55, 0.20, 0.25]  # Filo hala çoğunlukla dizel

# Araç tipine göre gerçekçi CO2 emisyon faktörü (kg CO2 / ton-km)
# Referans aralıklar (yaklaşık, literatürden esinlenilmiştir):
#   Dizel kamyon (ağır yük):  ~0.10 - 0.16 kgCO2/ton-km
#   Elektrikli yarı römork:    ~0.02 - 0.045 kgCO2/ton-km (şebeke karbon yoğunluğuna bağlı)
#   Hibrit van:                ~0.06 - 0.10 kgCO2/ton-km
EMISSION_FACTOR = {
    "Diesel Truck": 0.13,
    "Electric Semi": 0.035,
    "Hybrid Van": 0.08,
}

# Araç tipine göre sabit (idle/başlangıç) emisyon bileşeni (kg) - motor çalıştırma,
# yükleme/boşaltma sırasında oluşan ek emisyon
BASE_EMISSION = {
    "Diesel Truck": 8.0,
    "Electric Semi": 1.5,
    "Hybrid Van": 4.0,
}

# ---------------------------------------------------------------------------
# 2) Temel sütunların üretimi
# ---------------------------------------------------------------------------

shipment_ids = [f"SHP-{str(i).zfill(5)}" for i in range(1, N_ROWS + 1)]
vendor_ids = rng.choice(VENDOR_IDS, size=N_ROWS)

# Vendor_Rating: tedarikçinin temel puanı etrafında küçük gürültü ekleyerek
# gerçekçi çeşitlilik sağlanır (sevkiyat bazında hafif dalgalanma).
vendor_ratings = np.array([
    np.clip(vendor_base_rating[vid] + rng.normal(0, 0.15), 1.0, 5.0)
    for vid in vendor_ids
]).round(1)

# Origin / Destination: aynı şehir olamaz
origins = rng.choice(CITIES, size=N_ROWS)
destinations = []
for o in origins:
    choices = [c for c in CITIES if c != o]
    destinations.append(rng.choice(choices))
destinations = np.array(destinations)

distance_km = rng.uniform(50, 1200, size=N_ROWS).round(1)
weight_tons = rng.uniform(1, 25, size=N_ROWS).round(2)

# ---------------------------------------------------------------------------
# 2b) Shipment_Date - son 12 aya mevsimsel hacimle dağıt
# ---------------------------------------------------------------------------
all_days = pd.date_range(DATA_START_DATE, DATA_END_DATE, freq="D")
day_weights = np.array([MONTHLY_VOLUME_WEIGHTS[d.month] for d in all_days], dtype=float)
day_weights /= day_weights.sum()

shipment_dates = pd.DatetimeIndex(
    rng.choice(all_days, size=N_ROWS, p=day_weights)
).sort_values()          # kronolojik sıra: Shipment_ID zaman sırasını takip etsin

months = shipment_dates.month.to_numpy()

# Hava koşulu AYA BAĞLI çekiliyor (mevsimsellik burada devreye giriyor).
weather = np.empty(N_ROWS, dtype=object)
for m, probs in MONTHLY_WEATHER_PROBS.items():
    mask = months == m
    if mask.any():
        weather[mask] = rng.choice(WEATHER_CONDITIONS, size=int(mask.sum()), p=probs)
weather = weather.astype(str)

# Trafik ve araç tipi mevsimsel DEĞİL - kapsamı bilinçli olarak dar tutuyoruz.
traffic = rng.choice(TRAFFIC_LEVELS, size=N_ROWS, p=TRAFFIC_PROBS)
vehicle = rng.choice(VEHICLE_TYPES, size=N_ROWS, p=VEHICLE_PROBS)

# ---------------------------------------------------------------------------
# 3) Actual_Delay_Days - iki aşamalı (zero-inflated / hurdle) gecikme modeli
# ---------------------------------------------------------------------------
# AŞAMA 1 - Gecikecek mi?   P(gecikme) = sigmoid(logit),  Bernoulli çekilişi
# AŞAMA 2 - Kaç gün?        Yalnızca gecikenler için 1..MAX_DELAY_DAYS arası
#                           kesilmiş (truncated) Poisson
#
# İki aşamayı ayırmak, "çoğu sevkiyat zamanında; gecikenler ise ciddi gecikiyor"
# yapısını doğal olarak üretir. Tek aşamalı toplamsal model bunu üretemez, çünkü
# tüm bileşenler pozitif olduğunda 0'a düşmek imkânsızlaşır.

MAX_DELAY_DAYS = 5
TARGET_DELAY_RATE = 0.22          # hedeflenen genel gecikme oranı

# --- Aşama 1: gecikme olasılığının logit bileşenleri ------------------------
weather_logit = {"Normal": 0.00, "Rain": 0.60, "Storm": 2.00, "Snow": 1.40}
traffic_logit = {"Low": 0.00, "Medium": 0.50, "High": 1.20}

VENDOR_PIVOT = 3.25               # ortalama tedarikçi puanı -> merkezleme noktası
VENDOR_SLOPE = 0.80
DISTANCE_PIVOT = 625.0
DISTANCE_SLOPE = 0.50

# Hava x Tedarikçi ETKİLEŞİMİ.
# Toplamsal bir modelde kötü hava herkesi eşit vurur ve tedarikçi kalitesinin
# önemi tavan etkisiyle silinir. Gerçekte kötü koşullarda operasyonel yetkinlik
# DAHA ÇOK fark eder: iyi tedarikçi fırtınayı yönetir, kötüsü yönetemez.
#
# Tedarikçi terimi VENDOR_PIVOT etrafında MERKEZLENDİĞİ için bu çarpanı
# büyütmek ortalamayı değil YAYILIMI artırır:
#   - iyi tedarikçi (puan > 3.25) -> terim negatif -> kötü havada daha da negatif
#   - kötü tedarikçi (puan < 3.25) -> terim pozitif -> kötü havada daha da pozitif
# Yani "kötü hava herkesi vurur, ama kaliteli tedarikçiyi daha az vurur" ve
# merkezleme sayesinde havanın MARJİNAL etkisi büyük ölçüde korunur.
weather_vendor_amp = {"Normal": 0.00, "Rain": 0.20, "Storm": 0.75, "Snow": 0.50}

vendor_effect = (VENDOR_PIVOT - vendor_ratings) * VENDOR_SLOPE
amp = np.array([1.0 + weather_vendor_amp[w] for w in weather])

logit_wo_intercept = (
    np.array([weather_logit[w] for w in weather])
    + np.array([traffic_logit[t] for t in traffic])
    + vendor_effect * amp
    + ((distance_km - DISTANCE_PIVOT) / 1200.0) * DISTANCE_SLOPE
)


def _solve_intercept(linear, target, lo=-8.0, hi=4.0, iters=90):
    """E[sigmoid(linear + c)] = target olacak sabiti ikili aramayla bul.

    Taban gecikme oranını elle bir sihirli sayı olarak yazmak yerine
    türetiyoruz; böylece diğer katsayılar değiştiğinde oran hedefte kalır.
    """
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if (1.0 / (1.0 + np.exp(-(linear + mid)))).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


INTERCEPT = _solve_intercept(logit_wo_intercept, TARGET_DELAY_RATE)
delay_probability = 1.0 / (1.0 + np.exp(-(logit_wo_intercept + INTERCEPT)))
is_delayed = rng.random(N_ROWS) < delay_probability

# --- Aşama 2: gecikme şiddeti (yalnızca gecikenler için) -------------------
weather_severity = {"Normal": 0.00, "Rain": 0.25, "Storm": 1.10, "Snow": 0.75}
traffic_severity = {"Low": 0.00, "Medium": 0.15, "High": 0.45}
SEVERITY_BASE = 0.30

lam = (
    SEVERITY_BASE
    + np.array([weather_severity[w] for w in weather])
    + np.array([traffic_severity[t] for t in traffic])
    + (VENDOR_PIVOT - vendor_ratings) * 0.18
).clip(0.05, None)


def _truncated_poisson_days(lam_vec, max_days, generator):
    """1..max_days aralığında DOĞRU ŞEKİLDE kesilmiş Poisson çekilişi.

    Basitçe `np.clip(1 + Poisson(lam), 1, max_days)` yazmak kuyruğu max_days'te
    yığar ve dağılımı monoton olmaktan çıkarır (örn. 4 günden çok 5 gün üretir).
    Bunun yerine k = 0..max_days-1 üzerindeki pmf'i normalize edip ters-CDF ile
    örnekliyoruz. scipy gerektirmez.
    """
    k = np.arange(max_days)
    log_pmf = -lam_vec[:, None] + k[None, :] * np.log(lam_vec[:, None]) - _log_factorial(k)[None, :]
    pmf = np.exp(log_pmf)
    pmf /= pmf.sum(axis=1, keepdims=True)
    u = generator.random(len(lam_vec))[:, None]
    return 1 + (u > pmf.cumsum(axis=1)).sum(axis=1)


def _log_factorial(k):
    return np.cumsum(np.concatenate(([0.0], np.log(np.arange(1, k.max() + 1)))))[k]


actual_delay_days = np.where(
    is_delayed, _truncated_poisson_days(lam, MAX_DELAY_DAYS, rng), 0
).astype(int)

# ---------------------------------------------------------------------------
# 4) CO2_Emission_kg - mesafe, tonaj ve araç tipine göre hesaplama
# ---------------------------------------------------------------------------
# Formül:
#   CO2 = (Distance_km * Weight_tons * Emisyon_Faktörü[Araç]) + Baz_Emisyon[Araç]
#         + hava/trafik kaynaklı ek yakıt tüketimi (motor rölantide/duraklamada daha fazla yakar)
#         + küçük rastgele gürültü (gerçekçilik için, +-%5)

base_co2 = distance_km * weight_tons * np.array([EMISSION_FACTOR[v] for v in vehicle])
fixed_co2 = np.array([BASE_EMISSION[v] for v in vehicle])

# Kötü hava ve yoğun trafik -> daha fazla rölanti / verimsiz sürüş -> ek emisyon (%)
weather_co2_mult = {"Normal": 1.00, "Rain": 1.03, "Storm": 1.08, "Snow": 1.06}
traffic_co2_mult = {"Low": 1.00, "Medium": 1.04, "High": 1.10}

weather_mult = np.array([weather_co2_mult[w] for w in weather])
traffic_mult = np.array([traffic_co2_mult[t] for t in traffic])

co2_before_noise = (base_co2 + fixed_co2) * weather_mult * traffic_mult

# %5'lik rastgele gürültü ekleyerek gerçekçi varyasyon sağlanır
co2_noise = rng.normal(1.0, 0.03, size=N_ROWS)
co2_emission_kg = np.clip(co2_before_noise * co2_noise, 0.5, None).round(2)

# ---------------------------------------------------------------------------
# 5) DataFrame oluşturma ve dışa aktarma
# ---------------------------------------------------------------------------

df = pd.DataFrame({
    "Shipment_ID": shipment_ids,
    "Shipment_Date": shipment_dates.strftime("%Y-%m-%d"),
    "Vendor_ID": vendor_ids,
    "Vendor_Rating": vendor_ratings,
    "Origin": origins,
    "Destination": destinations,
    "Distance_km": distance_km,
    "Weight_tons": weight_tons,
    "Weather_Condition": weather,
    "Traffic_Density": traffic,
    "Vehicle_Type": vehicle,
    "Actual_Delay_Days": actual_delay_days,
    "CO2_Emission_kg": co2_emission_kg,
})

output_path = RAW_DIR / "smart_logistics_data.csv"
df.to_csv(output_path, index=False)

# ---------------------------------------------------------------------------
# 6) Hızlı doğrulama / özet istatistikler (konsola yazdırılır)
# ---------------------------------------------------------------------------
print(f"Veri seti oluşturuldu: {output_path}  ({len(df)} satır, {len(df.columns)} sütun)\n")

print("--- Sütun özet istatistikleri ---")
print(df[["Vendor_Rating", "Distance_km", "Weight_tons",
          "Actual_Delay_Days", "CO2_Emission_kg"]].describe().round(2))

delayed_mask = df["Actual_Delay_Days"] > 0
print(f"\n--- Zero-inflated yapı doğrulaması (INTERCEPT = {INTERCEPT:.4f}) ---")
print(f"Gecikme oranı           : %{delayed_mask.mean() * 100:.1f}   (hedef: %{TARGET_DELAY_RATE * 100:.0f})")
print(f"Ortalama gecikme (tümü) : {df['Actual_Delay_Days'].mean():.2f} gün")
print(f"Ortalama (gecikenler)   : {df.loc[delayed_mask, 'Actual_Delay_Days'].mean():.2f} gün")
print("Gün dağılımı:")
print(df["Actual_Delay_Days"].value_counts().sort_index().to_string())

print("\n--- Gecikme mantığı doğrulaması: hava durumuna göre ---")
print(df.groupby("Weather_Condition")["Actual_Delay_Days"]
      .agg(gecikme_orani=lambda s: round((s > 0).mean() * 100, 1),
           ortalama_gun=lambda s: round(s.mean(), 2))
      .sort_values("gecikme_orani", ascending=False))

print("\n--- Gecikme mantığı doğrulaması: trafik yoğunluğuna göre ---")
print(df.groupby("Traffic_Density")["Actual_Delay_Days"]
      .agg(gecikme_orani=lambda s: round((s > 0).mean() * 100, 1),
           ortalama_gun=lambda s: round(s.mean(), 2))
      .sort_values("gecikme_orani", ascending=False))

# Etkileşim terimi çalışıyor mu? Kötü havada tedarikçi kalitesi ayrımı
# NORMAL havadakine yakın veya daha belirgin olmalı; toplamsal modelde tavan
# etkisi yüzünden çok daha zayıf çıkıyordu.
print("\n--- Hava x Tedarikçi etkileşimi doğrulaması (gecikme oranı %) ---")
quartile = pd.qcut(df["Vendor_Rating"], 4, labels=["Q1 (düşük)", "Q2", "Q3", "Q4 (yüksek)"])
pivot = (df.assign(_q=quartile, _d=delayed_mask)
         .pivot_table(index="Weather_Condition", columns="_q", values="_d",
                      aggfunc="mean", observed=True) * 100).round(1)
pivot = pivot.reindex(["Storm", "Snow", "Rain", "Normal"])
print(pivot.to_string())
print("NOT: Storm/Snow hücrelerinde satır sayısı düşüktür (~25-55); hücre bazlı "
      "yüzdeler gürültülüdür, yalnızca genel eğilim yorumlanmalıdır.")

# ---------------------------------------------------------------------------
# 6b) Mevsimsellik doğrulaması
# ---------------------------------------------------------------------------
_dates = pd.to_datetime(df["Shipment_Date"])
print(f"\n--- Tarih aralığı: {_dates.min():%Y-%m-%d} .. {_dates.max():%Y-%m-%d} "
      f"({_dates.nunique()} farklı gün) ---")

# KRİTİK KONTROL: aylık dağılımlar farklılaştı ama YILLIK MARJİNAL dağılım
# eski sabit değere yakın kalmalı. Kalmazsa mevsimsellik, genel gecikme
# oranını da değiştirmiş olur ve "sadece zaman düzeni ekledik" iddiası bozulur.
marginal = df["Weather_Condition"].value_counts(normalize=True).reindex(
    WEATHER_CONDITIONS).round(3)
print("\n--- Yıllık marjinal hava dağılımı (eski sabit değerle kıyas) ---")
print(pd.DataFrame({
    "Yeni (mevsimsel)": marginal,
    "Eski (sabit)": pd.Series([0.60, 0.20, 0.10, 0.10], index=WEATHER_CONDITIONS),
}).to_string())

_seasonal = (
    df.assign(Ay=_dates.dt.month)
      .pivot_table(index="Ay", columns="Weather_Condition",
                   values="Shipment_ID", aggfunc="count", observed=True)
      .reindex(columns=WEATHER_CONDITIONS).fillna(0)
)
_seasonal_pct = (_seasonal.div(_seasonal.sum(axis=1), axis=0) * 100).round(1)
_seasonal_pct["Sevkiyat"] = _seasonal.sum(axis=1).astype(int)
_seasonal_pct["Gecikme_%"] = (
    df.assign(Ay=_dates.dt.month).groupby("Ay")["Actual_Delay_Days"]
      .apply(lambda s: round((s > 0).mean() * 100, 1))
)
print("\n--- Aylık hava dağılımı (%) ve gerçekleşen gecikme oranı ---")
print(_seasonal_pct.to_string())
print("Kış aylarında Snow/Storm payı ve gecikme oranı birlikte yükselmeli;")
print("yaz aylarında ikisi de düşmeli. Bu, mevsimsel sinyalin kanıtıdır.")

print("\n--- CO2 doğrulaması: Ortalama emisyon, araç tipine göre (kg) ---")
print(df.groupby("Vehicle_Type")["CO2_Emission_kg"].mean().round(2)
      .sort_values(ascending=False))

print("\n--- İlk 5 satır ---")
print(df.head())
