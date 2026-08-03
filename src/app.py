"""
Streamlit demo: Sevkiyat Gecikme Riski & Karbon Maliyeti
=========================================================
Çalıştırma:   streamlit run src/app.py

Bu uygulama HİÇBİR ML mantığını yeniden kurmaz. `models/production_risk_model.pkl`
içindeki tam pipeline'ı (ColumnTransformer + OneHotEncoder + CalibratedClassifierCV)
olduğu gibi yükler ve ham bir DataFrame verir. Encoding'i burada elle yapmak
train/serve skew'in klasik kaynağıdır; sütun sırası veya kategori işleme ufak bir
şekilde farklılaşır ve model sessizce yanlış tahmin üretir.

Eşikler, feature sırası ve metrikler de HARDCODE EDİLMEZ - aynı dosyadaki
metadata'dan okunur. Böylece model yeniden eğitilip eşikler değişse bile uygulama
otomatik uyumlu kalır.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# CO2 hesabı - generate_logistics_data.py'den BİREBİR kopyalanmış sabitler
# ---------------------------------------------------------------------------
# ÖNEMLİ: CO2 bir MODEL TAHMİNİ DEĞİLDİR. Mesafe, tonaj ve araç tipinden
# deterministik olarak hesaplanan mühendislik değeridir. Jeneratördeki formülün
# aynısı kullanılır; tek fark, oradaki %3'lük rastgele gürültü burada YOKTUR
# (aynı girdi her zaman aynı sonucu vermelidir).
EMISSION_FACTOR = {"Diesel Truck": 0.13, "Electric Semi": 0.035, "Hybrid Van": 0.08}
BASE_EMISSION = {"Diesel Truck": 8.0, "Electric Semi": 1.5, "Hybrid Van": 4.0}
WEATHER_CO2_MULT = {"Normal": 1.00, "Rain": 1.03, "Storm": 1.08, "Snow": 1.06}
TRAFFIC_CO2_MULT = {"Low": 1.00, "Medium": 1.04, "High": 1.10}

# Power BI'daki `Carbon Tax Impact ($)` ölçüsüyle aynı varsayım.
CARBON_PRICE_PER_TON = 50.0

# Örneklem azlığı nedeniyle model bu ikisini güvenilir ayıramıyor (bkz. README).
LOW_SAMPLE_WEATHER = {"Storm", "Snow"}


def compute_co2_kg(distance_km: float, weight_tons: float, vehicle: str,
                   weather: str, traffic: str) -> float:
    """CO2 = (mesafe x tonaj x emisyon_faktörü + sabit taban) x hava x trafik."""
    base = distance_km * weight_tons * EMISSION_FACTOR[vehicle] + BASE_EMISSION[vehicle]
    return max(base * WEATHER_CO2_MULT[weather] * TRAFFIC_CO2_MULT[traffic], 0.5)


def classify_risk(p: float, thresholds: dict) -> str:
    if p >= thresholds["high_risk"]:
        return "High Risk"
    if p >= thresholds["medium_risk"]:
        return "Medium Risk"
    return "Low Risk"


def find_model_path() -> Path:
    """Modeli hem düz hem src/ yerleşiminde bul (Streamlit Cloud'da cwd repo kökü)."""
    here = Path(__file__).resolve()
    for base in (here.parent.parent, here.parent, Path.cwd()):
        candidate = base / "models" / "production_risk_model.pkl"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "models/production_risk_model.pkl bulunamadı. "
        "Önce `python ml_delay_risk_pipeline.py` çalıştırın."
    )


@st.cache_resource
def load_bundle():
    bundle = joblib.load(find_model_path())
    return bundle["model"], bundle["metadata"]


# ---------------------------------------------------------------------------
st.set_page_config(page_title="Sevkiyat Gecikme Riski", page_icon="🚚",
                   layout="wide")

try:
    model, meta = load_bundle()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

thresholds = meta["thresholds"]
levels = meta["categorical_levels"]

st.title("🚚 Sevkiyat Gecikme Riski & Karbon Maliyeti")
st.caption(
    f"Kalibre Random Forest · OOF ROC-AUC {meta['metrics']['oof_roc_auc']:.3f} · "
    f"kronolojik holdout {meta['metrics']['chronological_holdout_roc_auc']:.3f} · "
    f"model eğitim tarihi {meta['trained_at'][:10]}"
)

# ---------------------------------------------------------------------------
# Sidebar - girdiler
# ---------------------------------------------------------------------------
st.sidebar.header("Sevkiyat Bilgileri")

vendor_rating = st.sidebar.slider("Tedarikçi Puanı (Vendor Rating)", 2.5, 5.0, 4.0, 0.1)
traffic = st.sidebar.selectbox("Trafik Yoğunluğu (Traffic Density)",
                               ["Low", "Medium", "High"], index=1)
weather = st.sidebar.selectbox("Hava Durumu (Weather Condition)",
                               ["Normal", "Rain", "Snow", "Storm"], index=0)
vehicle = st.sidebar.selectbox("Araç Tipi (Vehicle Type)",
                               ["Diesel Truck", "Electric Semi", "Hybrid Van"], index=0)
distance_km = st.sidebar.slider("Mesafe (km)", 50, 1200, 450, 10)
weight_tons = st.sidebar.slider("Ağırlık (ton)", 1.0, 24.0, 12.0, 0.5)

# Metadata ile arayüz seçenekleri tutarlı mı? Model yeniden eğitilip kategoriler
# değişirse sessizce yanlış tahmin üretmek yerine burada uyaralım.
for col, chosen in [("Weather_Condition", weather), ("Traffic_Density", traffic),
                    ("Vehicle_Type", vehicle)]:
    if chosen not in levels.get(col, [chosen]):
        st.sidebar.warning(
            f"'{chosen}' modelin eğitim verisinde yok ({col}). "
            "Tahmin bu kategoriyi yok sayarak üretilir."
        )

st.sidebar.divider()
st.sidebar.caption(
    "Bu bir **demo**dur: model sentetik veriyle eğitilmiştir ve gerçek dünya "
    "performansını temsil etmez. Ayrıntı için ana ekrandaki "
    "*Bu demo hakkında* bölümüne bakın."
)

# ---------------------------------------------------------------------------
# Ana ekran
# ---------------------------------------------------------------------------
if st.button("🚀 Riski ve Maliyeti Hesapla", type="primary"):
    # Sütun sırası metadata'dan geliyor - hardcode değil.
    features = pd.DataFrame([{
        "Vendor_Rating": vendor_rating,
        "Traffic_Density": traffic,
        "Weather_Condition": weather,
        "Vehicle_Type": vehicle,
        "Distance_km": float(distance_km),
        "Weight_tons": float(weight_tons),
    }])[meta["feature_order"]]

    probability = float(model.predict_proba(features)[0, 1])
    risk_level = classify_risk(probability, thresholds)

    co2_kg = compute_co2_kg(float(distance_km), float(weight_tons),
                            vehicle, weather, traffic)
    co2_tons = co2_kg / 1000.0
    carbon_tax = co2_tons * CARBON_PRICE_PER_TON

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gecikme Olasılığı", f"%{probability * 100:.1f}",
              help="Modelin kalibre edilmiş tahmini (ML çıktısı).")
    c2.metric("Risk Seviyesi", risk_level,
              help=f"Eşikler: High ≥ {thresholds['high_risk']}, "
                   f"Medium ≥ {thresholds['medium_risk']}")
    c3.metric("CO₂ (ton) · hesaplanan", f"{co2_tons:.3f}",
              help="Model tahmini DEĞİL - mesafe, tonaj ve araç tipinden "
                   "deterministik formülle hesaplanır.")
    c4.metric("Karbon Vergisi ($) · hesaplanan", f"${carbon_tax:,.2f}",
              help=f"CO₂ tonajı × ${CARBON_PRICE_PER_TON:.0f}/ton varsayımı.")

    if risk_level == "High Risk":
        st.error(
            f"**Yüksek Risk — %{probability * 100:.1f}**\n\n"
            "Yanlış alarm ile kaçırılan gecikme *aynı* maliyette olsa bile "
            "müdahale etmeye değer. Tedarikçiyle temasa geçin, tampon kapasite "
            "ayırın veya rota/araç değişimi değerlendirin."
        )
    elif risk_level == "Medium Risk":
        st.warning(
            f"**Orta Risk — %{probability * 100:.1f}**\n\n"
            f"Kaçırılan gecikmenin yanlış alarmdan "
            f"{thresholds['cost_fn_over_fp']:.0f}× pahalı olduğu varsayımı altında "
            "müdahale etmeye değer. İzlemeye alın."
        )
    else:
        st.success(
            f"**Düşük Risk — %{probability * 100:.1f}**\n\n"
            f"{thresholds['cost_fn_over_fp']:.0f}:1 maliyet varsayımı altında bile "
            "müdahale etmeye değmez. Standart akışta ilerleyebilir."
        )

    # Kullanıcı Storm <-> Snow çevirip riskin ters yönde hareket ettiğini
    # görebilir. Bunu gizlemek yerine tam burada açıklıyoruz.
    if weather in LOW_SAMPLE_WEATHER:
        st.info(
            f"ℹ️ **{weather} için örneklem sınırlı.** Eğitim verisinde Storm 128, "
            "Snow 193 sevkiyat içeriyor. Bu iki koşul arasındaki *sıralama* "
            "güvenilir değildir — Snow'dan Storm'a geçtiğinizde riskin bir miktar "
            "**düştüğünü** görebilirsiniz. Bu bir veri kısıtıdır, model hatası "
            "değil: model, veride gözlemlenen oranları öğrenir ve bu hücrelerde "
            "yeterli gözlem yoktur. Her iki koşulun *Normal/Rain'e göre* daha "
            "riskli olduğu ise güvenilirdir."
        )

    st.divider()
    st.markdown("**Girdi özeti**")
    st.dataframe(
        features.T.rename(columns={0: "Değer"}),
        use_container_width=False,
    )

else:
    st.info(
        "👈 Soldaki panelden sevkiyat bilgilerini girin, ardından "
        "**🚀 Riski ve Maliyeti Hesapla** butonuna basın."
    )

# ---------------------------------------------------------------------------
# Şeffaflık
# ---------------------------------------------------------------------------
with st.expander("ℹ️ Bu demo hakkında — sınırlar ve varsayımlar", expanded=False):
    st.markdown(
        f"""
**Veri sentetiktir.** Model, kural tabanlı bir simülasyonla üretilmiş
{meta['training_data']['n_rows']} sevkiyatla eğitilmiştir
({meta['training_data']['date_min']} – {meta['training_data']['date_max']}).
Buradaki hiçbir sayı gerçek dünya performansını temsil etmez; gösterilen şey
**metodolojidir**, ticari bir sonuç değildir.

**Olasılıklar kalibre edilmiştir — ama uçlarda zayıftır.** Ham Random Forest
çıktısı ağaç oylarının oranıdır, olasılık değildir. `CalibratedClassifierCV`
(isotonic) ile düzeltildi: kalibrasyon hatası (ECE) 0.137 → {meta['metrics']['oof_ece']:.3f}.
Yani "%30" gerçekten *yaklaşık %30* anlamına gelir. Ancak p > 0.8 bandında gözlem
azlığı nedeniyle güvenilirlik düşer — çok yüksek olasılıklara orta banttaki kadar
güvenmeyin.

**Eşikler bir maliyet varsayımından türetilmiştir**, kantilden değil.
Kaçırılan bir gecikme (SLA ihlali, ceza, ekspres taşıma) yanlış alarmdan
(bir planlamacı telefonu) **{thresholds['cost_fn_over_fp']:.0f}× pahalı** kabul
edilmiştir. Kalibre olasılıkla optimum müdahale eşiği
`p* = 1 / (1 + {thresholds['cost_fn_over_fp']:.0f}) = {thresholds['medium_risk']}` olur.
`High Risk` eşiği ({thresholds['high_risk']}) ise 1:1 oranın karşılığıdır: maliyetler
eşit olsa bile müdahale edilecek sevkiyatlar. **Bu oran gerçek SLA ceza tarifeleriyle
doğrulanmamıştır.**

**Storm ve Snow güvenilir ayrılamıyor.** Eğitim verisinde Storm yalnızca 128
sevkiyat içerir (Snow 193). 216 kombinasyonluk bir taramada gerçek üretim süreci
Storm'u Snow'dan riskli yapıyor (%100), model ise bunu ancak %61 oranında
yakalıyor. Bu bir **veri kısıtıdır, model hatası değil** — model veride gözlemlenen
oranları öğrenir ve bu hücrelerde yeterli gözlem yoktur.

**CO₂ ve karbon vergisi model çıktısı DEĞİLDİR.** Mesafe × tonaj × araç emisyon
faktörü + sabit taban, üzerine hava/trafik çarpanı — `generate_logistics_data.py`
içindeki deterministik formülün aynısı. Karbon vergisi ${CARBON_PRICE_PER_TON:.0f}/ton
varsayımıyla çarpımdır. Burada bir tahmin yoktur, birim dönüşümü ve fiyatlandırma vardır.

**Bu demo, Power BI panelinden biraz daha "keskin" skorlar üretir.** Panelin
beslendiği CSV'deki skorlar *zaman-ileri* üretilir: her sevkiyat, yalnızca kendisinden
önce gerçekleşmiş sevkiyatlarla eğitilmiş bir modelden skor alır. Buradaki model ise
tüm veriyle eğitilmiştir (üretime dağıtılacak model için doğrusu budur). Sonuç olarak
aynı sevkiyat panelde biraz daha düşük, burada biraz daha yüksek olasılık alabilir —
`High Risk` payı panelde %10.3, bu modelde %17.7. İkisi de doğrudur, farklı soruları
yanıtlarlar: panel *geçmişi dürüstçe raporlar*, demo *yeni bir sevkiyatı skorlar*.

**Model performansı (hepsi örneklem-dışı):**

| Metrik | Değer |
|---|---|
| ROC-AUC (out-of-fold) | {meta['metrics']['oof_roc_auc']:.3f} |
| ROC-AUC (kronolojik holdout) | {meta['metrics']['chronological_holdout_roc_auc']:.3f} |
| ROC-AUC (kronolojik CV) | {meta['metrics']['chronological_cv_roc_auc_mean']:.3f} ± {meta['metrics']['chronological_cv_roc_auc_std']:.3f} |
| Brier score | {meta['metrics']['oof_brier']:.4f} |
| Kalibrasyon hatası (ECE) | {meta['metrics']['oof_ece']:.4f} |

Kronolojik skor daha düşüktür çünkü model *gelecek* bir dönemde test edilir ve o
dönem mevsimsel olarak farklıdır. Bu bir gerileme değil, tek dürüst ölçümdür.
Kronolojik katlar arasındaki ±{meta['metrics']['chronological_cv_roc_auc_std']:.2f}
oynaklık da mevsimden gelir: kışın kötü hava ayrım gücü yaratır, yazın yaratmaz.
"""
    )
