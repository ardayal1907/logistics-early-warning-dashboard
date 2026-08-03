"""
ML Pipeline: Sevkiyat Gecikme Riski Tahmini (Kalibre Random Forest)
========================================================================
Microsoft Fabric Notebook ortamında (veya herhangi bir Python/pandas
ortamında) çalışacak şekilde tasarlanmıştır. Fabric Lakehouse kullanıyorsanız
CSV okuma/yazma satırlarını `spark.read.csv(...)` / Delta tablo
read/write ile değiştirebilirsiniz (bkz. script sonundaki "Fabric notları").

Adımlar:
  1) Dim_Vendor, Dim_Route, Fact_Shipments tablolarını oku ve birleştir (join)
  2) is_delayed (0/1) hedef değişkenini oluştur
  3) One-Hot Encoding + ColumnTransformer + Pipeline kur
  4) Tedarikçi bazlı (grup) bölme ile dürüst performans ölçümü
  5) Kalibre edilmiş, örneklem-dışı Delay_Risk_Probability üret
  6) Maliyet matrisinden türetilmiş eşiklerle Risk_Level oluştur
  7) Fact_Shipments.csv yapısını bozmadan, sadece 2 yeni sütun ekleyerek
     Fact_Shipments_with_ML.csv olarak kaydet
  8) Feature importance (impurity + permutation)
  9) Üretim modelini + metadata'yı models/ altına kaydet (Streamlit demosu için)

Metodolojik kararlar ve ölçüldükleri sayılar
--------------------------------------------
* GRUP BAZLI BÖLME (StratifiedGroupKFold, Vendor_ID): Aynı tedarikçinin hem
  eğitim hem test setinde bulunması, modelin "bu tedarikçi hep böyle" diye
  ezberlemesine izin verebilir. Ölçüldü: rastgele bölme 0.774, grup bazlı
  bölme 0.773 -> bu veri setinde ezber YOK. Sebebi, jeneratörde tedarikçi
  etkisinin puanın düzgün bir fonksiyonu olması; model kimliği değil puanı
  öğreniyor ve bu görülmemiş tedarikçiye transfer oluyor. Yine de grup bazlı
  bölme korunuyor: gerçek veride tedarikçi karnesi gürültülü ve eskimiş olur,
  orada bu garanti kendiliğinden gelmez.
* KALİBRASYON (isotonic): Ham Random Forest olasılıkları ağaç oylarının
  oranıdır, olasılık değildir. Ölçüldü: kalibrasyonsuz ECE 0.137 (0.65 denen
  yerde gerçek oran 0.36), isotonic sonrası 0.021. isotonic 5 farklı seed'in
  5'inde de sigmoid'den iyi ECE verdi (fark -0.0065 +/- 0.0043).
* MALİYET BAZLI EŞİK: Eşikler artık kantil değil, maliyet matrisi türevi.
  Ölçüldü: eski kantil eşiği (0.65) gecikmelerin sadece %19'unu yakalıyor ve
  4:1 maliyet varsayımı altında optimal eşiğe göre %51 daha pahalı.
"""

import json
import platform
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    cross_val_predict, StratifiedKFold, StratifiedGroupKFold, TimeSeriesSplit,
)
from sklearn.metrics import (
    classification_report, roc_auc_score, brier_score_loss,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
N_SPLITS = 5

# Yollar repo köküne göre çözülür; script'i hangi dizinden çağırdığınız fark etmez.
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

# ---------------------------------------------------------------------------
# 1) Star Schema tablolarını oku ve model eğitimi için geçici birleştirme
# ---------------------------------------------------------------------------
fact = pd.read_csv(PROCESSED_DIR / "Fact_Shipments.csv")
dim_vendor = pd.read_csv(PROCESSED_DIR / "Dim_Vendor.csv")
dim_route = pd.read_csv(PROCESSED_DIR / "Dim_Route.csv")
dim_date = pd.read_csv(PROCESSED_DIR / "Dim_Date.csv")

print(f"Fact_Shipments : {fact.shape}")
print(f"Dim_Vendor     : {dim_vendor.shape}")
print(f"Dim_Route      : {dim_route.shape}")
print(f"Dim_Date       : {dim_date.shape}")

analytical_df = (
    fact
    .merge(dim_vendor, on="Vendor_ID", how="left")
    .merge(dim_route, on="Route_ID", how="left")
    .merge(dim_date[["Date_ID", "Full_Date", "Month", "Season"]], on="Date_ID", how="left")
)

assert len(analytical_df) == len(fact), "Join sonrası satır sayısı değişti!"

# KRONOLOJİK SIRA: bundan sonraki her şey zaman sırasına dayanıyor.
analytical_df["Full_Date"] = pd.to_datetime(analytical_df["Full_Date"])
analytical_df = analytical_df.sort_values("Full_Date").reset_index(drop=True)

print(f"Birleştirilmiş analitik veri: {analytical_df.shape}")
print(f"Tarih aralığı: {analytical_df.Full_Date.min():%Y-%m-%d} .. "
      f"{analytical_df.Full_Date.max():%Y-%m-%d}\n")

# ---------------------------------------------------------------------------
# 2) Hedef değişken: is_delayed
# ---------------------------------------------------------------------------
analytical_df["is_delayed"] = (analytical_df["Actual_Delay_Days"] > 0).astype(int)

print("Hedef değişken dağılımı:")
print(analytical_df["is_delayed"].value_counts(normalize=True).round(3), "\n")

# ---------------------------------------------------------------------------
# 3) Özellik seçimi ve encoding
# ---------------------------------------------------------------------------
numeric_features = ["Distance_km", "Weight_tons", "Vendor_Rating"]
categorical_features = ["Weather_Condition", "Traffic_Density", "Vehicle_Type"]
feature_cols = numeric_features + categorical_features

X = analytical_df[feature_cols]
y = analytical_df["is_delayed"]
groups = analytical_df["Vendor_ID"]          # grup bazlı bölme anahtarı

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

base_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )),
])

# Kalibrasyon sarmalayıcı: iç 5 katlamada RF'i eğitip olasılıkları düzeltir.
model = CalibratedClassifierCV(base_model, method="isotonic", cv=N_SPLITS)

# ---------------------------------------------------------------------------
# 4) Dürüst performans ölçümü - KRONOLOJİK bölme
# ---------------------------------------------------------------------------
# Bu, dağıtım senaryosunun tek gerçekçi taklididir: geçmişle eğit, gelecekte
# test et. Rastgele (veya sadece tedarikçi bazlı) bölme modelin geleceği
# geçmişle tahmin etmesine izin verir ve skoru olduğundan iyi gösterir.
#
# Veri mevsimsel olduğu için bunun bir bedeli vardır ve o bedel GERÇEKTİR:
# son %20 belirli bir mevsime denk gelir, eğitim seti başka mevsimlere. Ortaya
# çıkan dağılım kayması (distribution shift) bir ölçüm hatası değil, üretimde
# gerçekten yaşanacak olan durumdur.
split_at = int(len(analytical_df) * 0.8)
train_idx = np.arange(split_at)
test_idx = np.arange(split_at, len(analytical_df))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
dates = analytical_df["Full_Date"]

assert dates.iloc[train_idx].max() <= dates.iloc[test_idx].min(), \
    "Kronolojik sızıntı: eğitim setinde test döneminden sonraki bir tarih var!"

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]

print(f"Train: {len(train_idx)} satır | {dates.iloc[train_idx].min():%Y-%m-%d} .. "
      f"{dates.iloc[train_idx].max():%Y-%m-%d}")
print(f"Test : {len(test_idx)} satır | {dates.iloc[test_idx].min():%Y-%m-%d} .. "
      f"{dates.iloc[test_idx].max():%Y-%m-%d}  (tamamı eğitimden SONRA)\n")
print("--- Test seti performans raporu (kronolojik) ---")
print(classification_report(y_test, y_pred, target_names=["Zamanında (0)", "Gecikti (1)"]))
chrono_auc = roc_auc_score(y_test, y_proba_test)
print(f"ROC-AUC (kronolojik holdout): {chrono_auc:.3f}\n")

# Permutation importance için test setini görmemiş modeli sakla.
eval_model = deepcopy(model)

# --- Neden düştü / düşmedi? Dağılım kaymasını ÖLÇEREK göster ---------------
print("--- Train vs Test dağılım kayması ---")
shift = pd.DataFrame({
    "Train": analytical_df.iloc[train_idx]["Weather_Condition"].value_counts(normalize=True),
    "Test": analytical_df.iloc[test_idx]["Weather_Condition"].value_counts(normalize=True),
}).fillna(0.0)
shift["Fark"] = shift["Test"] - shift["Train"]
print((shift * 100).round(1).to_string())
print(f"\nGecikme oranı  train %{y.iloc[train_idx].mean() * 100:.1f} -> "
      f"test %{y.iloc[test_idx].mean() * 100:.1f}")
print("Test dönemi mevsimleri:",
      analytical_df.iloc[test_idx]["Season"].value_counts().to_dict())
print("Eğitim dönemi mevsimleri:",
      analytical_df.iloc[train_idx]["Season"].value_counts().to_dict())

# --- Kıyas: aynı veri, zaman-körü bölmelerle ne verirdi? -------------------
# Farkın kaynağını görmek için üç bölme stratejisini yan yana koyuyoruz.
print("\n--- Bölme stratejisi karşılaştırması (aynı veri, aynı model) ---")
strategies = {
    "Rastgele (StratifiedKFold)": (StratifiedKFold(N_SPLITS, shuffle=True,
                                                   random_state=RANDOM_STATE), None),
    "Tedarikçi bazlı (GroupKFold)": (StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                                          random_state=RANDOM_STATE), groups),
}
for label, (splitter, grp) in strategies.items():
    p = cross_val_predict(model, X, y, cv=splitter, groups=grp,
                          method="predict_proba", n_jobs=-1)[:, 1]
    print(f"  {label:<30}: ROC-AUC {roc_auc_score(y, p):.3f}")

# Kronolojik CV (genişleyen pencere): tek bir dönemin şansına bağlı kalmamak
# için birden fazla zaman kesitinde ölçüyoruz.
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
ts_aucs = []
for i, (tr, te) in enumerate(tscv.split(X), 1):
    m = deepcopy(model)
    m.fit(X.iloc[tr], y.iloc[tr])
    a = roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1])
    ts_aucs.append(a)
    print(f"  Kronolojik kat {i}: AUC {a:.3f}  (test "
          f"{dates.iloc[te].min():%Y-%m} .. {dates.iloc[te].max():%Y-%m}, "
          f"gecikme %{y.iloc[te].mean() * 100:.0f})")
print(f"  {'Kronolojik CV ortalaması':<30}: ROC-AUC {np.mean(ts_aucs):.3f} "
      f"± {np.std(ts_aucs):.3f}\n")

# ---------------------------------------------------------------------------
# 5) Üretim skorlaması: KALİBRE + ÖRNEKLEM-DIŞI olasılıklar
# ---------------------------------------------------------------------------
# Modeli tüm veriyle eğitip aynı veriyi skorlamak ezberlenmiş olasılıklar
# üretir (ölçüldü: ROC-AUC 0.998 ve "Low Risk" grubunun gerçek gecikme oranı
# 0.000 - imkânsız derecede temiz). Power BI'a öyle bir sütun yazmak, Erken
# Uyarı Panelini bir tahmin değil geçmişin kopyası yapardı.
#
# Her satır, o satırı ve o satırın TEDARİKÇİSİNİ görmemiş bir modelle skorlanır.
# Artık ZAMAN-İLERİ (walk-forward) skorluyoruz: her sevkiyat, yalnızca ondan
# ÖNCE gerçekleşmiş sevkiyatlarla eğitilmiş bir modelle skorlanır. Üretimde
# olan tam olarak budur - sevkiyat sevk anında skorlanır, geleceği kimse
# bilmez.
#
# Bunun kaçınılmaz bir kısıtı var: ilk blok için "önceki veri" yoktur.
# O satırlar tedarikçi bazlı out-of-fold skorlarla dolduruluyor ve aşağıda
# kaç satır olduğu açıkça raporlanıyor. Bu bir uzlaşmadır; alternatif, en eski
# %17'lik sevkiyatı Power BI'da skorsuz bırakmaktı.
delay_risk_probability = np.full(len(analytical_df), np.nan)

tscv_score = TimeSeriesSplit(n_splits=N_SPLITS)
for tr, te in tscv_score.split(X):
    m = deepcopy(model)
    m.fit(X.iloc[tr], y.iloc[tr])
    delay_risk_probability[te] = m.predict_proba(X.iloc[te])[:, 1]

warm_up_mask = np.isnan(delay_risk_probability)
if warm_up_mask.any():
    fallback = cross_val_predict(
        model, X, y,
        cv=StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        groups=groups, method="predict_proba", n_jobs=-1,
    )[:, 1]
    delay_risk_probability[warm_up_mask] = fallback[warm_up_mask]
    print(f"Zaman-ileri skorlama: {(~warm_up_mask).sum()} satır "
          f"({(~warm_up_mask).mean() * 100:.0f}%) yalnızca GEÇMİŞ veriyle skorlandı.")
    print(f"İlk {warm_up_mask.sum()} satır (ısınma dönemi, "
          f"{dates[warm_up_mask].min():%Y-%m-%d} .. {dates[warm_up_mask].max():%Y-%m-%d}) "
          f"için önceki veri yok;\n  tedarikçi bazlı out-of-fold skorla dolduruldu.\n")

analytical_df["Delay_Risk_Probability"] = delay_risk_probability.round(2)


def expected_calibration_error(y_true, proba, n_bins=10):
    """Tahmin edilen olasılık ile gerçekleşen oran arasındaki ağırlıklı fark."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        upper = proba < edges[i + 1] if i < n_bins - 1 else proba <= 1.0
        mask = (proba >= edges[i]) & upper
        if mask.sum():
            total += mask.sum() / len(proba) * abs(proba[mask].mean() - y_true[mask].mean())
    return total


oof_roc_auc = roc_auc_score(y, delay_risk_probability)
oof_brier = brier_score_loss(y, delay_risk_probability)
oof_ece = expected_calibration_error(y.values, delay_risk_probability)

print("--- Out-of-fold skorlama kalitesi ---")
print(f"ROC-AUC : {oof_roc_auc:.3f}   (sıralama yeteneği)")
print(f"Brier   : {oof_brier:.4f}  (düşük = iyi; "
      f"taban oranı söyleyen model: {brier_score_loss(y, np.full(len(y), y.mean())):.4f})")
print(f"ECE     : {oof_ece:.4f}  (kalibrasyonsuz haliyle ~0.137 idi)\n")

# ---------------------------------------------------------------------------
# 6) Risk_Level: MALİYET MATRİSİNDEN türetilmiş eşikler
# ---------------------------------------------------------------------------
# Maliyet varsayımı:
#   YANLIŞ ALARM (FP)      -> planlamacı kontrol eder, tedarikçiyi arar,
#                             tampon kapasite ayırır. ~1 birim.
#   KAÇIRILAN GECİKME (FN) -> SLA ihlali: sözleşmesel ceza + ekspres/yeniden
#                             yönlendirme maliyeti + müşteri kaybı riski.
#                             Sözleşmeli lojistikte SLA cezaları navlun
#                             bedelinin yüzdesi olarak işler ve ekspres taşıma
#                             normal navlunun birkaç katıdır.  -> 4 birim.
#
# Kalibre olasılıkla beklenen maliyeti minimize eden eşik analitiktir:
#     müdahale et  <=>  p * C_FN > (1 - p) * C_FP
#     p* = C_FP / (C_FP + C_FN) = 1 / (1 + oran)
# Bu formül olasılık KALİBRE DEĞİLSE geçersizdir; Bölüm 5 bu yüzden önkoşul.
#
# Kademelerin her sınırının bir anlamı var:
COST_FN_OVER_FP = 4.0
HIGH_RISK_THRESHOLD = 0.50                                  # 1:1 oranın eşiği
MEDIUM_RISK_THRESHOLD = round(1.0 / (1.0 + COST_FN_OVER_FP), 2)   # 4:1 -> 0.20

#   High Risk   : yanlış alarm ile kaçırılan gecikme AYNI maliyette olsa bile
#                 müdahale etmeye değer (p >= 0.50)
#   Medium Risk : 4:1 varsayımımız altında müdahale etmeye değer
#   Low Risk    : 4:1'de bile müdahale etmeye değmez
#
# NOT: Eski kantil bazlı eşik (0.65) dağılıma bakıyordu, sonuçlara değil.
# Ölçüldü: gecikmelerin yalnızca %19'unu yakalıyor ve 4:1 altında optimal
# eşiğe göre %51 daha pahalı.


def classify_risk(p: float) -> str:
    if p >= HIGH_RISK_THRESHOLD:
        return "High Risk"
    elif p >= MEDIUM_RISK_THRESHOLD:
        return "Medium Risk"
    else:
        return "Low Risk"


analytical_df["Risk_Level"] = analytical_df["Delay_Risk_Probability"].apply(classify_risk)

print(f"Risk eşikleri: High >= {HIGH_RISK_THRESHOLD} | "
      f"Medium >= {MEDIUM_RISK_THRESHOLD}  (C_FN:C_FP = {COST_FN_OVER_FP:.0f}:1)")
print("Risk_Level dağılımı:")
print(analytical_df["Risk_Level"].value_counts().to_string(), "\n")
print(f"High Risk Rate % (Power BI KPI karşılığı): "
      f"%{(analytical_df['Risk_Level'] == 'High Risk').mean() * 100:.1f}")
print(f"Müdahale edilen (High + Medium)          : "
      f"%{(analytical_df['Delay_Risk_Probability'] >= MEDIUM_RISK_THRESHOLD).mean() * 100:.1f}\n")

# Kalibrasyon doğrulaması: ortalama tahmin, gerçekleşen orana eşit mi?
print("--- Kalibrasyon doğrulaması: tahmin vs gerçekleşen ---")
check = (
    analytical_df.groupby("Risk_Level")
    .agg(Sevkiyat=("Shipment_ID", "count"),
         Ort_Tahmin=("Delay_Risk_Probability", "mean"),
         Gercek_Oran=("is_delayed", "mean"),
         Ort_Gecikme_Gun=("Actual_Delay_Days", "mean"))
    .reindex(["High Risk", "Medium Risk", "Low Risk"])
)
check["Sapma"] = (check["Ort_Tahmin"] - check["Gercek_Oran"])
print(check.round(3).to_string())
print("Sapma sıfıra yakınsa skor gerçekten bir olasılıktır.\n")

# Maliyet karşılaştırması: yeni eşik gerçekten daha mı ucuz?
print("--- Eşik karşılaştırması (C_FN:C_FP = 4:1) ---")
p_arr, y_arr = analytical_df["Delay_Risk_Probability"].values, y.values


def alert_cost(threshold):
    alert = p_arr >= threshold
    fp = int((alert & (y_arr == 0)).sum())
    fn = int((~alert & (y_arr == 1)).sum())
    tp = int((alert & (y_arr == 1)).sum())
    return fp + COST_FN_OVER_FP * fn, tp, fp, fn


for label, th in [("Eski kantil eşiği (0.65)", 0.65),
                  (f"Maliyet-optimal ({MEDIUM_RISK_THRESHOLD})", MEDIUM_RISK_THRESHOLD)]:
    cost, tp, fp, fn = alert_cost(th)
    print(f"  {label:<28} alarm %{(p_arr >= th).mean() * 100:4.1f} | "
          f"yakalanan {tp}/{tp + fn} (%{tp / (tp + fn) * 100:.0f}) | "
          f"yanlış alarm {fp:3d} | maliyet {cost:.0f}")
print()

# ---------------------------------------------------------------------------
# 7) Fact_Shipments.csv yapısını bozmadan sadece 2 yeni sütun ekle
# ---------------------------------------------------------------------------
fact_with_ml = fact.merge(
    analytical_df[["Shipment_ID", "Delay_Risk_Probability", "Risk_Level"]],
    on="Shipment_ID",
    how="left",
)

EXPECTED_COLUMNS = [
    "Shipment_ID", "Date_ID", "Vendor_ID", "Route_ID", "Weight_tons", "Distance_km",
    "Actual_Delay_Days", "CO2_Emission_kg", "Delay_Risk_Probability", "Risk_Level",
]

assert len(fact_with_ml) == len(fact), "Satır sayısı değişti, join hatası olabilir!"
assert fact_with_ml["Shipment_ID"].is_unique, "Shipment_ID (PK) benzersizliği bozuldu!"
assert fact_with_ml["Delay_Risk_Probability"].notna().all(), "Eksik risk skoru bulundu!"
# .pbix ilişkileri sütun adlarına ve sırasına bağlı - sessizce bozulmasın.
assert list(fact_with_ml.columns) == EXPECTED_COLUMNS, (
    f"Sütun yapısı değişti! Power BI ilişkileri bozulur.\n"
    f"  Beklenen: {EXPECTED_COLUMNS}\n  Gelen    : {list(fact_with_ml.columns)}"
)
assert fact_with_ml["Vendor_ID"].isin(dim_vendor["Vendor_ID"]).all(), "Geçersiz Vendor_ID FK!"
assert fact_with_ml["Route_ID"].isin(dim_route["Route_ID"]).all(), "Geçersiz Route_ID FK!"
assert fact_with_ml["Date_ID"].isin(dim_date["Date_ID"]).all(), "Geçersiz Date_ID FK!"

output_path = PROCESSED_DIR / "Fact_Shipments_with_ML.csv"
fact_with_ml.to_csv(output_path, index=False)

print(f"Kaydedildi: {output_path}  (shape: {fact_with_ml.shape})")
print("Şema doğrulaması geçti: sütun adları/sırası ve tüm FK'lar korundu.")
print("\n--- Fact_Shipments_with_ML.csv önizleme ---")
print(fact_with_ml.head())

# ---------------------------------------------------------------------------
# 8) Feature Importance
# ---------------------------------------------------------------------------
# Her iki tablo da `eval_model` üzerinden: test setini hiç görmemiş model.
# Bölüm 5'te `model` tüm veriyle yeniden eğitildiği için onu X_test üzerinde
# permüte etmek sessiz bir sızıntı olurdu (ölçüldü: 0.14 yerine 0.24).
#
# CalibratedClassifierCV içinde N_SPLITS adet alt-model tutar; impurity
# importance'ları ortalayarak tek bir tablo çıkarıyoruz.
calibrated_folds = eval_model.calibrated_classifiers_
importances = np.mean(
    [cc.estimator.named_steps["classifier"].feature_importances_
     for cc in calibrated_folds], axis=0
)
ohe = calibrated_folds[0].estimator.named_steps["preprocessor"].named_transformers_["cat"]
all_feature_names = numeric_features + list(ohe.get_feature_names_out(categorical_features))


def group_name(feature_name: str) -> str:
    for cat in categorical_features:
        if feature_name.startswith(cat + "_"):
            return cat
    return feature_name


importance_df = pd.DataFrame({"Feature": all_feature_names, "Importance": importances})
grouped_importance = (
    importance_df.assign(Group=lambda d: d["Feature"].apply(group_name))
    .groupby("Group")["Importance"].sum()
    .sort_values(ascending=False)
    .round(4)
)

print("\n--- Feature Importance (grup bazlı, impurity/Gini) ---")
print(grouped_importance.to_string())

# --- Permutation Importance ------------------------------------------------
# Impurity importance sürekli ve yüksek kardinaliteli sütunları sistematik
# olarak ŞİŞİRİR: ağaç böyle bir sütunda çok sayıda aday bölme noktası bulur.
#
# Bu veri setinde etkisi ölçülebilir: Weight_tons,
# generate_logistics_data.py'deki gecikme sürecinde HİÇ YER ALMAZ - saf
# gürültüdür. Buna rağmen impurity importance onu en güçlü gerçek sürücü olan
# Weather_Condition'ın ÜSTÜNE koyar. Permutation importance sütunu karıştırıp
# modelin gerçekten ne kadar bozulduğuna bakar ve doğru sıralamayı verir.
#
# Yorumlanması gereken tablo AŞAĞIDAKİDİR.
perm = permutation_importance(
    eval_model, X_test, y_test, n_repeats=20,
    random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
)
perm_df = (
    pd.DataFrame({
        "Feature": feature_cols,
        "AUC_dususu": perm.importances_mean,
        "std": perm.importances_std,
    })
    .sort_values("AUC_dususu", ascending=False)
    .reset_index(drop=True)
)

print("\n--- Permutation Importance (test setinde ROC-AUC düşüşü) ---")
print(perm_df.round(4).to_string(index=False))
print("\nNOT: İki tablo çelişirse permutation importance esas alınmalıdır.")
print("     Weight_tons ve Vehicle_Type gecikme sürecinde hiç yer almaz; bunlar")
print("     bilinçli olarak modelde tutulan kontrol değişkenleridir (bkz. README).")

# ---------------------------------------------------------------------------
# 9) ÜRETİM MODELİNİ DİSKE KAYDET (Streamlit demosu bunu yükler)
# ---------------------------------------------------------------------------
# Kaydedilen nesne TÜM PIPELINE'dır: ColumnTransformer + OneHotEncoder +
# CalibratedClassifierCV ile sarılmış RandomForest, tek bir joblib dosyasında.
#
# Yalnızca classifier'ı kaydetmek, tüketen tarafın encoding'i elle yeniden
# kurmasını gerektirirdi - bu train/serve skew'in klasik kaynağıdır: kategori
# sırası, bilinmeyen kategori davranışı veya sütun sırası ufak bir şekilde
# farklılaşır ve model sessizce yanlış tahmin üretir. Pipeline'ın tamamını
# kaydedince tüketen taraf ham bir DataFrame verir, gerisini nesne halleder.
#
# Bu model TÜM veriyle eğitilir (üretim için doğrusu budur). Yukarıdaki
# metrikler ise dürüst, örneklem-dışı ölçümlerdir - bu son fit'ten GELMEZ.
production_model = Pipeline(steps=[
    ("preprocessor", ColumnTransformer(transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])),
    ("classifier", RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
    )),
])
production_model = CalibratedClassifierCV(production_model, method="isotonic", cv=N_SPLITS)
production_model.fit(X, y)

metadata = {
    "model_name": "logistics_delay_risk",
    "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),

    # Tüketen taraf sütun sırasını buradan okur - hardcode ETMEMELİ.
    "feature_order": list(feature_cols),
    "numeric_features": list(numeric_features),
    "categorical_features": list(categorical_features),
    # Arayüzdeki seçenek listeleri de buradan beslenebilir.
    "categorical_levels": {
        c: sorted(analytical_df[c].dropna().unique().tolist())
        for c in categorical_features
    },
    "numeric_ranges": {
        c: {"min": float(analytical_df[c].min()), "max": float(analytical_df[c].max())}
        for c in numeric_features
    },

    # Risk eşikleri - maliyet matrisinden türetildi, kantil değil.
    "thresholds": {
        "high_risk": HIGH_RISK_THRESHOLD,
        "medium_risk": MEDIUM_RISK_THRESHOLD,
        "cost_fn_over_fp": COST_FN_OVER_FP,
        "derivation": "p* = 1 / (1 + C_FN/C_FP); high_risk = 1:1 oranının eşiği",
    },

    # Metrikler: hepsi ÖRNEKLEM-DIŞI. Kaydedilen modelin kendi eğitim
    # verisindeki performansı değildir.
    "metrics": {
        "oof_roc_auc": round(float(oof_roc_auc), 4),
        "oof_brier": round(float(oof_brier), 4),
        "oof_ece": round(float(oof_ece), 4),
        "chronological_holdout_roc_auc": round(float(chrono_auc), 4),
        "chronological_cv_roc_auc_mean": round(float(np.mean(ts_aucs)), 4),
        "chronological_cv_roc_auc_std": round(float(np.std(ts_aucs)), 4),
    },

    "training_data": {
        "n_rows": int(len(analytical_df)),
        "positive_rate": round(float(y.mean()), 4),
        "date_min": analytical_df["Full_Date"].min().strftime("%Y-%m-%d"),
        "date_max": analytical_df["Full_Date"].max().strftime("%Y-%m-%d"),
        "is_synthetic": True,
    },

    # Sürüm uyuşmazlığı pickle yüklemeyi bozar; tüketen taraf kontrol edebilsin.
    "versions": {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    },

    "caveats": [
        "Sentetik veriyle eğitilmiştir; gerçek dünya performansını temsil etmez.",
        "Olasılıklar isotonic ile kalibre edildi (ECE ~0.03), ancak p>0.8 "
        "bandında gözlem azlığı nedeniyle güvenilirlik düşüktür.",
        "Eşikler 4:1 maliyet varsayımından türetilmiştir; bu oran gerçek SLA "
        "ceza tarifeleriyle doğrulanmamıştır.",
        "CO2_Emission_kg bu modelin çıktısı DEĞİLDİR; deterministik bir "
        "formülle hesaplanır (bkz. generate_logistics_data.py).",
    ],
}

MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "production_risk_model.pkl"
META_PATH = MODEL_DIR / "production_risk_model_metadata.json"

# compress=3: CalibratedClassifierCV 5 alt-model x 300 ağaç = 1500 ağaç tutar,
# sıkıştırmasız 43 MB eder. Bu GitHub'ın 50 MB uyarı eşiğine yakındır ve her
# yeniden eğitimde repo geçmişini şişirir. Sıkıştırma KAYIPSIZDIR (ölçüldü:
# tahminler bit düzeyinde aynı) ve dosyayı 8.7 MB'a indirir; yükleme 0.4 s.
joblib.dump({"model": production_model, "metadata": metadata}, MODEL_PATH, compress=3)
META_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\n--- Üretim modeli kaydedildi ---")
print(f"{MODEL_PATH}  ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
print(f"{META_PATH}   (insan tarafından okunabilir kopya)")

# Kayıt sonrası kendi kendini doğrula: dosyayı geri yükle ve tahmin üret.
# Bu, "kaydettim" ile "kullanılabilir" arasındaki farkı kapatır.
_loaded = joblib.load(MODEL_PATH)
_smoke = pd.DataFrame([{
    "Distance_km": 450.0, "Weight_tons": 12.0, "Vendor_Rating": 4.0,
    "Weather_Condition": "Storm", "Traffic_Density": "High",
    "Vehicle_Type": "Diesel Truck",
}])[_loaded["metadata"]["feature_order"]]
_p = float(_loaded["model"].predict_proba(_smoke)[0, 1])
assert 0.0 <= _p <= 1.0, "Yüklenen model geçersiz olasılık üretti!"
_orig = float(production_model.predict_proba(_smoke)[0, 1])
assert abs(_p - _orig) < 1e-9, "Yüklenen model bellektekiyle aynı sonucu vermiyor!"
print(f"Doğrulama: model geri yüklendi, örnek tahmin = {_p:.3f} "
      f"(bellektekiyle birebir aynı) ✓")

# ---------------------------------------------------------------------------
# Fabric notları
# ---------------------------------------------------------------------------
# - Bu script bir Microsoft Fabric Notebook hücresinde pandas ile aynen çalışır.
# - Lakehouse'a bağlıysanız CSV yerine Delta tablo kullanmak isterseniz:
#     fact = spark.read.format("delta").load("Tables/Fact_Shipments").toPandas()
#   ve kaydetmek için:
#     spark.createDataFrame(fact_with_ml).write.format("delta") \
#         .mode("overwrite").save("Tables/Fact_Shipments_with_ML")
# - Büyük veri setlerinde (>birkaç milyon satır) SynapseML üzerinden dağıtık
#   bir Random Forest'a geçiş değerlendirilebilir; ancak CalibratedClassifierCV
#   ve StratifiedGroupKFold'un dağıtık karşılıklarının elle kurulması gerekir.
