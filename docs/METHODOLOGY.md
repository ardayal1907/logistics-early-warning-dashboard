# Methodology & Technical Reference

Full technical documentation for
[Logistics Delay-Risk Prediction & Carbon Cost Dashboard](../README.md).

The README is deliberately short. Everything that a reviewer, a future maintainer or
an interviewer might want to dig into lives here: how each modelling decision was
made, what was measured, what was rejected, and what remains unsolved.

## Contents

- [Pipeline stages in detail](#pipeline-stages-in-detail)
- [Data Model (Star Schema)](#data-model-star-schema)
- [Tech Stack](#-tech-stack)
- [DAX Measures & Business Logic](#dax-measures)
- [Report Structure (2 Pages)](#-report-structure-2-pages)
- [What This Project Demonstrates](#what-this-demonstrates)
- [Streamlit App — Design Notes](#streamlit-app--design-notes)
- [Methodology Checklist](#methodology-checklist)
- [Known Limitations (full list)](#known-limitations)
- [Possible Next Steps](#next-steps)
- [Power BI Setup After Adding Dim_Date](#powerbi-setup)

---

## Architecture & Data Model

### Pipeline stages in detail

| Stage | Script | Input | Output |
|---|---|---|---|
| **1. Synthetic Data Generation** | `src/generate_logistics_data.py` | — | `data/raw/smart_logistics_data.csv` (1,500 shipments) |
| **2. Star Schema ETL** | `src/etl_star_schema.py` | `data/raw/smart_logistics_data.csv` | `data/processed/`: `Dim_Vendor.csv`, `Dim_Route.csv`, `Dim_Date.csv`, `Fact_Shipments.csv` |
| **3. ML Risk Scoring** | `src/ml_delay_risk_pipeline.py` | `data/processed/` star schema | `data/processed/Fact_Shipments_with_ML.csv` + `models/production_risk_model.pkl` & metadata |
| **4. BI Layer** | `powerbi/Lojistik_Erken_Uyari_Paneli.pbix` | `data/processed/*.csv` | Interactive 2-page Power BI report |
| **5. Web demo** | `src/app.py` | `models/production_risk_model.pkl` | Streamlit single-shipment scorer |
| *(side branch)* **Teaching notebook** | `notebooks/Logistics_Delay_Risk_ML_Pipeline.ipynb` | self-generated (schema-compatible) | `demo_risk_model_output.csv` — **not** consumed by Power BI |

> **Why the notebook writes to a different filename.** The notebook is a narrative
> walkthrough of *how* the model works and generates its own schema-compatible data so
> it can run standalone. Its output is deliberately named `demo_risk_model_output.csv`
> rather than `Fact_Shipments_with_ML.csv`. An earlier version used the production
> filename, which meant running the notebook in the repo root silently overwrote the
> table the `.pbix` depends on — breaking the `Dim_Route` and `Dim_Vendor`
> relationships and blanking the CO₂ measures without raising any error. Only
> `src/ml_delay_risk_pipeline.py` writes tables that Power BI consumes, and they now
> live in `data/processed/` while the notebook writes to its own working directory.

### Data Model (Star Schema)

```
       Dim_Vendor                Dim_Date                    Dim_Route
  ┌──────────────────┐   ┌────────────────────┐   ┌─────────────────────┐
  │ Vendor_ID (PK)   │   │ Date_ID (PK)        │   │ Route_ID (PK)        │
  │ Vendor_Rating    │   │ Full_Date           │   │ Origin                │
  └────────┬─────────┘   │ Year / Quarter      │   │ Destination           │
            │             │ Month / Month_Name  │   │ Vehicle_Type          │
            │             │ Month_Year          │   │ Weather_Condition     │
            │             │ Season              │   │ Traffic_Density       │
            │             │ Day_Of_Week         │   └───────────┬───────────┘
            │             │ Is_Weekend          │               │
            │             └─────────┬──────────┘               │
            │  1                     │  1                       │  1
            │                        │                          │
            │  N                     │  N                       │  N
            └──────────┐   ┌─────────┘   ┌────────────────────┘
                        ▼   ▼             ▼
             ┌────────────────────────────────┐
             │  Fact_Shipments_with_ML         │
             ├────────────────────────────────┤
             │ Shipment_ID (PK)                │
             │ Date_ID (FK)      ◀── new       │
             │ Vendor_ID (FK)                  │
             │ Route_ID (FK)                   │
             │ Weight_tons                     │
             │ Distance_km                     │
             │ Actual_Delay_Days               │
             │ CO2_Emission_kg                 │
             │ Delay_Risk_Probability          │
             │ Risk_Level                      │
             └────────────────────────────────┘
```

`Dim_Date` is a **contiguous** calendar covering every day in the shipment window,
including days with no shipments. This is deliberate: time-intelligence measures
(moving averages, YTD, year-over-year) silently return wrong answers on a date table
with gaps.

---

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data generation & preprocessing | Python, pandas, NumPy |
| Machine Learning | scikit-learn (Random Forest, One-Hot Encoding, `StratifiedGroupKFold`, `CalibratedClassifierCV`, permutation importance) |
| Data modeling | Star Schema (Kimball-style dimensional modeling) |
| BI & Visualization | Power BI Desktop, Power Query (M), DAX |
| Web demo | Streamlit (`src/app.py`), joblib model artefact + metadata |
| AI-assisted analytics in BI | Power BI Key Influencers visual |
| Version control | Git / GitHub |

---

---

<a id="dax-measures"></a>

## 📐 DAX Measures & Business Logic

> 📄 These measures are also mirrored as plain text in
> [`powerbi/measures.dax`](../powerbi/measures.dax), so the semantic model can be
> reviewed and diffed without opening Power BI Desktop.
>
> **The mirror is verified, not assumed.** The expressions were read out of the live
> model with DAX Studio (`SELECT [Name], [Expression] FROM $SYSTEM.TMSCHEMA_MEASURES`);
> the field bindings were extracted by parsing the `.pbix` report-definition JSON. The
> query returned **exactly four rows**, so the model contains these four measures and
> no others — nothing hidden, nothing unused. The report adds two implicit
> `Count of Shipment_ID` aggregations, which carry no DAX and are listed in
> `measures.dax` for completeness.

The Power BI semantic model implements exactly these four measures:

```dax
Total CO2 Tons =
DIVIDE ( SUM ( Fact_Shipments_with_ML[CO2_Emission_kg] ), 1000 )
```
> Converts the raw kg-level emission figures aggregated in the fact table into metric tons — the standard unit for corporate carbon reporting.

```dax
High Risk Rate % =
DIVIDE (
    CALCULATE (
        COUNTROWS ( Fact_Shipments_with_ML ),
        Fact_Shipments_with_ML[Risk_Level] = "High Risk"
    ),
    COUNTROWS ( Fact_Shipments_with_ML )
)
```
> Shows what share of the current shipment population (respecting all active slicers/filters) falls into the highest-risk delay tier — the primary KPI for the Early Warning page.

```dax
Carbon Tax Impact ($) =
[Total CO2 Tons] * 50
```
> Applies a $50/ton carbon price assumption (a commonly cited mid-range estimate used in voluntary carbon pricing and internal carbon fee models) to translate emissions into a projected financial liability. The multiplier is intentionally isolated as a hardcoded assumption so it can be swapped for a jurisdiction-specific carbon tax or internal shadow price without touching the rest of the model.

```dax
Average Delay Days =
AVERAGE ( Fact_Shipments_with_ML[Actual_Delay_Days] )
```
> The realised average across *all* shipments, not only the late ones — so it reads 0.39 rather than the ~1.8-day mean of the delayed subset. It is the observed outcome the model is trained to predict, kept on the Executive Summary as the ground-truth counterweight to the predicted risk KPIs.

**Business logic behind the model:** the three carbon-and-risk measures are designed to be read together — a shipment that is both high-risk *and* high-emission (e.g., a Diesel Truck on a long route in Storm conditions) represents compounded exposure: it is more likely to fail its SLA *and* carries a disproportionate carbon cost, making it a candidate for vendor renegotiation or a mode shift (e.g., to Electric Semi).

> Note on `High Risk Rate %`: this KPI currently reads **~10.3%**, with a further
> ~27% in the Medium tier — about 37% of shipments cross the cost-optimal
> intervention threshold. An earlier version of the data generator produced an 84.3%
> reading; a panel where most shipments are flagged red carries no information, since
> an alert is only informative when it is rare. The thresholds behind these tiers are
> derived from a cost matrix, not from percentiles — see
> [Key point 8](#what-this-demonstrates).

---

---

## 📊 Report Structure (2 Pages)

### Page 1 — Executive Summary
- KPI cards: Total Shipments, Total CO2 Tons, Carbon Tax Impact ($), Average Delay Days
- Column chart: CO2 Emission distribution by Vehicle Type (highlights the emissions gap between Diesel Truck, Hybrid Van, and Electric Semi)

### Page 2 — Early Warning Panel
- KPI card: High Risk Rate %
- Donut chart: Shipment distribution by Risk_Level (High / Medium / Low)
- **Key Influencers** visual: AI-driven analysis of which factors most increase the likelihood of a shipment being classified as High Risk
- Detailed, filterable risk table at the shipment level (Shipment_ID, Vendor, Route, Delay_Risk_Probability, Risk_Level)

---

---

<a id="what-this-demonstrates"></a>

## 💡 What This Project Demonstrates

> **Read this first.** The data in this project is **synthetic and generated by
> `generate_logistics_data.py`**. That script defines the process that produces
> delays, so anything the model "discovers" about *why* shipments are late is a
> **readback of assumptions written into the generator**, not an empirical finding.
> The items below are therefore stated as *what the pipeline demonstrates*, not as
> insights about real logistics operations. Every number is reproducible by running
> the three scripts in order.

1. **The model recovers a genuinely uncertain signal, not a formula.** Delay is
   generated as a two-stage stochastic outcome (a Bernoulli draw on a logistic
   probability, then a truncated-Poisson severity), so identical inputs can produce
   different outcomes. This puts a hard ceiling on achievable accuracy:

   | Metric | Value |
   |---|---|
   | Model ROC-AUC (chronological holdout, last 20%) | **0.737** |
   | Model ROC-AUC (chronological CV, 5 expanding folds) | **0.756 ± 0.050** |
   | Bayes ceiling — AUC of the *true* generating probability | ~0.81 |

   A model cannot beat the noise it was asked to predict through, so the right
   reference point is the ceiling, not 1.0.

2. **How you split the data matters more than which model you pick.** Three
   strategies, same data, same model:

   | Split strategy | ROC-AUC | What it allows |
   |---|---|---|
   | `StratifiedKFold` (random) | 0.781 | Uses the future to predict the past |
   | `StratifiedGroupKFold` on `Vendor_ID` | 0.758 | Still uses the future |
   | **Chronological holdout (last 20%)** | **0.737** | **Nothing the model wouldn't have in production** |

   Each restriction costs a few points, and **that cost is the measurement being
   corrected, not performance being lost.** The random-split figure was never real.

   The vendor-grouped row deserves a note of its own. `Vendor_Rating` is stored once
   per vendor, so it is effectively a vendor identifier, and the worry was that the
   model memorises "this vendor is always late". Measured against an earlier dataset
   the difference was 0.774 vs 0.773 — no memorisation, because vendor effect enters
   the generator as a smooth function of the rating, so the model learns the
   *rating→risk* mapping and it transfers to unseen vendors. (Removing
   `Vendor_Rating` entirely drops AUC to 0.615, confirming it carries most of the
   signal.) The grouped split is kept regardless: real vendor scorecards are noisy and
   stale, and correctness should rest on the design rather than on a measurement that
   happened to come out favourably.

3. **The chronological drop is caused by seasonal distribution shift — measured, not
   assumed.** The test window is the last 20% of the year, which lands in summer:

   | Weather | Train | Test | Δ |
   |---|---|---|---|
   | Normal | 51.9% | 82.0% | **+30.1** |
   | Rain | 22.6% | 13.0% | −9.6 |
   | Snow | 16.1% | **0.0%** | −16.1 |
   | Storm | 9.4% | 5.0% | −4.4 |

   Delay rate falls from 23.6% (train) to 13.3% (test). In the test window
   `Weather_Condition` — the second-strongest driver — is almost constant, so it has
   nearly no discriminating power left and the model must lean on `Vendor_Rating`
   alone. The per-fold results confirm the mechanism directly:

   | Chronological fold | Test window | Delay rate | ROC-AUC |
   |---|---|---|---|
   | 1 | Oct–Nov | 22% | 0.682 |
   | 2 | Nov–Jan | 26% | 0.765 |
   | 3 | **Jan–Mar** | **29%** | **0.824** |
   | 4 | Mar–Jun | 17% | 0.787 |
   | 5 | **Jun–Jul** | **14%** | **0.720** |

   AUC tracks the season: highest in the winter fold, lowest in the summer fold. The
   ±0.05 spread is not instability, it is the phenomenon. **Practical consequence: a
   single-period backtest of this model is not a reliable estimate — it must be
   evaluated across at least a full seasonal cycle.**

4. **Seasonality was added without changing the underlying signal.** Weather is now
   drawn per-month on a Northern-Hemisphere cycle (Nov–Feb: Snow/Storm heavy; Jun–Aug:
   Normal dominant, zero Snow). The chain is `month → weather distribution →
   (weather × vendor) → delay`. The weather→delay coefficients and the weather×vendor
   interaction are **untouched** — only *when* each weather type appears changed. The
   monthly distributions were chosen so the **annual marginal** stays close to the
   previous fixed one, which is what keeps the overall delay rate and interaction
   structure intact:

   | | Normal | Rain | Storm | Snow |
   |---|---|---|---|---|
   | Previous (fixed all year) | 0.60 | 0.20 | 0.10 | 0.10 |
   | New (seasonal, annual marginal) | 0.579 | 0.207 | 0.085 | 0.129 |

   The result is visible in the data: February runs 47% Snow with a 33% delay rate,
   July runs 0% Snow with a 12.8% delay rate.

5. **Feature importance rankings depend heavily on which importance metric is used —
   and the default one is misleading here.** `Weight_tons` does **not** appear
   anywhere in the delay-generating process; it is pure noise. Impurity (Gini)
   importance nevertheless ranks it 3rd, above the strongest true driver:

   | Feature | Impurity (Gini) | Permutation (ROC-AUC drop) | In the generator? |
   |---|---|---|---|
   | `Vendor_Rating` | 0.332 | **0.155** | yes — strongest |
   | `Traffic_Density` | 0.104 | **0.048** | yes — moderate |
   | `Weather_Condition` | 0.110 | **0.033** | yes — strong |
   | `Weight_tons` | 0.196 | 0.008 ± 0.009 | **no — pure noise** |
   | `Vehicle_Type` | 0.056 | 0.007 ± 0.016 | **no — pure noise** |
   | `Distance_km` | 0.201 | −0.005 ± 0.015 | yes — very weak |

   (`Weather_Condition` ranks below `Traffic_Density` here only because permutation
   importance is computed on the *summer* test window, where weather barely varies —
   another reminder that these numbers are period-specific.)

   Impurity importance systematically inflates continuous, high-cardinality columns
   because a tree finds many candidate split points in them. It ranks `Weight_tons`
   (0.196) *above* `Weather_Condition` (0.110) — noise above a real driver — and gives
   `Distance_km` the second-highest score despite permutation importance measuring it
   as slightly *negative*. Permutation importance puts both noise columns at the
   bottom, with effects within roughly one standard deviation of zero.
   **`ml_delay_risk_pipeline.py` prints both and treats permutation importance as
   authoritative.** This is the most transferable lesson in the project: on real data
   the same distortion occurs, but there is no answer key to catch it with.

   > **Why `Weight_tons` and `Vehicle_Type` are deliberately kept in the model.**
   > Both are provably irrelevant *here* — but only because we wrote the generator and
   > can check. On real data you never know in advance which column is noise; that is
   > precisely what feature importance is supposed to tell you. Dropping them would
   > make the model marginally simpler while removing the very demonstration that
   > matters: showing that the default importance metric will confidently promote a
   > useless column, and that permutation importance catches it. They function as
   > **control variables** — a built-in check that the diagnostic works. Removing them
   > would also make `CO2_Emission_kg` reporting inconsistent, since `Vehicle_Type` is
   > the main driver of emissions even though it does not affect delay.

6. **Weather and vendor quality interact — and the interaction had to be modelled
   explicitly.** In an additive model, bad weather pushes nearly every shipment over
   the delay threshold and vendor quality stops mattering (a ceiling effect). The
   generator therefore includes a weather × vendor interaction term, centred on the
   mean vendor rating so that it widens the *spread* without inflating the *mean*.
   Resulting delay rates by vendor-rating quartile:

   | Weather | Q1 (worst vendors) | Q2 | Q3 | Q4 (best) |
   |---|---|---|---|---|
   | Storm | 85.0% | 54.2% | 16.2% | 10.7% |
   | Normal | 29.7% | 14.3% | 7.8% | 4.7% |

   *Caveat:* Storm and Snow cells contain roughly 25–55 rows each, so individual
   cell percentages carry substantial sampling error. Only the direction of the
   gradient should be read, not the specific values.

7. **A Random Forest's `predict_proba` is not a probability until you calibrate it.**
   It is the fraction of trees voting positive. Measured on out-of-fold scores, the
   raw model was systematically overconfident — where it said 0.65, the realised delay
   rate was 0.36:

   | | ROC-AUC | Brier | ECE |
   |---|---|---|---|
   | Raw Random Forest | 0.757 | 0.1655 | **0.1381** |
   | + `CalibratedClassifierCV(isotonic)` | 0.756 | 0.1401 | **0.0232** |
   | *(reference: always predict the base rate)* | 0.500 | 0.1690 | — |

   Calibration is a monotone transform, so it does not change ranking (ROC-AUC is
   essentially unmoved) — what changes is what the number *means*. **Isotonic vs
   sigmoid was decided by measurement, not by rule of thumb:** n=1,500 conventionally
   favours sigmoid, but isotonic gave better ECE in **5 out of 5** random seeds
   (mean difference −0.0065 ± 0.0043) with an identical Brier score.

   This matters beyond presentation: the cost-optimal threshold formula in the next
   point is **only valid on calibrated probabilities**.

8. **Risk thresholds are now derived from a cost matrix, not from percentiles.**
   Earlier versions used 0.70/0.35, then 0.65/0.33 — the latter anchored to the score
   distribution's p95/p80. But "the riskiest 5%" has no business justification. The
   cost assumption instead:

   | | What it is | Cost |
   |---|---|---|
   | **False alarm (FP)** | A planner checks the shipment, calls the vendor, reserves buffer capacity | ~1 unit |
   | **Missed delay (FN)** | SLA breach: contractual penalty + expedited re-routing + churn risk | ~4 units |

   In contract logistics SLA penalties are charged as a percentage of freight value
   and expediting a late shipment costs a multiple of standard freight, whereas a
   false alarm costs about an hour of planner time. A 4:1 ratio sits mid-range of a
   defensible 2–5 band; the pipeline prints a sensitivity table across that band.

   With calibrated probabilities the expected-cost-minimising threshold is analytic —
   intervene when `p·C_FN > (1−p)·C_FP`, i.e. `p* = 1/(1+ratio) = 0.20`. The tiers are
   set so that **each boundary means something**:

   | Risk_Level | Cut-off | Meaning |
   |---|---|---|
   | High Risk | p ≥ 0.50 | Worth intervening even if a false alarm cost the *same* as a missed delay |
   | Medium Risk | 0.20 ≤ p < 0.50 | Worth intervening under the 4:1 assumption |
   | Low Risk | p < 0.20 | Not worth intervening even at 4:1 |

   The percentile approach turns out to have been badly wrong:

   | Threshold | Alerts on | Delays caught | Expected cost |
   |---|---|---|---|
   | Percentile p95 (0.65) | 5.3% | 58/323 (**18%**) | 1081 |
   | Cost-optimal (0.20) | 37.4% | 222/323 (**69%**) | **743** |

   It missed 82% of delays and cost 45% more. Calibration check against realised
   outcomes — mean predicted probability should equal the realised rate:

   | Risk_Level | Shipments | Mean predicted | Realised rate | Deviation |
   |---|---|---|---|---|
   | High Risk | 154 | 0.672 | 0.623 | +0.048 |
   | Medium Risk | 407 | 0.301 | 0.310 | −0.009 |
   | Low Risk | 939 | 0.104 | 0.108 | −0.003 |

9. **Scoring must be out-of-fold, or the dashboard shows memorised history instead of
   predictions.** Training on all rows and scoring those same rows produced ROC-AUC
   **0.998**, with the "Low Risk" tier showing a 0.0% actual delay rate — an
   impossible number that reflects a Random Forest reciting its own training labels.
   The pipeline uses `cross_val_predict` with the grouped splitter, so every shipment
   is scored by a model that saw neither it nor its vendor. Without this, the Early
   Warning panel would be a report on the past disguised as a forecast.

10. **A $50/ton carbon price turns an abstract sustainability metric into a budget line
   item**, enabling a cost-based rather than purely compliance-based conversation
   about fleet electrification. The multiplier is deliberately isolated as a single
   DAX constant so it can be swapped for a jurisdiction-specific rate. Note that the
   emissions figures are a deterministic function of distance, tonnage and vehicle
   type — there is no modelling claim here, only unit conversion and pricing.

---

---

## Streamlit App — Design Notes

**Design notes:**

- The app loads the **entire pipeline** (`ColumnTransformer` + `OneHotEncoder` +
  `CalibratedClassifierCV`) as one object and passes it a raw DataFrame. It never
  re-implements encoding — doing that in the serving layer is the classic source of
  train/serve skew.
- Risk thresholds, feature order, category levels and reported metrics are read from
  the model's **metadata**, not hardcoded. Retrain the model with different thresholds
  and the app follows automatically.
- **CO₂ and carbon tax are not model outputs.** They are computed with the same
  deterministic formula as `generate_logistics_data.py` (distance × tonnage ×
  emission factor + fixed base, times weather/traffic multipliers) and are labelled
  *"hesaplanan"* (calculated) in the UI so they are not mistaken for predictions.
- An in-app expander documents the synthetic-data caveat, the calibration quality and
  its weakness in the tail, the 4:1 cost assumption behind the thresholds, and the
  Storm/Snow sample-size limitation.

> **Deployment note:** `requirements.txt` pins exact versions. The model is a pickle,
> and scikit-learn's internal object layout changes between releases — a version
> mismatch on Streamlit Cloud either fails to load the model or loads it and silently
> produces different predictions. Keep `scikit-learn` identical to the version that
> trained the model (recorded in `models/production_risk_model_metadata.json`).

---

<a id="methodology-checklist"></a>

## ✅ Methodology Checklist

What the pipeline does and does not guard against, at a glance:

| Concern | Status |
|---|---|
| Target leakage (`Actual_Delay_Days`, `CO2_Emission_kg` excluded from features) | ✅ handled |
| Circular target (derived from an observed outcome, not a rule) | ✅ handled |
| Encoding of nominal categories (One-Hot, not label encoding) | ✅ handled |
| Preprocessing leakage (all steps inside a `Pipeline`) | ✅ handled |
| In-sample scoring (`cross_val_predict`) | ✅ handled |
| Vendor leakage across splits (`StratifiedGroupKFold` on `Vendor_ID`) | ✅ handled |
| Probability calibration (`CalibratedClassifierCV`, isotonic) | ✅ handled |
| Cost-based decision thresholds | ✅ handled |
| Misleading impurity importance (permutation importance reported) | ✅ handled |
| Temporal validation (chronological split + walk-forward scoring) | ✅ handled |
| Seasonality present in the data at all | ✅ handled (`Dim_Date`, month-dependent weather) |
| Multi-year drift / concept drift in vendor quality | ❌ not addressed (only one year of data) |

---

<a id="known-limitations"></a>

## ⚠️ Known Limitations

Stated explicitly, because a model card without a limitations section is a sales
document:

1. **Synthetic data.** No figure here predicts real-world performance. On real data
   the hardest problem would likely not be modelling at all, but the fact that
   `Actual_Delay_Days` is rarely recorded consistently across carriers.

2. **Only one seasonal cycle — no drift measurable.** The data covers 12 months, so
   seasonality can be modelled but year-over-year change cannot ("was this winter
   worse than last?"). More importantly, **vendor quality is constant over time** in
   this data. In reality vendor performance degrades or improves, and that concept
   drift is what forces periodic retraining. Nothing here tests for it.

3. **Chronological performance depends heavily on which period you test on.** The
   folds range from 0.682 to 0.824 (±0.05) purely by season. A single-period backtest
   of this model would be misleading in either direction; at minimum a full seasonal
   cycle must be evaluated.

4. **The warm-up period is not scored time-forward.** The earliest ~17% of shipments
   have no prior history to train on, so they fall back to vendor-grouped out-of-fold
   scores. Those rows carry slightly optimistic probabilities. The pipeline prints
   exactly how many rows this affects.

5. **The 4:1 cost ratio is an assumption.** Thresholds are now *derived* from it
   rather than from percentiles, which is the right structure — but the ratio itself
   has not been validated against actual SLA penalty schedules or expediting invoices.
   It is isolated as a single constant (`COST_FN_OVER_FP`) so it can be replaced with
   a real figure without touching anything else.

6. **Calibration is weakest in the tail.** Overall ECE is 0.023, but the top bucket
   (p > 0.8) contains few observations, so probabilities there are less trustworthy
   than in the middle of the range. Treat "85% likely to be late" with more caution
   than "30% likely". The High Risk tier's deviation is +0.048 — larger than the other
   tiers, and larger than before the chronological split, because scoring a future
   period is genuinely harder.

7. **Small sample.** n = 1,500 with a 21.5% positive rate means ~323 delayed
   shipments, and the test window holds only 40 of them. Treat differences of a few
   points as noise.

8. **The `.pbix` file is not refreshed automatically.** Data source paths must be
   repointed and refreshed manually in Power BI Desktop after re-running the pipeline,
   and the new `Dim_Date` relationship must be created once by hand — see
   [Power BI setup](#powerbi-setup).

---

<a id="next-steps"></a>

## 🔮 Possible Next Steps

- Extend the data to 2–3 years so year-over-year seasonality and vendor concept drift become measurable, and add a scheduled retraining cadence justified by the observed drift rate.
- Replace the assumed 4:1 cost ratio with figures from actual SLA penalty schedules and expediting invoices, and re-derive the thresholds.
- Add a planner-capacity constraint to the cost model. The current thresholds flag ~37% of shipments for action; if the team cannot work that volume, the optimisation is solving the wrong problem.
- Deploy the scoring pipeline on Microsoft Fabric (Lakehouse + Notebook) for scheduled refresh instead of manual CSV runs.
- Consider a per-vendor hierarchical model, so a vendor with few shipments is shrunk toward the fleet average rather than scored on a noisy individual rating.

---

---

<a id="powerbi-setup"></a>

## 🔌 Power BI Setup After Adding `Dim_Date`

`Date_ID` is the first new column added to the fact table, so one relationship must be
created manually after refresh.

**1. Load `Dim_Date.csv`** as a new query alongside the existing tables.

**2. Set column data types** (Power Query → Transform → Data Type):

| Column | Type | Note |
|---|---|---|
| `Date_ID` | **Whole Number** | Join key, format `YYYYMMDD`. Must match the fact table's type exactly or the relationship will not be creatable. |
| `Full_Date` | **Date** | The actual date column — used for time intelligence. |
| `Year` | Whole Number | |
| `Quarter` | Text | Values `Q1`–`Q4`. |
| `Month` | Whole Number | 1–12. Use as the **Sort by Column** for `Month_Name`. |
| `Month_Name` | Text | Otherwise sorts alphabetically (April, August, …). |
| `Month_Year` | Text | `YYYY-MM`, already sorts correctly as text. |
| `Season` | Text | Winter / Spring / Summer / Autumn. |
| `Day_Of_Week` | Text | |
| `Is_Weekend` | **True/False** | |

**3. In `Fact_Shipments_with_ML`, confirm `Date_ID` is Whole Number.**

**4. Create the relationship** (Model view): drag `Fact_Shipments_with_ML[Date_ID]`
onto `Dim_Date[Date_ID]`. It should be **Many-to-One (\*:1)**, single cross-filter
direction, `Dim_Date` on the one side, and **Active**.

**5. Mark `Dim_Date` as a date table**: select the table → Table tools → *Mark as date
table* → pick `Full_Date`. Without this, DAX time-intelligence functions will not work
reliably.

**Suggested measures for a monthly trend page** (DAX not written here — these are the
ones worth adding):

- **Delay Rate %** — delayed shipments ÷ total shipments. The single most useful trend line; it should visibly rise in winter.
- **Avg Delay Days** — mean of `Actual_Delay_Days`, showing severity rather than frequency.
- **Predicted vs Actual gap** — average `Delay_Risk_Probability` minus the realised delay rate, by month. This is a *model-monitoring* measure: if the gap starts drifting away from zero in recent months, the model needs retraining. Arguably the most valuable measure on the page.
- **CO2 Tons per Shipment** — normalises emissions by volume, so the Q4 volume peak does not read as an efficiency problem.

All of these become sliceable by `Season`, `Quarter`, and `Month_Year` once the
relationship above is in place.

---
