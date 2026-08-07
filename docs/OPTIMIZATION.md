# The optimisation layer

The system predicts (calibrated delay probability) and it calculates (CO2). It
does not yet **decide** anything. This document specifies the decision layer:
what it optimises, why the formulation is shaped the way it is, and — at least
as important — what it cannot honestly claim.

Every number below was recomputed from `data/processed/` and `data/raw/`.

---

## 1. The finding that shapes everything: the problem decomposes

Read `build_delays()` (`src/generate_logistics_data.py:285-294`). The delay
logit is

```
WEATHER_LOGIT[w] + TRAFFIC_LOGIT[t] + (3.25 - rating)*0.80*amp + ((d-625)/1200)*0.50
```

`Vehicle_Type` and `Weight_tons` **do not appear**. And `compute_co2_kg` never
takes a vendor. So on this data there is no constraint coupling the two
decisions:

| Possible coupling | Present? |
|---|---|
| Vendor-specific fleet (`FleetCap_jk`) | No — no such table exists |
| (vendor, vehicle) interacting tariff | No |
| `e_ik` depending on the vendor | No — `compute_co2_kg` has no vendor argument |
| `p_ij` depending on the vehicle | Not in the generator; **must be asserted against the model** (§6) |

```
BLOCK B — MODAL CHOICE            BLOCK A — VENDOR ASSIGNMENT
  4,500 continuous vars             37,500 continuous vars
  Pareto frontier, shadow price     service-level plan
  NO ML INVOLVED                    ALL of the ML lives here
```

This is a fact to exploit, not to hide. Phase 1 solves Block B alone. Packaging
a deterministic engineering calculation and a machine-learning result together
as "AI optimisation" would be a misrepresentation, and the honest split is also
the faster one.

---

## 2. What was rejected, and why

**Consolidation / green VRP — rejected on measurement.** The entire lever is the
fixed `BASE_EMISSION` term. Measured: 9,137.5 kg across all 1,500 shipments,
**0.74%** of the 1,233.1 t total. Realistic corridor pooling recovers a fraction:

| Pooling window | Legs eliminated | Saving | at $50/ton |
|---|---|---|---|
| 1 day | 5 | 42.9 kg | **$2/year** |
| 7 days | 48 | 375.1 kg | **$19/year** |
| 30 days | 222 | 1,677.6 kg | **$84/year** |

The distance geometry does not support routing either: within a single
(Origin, Destination) pair, distances span 645 km on average. There is no
`D[o][d]`.

**Monthly-bucket multi-objective — rejected on statistics.** Vendor-month cells
hold 5.1 shipments on average. A capacity ceiling estimated from ~5 observations
makes every step of an ε-sweep indistinguishable from noise. The ε-constraint
*method* is kept; the monthly bucket is not.

---

## 3. Block B — modal allocation (Phase 1)

**Variables.** `z_ik ∈ [0,1]`, shipment `i` carried by vehicle type `k`.
1,500 × 3 = 4,500. **Continuous, not binary.**

> Written with capacities in **trip counts** rather than tons, the assignment
> and cardinality constraints form a transportation polytope, which is totally
> unimodular — so the LP relaxation is integral and binaries are unnecessary.
> The single side constraint (V4) breaks TU, but a basic optimal solution can
> then have at most one fractional shipment in this block. Adding a **tonnage**
> capacity would break TU properly and turn this into a Generalized Assignment
> Problem (NP-hard); Phase 1 deliberately omits it.

**Objective** — carbon is deliberately *not* in it:

```
min  Σ_i Σ_k  f_ik · z_ik   −   δ · s / range_CO2        (δ = 1e-3, AUGMECON2)
```

Pricing carbon in the objective *and* bounding it in a constraint double-counts,
and it makes the dual uninterpretable: λ stops being "the marginal cost of the
next ton" and becomes "the premium on top of $50". Keep the objective in pure
currency; keep carbon in physical kg on the constraint. Only then can λ be
compared against `Settings.carbon_price_per_ton`.

**Constraints**

| # | Constraint | Rows | Why |
|---|---|---|---|
| V1 | `Σ_k z_ik = 1  ∀i` | 1,500 | The fact grain is one row per shipment; split loads are not in the contract. |
| V2 | `Σ_i z_ik ≤ Fleet_k  ∀k` | 3 | **The constraint that stops the answer being trivial.** Electric Semi dominates on both the per-ton-km factor (0.035) and the fixed term (1.5). With no ceiling the answer is "electrify everything", which is a capex decision, not an optimisation. |
| V3 | `z_ik = 0` where infeasible | column removal | Payload and range. **Not supported by the data** — see W4. |
| V4 | `Σ_ik e_ik z_ik + s = ε` | 1 | The ε-constraint. Its dual is the shadow carbon price. |

**Measured frontier**, fleet mix held at the status quo (846 / 311 / 343 — no
vehicle purchases):

| | CO2 | vs status quo |
|---|---|---|
| status quo (as shipped) | 1,233.1 t | — |
| optimal, **same fleet mix** | 880.6 t | **−28.6%** = 352.5 t = **$17.6k** at $50/ton |
| all-electric | 434.8 t | −64.7% (capex, not optimisation) |

Sweeping λ over freight tariffs, the status quo turns out to sit **inside** the
frontier — both objectives improve at once. At λ = 175 emissions fall 14.3%
while freight cost falls $1,051. Tested across four tariff scenarios, a
free-abatement region exists in all of them (2.5% to 28.6%). The *existence* of
that region is tariff-independent; only its *size* is not.

**The defensible headline:**

> With the current fleet mix and without buying a single vehicle, 13-18% of
> emissions can be cut at no increase in freight cost. Going to 28.6% carries an
> implied shadow carbon price of about $158/ton — three times the $50/ton the
> dashboard currently assumes.

And immediately beside it, in the same size type: **this figure exists because
vehicle assignment in the synthetic data is random. In a real fleet it will be
much smaller.** (See W3.)

---

## 4. Block A — vendor assignment (Phase 2)

**Variables.** `x_ij ∈ [0,1]`, 1,500 × 25 = 37,500.

**Objective.** `min Σ_ij x_ij · (f_ij + Π · p_ij)`

`Π · p_ij` is an expected value, and it is only currency if `p` is a genuine
probability. `logistics/domain/risk.py` already states the requirement: *an
expected value is only linear in a probability if the number really is a
probability.* This is where the project's calibration work is finally spent.

| # | Constraint | Rows | Why |
|---|---|---|---|
| A1 | `Σ_j x_ij = 1  ∀i` | 1,500 | Assignment. |
| A2 | `Σ_i x_ij ≤ Cap_j  ∀j` | 25 | **Does the real work.** The objective is monotone in `p_ij`, so without capacity everything goes to VEND-023 (rating 4.89, mean p 0.083, realised delay rate 0.0%). |
| A3 | `Σ_i x_ij ≥ L_j  ∀j` | 25 | **Exploration floor.** Optimising against the model starves low-rated vendors of volume, so the next training run has no fresh data on them and walk-forward scoring silently degrades. This constraint costs money and that cost must be budgeted explicitly. |
| A4 | `Σ_ij p_ij x_ij ≤ α·\|I\|` | 1 | **Global service level.** By linearity of expectation the left side is the expected number of delays — **no independence assumption required** (independence is needed for the variance, not the mean). Mean p is 0.2159, so α = 0.15 binds. |
| A5 | `Σ_i (p_ij − α_j) x_ij ≤ 0  ∀j` | 25 | Per-vendor service level. A4 alone lets a pile-up on one bad vendor be masked by good ones. Looks like a ratio; is linear as written. |

---

## 5. Where the parameters come from

**Already in the star schema:** shipment ids, `Weight_tons`, `Distance_km`,
`Weather_Condition` and `Traffic_Density` (via `Route_ID`), `Vendor_Rating`,
dates, `Settings.carbon_price_per_ton`, and observed fleet/vendor volumes as a
baseline for `Fleet_k` / `Cap_j`.

**Computed, no new assumptions:**

- `e_ik` ← `logistics.domain.carbon.compute_co2_kg_many`, 1,500 × 3 = 4,500
  values.
  > **Do not use the `CO2_Emission_kg` column.** The generator adds N(1, 0.03)
  > noise, and the column only exists for the *actual* vehicle. Verified: stored
  > total 1,232.6 t vs deterministic recomputation 1,233.1 t. Counterfactual
  > vehicles require calling the formula.
- `p_ij` ← `ScoringService.score_frame` over a counterfactual grid, injecting
  `Vendor_Rating` from `Dim_Vendor`.
  > **Do not use the `Delay_Risk_Probability` column** — it is the probability of
  > the *actual* assignment only.

**Not in the data at all, and must come from the business:** vendor tariffs
`f_ij`, vehicle costs `f_ik`, the SLA penalty `Π`, real capacities, payload and
range limits, and the policy targets α, α_j, L_j. Their home is
`src/logistics/domain/economics.py`, under the same discipline as `carbon.py`
and overridable through `LOGISTICS_*`. `risk.expected_delay_cost(probability,
penalty_per_delay, intervention_cost)` already has the right signature — the
slot exists, the values do not.

### Schema impact — breaking and easy to miss

`ROUTE_COLUMNS` includes `Vehicle_Type`, so **changing a shipment's vehicle
changes its `Route_ID`**, and the proposed combination may not be among the
existing 1,346 rows (13,680 are possible). Do not write optimiser output back
into the fact table. Emit a separate
`Fact_Assignment_Plan (Shipment_ID, Vendor_ID_opt, Vehicle_Type_opt,
Route_ID_opt, p_opt, CO2_opt, scenario_id)` and extend `Dim_Route` with the
missing combinations. Step 6 of the migration (route re-graining) makes this
much cleaner and should come first.

---

## 6. Mandatory guard: decision invariance

Permutation importance (`docs/METHODOLOGY.md`) measures `Vehicle_Type` at
0.007 ± 0.016 — statistically indistinguishable from zero, and the generator
confirms it has no causal role. But the model still consumes it as a feature,
and **a solver mines every spurious signal in its objective to exhaustion**: it
will discover that switching vehicles "reduces risk" and prevent no delays
whatsoever.

A near-zero permutation importance does not bound the per-row spread, so assert
the spread directly, then marginalise:

```python
by = grid.groupby(["Shipment_ID", "Vendor_ID"])["Delay_Risk_Probability"]
leak = (by.max() - by.min()).max()
assert leak < 0.05, f"Vehicle_Type leaks into p (spread {leak:.3f})"
p = by.mean()          # p_ij is independent of k by construction
```

**This assertion is the first task of Phase 1 and has not yet been run** — it
needs the model loaded. If it fails, either drop the feature and retrain, or
accept the dependency and formulate with `y_ijk`. Continuing quietly is not an
option.

Also fire `ShipmentFeatures.check_extrapolation` on every `(i,j)` in the
solution: the optimiser pushes the covariate distribution away from the training
distribution by construction. Out-of-range assignments go to a human review
queue, not into the plan.

### 6a. v1 result: the assertion was run, and it failed

| statistic | measured | bound |
|---|---|---|
| worst per-shipment spread | **0.6196** | 0.05 |
| group means across vehicle types | 0.027 | — |

Twelve times the bound, worst case `SHP-00518`. The group-mean version — the
weaker check — passed at 0.027, which is exactly the trap this section warned
about: *a near-zero permutation importance does not bound the per-row spread*.

`logistics-optimize` therefore refused by default, exit 1, `DecisionInvarianceError`.

Rebuilding the data did not rescue it. Under the v2 generator (city-pair distance
matrix, dispatched rather than random vehicle assignment) the spread moved only
from 0.6196 to **0.5669**. That is the diagnosis: the number was never about the
data. `generate.build_delays` does not read `Vehicle_Type` at all, so the true
effect is zero by construction, and what the statistic measures is a forest
answering an out-of-distribution counterfactual with noise.

### 6b. v2 formulation — `y_ijk` on calibrated probabilities

**Decision taken: keep `Vehicle_Type` in the model, reformulate the optimiser.**
The alternative — drop the feature and retrain — was rejected because the feature
is legitimately predictive of the *dispatch*, and removing it would hide the
confounding rather than handle it.

The v1 objective assumed the modal choice does not move delay risk, which is why
it needed the invariance guard to police that assumption. v2 stops assuming and
starts measuring:

```
min  sum_ik [ f_ik + sla_penalty * p_ik ] y_ik  -  delta * s / range_CO2
s.t. V1  sum_k y_ik = 1                        (per shipment)
     V2  sum_i y_ik <= Fleet_k                 (per vehicle)
     V4  sum_ik e_ik y_ik + s = epsilon        (emission ceiling)
```

V1, V2, V4 and the AUGMECON2 slack are unchanged from §3. What is new is `p_ik`
and the SLA term it multiplies — the optimiser can now trade a cheaper vehicle
against a likelier delay explicitly, instead of being forbidden from noticing the
trade exists.

**`p_ik` is calibrated, not extrapolated.** Per vehicle type, a Platt sigmoid with
a **shared slope** and a per-vehicle intercept, fitted on the shipments that
actually ran on that vehicle against what actually happened to them:

```
p_ik = sigmoid( a * logit(s_i) + b_k )
```

`s_i` is the model's score for shipment *i* **as it actually ran**. The forest is
never asked the counterfactual; the vehicle only shifts the level by `b_k`, and
the difference between two `b_k` *is* the measured vehicle effect, in log-odds.

Three attempts were needed, and the failures are the argument for the final form:

| calibration | worst spread |
|---|---|
| none — raw counterfactual scores (v1) | 0.5669 |
| isotonic, fitted per vehicle | 1.0000 |
| sigmoid, per-vehicle slope **and** intercept | 0.9957 |
| sigmoid, shared slope + per-vehicle intercept, applied to `s_i` | **0.0460** ✅ |

Isotonic is a step function: two vehicles whose raw scores differ by a hair land
on opposite sides of a jump and come out 0 versus 1. Per-vehicle slopes are
subtler but no better — the fits gave 3.88, 5.55 and 2.96 on 848, 287 and 365
rows, and a steeper slope is more extreme in the tails, so the curves fan apart
at high scores. Both failures give the vehicle more freedom than the data
supports and it fills the space with noise. The shared slope removes exactly that
freedom and keeps the one that carries the question.

Measured level shifts, with Diesel Truck as reference:

| vehicle | n | observed delay rate | level shift |
|---|---|---|---|
| Diesel Truck | 848 | 0.2017 | 0.000 (ref) |
| Electric Semi | 287 | 0.1916 | +0.090 log-odds |
| Hybrid Van | 365 | 0.1945 | +0.184 log-odds |

Shared slope 3.775. Calibrated means reproduce the observed rates to four
decimals, which is the check that the fit is doing its job.

**The guard does not disappear — it moves.** `check_calibrated_vehicle_effect`
applies the same 0.05 bound and the same worst-per-shipment statistic to `p_ik`.
It now passes at 0.0460 (worst `SHP-00560`, mean spread 0.0051). A failure here
would mean the fleet really does differ enough that modal allocation and delay
risk cannot be separated — the thing §6 was reaching for all along. The v1
statistic is still computed and logged alongside, for comparison.

**What this does not do.** It does not make the model causal. The v2 generator
correlates vehicle assignment with distance, weight and vendor, so part of any
residual difference is confounding. Calibration on observational subsets cannot
separate that. It bounds how far the optimiser can run with a number; it does not
license a causal reading of it.

### 6c. v2 results on the rebuilt data

| plan | CO₂ | freight | vs status quo |
|---|---|---|---|
| status quo | 1497.0 t | $1,179,224 | — |
| min-cost | 1212.3 t | $1,079,277 | **−19.0%**, both axes improve |
| greenest | 1033.3 t | $1,123,031 | **−31.0%** |

Expected late shipments under the min-cost plan: 296.4 of 1,500 — the quantity
the SLA term trades against. 0 fractional shipments.

The abatement rose from v1's −28.6% to −31.0%. That is not an improvement in the
method; it is the v2 dispatch rule putting long, heavy legs on diesel, which
concentrates emissions where the optimiser can act on them. **This is synthetic
data and the headroom is the headroom in a generator's dispatch rule.** Quote it
as a demonstration of the method, never as a saving an operator can bank.

---

## 7. Solver and scale

`pyproject.toml` already pins `pulp==3.4.2` and `highspy==1.12.0` under the
`optimization` extra. **No new dependency is required.** Do not switch to CBC
(HiGHS is already pinned) and do not move to Pyomo yet — PuLP exposes LP duals
via `constraint.pi` and supports `changeRHS()` for an ε-sweep without rebuilding.
Pyomo earns its place in Phase 3, where the chance constraint needs SOC.

| Block | Variables | Rows | Expected time |
|---|---|---|---|
| B (vehicle) | 4,500 continuous | 1,504 | < 1 s |
| A (vendor) | 37,500 continuous | 1,551 | 1-5 s |
| ε-sweep, 20-30 points (B only) | — | — | < 30 s |
| `p_ij` grid scoring (once, cached) | 112,500 rows | — | 1-3 min |

Cache the probability panel to parquet and bind it to the artefact via
`compute_data_fingerprint`: if the model is retrained, the panel is stale.

**Deployment mode is a rolling horizon.** 1,500 shipments over 53 weeks is 28.3
per week — 700 variables, milliseconds. Solving all 1,500 at once is a
*backtest*, not an operating policy; you cannot reassign the past. Carry monthly
capacity as a running budget so `Cap_j` still binds at a weekly horizon.

**A MILP has no duals.** Report the marginal abatement cost by finite
difference: `MAC_j = (f*(ε_{j+1}) − f*(ε_j)) / (ε_j − ε_{j+1})`. In Phase 1 the
variables are continuous anyway, so the LP dual and the finite difference
coincide — a second dividend of the TU formulation.

---

## 8. Honest weaknesses

**W1 — Half the objective is not in the data.** `f_ij`, `f_ik` and `Π` are
assumptions; the only real monetary parameter is $50/ton. The `Π/f` ratio, which
determines the optimum, has never been measured. A single "$X saved" claim is
unfalsifiable and therefore worthless. Ship the Pareto frontier and a tornado
chart. Phase 1 is anchored on the physical axis (tons of CO2) for this reason.

**W2 — The capacity parameters are invented and they do all the work.** The
objective is monotone in `p_ij`, so Block A is determined entirely by `Cap_j`.
The observed 45-74 range is a by-product of `rng.choice(VENDOR_IDS)`, not a real
capacity. The model's most decisive parameter is its least evidenced.

**W3 — The 28.6% measures the generator's randomness, not fleet inefficiency.**
`generate_logistics_data.py:250` draws the vehicle independently of weight and
distance, so the status quo is literally a random assignment and cannot be
optimal for anything. In a real 3PL, vehicle choice already correlates with
ton-km and the free-abatement region is far smaller. Say
"vehicle assignment in this dataset is random; removing that randomness yields
28.6%" — never "we saved 28.6%".

**W4 — The data contains physically impossible combinations, and the optimiser
will exploit exactly those.** Measured: 295 of 343 Hybrid Van shipments (86%)
exceed 3.5 t, up to 24.97 t; 182 of 311 Electric Semi legs exceed 500 km, up to
1,196.9 km. Without V3 the abatement figures are fiction — and V3's limits are
not in the data either. Imposing a realistic 3.5 t van limit makes 86% of
historical assignments infeasible, which means part of the "saving" is just
correcting a generator defect. The clean exit is to fix the vehicle taxonomy,
which means regenerating data and retraining — the fingerprint mechanism will
force that anyway.

**W5 — "ML feeds the optimiser" is only half true.** Vehicle type has exactly
zero effect on delay. Block B is pure deterministic engineering. State plainly
that Phase 1 uses no ML at all.

**W6 — Weather and traffic are hindsight.** `Dim_Route.Weather_Condition` is
realised weather. Tactical assignment happens days in advance, when only a
forecast exists. Scoring counterfactuals against realised weather inflates the
backtest. Label the result an **upper bound**.

**W7 — A vendor is one scalar.** 25 vendors collapse onto a single rating axis,
and the ETL averages away within-vendor variation. 23 distinct ratings across 25
vendors means two pairs the model cannot tell apart. "Assignment" degenerates
into "sort by rating, fill to capacity".

**W8 — Goodhart.** A3 mitigates the feedback loop but costs money. The
fingerprint mechanism catches data/model drift; it does not catch policy drift.

---

## 9. Invalidation conditions

| # | If | Then |
|---|---|---|
| G1 | the leakage assertion fails (`max_k p − min_k p ≥ 0.05`) | the blocks do not decompose, `y_ijk` becomes mandatory, and the optimiser is mining noise |
| G2 | `Fleet_k` is not actually binding | Block B is trivial ("electrify everything"); the layer is theatre |
| G3 | **real data replaces synthetic** | the most serious one. Counterfactual validity rests entirely on `rng.choice(VENDOR_IDS)` being uniform. Real planners assign *strategically* — hard loads to good vendors — so `Vendor_Rating` absorbs the selection effect and the optimiser **systematically overstates** the gain from reassignment. Nothing in the current pipeline detects this. Propensity weighting or a doubly-robust estimator becomes mandatory |
| G4 | carbon stays at $50/ton | carbon is ~$41/shipment against $800-1,500 of freight, 3-5%. It flips no decision in a weighted-sum objective — which is precisely why the ε-constraint is mandatory rather than optional |
| G5 | Phase 3 CVaR assumes independence | independent Bernoulli is true here *by construction* (σ = 14.1). Real delays are strongly correlated through shared weather, and σ grows several-fold. The risk-aware variant is most misleading exactly where it is most needed |
| G6 | you report a tail number | only 154 of 1,500 shipments are High Risk, and the metadata already warns that reliability drops above p > 0.8. ECE 0.023 is an *average*. Measure tail calibration separately before any CVaR figure goes to management |

---

## 10. Phasing

**Phase 0 (half a day, before any code).** Run the leakage assertion (§6). Add
`Dim_Parameters` to the ETL — scenario-based carbon pricing makes the hardcoded
`* 50` in `measures.dax` not merely stale but *inconsistent across scenarios*,
and `tests/test_dax_constants.py` pins a single value.

**Phase 1 (works today).** Block B only, fleet mix fixed at the status quo.
4,500 continuous variables, 20-30 ε points, PuLP + HiGHS. Output:
`Fact_Assignment_Plan`, the Pareto frontier, the MAC curve and the shadow carbon
price. No new dependency, no new data, no ML. Enable the `logistics-optimize`
console script. Keep the model-building functions pure and separable from the
solver so they can be tested without one.

**Phase 2 (needs three numbers).** Block A, given `Π`, `f_ij` and `Cap_j`. `Π`
alone is enough to start: with equal vendor tariffs the problem reduces to
"assignment under a risk budget" and A4/A5 stay meaningful. Blocks still solved
separately.

**Phase 3 (the first phase that earns a MILP).** Needs
`Dim_Vendor_Fleet (Vendor_ID, Vehicle_Type, Capacity)`. That table couples the
blocks for real: `y_ijk`, `x_ij = Σ_k y_ijk`, `z_ik = Σ_j y_ijk`,
`Σ_i y_ijk ≤ FleetCap_jk`. TU breaks, binaries return, and the solver is finally
justified. Also: tonnage capacity, chance constraints / CVaR (Pyomo + SOC), and
delay *severity* (`Π = C_SLA + c_day · E[days | delay]`; the data supports it —
delayed shipments average 1.82 days and the generator's `lam` already depends on
weather, traffic and rating).
