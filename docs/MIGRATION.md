# Migration to a layered architecture

This document is referenced from `src/logistics/__init__.py`. It records the
order in which the pre-refactor flat modules in `src/` move into the
`logistics` package, and the exit criterion of each step.

Two rules govern the whole sequence:

1. **Every step leaves all three run contexts working**: `python src/<file>.py`,
   `pytest`, and `streamlit run src/app.py`. There is no "broken for a week"
   phase.
2. **Build the gate before changing the code behind it.** A refactor that cannot
   be verified is not a refactor, it is a rewrite with extra steps.

---

## Status

| | Done | Open |
|---|---|---|
| Packaging | `pyproject.toml`, `src/logistics/`, console scripts | — |
| Domain | `carbon`, `risk`, `enums`, `models` | `economics.py` (freight tariffs, SLA penalty) |
| Infrastructure | `fingerprint`, `model_repository`, `prediction_log` | `star_schema.py`, `route_keymap.py` |
| Services | `scoring.py` | `optimization_service.py` |
| Delivery | — | `ui/`, `api/` |
| Pipelines | — | `generate`, `etl`, `train`, `report` |
| Optimization | — | all of it |

`src/config.py`, `src/app.py`, `src/etl_star_schema.py`,
`src/generate_logistics_data.py` and `src/ml_delay_risk_pipeline.py` are the
pre-refactor codebase. They still work and are still the code that runs.

---

## Dependency rule

```
        domain/          imports nothing (stdlib + pydantic + logistics.errors)
           ^
    infrastructure/      may import domain; never services/ui/api/optimization
           ^
       services/         domain + infrastructure
           ^
   optimization/  api/  ui/  pipelines/  cli   <- delivery and composition roots
```

Four hard rules:

1. **Domain never looks outward.** No `pandas`, `sklearn`, `joblib`,
   `streamlit`, no filesystem access, no `logistics.settings` import.
2. **Only composition roots read `Settings`** (`cli.py`, `api/deps.py`,
   `ui/streamlit_app.py`, each `pipelines/*.main()`). A service receives
   resolved collaborators, not configuration.
3. **Frameworks live only in the delivery layer.** `streamlit`, `fastapi` and
   `argparse` may not be imported outside `ui/`, `api/` and `cli.py`.
4. **Contracts flow up, code flows down.** Thresholds, column order and category
   levels are always read from the artefact's metadata; no consumer hardcodes
   them.

Rules 1-3 are pinned by `tests/unit/test_layering.py` (step 1), an AST scan in
the same spirit as the existing `test_consumers_never_assign_a_shared_constant`.

### Known violations

| # | Violation | Evidence | Closed in |
|---|---|---|---|
| V1 | UI takes domain from the legacy flat module and binds artefact loading to the presentation framework | `app.py:36`, `app.py:60-63` | Step 3 |
| V2 | Path resolution lives in the presentation module, including a `Path.cwd()` branch | `app.py:46-57`, `:66-73` | Step 3 |
| V3 | Domain and infrastructure in one module | `config.py:102`+`:111-117` vs `:166-198` | Step 2 |
| V4 | `sys.path` injection, top-level `config`/`app` shadowing | `app.py:25`, `generate_logistics_data.py:46`, `ml_delay_risk_pipeline.py:75`, `tests/conftest.py:18` | Step 2 |
| V5 | Inner layer calls outer: the new CLI imports the legacy scripts | `cli.py:92,101,110` (deliberate, documented bridge) | Step 4 |
| V6 | Persistence in three places, no contract, platform-dependent bytes | `etl_star_schema.py:197-200`, `ml_delay_risk_pipeline.py:695`, `generate_logistics_data.py:467` | Step 0 / 4 |
| V7 | Integrity guarantees carried by bare `assert` (stripped by `python -O`) | `etl_star_schema.py:162-187`, `ml_delay_risk_pipeline.py:378-398` | Step 4 |
| V8 | Tests depend on a UI framework | `test_single_source_of_truth.py:17` -> `app.py:23` | Step 3 |
| V9 | The column contract is copied into three files | `ml_delay_risk_pipeline.py:389`, `tests/test_star_schema.py:14-19`, `etl_star_schema.py:48-51` | Step 6 |

---

## Step 0 — Byte determinism (half a day) — DO THIS BEFORE `git init`

**Why first.** `to_csv()` uses `os.linesep`, so the committed CSVs are CRLF
(verified). Git for Windows defaults to `core.autocrlf=true`, so the first
`git add` normalises them to LF — and `training_data_sha256` in the model
metadata, computed over CRLF bytes, becomes wrong at that instant. After that,
fixing it means retraining and recommitting.

1. Add `.gitattributes`: `*.csv -text`, `*.pkl binary`, `*.pbix binary`,
   `*.dax text eol=lf`, `* text=auto`.
2. Route the three writers through
   `logistics.infrastructure.fingerprint.write_csv_deterministically`.
3. Regenerate the six CSVs as LF and retrain so the hashes are rewritten.

**Done when** no CSV under `data/` contains `\r\n`; a new
`tests/contract/test_byte_determinism.py` asserts that, asserts two writes of
the same frame are byte-identical, and asserts the fingerprint matches the
metadata; `git add --renormalize .` leaves a clean status.

**Blocks** steps 5, 7 and the Windows CI matrix.

---

## Step 1 — Make the declared gates actually run (1 day)

`pyproject.toml` declares ruff, mypy strict, `fail_under = 85`,
`error::InconsistentVersionWarning` and `--strict-markers`. **CI installs
`requirements-dev.txt`, which contains only `pytest`.** None of those gates
currently execute.

1. `requirements-dev.txt` -> `-r requirements.txt` + `-e ".[dev]"`.
2. Delete `sys.path.insert` from `tests/conftest.py` (`pythonpath = ["src"]`
   already covers it) and fix its now-stale docstring.
3. `_require()` -> `pytest.fail` under `CI`, `pytest.skip` locally. 29 of the
   tests currently vanish silently if an artefact is missing, and the badge
   stays green.
4. Generate and commit `models/production_risk_model.pkl.sha256`
   (`write_checksum_sidecar`), add `!models/*.sha256` to `.gitignore`.
5. Extend `tests.yml`: `lint` -> `unit` (matrix ubuntu+windows, python 3.12.10)
   -> `contract` -> `bi-contract`, plus weekly `security` (`pip-audit`,
   `bandit`) and a nightly `e2e-rebuild`. `timeout-minutes: 15` on every job,
   workflow-level `concurrency`. Keep the existing `test` job name so branch
   protection keeps working.
6. Add `tests/unit/test_layering.py` and
   `tests/contract/test_requirements_mirror_pyproject.py`.

**Done when** all four gates are green in CI, `pytest -q` reports `0 skipped`,
and `load_bundle` no longer logs "No checksum recorded".

**Blocks** everything after it.

> Do **not** enable Dependabot for pip. An automatic `scikit-learn` bump
> produces exactly the silent prediction drift `requirements.txt:1-14` exists to
> prevent. Weekly `pip-audit` report + manual bump + mandatory retrain.

---

## Step 2 — `config.py` becomes a shim (half a day)

Replace the body of `src/config.py` with re-exports from
`logistics.domain.carbon`, `logistics.domain.risk` and
`logistics.infrastructure.fingerprint`. Delete the three `sys.path.insert`
lines and the three `# noqa: E402` comments, and empty the
`[tool.ruff.lint] exclude` list.

**Why this is safe:** the identity assertions in
`test_single_source_of_truth.py:34-51` (`gen.EMISSION_FACTOR is
config.EMISSION_FACTOR`) still hold, because a re-export binds the same object.
That is the technical reason the existing suite survives untouched.

Keep the DAX warning comment in the shim — `test_single_source_of_truth.py`
asserts the string `"DAX"` appears in `config.py`.

**Then delete `tests/test_migration_bridge.py`.** Its own docstring says so: once
`config.py` re-exports, that file only asserts a module agrees with itself.

**Done when** `grep -rn "sys.path.insert" src tests` and
`grep -rn "noqa: E402" src` both return nothing, `src/config.py` is ~30 lines,
and all three scripts plus the Streamlit app still run.

---

## Step 3 — `ui/streamlit_app.py` (1 day)

Move `app.py:96-313` into `logistics/ui/streamlit_app.py::main()`. `src/app.py`
becomes a permanent shim of at most 15 lines — Streamlit Cloud's entrypoint is a
file path, so it is never deleted — and **keeps its `if __name__ ==
"__main__"` guard** (Streamlit runs the main script as `__main__`, so the shim
works and `test_scripts_guard_execution_behind_main` stays green).

`@st.cache_resource` stays in the UI, but what it wraps becomes
`build_scoring_service(Settings.from_env())`, which knows nothing about
Streamlit. Delete `find_model_path` and `find_processed_dir` including the
`Path.cwd()` branch.

Derive the slider bounds from `service.numeric_ranges`. This closes the verified
mismatch: the Vendor_Rating slider runs `[2.5, 5.0]` while the model was trained
on `[1.71, 4.89]`, so the worst vendors — the ones an early-warning panel exists
to catch — cannot currently be entered at all.

**Regression gate:** `tests/contract/test_ui_matches_legacy_scoring.py` compares
`ScoringService.score_frame` against the legacy path over all 1,500 rows of
`Fact_Shipments_with_ML.csv` with `rtol=0`. The numbers must be bit-identical.

---

## Step 4 — `pipelines/` (2 days)

Move the three scripts into `logistics/pipelines/`. Their `main()` signatures
take `Settings`; the seams already exist (`load_raw_data(path=...)`,
`write_star_schema(out_dir=...)`, `load_star_schema(processed_dir=...)`,
`save_production_model(model_dir=...)` — every one of them is currently called
with no arguments).

Delete `ROOT = Path(__file__).resolve().parent.parent`. Once the package is
installed, that points at site-packages and the data is never found — it is the
line that actively prevents packaging.

Split `ml_delay_risk_pipeline.main()` (143 lines) into `run_evaluation`,
`run_scoring` and `run_training`, each returning a value, and add
`--stage {evaluate,score,train,all}`. Collect the 99 `print()` calls into
`pipelines/report.py::render_report()`; computation paths use `logger`.

Convert the 21 bare `assert` statements to `DataIntegrityError`
(`logistics.errors` already defines it). `python -O` strips `assert`, which
turns `validate_star_schema` and `validate_output_schema` into empty functions
and restores exactly the silent corruption their docstrings say they prevent.

Delete the three legacy imports in `cli.py` — that closes V5.

**Done when** `pip install .` + `LOGISTICS_PROCESSED_DIR=/tmp/out logistics-etl`
runs **without a repository checkout**, and `python -O -m logistics.pipelines.etl`
still enforces its integrity checks.

---

## Step 5 — End-to-end tests and a data contract (2 days)

`tests/e2e/test_full_pipeline.py` runs generate -> etl -> train in `tmp_path`
with `n_rows=120`. Today `main()` is never called by any test, and 16 of the ML
pipeline's 26 functions have no coverage at all — including
`score_walk_forward` and `scoring_quality`, which produce the headline metrics.

Add Pandera schemas under `contracts/`, deriving the categorical domains from
the enums rather than retyping them. Include the cross-column rule: the
coefficient of variation of `Distance_km` within an `(Origin, Destination)` pair
must be bounded.

> **That rule fails on today's data, and it is right to fail.** Verified:
> `Istanbul -> Denizli` carries distances from 67.9 km to 1,169.2 km across six
> shipments, because `generate_logistics_data.py:224` draws distance
> independently of the city pair. Fix: a 20x20 distance matrix with ±5% noise.

---

## Step 6 — Fix the star schema (3 days, the one genuinely breaking step)

**Grain.** `Dim_Route` currently holds 1,346 rows against 1,500 facts (verified):
1,204 routes are used exactly once and the maximum is 4. Weather and traffic are
properties of a shipment *event*, not of a route. Split into `Dim_Route`
(Origin, Destination) = 375 rows, `Dim_Vehicle` = 3, and a junk dimension
`Dim_Condition` (Weather x Traffic) = 12. Total 390 dimension rows.

**Durable keys.** `Route_ID` is assigned from row position
(`etl_star_schema.py:102`), so inserting one new combination renumbers all 1,346
(verified by simulation). Replace with `sha1(natural_key)[:10]` or an
append-only keymap. Incremental loading is structurally impossible until this
is fixed, and the failure is silent: old facts point at the same key, now
meaning a different route.

**`Dim_Parameters.csv`**, generated from the domain constants, so the DAX measure
reads `MAX(Dim_Parameters[Carbon_Price_Per_Ton])` instead of a literal `50`.

**Audit columns** `_batch_id`, `_loaded_at`, `_source_sha256`, `_schema_version`
plus `data/processed/_manifest.json`.

**Column contract** moves to `contracts/fact_shipments.v1.json`; the exact-equality
assert becomes "required columns present in order, extras allowed". Adding a
column is backward-compatible for Power BI but currently breaks the build.

Do this in the **same commit** as the PBIP/TMDL conversion, so the relationship
changes show up in a plain-text diff.

---

## Step 7 — API and Docker (2 days)

`logistics/api/`: `/health`, `/score`, `/score/batch`, `/model`. Multi-stage
`docker/Dockerfile` with `builder -> runtime-base -> {api, ui, trainer}`, base
image `python:3.12.10-slim-bookworm` (patch-pinned, because the model is a
pickle). Copy the 8.7 MB artefact **last** so it does not invalidate the
dependency layer, and run `load_bundle` as a build-time smoke test so a
pickle/sklearn mismatch fails the image build rather than the first request.

---

## Step 8 — The optimisation layer

See `docs/OPTIMIZATION.md`. Phase 1 is a 4,500-variable LP with zero binaries,
zero new dependencies and zero new data.

---

## Backward compatibility

Four contracts leave this system. None of them may break.

| Contract | Guarantee |
|---|---|
| `streamlit run src/app.py` | The path never changes; `app.py` becomes a permanent shim, keeping its `__main__` guard. |
| `requirements.txt` | Stays a real pin list — Streamlit Cloud does not run `pip install -e .`. Mirrored against `pyproject.toml` by a test rather than by hand. |
| `data/processed/*.csv` | Column names, order and grain. Only step 6 breaks it; `Dim_Route_Legacy.csv` is emitted for one release as a fallback. |
| Model metadata keys | Additive only. Removing a key is a major version bump. |

Deprecations get a two-release window in `logistics/compat.py`, each entry
carrying a `__removed_in__` version that a test enforces — so temporary bridges
cannot become permanent. `tests/test_migration_bridge.py` already applies this
pattern to itself.
