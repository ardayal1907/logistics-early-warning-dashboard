"""Batch pipeline stages — a composition root.

Each module here owns one stage and exposes `main(settings)`. They may import
from `domain`, `infrastructure` and `services`; nothing imports back into them.

Three things changed when these moved out of `src/`:

1. **No `ROOT = Path(__file__).resolve().parent.parent`.** Once the project is
   installed as a wheel, `__file__` is inside site-packages, so that line
   pointed the pipeline at site-packages and the data was never found. It is
   the single line that prevented packaging. Directories now come from
   `Settings`, which reads the environment.

2. **Integrity guarantees are exceptions, not `assert`.** `python -O` strips
   `assert`, which turned `validate_star_schema` and `validate_output_schema`
   into empty functions and restored exactly the silent corruption their
   docstrings say they prevent. They raise `DataIntegrityError` now.

3. **Reporting is separated from computation.** Rendering lives in
   `report.py`; the computation paths log through `logging` so a scheduled run
   with no terminal still produces something an aggregator can filter on.
"""

__all__: list[str] = []
