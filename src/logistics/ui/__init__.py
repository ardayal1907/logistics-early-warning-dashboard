"""Presentation layer.

Streamlit is imported here and nowhere else. `logistics.ui` may import from
`domain`, `infrastructure` and `services`; nothing imports back into it.

Kept as a separate optional extra (`pip install .[ui]`) so that a nightly
ETL/training container does not pull in tornado, altair, pyarrow and protobuf,
and so that running the test suite does not require a UI framework.
"""

__all__: list[str] = []
