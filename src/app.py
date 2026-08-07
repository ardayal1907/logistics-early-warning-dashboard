"""Streamlit entrypoint — a permanent shim.

Run with:   streamlit run src/app.py

Streamlit Cloud's entrypoint is a FILE PATH, not an importable name, so this
file is never deleted or moved: the path is one of the four external contracts
in docs/MIGRATION.md. The application itself lives in
`logistics.ui.streamlit_app`.

The `if __name__ == "__main__"` guard stays. Streamlit runs the main script
with `__name__ == "__main__"`, so this launches the app, while importing the
module does nothing.
"""

from logistics.ui.streamlit_app import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
