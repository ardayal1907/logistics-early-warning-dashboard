"""HTTP delivery layer.

FastAPI is imported here and nowhere else. `logistics.api` may import from
`domain`, `infrastructure` and `services`; nothing imports back into it.

Optional extra (`pip install .[api]`), for the same reason as `ui`: a batch
training container has no business carrying a web framework.
"""

__all__: list[str] = []
