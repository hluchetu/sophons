from __future__ import annotations

import lazy_loader as lazy


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "chroma": ["ChromaVectorStore"],
        "in_memory": ["InMemoryVectorStore"],
        "numpy": ["NumPyVectorStore"],
    },
)
