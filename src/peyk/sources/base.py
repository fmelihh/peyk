"""Common interface for model catalog sources."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import ModelVariant


@runtime_checkable
class Source(Protocol):
    """A provider of model variants.

    Implementations must be best-effort: on any network / parsing failure they
    return an empty list rather than raising, so the pipeline degrades to
    whatever sources did succeed (at minimum the curated catalog).
    """

    name: str

    def fetch(self) -> list[ModelVariant]:  # pragma: no cover - protocol
        ...
