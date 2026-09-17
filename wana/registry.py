"""Explicit named factories, without global plugin discovery or inheritance."""

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], T]] = {}

    def register(self, name: str, factory: Callable[[], T]) -> None:
        if name in self._factories:
            raise ValueError(f"already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str) -> T:
        if name not in self._factories:
            raise ValueError(f"unknown component: {name}")
        return self._factories[name]()
