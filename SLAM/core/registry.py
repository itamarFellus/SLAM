from __future__ import annotations

from typing import Callable, Dict, Type, Any


_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register(name: str):
    def decorator(factory: Callable[..., Any]):
        _REGISTRY[name] = factory
        return factory
    return decorator


def make(name: str, **kwargs: Any) -> Any:
    if name not in _REGISTRY:
        raise KeyError(f"No component registered under name '{name}'")
    return _REGISTRY[name](**kwargs)


def available() -> Dict[str, Callable[..., Any]]:
    return dict(_REGISTRY)


