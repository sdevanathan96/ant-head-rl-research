"""Ant foraging environments: a grid world and a free (continuous) world."""
from .core import ACTION_NAMES, N_ACTIONS, ForageConfig, ForageEnv, gini
from .free import FreeConfig, FreeForage
from .grid import GridConfig, GridForage


def make_env(world: str, seed: int | None = None, **overrides) -> ForageEnv:
    """``make_env("grid", n_ants=3, observe_teammates=False)`` → a configured env."""
    if world == "grid":
        return GridForage(GridConfig(**overrides), seed=seed)
    if world == "free":
        return FreeForage(FreeConfig(**overrides), seed=seed)
    raise ValueError(f"unknown world {world!r} (expected 'grid' or 'free')")


__all__ = ["ACTION_NAMES", "N_ACTIONS", "ForageConfig", "ForageEnv", "FreeConfig", "FreeForage",
           "GridConfig", "GridForage", "gini", "make_env"]
