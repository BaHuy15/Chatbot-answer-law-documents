from __future__ import annotations

import logging
import os
import typing as t
import warnings
from functools import lru_cache

import numpy as np

if t.TYPE_CHECKING:
    from ragas.metrics.base import Metric
    from ragas.testset.evolutions import Evolution

DEBUG_ENV_VAR = "RAGAS_DEBUG"


@lru_cache(maxsize=1)
def get_cache_dir() -> str:
    "get cache location"
    DEFAULT_XDG_CACHE_HOME = "~/.cache"
    xdg_cache = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
    default_ragas_cache = os.path.join(xdg_cache, "ragas")
    return os.path.expanduser(os.getenv("RAGAS_CACHE_HOME", default_ragas_cache))