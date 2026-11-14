"""Convenience alias package for ``pip install conflux-state``.

Both ``import conflux`` (canonical) and ``import conflux_state`` work seamlessly
and re-export the exact same runtime classes and version.
"""

from conflux import *  # noqa: F401,F403
from conflux import __all__, __version__  # noqa: F401
