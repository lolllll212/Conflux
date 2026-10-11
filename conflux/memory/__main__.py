"""Entry point for ``python -m conflux.memory`` (see :mod:`conflux.memory.cli`)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
