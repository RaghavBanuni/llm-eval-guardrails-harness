"""Console entry point so ``python -m llmeval`` works as well as ``python -m llmeval.cli``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
