import sys

from benchmark.registry import REPO  # noqa: F401  (puts src/ on sys.path)
from workstreams.black_physics.granite.pipeline import main

sys.exit(main())
