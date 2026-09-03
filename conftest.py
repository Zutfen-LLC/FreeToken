"""Repo-root conftest: put the repo root on sys.path for tests that import
``benchmarks.inferswarm_*`` packages (the historical runs executed on the
editable-installed main tree where CWD made this implicit; worktree-based
gate runs need it explicitly).  ``freetoken`` itself resolves via
PYTHONPATH=python or the editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
