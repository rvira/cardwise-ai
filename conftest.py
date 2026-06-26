"""Make the project root importable so tests can `import src.*` the same way
scripts run with `PYTHONPATH=.`. Pytest auto-loads this before collecting tests."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
