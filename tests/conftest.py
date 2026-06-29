"""Pytest configuration: make the scripts/ package importable from any test.

Every test imports modules from ../scripts. Adding the directory here (once,
for the whole session) means individual test files run correctly in isolation
and regardless of collection order.
"""
import os
import sys

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
