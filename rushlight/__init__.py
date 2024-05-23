"""
Rushlight is an emission modeling package to generate synthetic observational data from numerical models
"""

from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass
