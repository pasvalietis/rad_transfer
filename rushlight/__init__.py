"""
Rushlight is an emission modeling package to generate synthetic observational data from numerical models
"""

# Checks if rushlight package is installed; always throws PackageNotFoundError?
from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

# Make an implicit call to config to ensure directories are established
# from rushlight import config
