"""
Configuration file for Nox.

Nox is a task automation tool that can be used to run tests, build
documentation, and perform other checks. Nox sessions are defined in
`noxfile.py`.

Running `nox` without arguments will run tests with the version of
Python that `nox` is installed under.

To invoke a nox session, enter the top-level directory of this
repository and run

    nox -s "<session>"

where <session> is replaced with the name of the session. The quotes are
necessary if the session name contains special characters.

To list available sessions, run `nox -l`.

Nox documentation: https://nox.thea.codes
"""

import nox

nox.options.default_venv_backend = "uv|virtualenv"


@nox.session(python=["3.12", "3.13"])
def tests(session):
    session.install(".", "pytest", "numpy", "scipy", "astropy", "yt", "sunpy", "scikit-image", "reproject>=0.12.0", "mpl-animators>=1.2.0")
    session.run("pytest", "tests", "--tb=short")
