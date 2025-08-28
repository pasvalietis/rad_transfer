import nox


supported_python_versions = ("3.12", "3.13")

@nox.session
def tests(session: nox.Session):
    session.install(".[tests]")
    session.run("pytest", "--tb=short")