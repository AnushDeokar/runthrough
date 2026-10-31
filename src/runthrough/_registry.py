"""Inspect AI entry point.

Inspect imports this module at startup (see ``[project.entry-points.inspect_ai]``
in pyproject.toml). Importing a module that defines ``@solver`` / ``@scorer``
functions registers them, so they can be referenced by name, e.g.
``inspect eval ... --solver runthrough/simulated_conversation``.

Add imports here as solvers and scorers are implemented.
"""
