class _elastica_numba:
    """The purpose is to throw deprecation error to people previously
    using _elastica_numba module. Please remove this after v0.3.
    """

    raise ImportError(
        "The module _elastica_numba is moved to main implementation. Please import the modules without _elastica_numba."
    )
