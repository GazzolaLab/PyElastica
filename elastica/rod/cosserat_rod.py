__doc__ = """ Rod base classes and implementation details that need to be hidden from the user"""
__all__ = ["CosseratRod"]


from elastica import IMPORT_NUMBA

if IMPORT_NUMBA:
    from elastica._elastica_numba._rod._cosserat_rod import CosseratRod
else:
    from elastica._elastica_numpy._rod._cosserat_rod import CosseratRod
