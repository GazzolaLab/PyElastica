from typing import Literal

CoarseDetectionType = Literal["hash_grid"]
FineDetectionType = Literal["sphere_sphere"]
BatchingType = Literal["union_find"]  # , "single_batch", "hybrid_batch"]
