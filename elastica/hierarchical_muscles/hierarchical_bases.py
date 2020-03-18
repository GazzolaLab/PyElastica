__doc__ = """Hierarchical bases functions"""
__all__ = [
    "Union",
    "SplineHierarchy",
    "SpatiallyInvariantSplineHierarchy",
    "SplineHierarchyMapper",
    "SpatiallyInvariantSplineHierarchyMapper",
    "SplineHierarchySegments",
    "Gaussian",
    "TruncatedCosine",
    "Filter",
    "ScalingFilter",
]
from elastica import IMPORT_NUMBA

if IMPORT_NUMBA:
    from elastica._elastica_numba._hierarchical_muscles._hierarchical_bases import (
        Union,
        SplineHierarchy,
        SpatiallyInvariantSplineHierarchy,
        SplineHierarchyMapper,
        SpatiallyInvariantSplineHierarchyMapper,
        SplineHierarchySegments,
        Gaussian,
        TruncatedCosine,
        Filter,
        ScalingFilter,
    )
else:
    from elastica._elastica_numpy._hierarchical_muscles._hierarchical_bases import (
        Union,
        SplineHierarchy,
        SpatiallyInvariantSplineHierarchy,
        SplineHierarchyMapper,
        SpatiallyInvariantSplineHierarchyMapper,
        SplineHierarchySegments,
        Gaussian,
        TruncatedCosine,
        Filter,
        ScalingFilter,
    )


# class Union(MutableSequence):
#     def __init__(self, *args):
#         self._bases = []
#         self._bases.extend(args)
#
#     def __len__(self):
#         return len(self._bases)
#
#     def __getitem__(self, idx):
#         return self._bases[idx]
#
#     def __delitem__(self, idx):
#         del self._bases[idx]
#
#     def __setitem__(self, idx, basis):
#         self._bases[idx] = basis
#
#     def insert(self, idx, basis):
#         self._bases.insert(idx, basis)
#
#
# class SplineHierarchy:
#     def __init__(self, basis_function_union: Union, spacing="equal", scaling_factor=3):
#         self.__scaling_factor = int(scaling_factor)  # increase in levels 3
#
#         assert self.__scaling_factor > 1
#         assert type(self.__scaling_factor) == int
#
#         # levels in hierarchy
#         # 0 is the top level, 1 is the next level and so on..
#         self.n_levels = len(basis_function_union)
#
#         # (from geometric progression)
#         self.n_bases = (self.__scaling_factor ** (self.n_levels) - 1) // (
#             self.__scaling_factor - 1
#         )
#
#         self.bases = [None] * self.n_bases
#         self.bases_origins = [None] * self.n_bases
#
#         """
#         start = 0
#         for level in range(self.n_levels):
#             stop = self.__scaling_factor ** level
#             level_spacing = 1.0 / float(stop)
#             for i in range(start, start + stop):
#                 self.bases[i] = basis_function_union[level]
#                 if spacing == 'equal':
#                     self.bases_origins[i] = (0.5 + i - start) * level_spacing
#             start = start + stop
#         """
#
#         for level in range(self.n_levels):
#             start = self.basis_start_idx(level)
#             stop = self.n_bases_at_level(level)
#             level_spacing = 1.0 / float(stop)
#             for i in range(start, start + stop):
#                 self.bases[i] = basis_function_union[level]
#                 if spacing == "equal":
#                     self.bases_origins[i] = (0.5 + i - start) * level_spacing
#
#     def basis_start_idx(self, basis_level):
#         return (self.__scaling_factor ** basis_level - 1) // (self.__scaling_factor - 1)
#
#     def n_bases_at_level(self, basis_level):
#         return self.__scaling_factor ** basis_level
#
#     def apply_filter(self, basis_level, filter_cls, *Filterargs, **Filterkwargs):
#         basis_level_start = self.basis_start_idx(basis_level)
#         basis_level_stop = basis_level_start + self.n_bases_at_level(basis_level)
#         self.bases[basis_level_start:basis_level_stop] = [
#             filter_cls(x, *Filterargs, **Filterkwargs)
#             for x in self.bases[basis_level_start:basis_level_stop]
#         ]
#
#     def __call__(self, centerline, activation):
#         output = 0.0 * centerline
#         for i in range(self.n_bases):
#             output += activation[i] * self.bases[i](centerline - self.bases_origins[i])
#         return output
#
#
# class SpatiallyInvariantSplineHierarchy(SplineHierarchy):
#     def __init__(self, basis_function_union: Union, scaling_factor, spacing="equal"):
#         super().__init__(basis_function_union, spacing, scaling_factor)
#         self.not_initialized = True
#
#     def __call__(self, centerline, activation):
#         """
#
#         :param centerline: Should always be between 0 and 1
#         :param activation:  Should always be between -1 and 1
#         :return:
#         """
#         if self.not_initialized:
#             # initialized by passing in centerlines but with differnt
#             # activation signals for each of the n_bases and store it
#             # cached_outputs
#             # assume centerline is a 1D rod
#             self.cached_outputs = np.zeros((self.n_bases, centerline.shape[0]))
#
#             for i in range(self.n_bases):
#                 local_activation = np.zeros((self.n_bases,))
#                 local_activation[i] = 1.0
#                 self.cached_outputs[i, ...] = super().__call__(
#                     centerline, local_activation
#                 )
#
#             self.not_initialized = False
#
#         # do einsum of activation and cached_outputs
#
#         # return np.einsum("i,ij->j", activation, self.cached_outputs)
#         return self.activation_cached_output_mult(activation, self.cached_outputs)
#
#     @staticmethod
#     @numba.njit()
#     def activation_cached_output_mult(activation, cached_outputs):
#         dim = activation.shape[0]
#         blocksize = cached_outputs.shape[1]
#         output_vector = np.zeros((blocksize))
#
#         for i in range(dim):
#             for k in range(blocksize):
#                 output_vector[k] += activation[i] * cached_outputs[i, k]
#
#         return output_vector
#
#
# class SplineHierarchyMapper:
#     """
#     Maps spline onto rod and returns the result
#     """
#
#     def __init__(self, hierarchy: SplineHierarchy, centerline_start_stop: tuple):
#         self.hierarchical_basis = hierarchy
#         self.start = centerline_start_stop[0]
#         self.stop = centerline_start_stop[1]
#         assert self.start <= self.stop
#
#     @property
#     def n_bases(self):
#         return self.hierarchical_basis.n_bases
#
#     def get_indices_and_starts(self, non_dimensional_cumulative_length):
#         idx = np.searchsorted(
#             non_dimensional_cumulative_length, (self.start, self.stop)
#         )
#         start = non_dimensional_cumulative_length[idx[0]]
#         end = non_dimensional_cumulative_length[idx[1]]
#         non_dimensional_centerline = (
#             non_dimensional_cumulative_length[idx[0] : idx[1]] - start
#         ) / (end - start)
#         return (idx[0], idx[1], non_dimensional_centerline)
#
#     def __call__(self, non_dimensional_cum_length, activation, output):
#         # cumulative_lengths = np.cumsum(rod_length)
#         # non_dimensional_cum_length = cumulative_lengths / cumulative_lengths[-1]
#         # output = 0.0 * non_dimensional_cum_length
#         start_idx, stop_idx, non_dimensional_centerline = self.get_indices_and_starts(
#             non_dimensional_cum_length
#         )
#         output[start_idx:stop_idx] += self.hierarchical_basis(
#             non_dimensional_centerline, activation
#         )
#
#
# class SpatiallyInvariantSplineHierarchyMapper(SplineHierarchyMapper):
#     def __init__(
#         self, hierarchy: SpatiallyInvariantSplineHierarchy, centerline_start_stop: tuple
#     ):
#         assert type(hierarchy) == SpatiallyInvariantSplineHierarchy
#         super().__init__(hierarchy, centerline_start_stop)
#         self.not_initialized = True
#
#     def __call__(self, non_dimensional_cum_length, activation, output):
#         if self.not_initialized:
#             (
#                 self.start_idx,
#                 self.stop_idx,
#                 self.non_dimensional_centerline,
#             ) = self.get_indices_and_starts(non_dimensional_cum_length)
#             self.not_initialized = False
#
#         # output = 0.0 * non_dimensional_cum_length
#         output[self.start_idx : self.stop_idx] += self.hierarchical_basis(
#             self.non_dimensional_centerline, activation
#         )
#
#
# class SplineHierarchySegments:
#     def __init__(self, *hierarchy_mappers):
#         self.n_segments = len(hierarchy_mappers)
#         self.mappers = hierarchy_mappers
#
#         # list of (start_idx, stop_idx) pairs denoting indices of activation
#         self.activation_start_stop = [None] * self.n_segments
#         start = 0
#         for idx, mapper in enumerate(hierarchy_mappers):
#             stop = mapper.n_bases + start
#             self.activation_start_stop[idx] = (start, stop)
#             start = stop
#
#     def __call__(self, rod_lengths, activation):
#
#         # cumulative_lengths = np.cumsum(rod_lengths)
#         # non_dimensional_cum_length = cumulative_lengths / cumulative_lengths[-1]
#         # output = 0.0 * rod_lengths
#
#         output, non_dimensional_cum_length = self.compute_non_dimensional_length(
#             rod_lengths
#         )
#
#         for idx, mapper in enumerate(self.mappers):
#             activation_start, activation_stop = self.activation_start_stop[idx]
#             local_activation = activation[activation_start:activation_stop]
#             mapper(non_dimensional_cum_length, local_activation, output)
#
#         return output
#
#     @staticmethod
#     @numba.njit()
#     def compute_non_dimensional_length(rod_lengths):
#         cumulative_lengths = np.cumsum(rod_lengths)
#         non_dimensional_cum_length = cumulative_lengths / cumulative_lengths[-1]
#         output = 0.0 * rod_lengths
#         return output, non_dimensional_cum_length
#
#
# """
# class SplineHierarchySegments:
#     def __init__(self, hierarchies: list, hierarchy_start_stops: list):
#         assert (len(hierarchies) == len(hierarchy_start_stops))
#
#         self.n_segments = len(hierarchies)
#         self.spline_hierarchies = hierarchies
#         # list of (start_idx, stop_idx) pairs denoting indices of activation
#         self.activation_start_stop = [None] * self.n_segments
#         start = 0
#         for idx, hierarchy in enumerate(hierarchies):
#             stop = hierarchy.n_bases + start
#             self.activation_start_stop[idx] = (start, stop)
#             start = stop
#
#         # list of (start, stop) pairs denoting lengths
#         self.start_stop = hierarchy_start_stops
#
#     def get_indices_and_starts(self, i_segment, non_dimensional_cumulative_length):
#         idx = np.searchsorted(non_dimensional_cumulative_length, self.start_stop[i_segment])
#         start = non_dimensional_cumulative_length[idx[0]]
#         end = non_dimensional_cumulative_length[idx[1]]
#         return (idx[0], idx[1], start, end)
#
#     def __call__(self, rod_lengths, activation):
#         cumulative_lengths = np.cumsum(rod_lengths)
#         non_dimensional_cum_length = cumulative_lengths / cumulative_lengths[-1]
#         output = 0.0 * rod_lengths
#         for i in range(self.n_segments):
#             start_idx, stop_idx, start, end = self.get_indices_and_starts(i, non_dimensional_cum_length)
#             non_dimensional_centerline = (non_dimensional_cum_length[start_idx: stop_idx] - start) / (end - start)
#             activation_start, activation_stop = self.activation_start_stop[i]
#             local_activation = activation[activation_start:activation_stop]
#             output[start_idx: stop_idx] += self.spline_hierarchies[i](non_dimensional_centerline, local_activation)
#
#         return output
#
#
# class SpatiallyInvariantSplineHierarchySegments(SplineHierarchySegments):
#     '''
#     Caches the spline and multiplies the output by activation
#     assumes separability between activation and spatial progile
#     '''
#     def __init__(self, hierarchies: list, hierarchy_start_stops: list):
#         super().__init__(hierarchies, hierarchy_start_stops)
#         for hierarchy in hierarchies:
#             assert (type(hierarchy) == SpatiallyInvariantSplineHierarchy)
#         self.not_initialized = True
#
#
#     def __call__(self, rod_lengths, activation):
#         cumulative_lengths = np.cumsum(rod_lengths)
#         non_dimensional_cum_length = cumulative_lengths / cumulative_lengths[-1]
#         output = 0.0 * rod_lengths
#
#         if self.not_initialized:
#             self.start_idx = [None] * self.n_segments
#             self.stop_idx = [None] * self.n_segments
#             self.non_dimensional_centerline = [None] * self.n_segments
#             # initialize and cache whatever variables that can be cached
#             for i in range(self.n_segments):
#                 start_idx, stop_idx, start, end = self.get_indices_and_starts(i, non_dimensional_cum_length)
#                 self.start_idx[i] = start_idx
#                 self.stop_idx[i] = stop_idx
#                 self.non_dimensional_centerline[i] = (non_dimensional_cum_length[start_idx: stop_idx] - start) / (end - start)
#
#             self.not_initialized = False
#
#         for i in range(self.n_segments):
#             activation_start, activation_stop = self.activation_start_stop[i]
#             local_activation = activation[activation_start:activation_stop]
#             output[self.start_idx[i]: self.stop_idx[i]] += self.spline_hierarchies[i](self.non_dimensional_centerline[i], local_activation)
#
#         return output
# """
#
#
# """
#     def __init__(self, n_segments: int, hierarchies: list, hierarchy_start_stops, rod_length):
#         assert (len(hierarchy_start_stops) == n_segments)
#         assert (len(hierarchies) == n_segments)
#
#         self.n_segments = n_segments
#         self.spline_hierarchies = hierarchies
#         # list of (start_idx, stop_idx) pairs denoting indices of activation
#         self.activation_start_stop = [None] * n_segments
#
#         start = 0
#         for idx, hierarchy in enumerate(hierarchies):
#             stop = hierarchy.n_bases + start
#             self.activation_start_stop[idx] = (start, stop)
#             start = stop
#
#         self.total_n_bases = self.activation_start_stop[-1][1]
#         self.start_stop = hierarchy_start_stops  # list of (start, stop) pairs denoting lengths
# """
#
#
# """
#
# Some canonical compact basis functions
#
# """
#
#
# class Gaussian:
#     def __init__(self, epsilon):
#
#         self.epsilon = epsilon * 0.5
#
#     def __call__(self, r):
#         return np.exp(-((r / self.epsilon) ** 2))
#
#
# class TruncatedCosine:
#     def __init__(self, epsilon):
#         self.epsilon = epsilon
#
#     def __call__(self, r):
#         output = 0.0 * r
#         idx = r ** 2 < self.epsilon ** 2
#         output[idx] = 0.5 * (1.0 + np.cos(np.pi * r[idx] / self.epsilon))
#         return output
#
#
# """ Can be functools objects too!"""
#
#
# # https://en.wikipedia.org/wiki/Bump_function
# def Bump(epsilon):
#     def bump_impl(r, eps):
#
#         output = 0.0 * r
#         idx = r ** 2 < eps ** 2
#         output[idx] = np.exp(-1.0 / (1.0 - (r / eps) ** 2))
#         return output
#
#     from functools import partial
#
#     return partial(bump_impl, eps=epsilon)
#
#
# """
#
# Some examples of filters that can be applied
#
# """
#
#
# class Filter:
#     pass
#
#
# class ScalingFilter(Filter):
#     def __init__(self, wraps, scale):
#         self.wrapped_callable = wraps
#         self.scale = scale
#
#     def __call__(self, *args, **kwargs):
#         return self.scale * self.wrapped_callable(*args, **kwargs)
