__doc__ = """
Rod constitutive model implementations.
"""
import numpy as np

from elastica._linalg import _batch_matvec
from elastica.utils import MaxDimension, Tolerance


class _LinearConstitutiveModelMixin:
    """
    Linear constitutive model mixin class for Cosserat rod.

        Attributes
        ----------
        rest_sigma: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type.
            Strain rest configuration defined on rod elements. Usually, rest strain is zero.
        rest_kappa: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type.
            Curvature of at rest configuration defined on rod elements. Usually, rest kappa is zero.
        shear_matrix: numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
            Shear/stretch matrix defined on rod elements.
        bend_matrix: numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
            Bending/twist matrix defined on rod voronoi domain.
    """

    # Needs
    # kappa, kappa0, strain (sigma), sigma0, B, S in specified formats
    # maybe use __init__ to initialize if not found?
    def __init__(
        self, n_elements, shear_matrix, bend_matrix, rest_lengths, *args, **kwargs
    ):
        """
        Parameters
        ----------
        n_elements: int
            The number of elements of the rod.
        shear_matrix: numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
            Shear/stretch matrix defined on rod elements.
        bend_matrix: numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
            Bending/twist matrix defined on rod voronoi domain.
        rest_lengths: numpy.ndarray
            1D (blocksize) array containing data with 'float' type.
            Rod element lengths at rest configuration.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        # set rest strains and curvature to be  zero at start
        # if found in kwargs modify (say for curved rod)
        self.rest_sigma = np.zeros((MaxDimension.value(), n_elements))
        self.rest_kappa = np.zeros((MaxDimension.value(), n_elements - 1))
        # sanity checks here
        # NOTE: assuming matrices to be diagonal here
        for i in range(0, MaxDimension.value()):
            assert shear_matrix[i, i] > Tolerance.atol()
            assert bend_matrix[i, i] > Tolerance.atol()

        self.shear_matrix = np.repeat(
            shear_matrix[:, :, np.newaxis], n_elements, axis=2
        )
        self.bend_matrix = np.repeat(bend_matrix[:, :, np.newaxis], n_elements, axis=2)

        # Compute bend matrix in Voronoi Domain
        self.bend_matrix = (
            self.bend_matrix[..., 1:] * rest_lengths[1:]
            + self.bend_matrix[..., :-1] * rest_lengths[0:-1]
        ) / (rest_lengths[1:] + rest_lengths[:-1])

    def _compute_internal_shear_stretch_stresses_from_model(self):
        """
        This method computes shear and stretch stresses on the rod elements.

        Returns
        -------

        """
        self._compute_shear_stretch_strains()  # concept : needs to compute sigma
        # TODO : the _batch_matvec kernel needs to depend on the representation of Shearmatrix
        self.internal_stress = _batch_matvec(
            self.shear_matrix, self.sigma - self.rest_sigma
        )

    def _compute_internal_bending_twist_stresses_from_model(self):
        """
        This method computes internal bending and twist stress on the rod voronoi.

        Returns
        -------

        """
        self._compute_bending_twist_strains()  # concept : needs to compute kappa
        # TODO : the _batch_matvec kernel needs to depend on the representation of Bendmatrix
        self.internal_couple = _batch_matvec(
            self.bend_matrix, self.kappa - self.rest_kappa
        )


class _LinearConstitutiveModelWithStrainRateMixin(_LinearConstitutiveModelMixin):
    """
    Linear constitutive model with strain rate mixin class for Cosserat rod.

        Attributes
        ----------
        rest_sigma: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type.
            Strain rest configuration defined on rod elements. Usually, rest strain is zero.
        rest_kappa: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type.
            Curvature of at rest configuration defined on rod elements. Usually, rest kappa is zero.
        shear_matrix: numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
            Shear/stretch matrix defined on rod elements.
        bend_matrix: numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
            Bending/twist matrix defined on rod voronoi domain.
        shear_rate_matrix: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type.
            Shear/stretch rate matrix defined on rod elements.
        bend_rate_matrix: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type.
            Bending/twist rate matrix defined on rod voronoi domain.
    """

    def __init__(
        self, n_elements, shear_matrix, bend_matrix, rest_lengths, *args, **kwargs
    ):

        _LinearConstitutiveModelMixin.__init__(
            self, n_elements, shear_matrix, bend_matrix, rest_lengths, *args, **kwargs
        )
        if "shear_rate_matrix" in kwargs.keys():
            self.shear_rate_matrix = np.repeat(
                kwargs["shear_rate_matrix"][:, :, np.newaxis], n_elements, axis=2
            )
        else:
            raise ValueError("shear rate matrix value missing!")
        if "bend_rate_matrix" in kwargs.keys():
            self.bend_rate_matrix = np.repeat(
                kwargs["bend_rate_matrix"][:, :, np.newaxis], n_elements, axis=2
            )
            # Compute bend rate matrix in Voronoi Domain
            self.bend_rate_matrix = (
                self.bend_rate_matrix[..., 1:] * rest_lengths[1:]
                + self.bend_rate_matrix[..., :-1] * rest_lengths[0:-1]
            ) / (rest_lengths[1:] + rest_lengths[:-1])
        else:
            raise ValueError("bend rate matrix value missing!")

    def _compute_internal_shear_stretch_stresses_from_model(self):

        # TODO : test this function
        # Calculates stress based purely on strain component
        super(
            _LinearConstitutiveModelWithStrainRateMixin, self
        )._compute_internal_shear_stretch_stresses_from_model()
        self._compute_shear_stretch_strains_rates()  # concept : needs to compute sigma_dot
        # TODO : the _batch_matvec kernel needs to depend on the representation of ShearStrainmatrix
        self.internal_stress += _batch_matvec(self.shear_rate_matrix, self.sigma_dot)

    def _compute_internal_bending_twist_stresses_from_model(self):

        # TODO : test this function
        # Calculates stress based purely on strain component
        super(
            _LinearConstitutiveModelWithStrainRateMixin, self
        )._compute_internal_bending_twist_stresses_from_model()
        self._compute_bending_twist_strain_rates()  # concept : needs to compute kappa rate
        # TODO : the _batch_matvec kernel needs to depend on the representation of Bendmatrix
        self.internal_couple += _batch_matvec(self.bend_rate_matrix, self.kappa_dot)
