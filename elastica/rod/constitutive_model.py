__doc__ = """
Rod constitutive model mixins but classes here does not uploaded
"""
import numpy as np

from elastica._linalg import _batch_matvec
from elastica.utils import MaxDimension, Tolerance


class _LinearConstitutiveModelMixin:
    __doc__ = """
    Linear Constitutive model for Cosserat Rods
    """

    # Needs
    # kappa, kappa0, strain (sigma), sigma0, B, S in specified formats
    # maybe use __init__ to initialize if not found?
    def __init__(
        self, n_elements, shear_matrix, bend_matrix, rest_lengths, *args, **kwargs
    ):
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
        Linear force functional
        Operates on
        S : (3,3,n) tensor and sigma (3,n)

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
        Linear force functional
        Operates on
        B : (3,3,n) tensor and curvature kappa (3,n)

        Returns
        -------

        """
        self._compute_bending_twist_strains()  # concept : needs to compute kappa
        # TODO : the _batch_matvec kernel needs to depend on the representation of Bendmatrix
        self.internal_couple = _batch_matvec(
            self.bend_matrix, self.kappa - self.rest_kappa
        )


class _LinearConstitutiveModelWithStrainRateMixin(_LinearConstitutiveModelMixin):
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
        """
        Linear force functional
        Operates on
        S : (3,3,n) tensor and sigma (3,n)

        Returns
        -------

        """
        # TODO : test this function
        # Calculates stress based purely on strain component
        super(
            _LinearConstitutiveModelWithStrainRateMixin, self
        )._compute_internal_shear_stretch_stresses_from_model()
        self._compute_shear_stretch_strains_rates()  # concept : needs to compute sigma_dot
        # TODO : the _batch_matvec kernel needs to depend on the representation of ShearStrainmatrix
        self.internal_stress += _batch_matvec(self.shear_rate_matrix, self.sigma_dot)

    def _compute_internal_bending_twist_stresses_from_model(self):
        """
        Linear force functional
        Operates on
        B : (3,3,n) tensor and curvature kappa (3,n)

        Returns
        -------

        """
        # TODO : test this function
        # Calculates stress based purely on strain component
        super(
            _LinearConstitutiveModelWithStrainRateMixin, self
        )._compute_internal_bending_twist_stresses_from_model()
        self._compute_bending_twist_strain_rates()  # concept : needs to compute kappa rate
        # TODO : the _batch_matvec kernel needs to depend on the representation of Bendmatrix
        self.internal_couple += _batch_matvec(self.bend_rate_matrix, self.kappa_dot)
