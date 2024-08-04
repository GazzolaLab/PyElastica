# TODO: implement more tests based on functionality

from elasticapp._PyCosseratRods import CosseratRod


def test_attributes():
    assert {
        "acceleration_collection",
        "alpha_collection",
        "dilatation",
        "director_collection",
        "dissipation_constant_for_forces",
        "dissipation_constant_for_torques",
        "external_forces",
        "external_torques",
        "internal_forces",
        "internal_torques",
        "inv_mass_second_moment_of_inertia",
        "lengths",
        "loc",
        "mass",
        "mass_second_moment_of_inertia",
        "material",
        "n_elems",
        "omega_collection",
        "position_collection",
        "radius",
        "rest_lengths",
        "rest_voronoi_lengths",
        "tangents",
        "velocity_collection",
        "volume",
        "voronoi_dilatation",
    }.issubset(dir(CosseratRod))
