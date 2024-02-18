"""To store muscle geometries and properties from Kinetic Research Group"""

import numpy as np
import os
from examples.ArtificialMusclesCases.muscle.muscle_utils import *


class Dict2Class(object):
    def __init__(self, my_dict, **kwargs):

        for key in my_dict:
            setattr(self, key, my_dict[key])


# data from Kinetic Research Group and internet
nylon_material_properties_dict = {
    "density": 1090,  # kg/m^3
    "youngs_modulus": 1925,  # MPa at 25 degrees
    "shear_modulus": 1925 / (2 * (1 + 0.5)),  # MPa at 25 degrees
    "thermal_expansion_coefficient": 8e-5,  # 1/°C
    "friction_coefficient": 5e-1,
    "velocity_damping_coefficient": 1e5,
    "youngs_modulus_coefficients": [
        2.26758447119,
        -0.00996645676489,
        0.0000323219668553,
        -3.8696662364 * 1e-7,
        -6.3964732027 * 1e-7,
        2.0149695202 * 1e-8,
        -2.5861167614 * 1e-10,
        1.680136396 * 1e-12,
        -5.4956153529 * 1e-15,
        7.2138065668 * 1e-18,
    ],  # coefficients of youngs modulus interpolation polynomial
}

# validated
class Liuyang_monocoil:
    def __init__(self) -> None:
        # from tuning PureContraction
        muscle_sim_settings_dict = {
            "n_elem_per_coil": 12,  # at least 24 for stable coil beyond 30 seconds
            "nu": 3e-3,
            "actuation_start_temperature": 25,  # °C
            "actuation_end_temperature": 120,  # °C
            "actuation_kappa_change": 0.02443781917600426,  # result from tuning to no_strain_actuation_force
            "E_scale": 1e6,
        }

        # data from Kinetic Research Group
        muscle_geometry_dict = {
            "fiber_radius": 0.37e-3,  # m
            "start_radius_list": [0.77 * 1.06e-3],  # m
            "taper_slope_list": [0],
            "start_position": np.array([0.0, 0.0, 0.0]),
            "direction": np.array([0.0, 0.0, 1.0]),
            "normal": np.array([1.0, 0.0, 0.0]),
            "angular_offset": np.pi / 2,  # rad
            "muscle_length": 25e-3,  # m
            "turns_per_length_list": [0.732e3],  # turns/m
            "initial_link_per_fiber_length": 0.487125e3,  # turns/m
            "CCW_list": [False],
            "n_ply_per_coil_level": [1],
        }
        self.sim_settings = Dict2Class(muscle_sim_settings_dict)
        self.geometry = Dict2Class(muscle_geometry_dict)
        self.properties = Dict2Class(nylon_material_properties_dict)
        self.name = "Liuyang_monocoil"
        self.strain_experimental = np.array(
            [
                0,
                0.02,
                0.04,
                0.06,
                0.08,
                0.1,
                0.12,
                0.14,
                0.16,
                0.18,
                0.2,
                0.22,
                0.24,
                0.26,
                0.28,
                0.3,
                0.32,
                0.34,
            ]
        )
        self.passive_force_experimental = np.array(
            [
                0,
                0.029488725,
                0.1003074,
                0.263201672,
                0.483295536,
                0.700134061,
                0.789446376,
                1.220160091,
                1.248365025,
                1.464562583,
                1.693720112,
                1.657772707,
                1.957271513,
                2.091979301,
                2.152205745,
                2.365574753,
                2.688491025,
                3.009691711,
            ]
        )
        self.passive_force_experimental = np.linspace(0, 20, 7)
        self.total_force_experimental = np.array(
            [
                2.008489373,
                2.158127796,
                2.282132745,
                2.463859078,
                2.628392363,
                2.776646399,
                2.948111006,
                3.126448301,
                3.293623327,
                3.432654265,
                3.57477033,
                3.727534782,
                3.866610229,
                4.083184061,
                4.301790654,
                4.383303385,
                4.585479343,
                4.761292441,
            ]
        )


class Liuyang_supercoil:
    def __init__(self) -> None:
        # from tuning PureContraction
        muscle_sim_settings_dict = {
            "n_elem_per_coil": 48,  # at least 24 for stable coil beyond 30 seconds
            "nu": 3e-3,
            "actuation_start_temperature": 25,  # °C
            "actuation_end_temperature": 150,  # °C
            "actuation_kappa_change": 0.1,  # result from tuning to data (-1,+1)
            "E_scale": 1e6,
        }
        fiber_radius = 0.37e-3
        ply_fiber_ratio = 2 / np.sqrt(3)
        ply_radius = ply_fiber_ratio * fiber_radius
        external_coil_radius = 3.38e-3 / 2
        coil_radius = external_coil_radius - fiber_radius - ply_radius

        # data from Kinetic Research Group
        muscle_geometry_dict = {
            "fiber_radius": fiber_radius,  # m
            "start_radius_list": [coil_radius, ply_radius],  # m
            "taper_slope_list": [0, 0],
            "start_position": np.array([0.0, 0.0, 0.0]),
            "direction": np.array([0.0, 0.0, 1.0]),
            "normal": np.array([1.0, 0.0, 0.0]),
            "angular_offset": np.pi / 2,  # rad
            "muscle_length": 23.04e-3,  # m
            "turns_per_length_list": [
                0.3462e3,
                500,
            ],  # turns/m [from data, 1805.054988131072]
            "initial_link_per_fiber_length": 0.2534e3,  # turns/m
            "CCW_list": [False, False],
            "n_ply_per_coil_level": [1, 3],
        }
        self.sim_settings = Dict2Class(muscle_sim_settings_dict)
        self.geometry = Dict2Class(muscle_geometry_dict)
        self.properties = Dict2Class(nylon_material_properties_dict)
        self.name = "Liuyang_supercoil"
        self.strain_experimental = np.array(
            [
                0,
                0.021701389,
                0.043402778,
                0.065104167,
                0.086805556,
                0.108506944,
                0.130208333,
                0.151909722,
                0.173611111,
                0.1953125,
                0.217013889,
                0.238715278,
                0.260416667,
                0.282118056,
                0.303819444,
                0.325520833,
                0.347222222,
                0.368923611,
                0.390625,
                0.412326389,
                0.434027778,
            ]
        )
        self.passive_force_experimental = np.array(
            [
                0,
                -0.154967758,
                0.100453819,
                0.276043017,
                0.385782032,
                0.495135468,
                0.75147924,
                1.050978661,
                1.284402531,
                1.412206327,
                1.865518986,
                2.33284984,
                2.773099845,
                2.928076664,
                3.454961448,
                3.795152497,
                4.141022647,
                4.688836321,
                4.453549325,
                5.468513473,
                6.016068264,
            ]
        )
        self.total_force_experimental = np.array(
            [
                4.191093659,
                4.509094541,
                4.861036265,
                5.166190402,
                5.539548424,
                6.114387784,
                6.284487844,
                6.728777692,
                7.235194867,
                7.626931844,
                7.926203945,
                8.432747028,
                8.897345235,
                9.44426316,
                10.13296067,
                10.83752863,
                11.44055124,
                12.25497973,
                12.65930362,
                13.07486844,
                13.38003541,
            ]
        )


class Jeongmin_monocoil:
    def __init__(self) -> None:
        # from tuning PureContraction
        muscle_sim_settings_dict = {
            "n_elem_per_coil": 12,  # at least 24 for stable coil beyond 30 seconds
            "nu": 3e-3,
            "actuation_start_temperature": 25,  # °C
            "actuation_end_temperature": 150,  # °C
            "actuation_kappa_change": 0.01653146246002244,  # result from tuning to no_strain_actuation_force
            "E_scale": 1e6,
        }

        # data from Kinetic Research Group
        muscle_geometry_dict = muscle_geometry_dict = {
            "fiber_radius": 0.5e-3,  # m
            "start_radius_list": [1.25e-3],  # m
            "taper_slope_list": [0],
            "start_position": np.array([0.0, 0.0, 0.0]),
            "direction": np.array([0.0, 0.0, 1.0]),
            "normal": np.array([1.0, 0.0, 0.0]),
            "angular_offset": np.pi / 2,  # rad
            "muscle_length": 20e-3,  # m
            "turns_per_length_list": [NotImplementedError],  # turns/m
            "initial_link_per_fiber_length": NotImplementedError,  # turns/m
            "CCW_list": [False],
            "n_ply_per_coil_level": [1],
        }
        self.sim_settings = Dict2Class(muscle_sim_settings_dict)
        self.geometry = Dict2Class(muscle_geometry_dict)
        self.properties = Dict2Class(nylon_material_properties_dict)
        self.name = "Jeongmin_monocoil"
        self.strain_experimental = np.array(
            [
                0,
                0.025,
                0.05,
                0.075,
                0.1,
                0.125,
                0.15,
                0.175,
                0.2,
                0.225,
                0.25,
                0.275,
                0.3,
                0.325,
                0.35,
                0.375,
                0.4,
                0.425,
                0.45,
                0.475,
                0.5,
            ]
        )
        self.passive_force_experimental = np.array(
            [
                0,
                0.206348321,
                0.30345,
                0.49281,
                0.68102,
                0.81816,
                0.88204,
                1.1355,
                1.31214,
                1.47532,
                1.65478,
                1.8325,
                2.03345,
                2.23634,
                2.46626,
                2.69126,
                2.83849,
                3.05689,
                3.22737,
                3.4833,
                3.60965,
            ]
        )
        self.total_force_experimental = np.array(
            [
                2.791219172,
                3.03263832,
                3.22230481,
                3.438914732,
                3.63796145,
                3.873032582,
                4.073191126,
                4.326347,
                4.537140137,
                4.7717514,
                5.019232887,
                5.231982276,
                5.488551447,
                5.750055811,
                5.960712285,
                6.232450646,
                6.472476152,
                6.751255257,
                6.996955194,
                7.282620496,
                7.53415179,
            ]
        )


class Jeongmin_supercoil_ply:
    def __init__(self) -> None:
        # from tuning PureContraction
        muscle_sim_settings_dict = {
            "n_elem_per_coil": 36,  # at least 24 for stable coil beyond 30 seconds
            "nu": 3e-3,
            "actuation_start_temperature": 25,  # °C
            "actuation_end_temperature": 150,  # °C
            "actuation_kappa_change": 0.008955318869719914,  # result from tuning to data
            "E_scale": 1e6,
        }

        # data from Kinetic Research Group
        muscle_geometry_dict = {
            "fiber_radius": 0.235e-3,  # m
            "start_radius_list": [1e-14, 0.271e-3],  # m
            "taper_slope_list": [0, 0],
            "start_position": np.array([0.0, 0.0, 0.0]),
            "direction": np.array([0.0, 0.0, 1.0]),
            "normal": np.array([1.0, 0.0, 0.0]),
            "angular_offset": np.pi / 2,  # rad
            "muscle_length": 23.04e-3,  # m
            "turns_per_length_list": [187, 0],  # turns/m
            "initial_link_per_fiber_length": 0,  # turns/m
            "CCW_list": [False, False],
            "n_ply_per_coil_level": [1, 3],
        }
        self.sim_settings = Dict2Class(muscle_sim_settings_dict)
        self.geometry = Dict2Class(muscle_geometry_dict)
        self.properties = Dict2Class(nylon_material_properties_dict)
        self.name = "Jeongmin_supercoil_ply"
        self.strain_experimental = np.array(
            [
                0,
                0.025,
                0.05,
                0.075,
                0.1,
                0.125,
                0.15,
                0.175,
                0.2,
                0.225,
                0.25,
                0.275,
                0.3,
                0.325,
                0.35,
                0.375,
                0.4,
                0.425,
                0.45,
                0.475,
                0.5,
            ]
        )
        self.passive_force_experimental = np.array(
            [
                0,
                0.007600581,
                0.034586405,
                0.072674498,
                0.115746413,
                0.154386122,
                0.231933607,
                0.313594936,
                0.357142142,
                0.316202606,
                0.403065982,
                0.531555746,
                0.657329432,
                0.565453472,
                0.709946966,
                0.826932056,
                0.95674734,
                1.115898178,
                1.198010685,
                1.362933693,
                1.413026819,
            ]
        )
        self.total_force_experimental = np.array(
            [
                1.91348616,
                2.019153588,
                2.222713352,
                2.42758298,
                2.69602473,
                2.840115851,
                3.007196055,
                3.056482527,
                3.314821906,
                3.110344076,
                3.293236986,
                3.702780884,
                3.80130241,
                3.580861806,
                3.890330955,
                4.23299255,
                4.559182593,
                4.937360917,
                4.968784216,
                5.290097442,
                5.569362061,
            ]
        )


class Jeongmin_supercoil:
    def __init__(self) -> None:
        # from tuning PureContraction
        muscle_sim_settings_dict = {
            "n_elem_per_coil": 36,  # at least 24 for stable coil beyond 30 seconds
            "nu": 3e-3,
            "actuation_start_temperature": 25,  # °C
            "actuation_end_temperature": 150,  # °C
            "actuation_kappa_change": 0.02695193464380536,  # result from tuning to data
            "E_scale": 1e6,
        }

        supercoils_per_coil_length = 1 / (6e-3)  # 1 coil per 8 mm (guess)
        # supercoils_per_coil_length = 0./length_scale
        k_m = 90
        # k_s = supercoils_per_coil_length*helix_length_per_coil(k_m,2.3e-3)*k_m #supercoils per unit height
        k_s = supercoils_per_coil_length * helix_length_per_coil(k_m, 2.3e-3) * k_m

        # data from Kinetic Research Group
        muscle_geometry_dict = {
            "fiber_radius": 0.235e-3,  # m
            "start_radius_list": [1.15e-3, 0.5e-3],  # m
            "taper_slope_list": [0, 0],
            "start_position": np.array([0.0, 0.0, 0.0]),
            "direction": np.array([0.0, 0.0, 1.0]),
            "normal": np.array([1.0, 0.0, 0.0]),
            "angular_offset": np.pi / 2,  # rad
            "muscle_length": 20.0e-3,  # m
            "turns_per_length_list": [
                k_m,
                k_s,
            ],  # turns/m [could be incorrect, from data]
            "initial_link_per_fiber_length": 187 + 78 + 90,  # turns/m
            "CCW_list": [False, False],
            "n_ply_per_coil_level": [1, 3],
        }
        self.sim_settings = Dict2Class(muscle_sim_settings_dict)
        self.geometry = Dict2Class(muscle_geometry_dict)
        self.properties = Dict2Class(nylon_material_properties_dict)
        self.name = "Jeongmin_supercoil"
        self.strain_experimental = np.array(
            [
                0,
                0.025,
                0.05,
                0.075,
                0.1,
                0.125,
                0.15,
                0.175,
                0.2,
                0.225,
                0.25,
                0.275,
                0.3,
                0.325,
                0.35,
                0.375,
                0.4,
                0.425,
                0.45,
                0.475,
                0.5,
            ]
        )
        self.passive_force_experimental = np.array(
            [
                0,
                0.007600581,
                0.034586405,
                0.072674498,
                0.115746413,
                0.154386122,
                0.231933607,
                0.313594936,
                0.357142142,
                0.316202606,
                0.403065982,
                0.531555746,
                0.657329432,
                0.565453472,
                0.709946966,
                0.826932056,
                0.95674734,
                1.115898178,
                1.198010685,
                1.362933693,
                1.413026819,
            ]
        )
        self.total_force_experimental = np.array(
            [
                1.91348616,
                2.019153588,
                2.222713352,
                2.42758298,
                2.69602473,
                2.840115851,
                3.007196055,
                3.056482527,
                3.314821906,
                3.110344076,
                3.293236986,
                3.702780884,
                3.80130241,
                3.580861806,
                3.890330955,
                4.23299255,
                4.559182593,
                4.937360917,
                4.968784216,
                5.290097442,
                5.569362061,
            ]
        )


class mock_supercoil:
    def __init__(self) -> None:
        # from tuning PureContraction
        muscle_sim_settings_dict = {
            "n_elem_per_coil": 36,  # at least 24 for stable coil beyond 30 seconds
            "nu": 3e-3,
            "actuation_start_temperature": 25,  # °C
            "actuation_end_temperature": 120,  # °C
            "actuation_kappa_change": 0.017102412067874193
            / 81,  # result from tuning to data
            "E_scale": 1e6,
        }

        coil_radius = 0.92 * 1.06e-3
        ply_radius = 0.37e-3
        supercoils_per_coil_length = (
            647.9653947061116 / 3
        )  # twist from Liuyang monocoil
        k_m = 0.732e3
        # k_s = supercoils_per_coil_length*helix_length_per_coil(k_m,2.3e-3)*k_m #supercoils per unit height
        k_s = (
            supercoils_per_coil_length * helix_length_per_coil(k_m, coil_radius) * k_m
        )  # (supercoils/coil_length)(coil_length/coil)(coils/height) = supercoils/height
        print(k_s)

        # data from Kinetic Research Group
        muscle_geometry_dict = {
            "fiber_radius": ply_radius * (np.sqrt(3) / 2),  # m
            "start_radius_list": [coil_radius, ply_radius],  # m
            "taper_slope_list": [0, 0],
            "start_position": np.array([0.0, 0.0, 0.0]),
            "direction": np.array([0.0, 0.0, 1.0]),
            "normal": np.array([1.0, 0.0, 0.0]),
            "angular_offset": np.pi / 2,  # rad
            "muscle_length": 25.0e-3,  # m
            "turns_per_length_list": [
                k_m,
                k_s,
            ],  # turns/m [could be incorrect, from data]
            "initial_link_per_fiber_length": 0,  # turns/m
            "CCW_list": [False, False],
            "n_ply_per_coil_level": [1, 3],
        }
        self.sim_settings = Dict2Class(muscle_sim_settings_dict)
        self.geometry = Dict2Class(muscle_geometry_dict)
        self.properties = Dict2Class(nylon_material_properties_dict)
        self.name = "mock_supercoil"
        self.strain_experimental = np.array(
            [
                0,
                0.02,
                0.04,
                0.06,
                0.08,
                0.1,
                0.12,
                0.14,
                0.16,
                0.18,
                0.2,
                0.22,
                0.24,
                0.26,
                0.28,
                0.3,
                0.32,
                0.34,
            ]
        )
        self.passive_force_experimental = np.array(
            [
                0,
                0.029488725,
                0.1003074,
                0.263201672,
                0.483295536,
                0.700134061,
                0.789446376,
                1.220160091,
                1.248365025,
                1.464562583,
                1.693720112,
                1.657772707,
                1.957271513,
                2.091979301,
                2.152205745,
                2.365574753,
                2.688491025,
                3.009691711,
            ]
        )
        self.total_force_experimental = np.array(
            [
                2.008489373,
                2.158127796,
                2.282132745,
                2.463859078,
                2.628392363,
                2.776646399,
                2.948111006,
                3.126448301,
                3.293623327,
                3.432654265,
                3.57477033,
                3.727534782,
                3.866610229,
                4.083184061,
                4.301790654,
                4.383303385,
                4.585479343,
                4.761292441,
            ]
        )


class Samuel_monocoil:
    def __init__(self) -> None:
        # from tuning PureContraction
        muscle_sim_settings_dict = {
            "n_elem_per_coil": 12,
            "nu": 3e-3,
            "actuation_start_temperature": 25,  # °C
            "actuation_end_temperature": 150,  # °C
            "actuation_kappa_change": 0.06774102614226328,  # result from tuning to no_strain_actuation_force
            "E_scale": 1e6,
        }

        # data from Kinetic Research Group
        coils_per_length = 10e3 / 16  # 46/(61.03e-3)
        fiber_diameter = 0.91e-3  # 0.8e-3
        external_coil_diameter = 2.43e-3
        coil_radius = (external_coil_diameter - fiber_diameter) / 2
        # coil_radius = 1.2*external_coil_diameter/2

        initial_link_per_fiber_length = 366  # turns/meter
        muscle_geometry_dict = {
            "fiber_radius": fiber_diameter / 2,  # m
            "start_radius_list": [coil_radius],  # m
            "taper_slope_list": [0],
            "start_position": np.array([0.0, 0.0, 0.0]),
            "direction": np.array([0.0, 0.0, 1.0]),
            "normal": np.array([1.0, 0.0, 0.0]),
            "angular_offset": 0,  # rad
            "muscle_length": 20e-3,  # m
            "turns_per_length_list": [coils_per_length],  # turns/m
            "initial_link_per_fiber_length": initial_link_per_fiber_length,  # turns/m
            "CCW_list": [False],
            "n_ply_per_coil_level": [1],
        }
        self.sim_settings = Dict2Class(muscle_sim_settings_dict)
        self.geometry = Dict2Class(muscle_geometry_dict)
        self.properties = Dict2Class(nylon_material_properties_dict)
        self.name = "Samuel_monocoil"
        self.cross_sectional_area = 0.5027
        self.strain_experimental = 0.5 * np.array(
            [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        )
        # self.strain_experimental = np.array([
        #                                     0,
        #                                     0.05,
        #                                     0.1,
        #                                     0.15,
        #                                     0.2,
        #                                     0.25,
        #                                     0.3,
        #                                     0.35,
        #                                     0.4,
        #                                     0.45,
        #                                     0.5])
        self.passive_force_experimental = np.linspace(0, 10, 11)
        # self.total_force_experimental =  self.cross_sectional_area*np.array([8.08254,
        #                                                                 9.33505,
        #                                                                 10.64047,
        #                                                                 12.38656,
        #                                                                 13.80071,
        #                                                                 14.25318,
        #                                                                 15.73646,
        #                                                                 17.22668,
        #                                                                 17.50115,
        #                                                                 19.38474,
        #                                                                 19.652])

        self.tensile_test_list = []
        for file in os.listdir("ExperimentalData/SamuelMuscles/Monocoil"):
            if file.endswith(".txt"):
                self.tensile_test_list.append(
                    np.loadtxt(
                        os.path.join("ExperimentalData/SamuelMuscles/Monocoil", file)
                    )
                )
        self.strain_relaxtion = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) / 14
        self.passive_force_relaxtion_max = np.array(
            [1.38877, 2.67347, 3.81725, 4.9988, 6.17261, 7.40619, 8.73567]
        )
        self.passive_force_relaxtion_min = np.array(
            [0.812277, 1.77133, 2.73674, 3.70495, 4.66849, 5.75411, 6.91049]
        )
        self.trained_tensile_test_list = []
        for file in os.listdir("ExperimentalData/SamuelMuscles/TrainedMonocoil"):
            if file.endswith(".txt"):
                self.trained_tensile_test_list.append(
                    np.loadtxt(
                        os.path.join(
                            "ExperimentalData/SamuelMuscles/TrainedMonocoil", file
                        )
                    )
                )


class Samuel_supercoil:
    def __init__(self) -> None:
        # from tuning PureContraction
        muscle_sim_settings_dict = {
            "n_elem_per_coil": 36,  # at least 24 for stable coil beyond 30 seconds
            "nu": 3e-3,
            "actuation_start_temperature": 25,  # °C
            "actuation_end_temperature": 80,  # °C
            "actuation_kappa_change": 1e-7,  # result from tuning to no strain actuation force
            "E_scale": 1e6,
        }

        fiber_diameter = 0.47e-3

        ply_external_diameter = 1e-3
        ply_radius = (ply_external_diameter - fiber_diameter) / 2
        external_coil_diameter = 2.48e-3
        coil_radius = (external_coil_diameter / 2) - ply_radius
        initial_link_per_fiber_length = 387  # turns/meter
        coils_per_length = 53 / (75.13e-3)  # 10e3/16 #
        supercoils_per_coil_length = 1 / (6e-3)
        supercoils_per_length = (
            supercoils_per_coil_length
            * helix_length_per_coil(coils_per_length, coil_radius)
            * coils_per_length
        )  # (supercoils/coil_length)(coil_length/coil)(coils/height) = supercoils/height

        # data from Kinetic Research Group
        muscle_geometry_dict = {
            "fiber_radius": fiber_diameter / 2,  # m
            "start_radius_list": [coil_radius, ply_radius],  # m
            "taper_slope_list": [0, 0],
            "start_position": np.array([0.0, 0.0, 0.0]),
            "direction": np.array([0.0, 0.0, 1.0]),
            "normal": np.array([1.0, 0.0, 0.0]),
            "angular_offset": np.pi / 2,  # rad
            "muscle_length": 20e-3,  # m
            "turns_per_length_list": [
                coils_per_length,
                supercoils_per_length,
            ],  # turns/m
            "initial_link_per_fiber_length": initial_link_per_fiber_length,  # turns/m
            "CCW_list": [False, False],
            "n_ply_per_coil_level": [1, 3],
        }
        self.sim_settings = Dict2Class(muscle_sim_settings_dict)
        self.geometry = Dict2Class(muscle_geometry_dict)
        self.properties = Dict2Class(nylon_material_properties_dict)
        self.name = "Samuel_supercoil"
        self.cross_sectional_area = 0.5205
        self.strain_experimental_tensile = np.loadtxt(
            "ExperimentalData/SamuelMuscles/strain.txt"
        )
        self.passive_force_experimental_tensile = (
            self.cross_sectional_area
            * np.loadtxt("ExperimentalData/SamuelMuscles/supercoiled_stress.txt")
        )
        self.strain_experimental_tensile2 = np.loadtxt(
            "ExperimentalData/SamuelMuscles/strain2_supercoiled.txt"
        )
        self.passive_force_experimental_tensile2 = np.loadtxt(
            "ExperimentalData/SamuelMuscles/supercoiled_force.txt"
        )
        # self.strain_experimental =  0.5*np.array([
        #                                     0,
        #                                     0.05,
        #                                     0.1,
        #                                     0.15,
        #                                     0.2,
        #                                     0.25,
        #                                     0.3,
        #                                     0.35,
        #                                     0.4,
        #                                     0.45,
        #                                     0.5])
        self.strain_experimental = np.linspace(0, 0.25, 6)
        # self.passive_force_experimental = np.linspace(0,3,7)
        self.passive_force_experimental = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.total_force_experimental = self.cross_sectional_area * np.array(
            [
                5.77819,
                7.35001,
                8.66797,
                9.47205,
                10.66432,
                12.053,
                13.58367,
                14.58834,
                15.30351,
                16.88798,
                18.53586,
            ]
        )


class Samuel_supercoil_stl:
    def __init__(self) -> None:
        # from tuning PureContraction
        muscle_sim_settings_dict = {
            "n_elem_per_coil": 36,  # at least 24 for stable coil beyond 30 seconds
            "nu": 3e-3,
            "actuation_start_temperature": 25,  # °C
            "actuation_end_temperature": 80,  # °C
            "actuation_kappa_change": 1e-7,  # result from tuning to no strain actuation force
            "E_scale": 1e6,
        }

        fiber_diameter = 0.47e-3
        external_coil_diameter = 2.6188e-3  # from stl file
        internal_coil_diameter = 0.5569e-3  # from stl file
        coil_radius = (external_coil_diameter + internal_coil_diameter) / 4
        ply_external_diameter = (external_coil_diameter - internal_coil_diameter) / 2
        ply_radius = (ply_external_diameter - fiber_diameter) / 2

        initial_link_per_fiber_length = 387  # turns/meter
        coils_per_length = 1 / (1.7e-3)
        supercoils_per_length = 8 / (5.4e-3)
        # data from Kinetic Research Group
        muscle_geometry_dict = {
            "fiber_radius": fiber_diameter / 2,  # m
            "start_radius_list": [coil_radius, ply_radius],  # m
            "taper_slope_list": [0, 0],
            "start_position": np.array([0.0, 0.0, 0.0]),
            "direction": np.array([0.0, 0.0, 1.0]),
            "normal": np.array([1.0, 0.0, 0.0]),
            "angular_offset": np.pi / 2,  # rad
            "muscle_length": 20e-3,  # m
            "turns_per_length_list": [
                coils_per_length,
                supercoils_per_length,
            ],  # turns/m
            "initial_link_per_fiber_length": initial_link_per_fiber_length,  # turns/m
            "CCW_list": [False, False],
            "n_ply_per_coil_level": [1, 3],
        }
        self.sim_settings = Dict2Class(muscle_sim_settings_dict)
        self.geometry = Dict2Class(muscle_geometry_dict)
        self.properties = Dict2Class(nylon_material_properties_dict)
        self.name = "Samuel_supercoil_stl"
        # self.experimental_tensile_test = np.loadtxt(
        #     "ExperimentalData/SamuelMuscles/Supercoil/supercoiled_tensile_test.txt"
        # )
        # self.experimental_tensile_test_single_fiber_times_3 = np.loadtxt(
        #     "ExperimentalData/SamuelMuscles/Supercoil/supercoiled_one_fiber_tensile_test.txt"
        # )
        # self.experimental_tensile_test_single_fiber_times_3[:, 1] *= 3
        self.strain_experimental = np.linspace(0, 1.5, 6)  # np.linspace(0,0.35,6)
        self.passive_force_experimental = np.linspace(0, 50, 6)


class Samuel_supercoil_stl_single_fiber:
    def __init__(self) -> None:
        # from tuning PureContraction
        muscle_sim_settings_dict = {
            "n_elem_per_coil": 36,  # at least 24 for stable coil beyond 30 seconds
            "nu": 3e-3,
            "actuation_start_temperature": 25,  # °C
            "actuation_end_temperature": 80,  # °C
            "actuation_kappa_change": 1e-7,  # result from tuning to no strain actuation force
            "E_scale": 1e6,
        }

        fiber_diameter = 0.47e-3
        external_coil_diameter = 2.6188e-3  # from stl file
        internal_coil_diameter = 0.5569e-3  # from stl file
        coil_radius = (external_coil_diameter + internal_coil_diameter) / 4
        ply_external_diameter = (external_coil_diameter - internal_coil_diameter) / 2
        ply_radius = (ply_external_diameter - fiber_diameter) / 2

        initial_link_per_fiber_length = 387  # turns/meter
        coils_per_length = 1 / (1.7e-3)
        supercoils_per_length = 8 / (5.4e-3)
        # data from Kinetic Research Group
        muscle_geometry_dict = {
            "fiber_radius": fiber_diameter / 2,  # m
            "start_radius_list": [coil_radius, ply_radius],  # m
            "taper_slope_list": [0, 0],
            "start_position": np.array([0.0, 0.0, 0.0]),
            "direction": np.array([0.0, 0.0, 1.0]),
            "normal": np.array([1.0, 0.0, 0.0]),
            "angular_offset": np.pi / 2,  # rad
            "muscle_length": 20e-3,  # m
            "turns_per_length_list": [
                coils_per_length,
                supercoils_per_length,
            ],  # turns/m
            "initial_link_per_fiber_length": initial_link_per_fiber_length,  # turns/m
            "CCW_list": [False, False],
            "n_ply_per_coil_level": [1, 1],
        }
        self.sim_settings = Dict2Class(muscle_sim_settings_dict)
        self.geometry = Dict2Class(muscle_geometry_dict)
        self.properties = Dict2Class(nylon_material_properties_dict)
        self.name = "Samuel_supercoil_stl_single_fiber"
        self.experimental_tensile_test_single_fiber = np.loadtxt(
            "ExperimentalData/SamuelMuscles/Supercoil/supercoiled_one_fiber_tensile_test.txt"
        )
        self.strain_experimental = np.linspace(0, 0.35, 6)
        self.passive_force_experimental = np.linspace(0, 0.4, 6)


class Samuel_supercoil_ply:
    def __init__(self) -> None:
        # from tuning PureContraction
        muscle_sim_settings_dict = {
            "n_elem_per_coil": 36,  # at least 24 for stable coil beyond 30 seconds
            "nu": 3e-3,
            "actuation_start_temperature": 25,  # °C
            "actuation_end_temperature": 150,  # °C
            "actuation_kappa_change": 0.05976653703417187,  # result from tuning to no strain actuation force
            "E_scale": 1e6,
        }

        fiber_diameter = 0.47e-3

        ply_external_diameter = 1e-3
        ply_radius = (ply_external_diameter - fiber_diameter) / 2
        external_coil_diameter = 2.48e-3
        coil_radius = (external_coil_diameter / 2) - ply_radius
        initial_link_per_fiber_length = 387  # turns/meter
        coils_per_length = 53 / (75.13e-3)  # 10e3/16 #
        supercoils_per_coil_length = 1 / (6e-3)
        supercoils_per_length = (
            supercoils_per_coil_length
            * helix_length_per_coil(coils_per_length, coil_radius)
            * coils_per_length
        )  # (supercoils/coil_length)(coil_length/coil)(coils/height) = supercoils/height

        # data from Kinetic Research Group
        muscle_geometry_dict = {
            "fiber_radius": fiber_diameter / 2,  # m
            "start_radius_list": [1e-14, ply_radius],  # m
            "taper_slope_list": [0, 0],
            "start_position": np.array([0.0, 0.0, 0.0]),
            "direction": np.array([0.0, 0.0, 1.0]),
            "normal": np.array([1.0, 0.0, 0.0]),
            "angular_offset": np.pi / 2,  # rad
            "muscle_length": 20e-3,  # m
            "turns_per_length_list": [supercoils_per_coil_length, 0],  # turns/m
            "initial_link_per_fiber_length": 0,  # turns/m
            "CCW_list": [False, False],
            "n_ply_per_coil_level": [1, 3],
        }
        self.sim_settings = Dict2Class(muscle_sim_settings_dict)
        self.geometry = Dict2Class(muscle_geometry_dict)
        self.properties = Dict2Class(nylon_material_properties_dict)
        self.name = "Samuel_supercoil_ply"
        self.cross_sectional_area = 0.5205
        self.strain_experimental_tensile = np.loadtxt(
            "ExperimentalData/SamuelMuscles/strain.txt"
        )
        self.passive_force_experimental_tensile = (
            self.cross_sectional_area
            * np.loadtxt("ExperimentalData/SamuelMuscles/supercoiled_stress.txt")
        )
        self.strain_experimental_tensile2 = np.loadtxt(
            "ExperimentalData/SamuelMuscles/strain2_supercoiled.txt"
        )
        self.passive_force_experimental_tensile2 = np.loadtxt(
            "ExperimentalData/SamuelMuscles/supercoiled_force.txt"
        )
        self.strain_experimental = np.array(
            [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        )
        self.passive_force_experimental = self.cross_sectional_area * np.array(
            [
                0,
                0.19987,
                0.46295,
                0.88857,
                1.40059,
                2.07213,
                2.73351,
                3.37198,
                4.1914,
                5.07965,
                5.84766,
            ]
        )
        self.total_force_experimental = self.cross_sectional_area * np.array(
            [
                5.77819,
                7.35001,
                8.66797,
                9.47205,
                10.66432,
                12.053,
                13.58367,
                14.58834,
                15.30351,
                16.88798,
                18.53586,
            ]
        )
