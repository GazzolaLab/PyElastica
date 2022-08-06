import numpy as np
from rigid_cylinder_rigid_cylinder_contact import cylinder_cylinder_contact_case

if __name__ == "__main__":
    inclination_angle = np.deg2rad(-30.0)
    cylinder_cylinder_contact_case(inclination_angle=inclination_angle)
