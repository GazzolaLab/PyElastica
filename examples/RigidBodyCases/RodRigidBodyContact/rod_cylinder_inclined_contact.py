from rod_cylinder_contact import rod_cylinder_contact_case

import numpy as np

if __name__ == "__main__":
    inclination_angle = np.deg2rad(-30.0)
    rod_cylinder_contact_case(inclination_angle=inclination_angle)
