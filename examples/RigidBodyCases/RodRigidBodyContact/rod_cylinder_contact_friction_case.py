if __name__ == "__main__":
    from examples.RigidBodyCases.RodRigidBodyContact.rod_cylinder_contact_friction import (
        rod_cylinder_contact_friction_case,
    )

    total_energy = rod_cylinder_contact_friction_case(
        force_coefficient=0.1, normal_force_mag=10, POST_PROCESSING=True
    )
