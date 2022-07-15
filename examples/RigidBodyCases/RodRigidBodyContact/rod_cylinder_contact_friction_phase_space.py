if __name__ == "__main__":
    import multiprocessing as mp
    from examples.RigidBodyCases.RodRigidBodyContact.rod_cylinder_contact_friction import (
        rod_cylinder_contact_friction_case,
    )
    from examples.RigidBodyCases.RodRigidBodyContact.post_processing import (
        plot_force_vs_energy,
    )

    # total_energy = rod_cylinder_contact_friction_case(friction_coefficient=1.0, normal_force_mag=10, velocity_damping_coefficient=1E4, POST_PROCESSING=True)

    friction_coefficient = list([(float(x)) / 100.0 for x in range(0, 100, 5)])

    with mp.Pool(mp.cpu_count()) as pool:
        result_total_energy = pool.map(
            rod_cylinder_contact_friction_case, friction_coefficient
        )

    plot_force_vs_energy(
        friction_coefficient,
        result_total_energy,
        friction_coefficient=0.5,
        filename="rod_energy_vs_force.png",
        SAVE_FIGURE=True,
    )
