## Systems

This folder implements the *systems* library in `Elastica++`. A *system* is defined as any quantity that has some
internal state (
usually positions, velocities etc) which evolves over time.

The functionality of systems is split into multiple files (each of which has a corresponding folder), with different
intentions:

- [Block](Block)
    - Exposed in [Block.hpp](Block.hpp)
    - Is the primary means of implementing systems in `Elastica++`, does automatic AoS -> SoA conversion, padding,
      aggregation, ghosting etc to optimize memory layouts for achieving performance.
- [common](common)
    - Internal to the systems library, containing functionality common to all systems (Rods, Rigidbodies etc).
- [CosseratRods](CosseratRods)
    - Exposed in [CosseratRods.hpp](CosseratRods.hpp)
    - Implements Cosserat Rods in `Elastica++`.
- [RigidBodies](RigidBodies)
    - Exposed in [RigidBodies.hpp](RigidBodies.hpp)
    - Implements Rigid bodies in `Elastica++`.
- [States](States)
    - For internal use in the systems library, and a bridge to the time-steppers.
    - Implements a poor man's expression template system to handle temporally evolving states in physical systems.
