# PyElastica Examples

This directory contains number of examples of elastica.
Each [example cases](#example-cases) are stored in separate subdirectories, containing case descriptions, run file, and all other data/script necessary to run.
More [advanced cases](#advanced-cases) are stored in separate repository with its description.

## Installing Requirements
In order to run examples, you will need to install additional dependencies.

```bash
make install_examples_dependencies
```

## Case Examples

Some examples provide additional files or links to published paper for a complete description.
Examples can serve as a starting template for customized usages.

* [AxialStretchingCase](./AxialStretchingCase)
    * __Purpose__: Physical convergence test of simple stretching rod.
    * __Features__: CosseratRod, OneEndFixedRod, EndpointForces
* [TimoshenkoBeamCase](./TimoshenkoBeamCase)
    * __Purpose__: Physical convergence test of simple Timoshenko beam.
    * __Features__: CosseratRod, OneEndFixedRod, EndpointForces,
* [FlexibleSwingingPendulumCase](./FlexibleSwingingPendulumCase)
    * __Purpose__: Physical convergence test of simple pendulum with flexible rod.
    * __Features__: CosseratRod, HingeBC, GravityForces
* [ContinuumSnakeCase](./ContinuumSnakeCase)
    * __Purpose__: Demonstrate simple case of modeling biological creature using PyElastica. The example use friction to create slithering snake, and optimize the speed using [CMA](https://github.com/CMA-ES/pycma).
    * __Features__: CosseratRod, MuscleTorques, AnisotropicFrictionalPlane, Gravity, CMA Optimization
    * [MuscularSnake](./MuscularSnake)
      * __Purpose__: Example of [Parallel connection module](../elastica/experimental/connection_contact_joint/parallel_connection.py) and customized [Force module](./MuscularSnake/muscle_forces.py) to implement muscular snake.
      * __Features__: MuscleForces(custom implemented)
* [ButterflyCase](./ButterflyCase)
    * __Purpose__: Demonstrate simple restoration with initial strain.
    * __Features__: CosseratRod
* [FrictionValidationCases](./FrictionValidationCases)
    * __Purpose__: Physical validation of rolling and translational friction.
    * __Features__: CosseratRod, UniformForces, AnisotropicFrictionalPlane
* [JointCases](./JointCases)
    * __Purpose__: Demonstrate various joint usage with Cosserat Rod.
    * __Features__: FreeJoint, FixedJoint, HingeJoint, OneEndFixedRod, EndpointForcesSinusoidal
* [RigidBodyCases](./RigidBodyCases)
    * __Purpose__: Demonstrate usage of rigid body on simulation.
    * __Features__: Cylinder, Sphere
    * [RodRigidBodyContact](./RigidBodyCases/RodRigidBodyContact)
      * __Purpose__: Demonstrate contact between cylinder and rod, for different intial conditions.
      * __Features__: Cylinder, CosseratRods, ExternalContact
* [HelicalBucklingCase](./HelicalBucklingCase)
    * __Purpose__: Demonstrate helical buckling with extreme twisting boundary condition.
    * __Features__: HelicalBucklingBC
* [ContinuumFlagellaCase](./ContinuumFlagellaCase)
    * __Purpose__: Demonstrate flagella modeling using PyElastica.
    * __Features__: SlenderBodyTheory, MuscleTorques,
    * [MuscularFlagella](./MuscularFlagella)
        * __Purpose__: Example of customizing [Joint module](./MuscularFlagella/connection_flagella.py) and [Force module](./MuscularFlagella/muscle_forces_flagella.py) to implement muscular flagella.
        * __Features__: MuscleForces(custom implemented)
* [RodContactCase](./RodContactCase)
  * [RodRodContact](./RodContactCase/RodRodContact)
    * __Purpose__: Demonstrates contact between two rods, for different initial conditions.
    * __Features__: CosseratRod, ExternalContact
  * [RodSelfContact](./RodContactCase/RodSelfContact)
    * [PlectonemesCase](./RodContactCase/RodSelfContact/PlectonemesCase)
      * __Purpose__: Demonstrates rod self contact with Plectoneme example, and how to use link-writhe-twist after simulation completed.
      * __Features__: CosseratRod, SelonoidsBC, SelfContact, Link-Writhe-Twist
    * [SolenoidsCase](./RodContactCase/RodSelfContact/SolenoidsCase)
      * __Purpose__: Demonstrates rod self contact with Solenoid example, and how to use link-writhe-twist after simulation completed.
      * __Features__: CosseratRod, SelonoidsBC, SelfContact, Link-Writhe-Twist
* [BoundaryConditionsCases](./BoundaryConditionsCases)
    * __Purpose__: Demonstrate the usage of boundary conditions for constraining the movement of the system.
    * __Features__: GeneralConstraint, CosseratRod
* [DynamicCantileverCase](./DynamicCantileverCase)
    * __Purpose__: Validation of dynamic cantilever vibration for multiple modes.
    * __Features__: CosseratRod, OneEndFixedRod
* [RingRodCase](./RingRodCase)
    * __Purpose__: Demonstrate simulation of ring rod.
    * __Features__: RingCosseratRod, OneEndFixedRod, GravityForce

## Functional Examples

* [RestartExample](./RestartExample)
   * __Purpose__: Demonstrate the usage of restart module.
   * __Features__: save_state, load_state
* [Visualization](./Visualization)
    * __Purpose__: Include simple examples of raytrace rendering data.
    * __Features__: POVray

## Advanced Cases

* [Elastica RL control](https://github.com/GazzolaLab/Elastica-RL-control) - Case presented in [<strong>Elastica: A compliant mechanics environment for soft robotic control</strong>](https://doi.org/10.1109/LRA.2021.3063698)
* [Gym Softrobot](https://github.com/skim0119/gym-softrobot) - Soft-robot control environment developed in OpenAI-gym format to study slender body control with reinforcement learning.

## Experimental Cases
* [ParallelConnectionExample](./ExperimentalCases/ParallelConnectionExample)
  * __Purpose__: Demonstrate the usage of parallel connection.
  * __Features__: connect two parallel rods
