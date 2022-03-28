# PyElastica Examples

This directory contains number of examples of elastica.
Each [example cases](#example-cases) are stored in separate subdirectories, containing case descriptions, run file, and all other data/script necessary to run.
More [advanced cases](#advanced-cases) are stored in separate repository with its description.

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
      * __Purpose__: Demonstrates rod self contact with Plectoneme example.
      * __Features__: CosseratRod, SelonoidsBC, SelfContact
    * [SolenoidsCase](./RodContactCase/RodSelfContact/SolenoidsCase)
      * __Purpose__: Demonstrates rod self contact with Solenoid example and how to use link-writhe-twist after simulation completed.
      * __Features__: CosseratRod, SelonoidsBC, SelfContact, Link-Writhe-Twist

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
