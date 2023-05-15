# Release 0.3.1

## New Features

* Ring Cosserat rods by @armantekinalp in https://github.com/GazzolaLab/PyElastica/pull/229
* Magnetic Cosserat rods functionality by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/245

## What's Changed

* Dynamic validation example by @sy-cui in https://github.com/GazzolaLab/PyElastica/pull/173
* Refactor: change typings in forcing/constraints/connections to SystemType or RodType by @sy-cui in https://github.com/GazzolaLab/PyElastica/pull/191
* Wildcard imports removed by @AsadNizami in https://github.com/GazzolaLab/PyElastica/pull/238
* Remove internal damping option for Cosserat rod by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/243

## Minor Fixes

* Fix main yml windows python version by @armantekinalp in https://github.com/GazzolaLab/PyElastica/pull/206
* Fix restart functionality by @armantekinalp in https://github.com/GazzolaLab/PyElastica/pull/249
* Replaced 'moviepy' with 'ffmpeg' for video generation by @Rahul-JOON in https://github.com/GazzolaLab/PyElastica/pull/232

## Repository Update

* Windows CI fix by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/192
* autoflake funtionality by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/175

## New Contributors

* @sy-cui made their first contribution in https://github.com/GazzolaLab/PyElastica/pull/173
* @Rahul-JOON made their first contribution in https://github.com/GazzolaLab/PyElastica/pull/232
* @AsadNizami made their first contribution in https://github.com/GazzolaLab/PyElastica/pull/238
* @erfanhamdi made their first contribution in https://github.com/GazzolaLab/PyElastica/pull/247

**Full Changelog**: https://github.com/GazzolaLab/PyElastica/compare/v0.3.0...v0.3.1


# Release v0.3.0

## New Features

* Refactor internal dissipation as external addon damping module  by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/114
  - New AnalyticalDamper
  * Update timestep values for the new damping module by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/120
* Filter Damper class by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/123
* Adding `ConfigurableFixedConstraint` boundary condition class by @mstoelzle in https://github.com/GazzolaLab/PyElastica/pull/143

## What's Changed

* Adds significant digits to shear coefficient (Alpha) (#79) by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/82
* Dissipation constant fix (#81) by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/87
  - Scale dissipation constant by mass instead of length.
* Update FixedJoints: restoring spring-damped-torques, initial rotation offset by @mstoelzle in https://github.com/GazzolaLab/PyElastica/pull/135
* Update: Damping values for rod-rigid body contact cases (#171) by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/172
* Fix damping force direction by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/170
* Refactor: `wrappers` -> `modules` by @skim0119 in https://github.com/GazzolaLab/PyElastica/pull/177

## Minor Fixes

* Fix compute shear energy function typo by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/88
* Track velocity norms as dynamic proxies in Axial stretching and Timoshenko examples by @tp5uiuc in https://github.com/GazzolaLab/PyElastica/pull/97
* Node to element interpolation fix by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/98
* Update: numba disable jit flag in poetry command by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/146
* Adjusting data structure of `fixed_positions` and `fixed_directors` by @mstoelzle in https://github.com/GazzolaLab/PyElastica/pull/147
* Docs: correct endpoint forces docstring by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/157
* Update: remove sys append calls in examples by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/162

## New Experimental Features

* Enabling joints to connect rods and rigid bodies  by @mstoelzle in https://github.com/GazzolaLab/PyElastica/pull/149

## Repository Updates

* Codeowners setup by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/115
* Remove _elastica_numba folder while keeping _elastica_numba.py by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/138
* Update CI: Add sphinx build by @skim0119 in https://github.com/GazzolaLab/PyElastica/pull/139
* Poetry setup for PyElastica (#101) by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/141
* Add pre commit to pyelastica by @bhosale2 in https://github.com/GazzolaLab/PyElastica/pull/151
* Update makefile commands: test by @skim0119 in https://github.com/GazzolaLab/PyElastica/pull/156

**Full Changelog**: https://github.com/GazzolaLab/PyElastica/compare/v0.2.4...v0.3.0


# Release Note (version 0.2.4)

## What's Changed

* Refactor EndPointForcesSinusoidal example and test cases by @armantekinalp in https://github.com/GazzolaLab/PyElastica/pull/110
* Fix save_every condition in ExportCallBack by @mstoelzle in https://github.com/GazzolaLab/PyElastica/pull/125
* Fix and update contact examples by @armantekinalp in https://github.com/GazzolaLab/PyElastica/pull/109
* Update rigid body rod contact by @armantekinalp in https://github.com/GazzolaLab/PyElastica/pull/117
* Update rigid body rod contact friction by @armantekinalp in https://github.com/GazzolaLab/PyElastica/pull/124
* Update ExportCallback by @skim0119 in https://github.com/GazzolaLab/PyElastica/pull/130

## New Contributors

* @mstoelzle made their first contribution in https://github.com/GazzolaLab/PyElastica/pull/125

---

# Release Note (version 0.2.3)

## Developer Note
The major updates are knot theory module added to the Cosserat rod as *mixin*, and muscular snake example is added. 

## Notable Changes
- #70: Knot theory module to compute topological quantities.
- #71: Reorganize rod constructor warning messages and collect messages in log.  
- #72: Muscular snake example is added.

---

# Release Note (version 0.2.2)

## Developer Note

The major documentation update is finished in this version.
Constraint and finalize module are refactored to enhance readability.

## Notable Changes
- #64: Core wrapper redesign. The finalizing code is refactored for easier integration.
- #65: Documentation update.
- #56: Constraint module has been refactored to include proper abstract base class. Additionally, new `FixedConstraint` is added for generalized fixed boundary condition.
- More test cases are added to increase code-coverage.

---

# Release Note (version 0.2.1)

## Developer Note

Contact model between two different rods and rod with itself is implemented. 
Testing the contact model is done through simulations. These simulation scripts can be found under
[RodContactCase](./RodContactCase). 
However, in future releases we have to add unit tests for contact model functions to test them and increase code coverage.

## Notable Changes
- #31: Merge contact model to master [PR #40 in public](https://github.com/GazzolaLab/PyElastica/pull/40)
- #46: The progress bar can be disabled by passing an argument to `integrate`.
- #48: Experimental modules are added to hold functions that are in test phase.
- 
### Release Note
<details>
  <summary>Click to expand</summary>

- Rod-Rod contact and Rod self contact is added.
- Two example cases for rod-rod contact is added, i.e. two rods colliding to each other in space. 
- Two example cases for rod self contact is added, i.e. plectonemes and solenoids.
- Progress bar can be disabled by passing an argument to `integrate` function.
- Experimental module added.
- Bugfix in callback mechanism

</details>

---

# Release Note (version 0.2)

## Developer Note

Good luck! If it explode, increase nu. :rofl: If it doesn't explode, thoroughly check for the bug.

## Notable Changes
- #84: Block implementation
- #75: Poisson ratio and definition of modulus [PR #26 in public](https://github.com/GazzolaLab/PyElastica/pull/26)
- #95: MuscularFlagella example case is added
- #100: ExportCallBack is added to export the rod-data into file.
- #109: Numpy-only version is now removed. Numba-implementation is set to default.
- #112: Save and load implementation with the example are added.
 
### Release Note
<details>
  <summary>Click to expand</summary>

- Block structure is included as part of optimization strategy.
- Different Poisson ratio is supported.
- Contributing guideline is added.
- Update readme
- Add MuscularFlagella example case
- Minimum requirement for dependencies is specified.
- Shear coefficient is corrected.
- Connection index assertion fixed.
- Remove numpy-only version.
- Save/Load example

</details>

