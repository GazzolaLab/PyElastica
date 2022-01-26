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

