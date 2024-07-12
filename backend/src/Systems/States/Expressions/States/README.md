## States

Frequently when temporally integrating mechanical systems we have the integration algebra defined in multiple spaces.
For example algebra for translations is described in straightforward (SE3) space, where one can do element wise
operations for all operations, such as `+` between states and `*` between state and ∆t. Meanwhile algebra for rotations
is defined in the so3 space (for rotation matrices, where the matrices are always orthonormal). To satisfy the
orthonormality to machine precision, its more convenient to define operators that act on the exponential space, i.e. `+`
between two states is transformed into matrix multiplications. To accommodate states with mixed algebra (one rotation
and one translation, as is quite frequent), we define a States class (or more meaningfully a StateCollection class).
This class can contain multiple, disjoint, smaller states each with its own algebra. States then acts as a convenient
wrapper for interfacing with the time-stepping algorithms---indeed States has its own expression template system. This
system builds up expression trees which are then forwarded to the constituent, smaller states. The smaller states can
then execute its own operations with the appropriate algebra.
