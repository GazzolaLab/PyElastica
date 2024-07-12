## Expr

Contains base templates for recognizing different expressions
(such as addition of states and multiplication with delta-time)
within the states library. Any custom definition of operators
such as `+` and `*` should derive from the templates provided
in this folder. Contents of this folder need NOT be exposed in
the `States.hpp` file, it is meant for internal use only.
