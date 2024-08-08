## States

* Implements a poor man's expression template system to handle temporally evolving "states" in a physical systems.
* Integrates dynamics in SO3 and SE3 space.
* The expression template system is not too "smart", but fits the needs---any temporary evaluations etc. is handled in
  the backend by libraries such as Blaze.
* Modern ODE libraries, like boost::odeint failed to fit our needs because of the following reasons.
    * TODO
* Hence we implement our own library with these features:
    * We implement data-structures using variadic to handle many number of disjointly evolving "states" : this can be
      explicitly contolled by the programmer
    * All data-structures are non-owning, so that memory is handled somewhere else. This models how simulators usually
      work---the crux of the memory is in a core simulation routine.
    * Algebra of temporal integration lies in the control of the user.
