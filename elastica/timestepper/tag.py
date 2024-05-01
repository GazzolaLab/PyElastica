from typing import Literal, TypeAlias, Final


ExplicitStepperTag: str = "ExplicitStepper"
SymplecticStepperTag: str = "SymplecticStepper"
StepperTags: TypeAlias = Literal["SymplecticStepper", "ExplicitStepper"]
