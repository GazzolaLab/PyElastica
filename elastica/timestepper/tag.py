from typing import Literal, TypeAlias, Final


ExplicitStepperTag: Final = "ExplicitStepper"
SymplecticStepperTag: Final = "SymplecticStepper"
StepperTags: TypeAlias = Literal["SymplecticStepper", "ExplicitStepper"]

allowed_stepper_tags: list[StepperTags] = [ExplicitStepperTag, SymplecticStepperTag]
