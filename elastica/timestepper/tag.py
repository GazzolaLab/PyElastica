from typing import Callable, Type


# TODO: Maybe move this for common utility
def tag(Tag: Type) -> Callable[[Type], Type]:
    """
    Tag a class with arbitrary type-class

    example:
    class ATag: ...

    @tag(ATag)
    class A1:
        ...

    assert isinstance(A1.tag, ATag)
    """

    def wrapper(cls: Type) -> Type:
        cls.Tag = Tag()
        return cls

    return wrapper


class SymplecticStepperTag:
    pass


class ExplicitStepperTag:
    pass
