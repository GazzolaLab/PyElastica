from typing import Type, List


# TODO: Maybe move this for common utility
def tag(Tag) -> Type:
    """
    Tag a class with arbitrary type-class

    example:
    class ATag: ...

    @tag(ATag)
    class A1:
        ...

    assert isinstance(A1.tag, ATag)
    """

    def wrapper(cls):
        cls.Tag = Tag()
        return cls

    return wrapper


class SymplecticStepperTag:
    pass


class ExplicitStepperTag:
    pass
