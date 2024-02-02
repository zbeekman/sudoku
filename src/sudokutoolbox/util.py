import math


def is_square(N: int) -> bool:
    """Return True if N is a perfect square."""
    return math.sqrt(N).is_integer()


def valid_square(instance, attribute, value):
    """Validate that the attribute is a perfect square."""
    if not is_square(value):
        raise ValueError(
            f"{attribute.name} of {instance.name} must have an integer square root."
        )
