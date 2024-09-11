"""
Collection of the core mathematical operators used throughout the code base.
"""

import math

# ## Task 0.1
from typing import Callable, Iterable, TypeVar

A = TypeVar("A")

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$
def mul(x: float, y: float) -> float:
    """
    Multiplies two numbers.

    Args:
            x: A float.
            y: A float.

    Returns:
            A float
    """
    return x * y


def id(x: A) -> A:
    """
    Returns the input unchanged

    Args:
            x: Input of Any type

    Returns:
            Value unchanged
    """
    return x


def add(x: float, y: float) -> float:
    """
    Returns the sum of two numbers

    Args:
            x: A float.
            y: A float

    Returns:
            The sum of two floats
    """
    return x + y


def neg(x: float) -> float:
    """
    Negates a number

    Args:
            x: A float.

    Returns:
            The negative of a number
    """
    return -x


def lt(x: float, y: float) -> float:
    """
    Checks if one number is less than another

    Args:
            x: A float.
            y: A float

    Returns:
            True if x < y, False otherwise
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """
    Checks if two numbers are equal

    Args:
            x: A float.
            y: A float

    Returns:
            True if x == y, False otherwise
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """
    returns the larger of two mumbers

    Args:
            x: A float.
            y: A float

    Returns:
            Returns the larger of two numbers
    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """
    Checks if two numbers are close in value

    Args:
            x: A float.
            y: A float

    Returns:
            A float
    """
    distance = 1e-2
    return 1.0 if abs(x - y) < distance else 0.0


def sigmoid(x: float) -> float:
    """
    Calculates the sigmoid

    Args:
            x: A float.

    Returns:
            A float
    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """
    Appllies ReLU

    Args:
            x: A float.

    Returns:
            A float
    """
    return x if x > 0.0 else 0.0


def log(x: float) -> float:
    """
    Calculate natural log

    Args:
            x: A float.

    Returns:
            A float
    """
    pseudocount = 1
    return math.log(x + pseudocount)


def exp(x: float) -> float:
    """
    Calculate exponential function

    Args:
            x: A float.

    Returns:
            A float
    """
    return math.exp(x)


def inv(x: float) -> float:
    """
    Calculate the reciprocal

    Args:
            x: A float.

    Returns:
            A float
    """
    return x**-1
    # return 1.0 / x


def log_back(x: str, y: str) -> str:
    """
    Computes the derivative of a log times a second arg

    Args:
            x: A float.
            y: A float

    Returns:
            A float
    """
    return y / x


def inv_back(x: float, y: float) -> float:
    """
    Computes the derivative of reciprocal times a second arg

    Args:
            x: A float.
            y: A float.

    Returns:
            A float
    """
    return -y / (x**2)


def relu_back(x: float, y: float) -> float:
    """
    Computes the derivative of a ReLU times a second arg

    Args:
            x: A float.
            y: A float.

    Returns:
            A float
    """
    return y if x > 0.0 else 0.0


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map function

    Args:
            fn: a function

    Returns:
            Higher-order function that applies a given function to each element of an iterable
    """

    def apply_fn(ls: Iterable[float]):
        return [fn(x) for x in ls]

    return apply_fn


def negList(ls: Iterable[float]) -> Iterable[float]:
    """
    Negate all elements in a list using map

    Args:
            ls: a list to negate

    Returns:
            A list
    """
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zip function

    Args:
            fn: a function

    Returns:
            Higher-order function that combines elements from two iterables using a given function
    """

    def zip_fn(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return zip_fn


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Add corresponding elements from two lists using zipWith

    Args:
            ls1: A list.
            ls2: A list

    Returns:
            A list that containes the corresponding elements form ls1 and ls2
    """
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """

    def apply(ls: Iterable[float]):
        total = start
        for element in ls:
            total = fn(total, element)
        return total

    return apply


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    return reduce(mul, 1)(ls)
