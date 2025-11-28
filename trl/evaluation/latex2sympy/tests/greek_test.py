from .context import assert_equal
import pytest
from sympy import Symbol

epsilon_upper = Symbol('char"000190', real=True)
epsilon_lower = Symbol('epsilon', real=True)
varepsilon = Symbol('varepsilon', real=True)


def test_greek_epsilon():
    assert_equal("\\epsilon", epsilon_lower)


def test_greek_epsilon_upper():
    assert_equal('\\char"000190', epsilon_upper)


def test_greek_varepsilon():
    assert_equal('\\varepsilon', varepsilon)
