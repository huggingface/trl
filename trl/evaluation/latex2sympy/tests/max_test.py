from .context import assert_equal
import pytest
from sympy import Symbol, Rational, Float, Max, sqrt, exp, pi, nsimplify

x = Symbol('x', real=True)
y = Symbol('y', real=True)
z = Symbol('z', real=True)


def test_max_usual():
    assert_equal("\\max(1, 5)", Max(1, 5))
    assert_equal("\\max(12, 4)", Max(12, 4))
    assert_equal("\\max(109, 120)", Max(109, 120))
    assert_equal("\\max(3, 3)", Max(3, 3))
    assert_equal("\\max(0, 0)", Max(0, 0))
    assert_equal("\\max(1)", Max(1))
    assert_equal("\\max(1092198374, 290348E32)", Max(1092198374, Rational('290348E32')))
    assert_equal("\\max(5, 2, 17, 4)", Max(5, 2, 17, 4))


def test_max_negative():
    assert_equal("\\max(-9, 4)", Max(-9, 4))
    assert_equal("\\max(4, -9)", Max(4, -9))
    assert_equal("\\max(-7)", Max(-7))
    assert_equal("\\max(-2, -2)", Max(-2, -2))
    assert_equal("\\max(-324E-3, -58)", Max(Rational('-324E-3'), -58))
    assert_equal("\\max(-1, 0, 1, -37, 42)", Max(-1, 0, 1, -37, 42))


def test_max_float():
    assert_equal("\\max(\\pi, 3)", Max(pi, 3))
    assert_equal("\\max(1234.56789, 1234.5678901)", Max(Rational('1234.56789'), Rational('1234.5678901')))
    assert_equal("\\max(12.4, 9.5)", Max(12.4, 9.5))
    assert_equal("\\max(6, 6.2)", Max(6, 6.2))
    assert_equal("\\max(-98.7)", Max(-98.7))
    assert_equal("\\max(7.1, 9)", Max(7.1, 9))
    assert_equal("\\max(-21E-12, 0.00005)", Max(nsimplify(Rational('-21E-12')), Rational('0.00005')), symbolically=True)
    assert_equal("\\max(\\sqrt{3}, 0, 1)", Max(sqrt(3), 0, 1))


def test_max_fraction():
    assert_equal("\\max(1/2, 1/4)", Max(Rational('1/2'), Rational('1/4')))
    assert_equal("\\max(6/2, 3)", Max(Rational('6/2'), 3))
    assert_equal("\\max(2/4, 1/2)", Max(Rational('2/4'), Rational('1/2')))
    assert_equal("\\max(-12/5, 6.4)", Max(Rational('-12/5'), Rational('6.4')))
    assert_equal("\\max(1/10)", Max(Rational('1/10')))
    assert_equal("\\max(1.5, \\pi/2)", Max(Rational('1.5'), pi / 2, evaluate=False))
    assert_equal("\\max(-4/3, -2/1, 0/9, -3)", Max(Rational('-4/3'), Rational('-2/1'), Rational('0/9'), -3))


def test_max_expr():
    assert_equal("\\max((1+6)/3, 7)", Max(Rational(1 + 6, 3), 7))
    assert_equal("\\max(58*9)", Max(58 * 9))
    assert_equal("\\max(1+6/3, -5)", Max(1 + Rational('6/3'), -5))
    assert_equal("\\max(7*4/5, 092) * 2", Max(7 * 4 / 5, 92) * 2)
    assert_equal("38+\\max(13, 15-2.3)", 38 + Max(13, 15 - Rational('2.3')))
    assert_equal("\\sqrt{\\max(99.9999999999999, 100)}", sqrt(Max(Rational('99.9999999999999'), 100)))
    assert_equal("\\max(274/(5+2), \\exp(12.4), 1.4E2)", Max(Rational(274, 5 + 2), exp(Rational('12.4')), Rational('1.4E2')))


def test_max_symbol():
    assert_equal("\\max(x)", Max(x), symbolically=True)
    assert_equal("\\max(x, y)", Max(x, y), symbolically=True)
    assert_equal("\\max(y, x)", Max(y, x), symbolically=True)
    assert_equal("\\max(x+y, y+x)", Max(x + y, y + x), symbolically=True)
    assert_equal("\\max(9x/4, z)", Max(9 * x / 4, z), symbolically=True)
    assert_equal("\\max(y\\pi, 9)", Max(y * pi, 9), symbolically=True)
    assert_equal("\\max(2y-y, y + 1)", Max(2 * y - y, y + 1), symbolically=True)
    assert_equal("\\max(z, y, x)", Max(z, y, x), symbolically=True)


def test_max_multiarg():
    assert_equal("\\max(1,2)", Max(1, 2))
    assert_equal("\\max(9,876,543)", Max(9, 876, 543))
    assert_equal("\\max(x, y,z)", Max(x, y, z), symbolically=True)
    assert_equal("\\max(5.8,7.4, 2.2,-10)", Max(Rational('5.8'), Rational('7.4'), Rational('2.2'), -10))
    assert_equal("\\max(\\pi,12E2,84,\\sqrt{5},12/5)", Max(pi, Rational('12E2'), 84, sqrt(5), Rational('12/5')))
    assert_equal("\\max(823,51)", Max(823, 51))
    assert_equal("\\max(72*4,23, 9)", Max(72 * 4, 23, 9))
