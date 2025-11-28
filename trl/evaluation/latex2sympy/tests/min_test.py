from .context import assert_equal
import pytest
from sympy import Symbol, Rational, Float, Min, sqrt, exp, pi, nsimplify

x = Symbol('x', real=True)
y = Symbol('y', real=True)
z = Symbol('z', real=True)


def test_min_usual():
    assert_equal("\\min(1, 5)", Min(1, 5))
    assert_equal("\\min(12, 4)", Min(12, 4))
    assert_equal("\\min(109, 120)", Min(109, 120))
    assert_equal("\\min(3, 3)", Min(3, 3))
    assert_equal("\\min(0, 0)", Min(0, 0))
    assert_equal("\\min(1)", Min(1))
    assert_equal("\\min(1092198374, 290348E32)", Min(1092198374, Rational('290348E32')))
    assert_equal("\\min(5, 2, 17, 4)", Min(5, 2, 17, 4))


def test_min_negative():
    assert_equal("\\min(-9, 4)", Min(-9, 4))
    assert_equal("\\min(4, -9)", Min(4, -9))
    assert_equal("\\min(-7)", Min(-7))
    assert_equal("\\min(-2, -2)", Min(-2, -2))
    assert_equal("\\min(-324E-3, -58)", Min(Rational('-324E-3'), -58))
    assert_equal("\\min(-1, 0, 1, -37, 42)", Min(-1, 0, 1, -37, 42))


def test_min_float():
    assert_equal("\\min(\\pi, 3)", Min(pi, 3))
    assert_equal("\\min(1234.56789, 1234.5678901)", Min(Rational('1234.56789'), Rational('1234.5678901')))
    assert_equal("\\min(12.4, 9.5)", Min(12.4, 9.5))
    assert_equal("\\min(6, 6.2)", Min(6, 6.2))
    assert_equal("\\min(-98.7)", Min(-98.7))
    assert_equal("\\min(7.1, 9)", Min(7.1, 9))
    assert_equal("\\min(-21E-12, 0.00005)", Min(nsimplify(Rational('-21E-12')), Rational('0.00005')), symbolically=True)
    assert_equal("\\min(\\sqrt{3}, 0, 1)", Min(sqrt(3), 0, 1))


def test_min_fraction():
    assert_equal("\\min(1/2, 1/4)", Min(Rational('1/2'), Rational('1/4')))
    assert_equal("\\min(6/2, 3)", Min(Rational('6/2'), 3))
    assert_equal("\\min(2/4, 1/2)", Min(Rational('2/4'), Rational('1/2')))
    assert_equal("\\min(-12/5, 6.4)", Min(Rational('-12/5'), Rational('6.4')))
    assert_equal("\\min(1/10)", Min(Rational('1/10')))
    assert_equal("\\min(1.5, \\pi/2)", Min(Rational('1.5'), pi / 2, evaluate=False))
    assert_equal("\\min(-4/3, -2/1, 0/9, -3)", Min(Rational('-4/3'), Rational('-2/1'), Rational('0/9'), -3))


def test_min_expr():
    assert_equal("\\min((1+6)/3, 7)", Min(Rational(1 + 6, 3), 7))
    assert_equal("\\min(58*9)", Min(58 * 9))
    assert_equal("\\min(1+6/3, -5)", Min(1 + Rational('6/3'), -5))
    assert_equal("\\min(7*4/5, 092) * 2", Min(7 * 4 / 5, 92) * 2)
    assert_equal("38+\\min(13, 15-2.3)", 38 + Min(13, 15 - Rational('2.3')))
    assert_equal("\\sqrt{\\min(99.9999999999999, 100)}", sqrt(Min(Rational('99.9999999999999'), 100)))
    assert_equal("\\min(274/(5+2), \\exp(12.4), 1.4E2)", Min(Rational(274, 5 + 2), exp(Rational('12.4')), Rational('1.4E2')))


def test_min_symbol():
    assert_equal("\\min(x)", Min(x), symbolically=True)
    assert_equal("\\min(x, y)", Min(x, y), symbolically=True)
    assert_equal("\\min(y, x)", Min(y, x), symbolically=True)
    assert_equal("\\min(x+y, y+x)", Min(x + y, y + x), symbolically=True)
    assert_equal("\\min(9x/4, z)", Min(9 * x / 4, z), symbolically=True)
    assert_equal("\\min(y\\pi, 9)", Min(y * pi, 9), symbolically=True)
    assert_equal("\\min(2y-y, y + 1)", Min(2 * y - y, y + 1), symbolically=True)
    assert_equal("\\min(z, y, x)", Min(z, y, x), symbolically=True)


def test_min_multiarg():
    assert_equal("\\min(1,2)", Min(1, 2))
    assert_equal("\\min(9,876,543)", Min(9, 876, 543))
    assert_equal("\\min(x, y,z)", Min(x, y, z), symbolically=True)
    assert_equal("\\min(5.8,7.4, 2.2,-10)", Min(Rational('5.8'), Rational('7.4'), Rational('2.2'), -10))
    assert_equal("\\min(\\pi,12E2,84,\\sqrt{5},12/5)", Min(pi, Rational('12E2'), 84, sqrt(5), Rational('12/5')))
    assert_equal("\\min(823,51)", Min(823, 51))
    assert_equal("\\min(72*4,23, 9)", Min(72 * 4, 23, 9))
