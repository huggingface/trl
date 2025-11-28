from .context import assert_equal
import pytest
from sympy import Symbol, Rational, UnevaluatedExpr, gcd, igcd, sqrt, pi

x = Symbol('x', real=True)
y = Symbol('y', real=True)
z = Symbol('z', real=True)


def test_gcd_usual():
    assert_equal("\\gcd(18, 3)", gcd(18, 3))
    assert_equal("\\gcd(3, 18)", gcd(3, 18))
    assert_equal("\\gcd(2, 2)", gcd(2, 2))
    assert_equal("\\gcd(0, 21)", UnevaluatedExpr(gcd(0, 21)))
    assert_equal("\\gcd(21, 0)", UnevaluatedExpr(gcd(21, 0)))
    assert_equal("\\gcd(0, 0)", UnevaluatedExpr(gcd(0, 0)))
    assert_equal("\\gcd(6128, 24)", gcd(6128, 24))
    assert_equal("\\gcd(24, 6128)", gcd(24, 6128))
    assert_equal("\\gcd(1E20, 1000000)", gcd(Rational('1E20'), 1000000))
    assert_equal("\\gcd(128*10^32, 1)", gcd(Rational('128E32'), 1))

    assert_equal("\\operatorname{gcd}(18, 3)", gcd(18, 3))
    assert_equal("\\operatorname{gcd}(3, 18)", gcd(3, 18))
    assert_equal("\\operatorname{gcd}(2, 2)", gcd(2, 2))
    assert_equal("\\operatorname{gcd}(0, 21)", UnevaluatedExpr(gcd(0, 21)))
    assert_equal("\\operatorname{gcd}(21, 0)", UnevaluatedExpr(gcd(21, 0)))
    assert_equal("\\operatorname{gcd}(0, 0)", UnevaluatedExpr(gcd(0, 0)))
    assert_equal("\\operatorname{gcd}(6128, 24)", gcd(6128, 24))
    assert_equal("\\operatorname{gcd}(24, 6128)", gcd(24, 6128))
    assert_equal("\\operatorname{gcd}(1E20, 1000000)", gcd(Rational('1E20'), 1000000))
    assert_equal("\\operatorname{gcd}(128*10^32, 1)", gcd(Rational('128E32'), 1))


def test_gcd_negative():
    assert_equal("\\gcd(-12, 4)", gcd(-12, 4))
    assert_equal("\\gcd(219, -9)", gcd(219, -9))
    assert_equal("\\gcd(-8, -64)", gcd(-8, -64))
    assert_equal("\\gcd(-5, -5)", gcd(-5, -5))
    assert_equal("\\gcd(-1, 182033)", gcd(-1, 182033))
    assert_equal("\\gcd(25, -6125)", gcd(25, -6125))
    assert_equal("\\gcd(243, -2.9543127E21)", gcd(243, Rational('-2.9543127E21')))

    assert_equal("\\operatorname{gcd}(-12, 4)", gcd(-12, 4))
    assert_equal("\\operatorname{gcd}(219, -9)", gcd(219, -9))
    assert_equal("\\operatorname{gcd}(-8, -64)", gcd(-8, -64))
    assert_equal("\\operatorname{gcd}(-5, -5)", gcd(-5, -5))
    assert_equal("\\operatorname{gcd}(-1, 182033)", gcd(-1, 182033))
    assert_equal("\\operatorname{gcd}(25, -6125)", gcd(25, -6125))
    assert_equal("\\operatorname{gcd}(243, -2.9543127E21)", gcd(243, Rational('-2.9543127E21')))


def test_gcd_float():
    assert_equal("\\gcd(2.4, 3.6)", gcd(Rational('2.4'), Rational('3.6')))
    assert_equal("\\gcd(3.6, 2.4)", gcd(Rational('3.6'), Rational('2.4')))
    assert_equal("\\gcd(\\pi, 3)", gcd(pi, 3))
    assert_equal("\\gcd(618, 1.5)", gcd(618, Rational('1.5')))
    assert_equal("\\gcd(-1.5, 618)", gcd(Rational('-1.5'), 618))
    assert_equal("\\gcd(0.42, 2)", gcd(Rational('0.42'), 2))
    assert_equal("\\gcd(1.43E-13, 21)", gcd(Rational('1.43E-13'), 21))
    assert_equal("\\gcd(21, -143E-13)", gcd(21, Rational('-143E-13')))
    assert_equal("\\gcd(9.80655, 9.80655)", gcd(Rational('9.80655'), Rational('9.80655')))
    assert_equal("\\gcd(0.0000923423, -8341.234802909)", gcd(Rational('0.0000923423'), Rational('-8341.234802909')))
    assert_equal("\\gcd(\\sqrt{5}, \\sqrt{2})", gcd(sqrt(5), sqrt(2)))

    assert_equal("\\operatorname{gcd}(2.4, 3.6)", gcd(Rational('2.4'), Rational('3.6')))
    assert_equal("\\operatorname{gcd}(3.6, 2.4)", gcd(Rational('3.6'), Rational('2.4')))
    assert_equal("\\operatorname{gcd}(\\pi, 3)", gcd(pi, 3))
    assert_equal("\\operatorname{gcd}(618, 1.5)", gcd(618, Rational('1.5')))
    assert_equal("\\operatorname{gcd}(-1.5, 618)", gcd(Rational('-1.5'), 618))
    assert_equal("\\operatorname{gcd}(0.42, 2)", gcd(Rational('0.42'), 2))
    assert_equal("\\operatorname{gcd}(1.43E-13, 21)", gcd(Rational('1.43E-13'), 21))
    assert_equal("\\operatorname{gcd}(21, -143E-13)", gcd(21, Rational('-143E-13')))
    assert_equal("\\operatorname{gcd}(9.80655, 9.80655)", gcd(Rational('9.80655'), Rational('9.80655')))
    assert_equal("\\operatorname{gcd}(0.0000923423, -8341.234802909)", gcd(Rational('0.0000923423'), Rational('-8341.234802909')))
    assert_equal("\\operatorname{gcd}(\\sqrt{5}, \\sqrt{2})", gcd(sqrt(5), sqrt(2)))


def test_gcd_fraction():
    assert_equal("\\gcd(1/2, 3)", gcd(Rational('1/2'), 3))
    assert_equal("\\gcd(3, 1/2)", gcd(3, Rational('1/2')))
    assert_equal("\\gcd(6/2, 3)", gcd(Rational('6/2'), 3))
    assert_equal("\\gcd(1/10, 1/10)", gcd(Rational('1/10'), Rational('1/10')))
    assert_equal("\\gcd(42, 42/6)", gcd(42, Rational('42/6')))
    assert_equal("\\gcd(10000000/10, 10000)", gcd(Rational('10000000/10'), 10000))

    assert_equal("\\operatorname{gcd}(1/2, 3)", gcd(Rational('1/2'), 3))
    assert_equal("\\operatorname{gcd}(3, 1/2)", gcd(3, Rational('1/2')))
    assert_equal("\\operatorname{gcd}(6/2, 3)", gcd(Rational('6/2'), 3))
    assert_equal("\\operatorname{gcd}(1/10, 1/10)", gcd(Rational('1/10'), Rational('1/10')))
    assert_equal("\\operatorname{gcd}(42, 42/6)", gcd(42, Rational('42/6')))
    assert_equal("\\operatorname{gcd}(10000000/10, 10000)", gcd(Rational('10000000/10'), 10000))


def test_gcd_expr():
    assert_equal("\\gcd(1+1, 8)", gcd(1 + 1, 8))
    assert_equal("920*\\gcd(9, 12*4/2)", 920 * gcd(9, 12 * Rational('4/2')))
    assert_equal("\\gcd(32-128, 10)*22", gcd(32 - 128, 10) * 22)
    assert_equal("\\sqrt{\\gcd(1.25E24, 1E12)}", sqrt(gcd(Rational('1.25E24'), Rational('1E12'))))
    assert_equal("\\gcd(92.0, 000+2)", gcd(Rational('92.0'), 000 + 2))

    assert_equal("\\operatorname{gcd}(1+1, 8)", gcd(1 + 1, 8))
    assert_equal("920*\\operatorname{gcd}(9, 12*4/2)", 920 * gcd(9, 12 * Rational('4/2')))
    assert_equal("\\operatorname{gcd}(32-128, 10)*22", gcd(32 - 128, 10) * 22)
    assert_equal("\\sqrt{\\operatorname{gcd}(1.25E24, 1E12)}", sqrt(gcd(Rational('1.25E24'), Rational('1E12'))))
    assert_equal("\\operatorname{gcd}(92.0, 000+2)", gcd(Rational('92.0'), 000 + 2))


def test_gcd_symbol():
    assert_equal("\\gcd(x, y)", gcd(x, y), symbolically=True)
    assert_equal("\\gcd(y, -x)", gcd(y, -x), symbolically=True)
    assert_equal("\\gcd(2y, x)", gcd(2 * y, x), symbolically=True)
    assert_equal("\\gcd(125, 50x)", gcd(125, 50 * x), symbolically=True)
    assert_equal("\\gcd(x + 76, \\sqrt{x} * 4)", gcd(x + 76, sqrt(x) * 4), symbolically=True)
    assert_equal("\\gcd(y, y)", gcd(y, y), symbolically=True)
    assert_equal("y + \\gcd(0.4x, 8/3) / 2", y + gcd(Rational('0.4') * x, Rational('8/3')) / 2, symbolically=True)
    assert_equal("6.673E-11 * (\\gcd(8.85418782E-12, 9x) + 4) / 8y", Rational('6.673E-11') * (gcd(Rational('8.85418782E-12'), 9 * x) + 4) / (8 * y), symbolically=True)

    assert_equal("\\operatorname{gcd}(x, y)", gcd(x, y), symbolically=True)
    assert_equal("\\operatorname{gcd}(y, -x)", gcd(y, -x), symbolically=True)
    assert_equal("\\operatorname{gcd}(2y, x)", gcd(2 * y, x), symbolically=True)
    assert_equal("\\operatorname{gcd}(125, 50x)", gcd(125, 50 * x), symbolically=True)
    assert_equal("\\operatorname{gcd}(x + 76, \\sqrt{x} * 4)", gcd(x + 76, sqrt(x) * 4), symbolically=True)
    assert_equal("\\operatorname{gcd}(y, y)", gcd(y, y), symbolically=True)
    assert_equal("y + \\operatorname{gcd}(0.4x, 8/3) / 2", y + gcd(Rational('0.4') * x, Rational('8/3')) / 2, symbolically=True)
    assert_equal("6.673E-11 * (\\operatorname{gcd}(8.85418782E-12, 9x) + 4) / 8y", Rational('6.673E-11') * (gcd(Rational('8.85418782E-12'), 9 * x) + 4) / (8 * y), symbolically=True)


def test_multiple_parameters():
    assert_equal("\\gcd(830,450)", gcd(830, 450))
    assert_equal("\\gcd(6,321,429)", igcd(6, 321, 429))
    assert_equal("\\gcd(14,2324)", gcd(14, 2324))
    assert_equal("\\gcd(3, 6, 2)", igcd(3, 6, 2))
    assert_equal("\\gcd(144, 2988, 37116)", igcd(144, 2988, 37116))
    assert_equal("\\gcd(144,2988, 37116,18, 72)", igcd(144, 2988, 37116, 18, 72))
    assert_equal("\\gcd(144, 2988, 37116, 18, 72, 12, 6)", igcd(144, 2988, 37116, 18, 72, 12, 6))
    assert_equal("\\gcd(32)", gcd(32, 32))
    assert_equal("\\gcd(-8, 4,-2)", gcd(-8, gcd(4, -2)))
    assert_equal("\\gcd(x, y,z)", gcd(x, gcd(y, z)), symbolically=True)
    assert_equal("\\gcd(6*4,48, 3)", igcd(6 * 4, 48, 3))
    assert_equal("\\gcd(6*4,48,3)", igcd(6 * 4, 48, 3))
    assert_equal("\\gcd(2.4,3.6, 0.6)", gcd(Rational('2.4'), gcd(Rational('3.6'), Rational('0.6'))))
    assert_equal("\\gcd(2.4,3.6,0.6)", gcd(Rational('2.4'), gcd(Rational('3.6'), Rational('0.6'))))
    assert_equal("\\gcd(\\sqrt{3},\\sqrt{2}, \\sqrt{100})", gcd(sqrt(3), gcd(sqrt(2), sqrt(100))))
    assert_equal("\\gcd(1E12, 1E6,1E3, 10)", igcd(Rational('1E12'), Rational('1E6'), Rational('1E3'), 10))

    assert_equal("\\operatorname{gcd}(830,450)", gcd(830, 450))
    assert_equal("\\operatorname{gcd}(6,321,429)", igcd(6, 321, 429))
    assert_equal("\\operatorname{gcd}(14,2324)", gcd(14, 2324))
    assert_equal("\\operatorname{gcd}(3, 6, 2)", igcd(3, 6, 2))
    assert_equal("\\operatorname{gcd}(144, 2988, 37116)", igcd(144, 2988, 37116))
    assert_equal("\\operatorname{gcd}(144,2988, 37116,18, 72)", igcd(144, 2988, 37116, 18, 72))
    assert_equal("\\operatorname{gcd}(144, 2988, 37116, 18, 72, 12, 6)", igcd(144, 2988, 37116, 18, 72, 12, 6))
    assert_equal("\\operatorname{gcd}(32)", gcd(32, 32))
    assert_equal("\\operatorname{gcd}(-8, 4,-2)", gcd(-8, gcd(4, -2)))
    assert_equal("\\operatorname{gcd}(x, y,z)", gcd(x, gcd(y, z)), symbolically=True)
    assert_equal("\\operatorname{gcd}(6*4,48, 3)", igcd(6 * 4, 48, 3))
    assert_equal("\\operatorname{gcd}(6*4,48,3)", igcd(6 * 4, 48, 3))
    assert_equal("\\operatorname{gcd}(2.4,3.6, 0.6)", gcd(Rational('2.4'), gcd(Rational('3.6'), Rational('0.6'))))
    assert_equal("\\operatorname{gcd}(2.4,3.6,0.6)", gcd(Rational('2.4'), gcd(Rational('3.6'), Rational('0.6'))))
    assert_equal("\\operatorname{gcd}(\\sqrt{3},\\sqrt{2}, \\sqrt{100})", gcd(sqrt(3), gcd(sqrt(2), sqrt(100))))
    assert_equal("\\operatorname{gcd}(1E12, 1E6,1E3, 10)", igcd(Rational('1E12'), Rational('1E6'), Rational('1E3'), 10))
