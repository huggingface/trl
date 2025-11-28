from .context import assert_equal
import pytest
from sympy import Symbol, Rational, UnevaluatedExpr, lcm, ilcm, sqrt, pi

x = Symbol('x', real=True)
y = Symbol('y', real=True)
z = Symbol('z', real=True)


def test_lcm_usual():
    assert_equal("\\lcm(6, 4)", lcm(6, 4))
    assert_equal("\\lcm(4, 6)", lcm(4, 6))
    assert_equal("\\lcm(2, 2)", lcm(2, 2))
    assert_equal("\\lcm(0, 21)", UnevaluatedExpr(lcm(0, 21)))
    assert_equal("\\lcm(21, 0)", UnevaluatedExpr(lcm(21, 0)))
    assert_equal("\\lcm(0, 0)", UnevaluatedExpr(lcm(0, 0)))
    assert_equal("\\lcm(9, 21)", lcm(9, 21))
    assert_equal("\\lcm(6128, 24)", lcm(6128, 24))
    assert_equal("\\lcm(24, 6128)", lcm(24, 6128))
    assert_equal("\\lcm(1E20, 1000000)", lcm(Rational('1E20'), 1000000))
    assert_equal("\\lcm(128*10^32, 1)", lcm(Rational('128E32'), 1))

    assert_equal("\\operatorname{lcm}(6, 4)", lcm(6, 4))
    assert_equal("\\operatorname{lcm}(4, 6)", lcm(4, 6))
    assert_equal("\\operatorname{lcm}(2, 2)", lcm(2, 2))
    assert_equal("\\operatorname{lcm}(0, 21)", UnevaluatedExpr(lcm(0, 21)))
    assert_equal("\\operatorname{lcm}(21, 0)", UnevaluatedExpr(lcm(21, 0)))
    assert_equal("\\operatorname{lcm}(0, 0)", UnevaluatedExpr(lcm(0, 0)))
    assert_equal("\\operatorname{lcm}(9, 21)", lcm(9, 21))
    assert_equal("\\operatorname{lcm}(6128, 24)", lcm(6128, 24))
    assert_equal("\\operatorname{lcm}(24, 6128)", lcm(24, 6128))
    assert_equal("\\operatorname{lcm}(1E20, 1000000)", lcm(Rational('1E20'), 1000000))
    assert_equal("\\operatorname{lcm}(128*10^32, 1)", lcm(Rational('128E32'), 1))


def test_lcm_negative():
    assert_equal("\\lcm(-12, 4)", lcm(-12, 4))
    assert_equal("\\lcm(219, -9)", lcm(219, -9))
    assert_equal("\\lcm(-8, -12)", lcm(-8, -12))
    assert_equal("\\lcm(-5, -5)", lcm(-5, -5))
    assert_equal("\\lcm(-1, 182033)", lcm(-1, 182033))
    assert_equal("\\lcm(25, -30)", lcm(25, -30))
    assert_equal("\\lcm(243, -2.9543127E21)", lcm(243, Rational('-2.9543127E21')))

    assert_equal("\\operatorname{lcm}(-12, 4)", lcm(-12, 4))
    assert_equal("\\operatorname{lcm}(219, -9)", lcm(219, -9))
    assert_equal("\\operatorname{lcm}(-8, -12)", lcm(-8, -12))
    assert_equal("\\operatorname{lcm}(-5, -5)", lcm(-5, -5))
    assert_equal("\\operatorname{lcm}(-1, 182033)", lcm(-1, 182033))
    assert_equal("\\operatorname{lcm}(25, -30)", lcm(25, -30))
    assert_equal("\\operatorname{lcm}(243, -2.9543127E21)", lcm(243, Rational('-2.9543127E21')))


def test_lcm_float():
    assert_equal("\\lcm(2.4, 3.6)", lcm(Rational('2.4'), Rational('3.6')))
    assert_equal("\\lcm(3.6, 2.4)", lcm(Rational('3.6'), Rational('2.4')))
    assert_equal("\\lcm(\\pi, 3)", lcm(pi, 3))
    assert_equal("\\lcm(618, 1.5)", lcm(618, Rational('1.5')))
    assert_equal("\\lcm(-1.5, 618)", lcm(Rational('-1.5'), 618))
    assert_equal("\\lcm(0.42, 2)", lcm(Rational('0.42'), 2))
    assert_equal("\\lcm(1.43E-13, 21)", lcm(Rational('1.43E-13'), 21))
    assert_equal("\\lcm(21, -143E-13)", lcm(21, Rational('-143E-13')))
    assert_equal("\\lcm(9.80655, 9.80655)", lcm(Rational('9.80655'), Rational('9.80655')))
    assert_equal("\\lcm(0.0000923423, -8341.234802909)", lcm(Rational('0.0000923423'), Rational('-8341.234802909')))
    assert_equal("\\lcm(\\sqrt{5}, \\sqrt{2})", lcm(sqrt(5), sqrt(2)))

    assert_equal("\\operatorname{lcm}(2.4, 3.6)", lcm(Rational('2.4'), Rational('3.6')))
    assert_equal("\\operatorname{lcm}(3.6, 2.4)", lcm(Rational('3.6'), Rational('2.4')))
    assert_equal("\\operatorname{lcm}(\\pi, 3)", lcm(pi, 3))
    assert_equal("\\operatorname{lcm}(618, 1.5)", lcm(618, Rational('1.5')))
    assert_equal("\\operatorname{lcm}(-1.5, 618)", lcm(Rational('-1.5'), 618))
    assert_equal("\\operatorname{lcm}(0.42, 2)", lcm(Rational('0.42'), 2))
    assert_equal("\\operatorname{lcm}(1.43E-13, 21)", lcm(Rational('1.43E-13'), 21))
    assert_equal("\\operatorname{lcm}(21, -143E-13)", lcm(21, Rational('-143E-13')))
    assert_equal("\\operatorname{lcm}(9.80655, 9.80655)", lcm(Rational('9.80655'), Rational('9.80655')))
    assert_equal("\\operatorname{lcm}(0.0000923423, -8341.234802909)", lcm(Rational('0.0000923423'), Rational('-8341.234802909')))
    assert_equal("\\operatorname{lcm}(\\sqrt{5}, \\sqrt{2})", lcm(sqrt(5), sqrt(2)))


def test_lcm_fraction():
    assert_equal("\\lcm(1/2, 3)", lcm(Rational('1/2'), 3))
    assert_equal("\\lcm(3, 1/2)", lcm(3, Rational('1/2')))
    assert_equal("\\lcm(6/2, 3)", lcm(Rational('6/2'), 3))
    assert_equal("\\lcm(1/10, 1/10)", lcm(Rational('1/10'), Rational('1/10')))
    assert_equal("\\lcm(42, 42/6)", lcm(42, Rational('42/6')))
    assert_equal("\\lcm(10000000/10, 10000)", lcm(Rational('10000000/10'), 10000))

    assert_equal("\\operatorname{lcm}(1/2, 3)", lcm(Rational('1/2'), 3))
    assert_equal("\\operatorname{lcm}(3, 1/2)", lcm(3, Rational('1/2')))
    assert_equal("\\operatorname{lcm}(6/2, 3)", lcm(Rational('6/2'), 3))
    assert_equal("\\operatorname{lcm}(1/10, 1/10)", lcm(Rational('1/10'), Rational('1/10')))
    assert_equal("\\operatorname{lcm}(42, 42/6)", lcm(42, Rational('42/6')))
    assert_equal("\\operatorname{lcm}(10000000/10, 10000)", lcm(Rational('10000000/10'), 10000))


def test_lcm_expr():
    assert_equal("\\lcm(1+1, 8)", lcm(1 + 1, 8))
    assert_equal("920*\\lcm(9, 12*4/2)", 920 * lcm(9, 12 * Rational('4/2')))
    assert_equal("\\lcm(32-128, 10)*22", lcm(32 - 128, 10) * 22)
    assert_equal("\\sqrt{\\lcm(1.25E24, 1E12)}", sqrt(lcm(Rational('1.25E24'), Rational('1E12'))))
    assert_equal("\\lcm(92.0, 000+2)", lcm(Rational('92.0'), 000 + 2))

    assert_equal("\\operatorname{lcm}(1+1, 8)", lcm(1 + 1, 8))
    assert_equal("920*\\operatorname{lcm}(9, 12*4/2)", 920 * lcm(9, 12 * Rational('4/2')))
    assert_equal("\\operatorname{lcm}(32-128, 10)*22", lcm(32 - 128, 10) * 22)
    assert_equal("\\sqrt{\\operatorname{lcm}(1.25E24, 1E12)}", sqrt(lcm(Rational('1.25E24'), Rational('1E12'))))
    assert_equal("\\operatorname{lcm}(92.0, 000+2)", lcm(Rational('92.0'), 000 + 2))


def test_lcm_symbol():
    assert_equal("\\lcm(x, y)", lcm(x, y), symbolically=True)
    assert_equal("\\lcm(y, -x)", lcm(y, -x), symbolically=True)
    assert_equal("\\lcm(2y, x)", lcm(2 * y, x), symbolically=True)
    assert_equal("\\lcm(125, 50x)", lcm(125, 50 * x), symbolically=True)
    assert_equal("\\lcm(x + 76, \\sqrt{x} * 4)", lcm(x + 76, sqrt(x) * 4), symbolically=True)
    assert_equal("\\lcm(y, y)", lcm(y, y), symbolically=True)
    assert_equal("y + \\lcm(0.4x, 8/3) / 2", y + lcm(Rational('0.4') * x, Rational('8/3')) / 2, symbolically=True)
    assert_equal("6.673E-11 * (\\lcm(8.85418782E-12, 9x) + 4) / 8y", Rational('6.673E-11') * (lcm(Rational('8.85418782E-12'), 9 * x) + 4) / (8 * y), symbolically=True)

    assert_equal("\\operatorname{lcm}(x, y)", lcm(x, y), symbolically=True)
    assert_equal("\\operatorname{lcm}(y, -x)", lcm(y, -x), symbolically=True)
    assert_equal("\\operatorname{lcm}(2y, x)", lcm(2 * y, x), symbolically=True)
    assert_equal("\\operatorname{lcm}(125, 50x)", lcm(125, 50 * x), symbolically=True)
    assert_equal("\\operatorname{lcm}(x + 76, \\sqrt{x} * 4)", lcm(x + 76, sqrt(x) * 4), symbolically=True)
    assert_equal("\\operatorname{lcm}(y, y)", lcm(y, y), symbolically=True)
    assert_equal("y + \\operatorname{lcm}(0.4x, 8/3) / 2", y + lcm(Rational('0.4') * x, Rational('8/3')) / 2, symbolically=True)
    assert_equal("6.673E-11 * (\\operatorname{lcm}(8.85418782E-12, 9x) + 4) / 8y", Rational('6.673E-11') * (lcm(Rational('8.85418782E-12'), 9 * x) + 4) / (8 * y), symbolically=True)


def test_multiple_parameters():
    assert_equal("\\lcm(830,450)", lcm(830, 450))
    assert_equal("\\lcm(6,321,429)", ilcm(6, 321, 429))
    assert_equal("\\lcm(14,2324)", lcm(14, 2324))
    assert_equal("\\lcm(3, 6, 2)", ilcm(3, 6, 2))
    assert_equal("\\lcm(8, 9, 21)", ilcm(8, 9, 21))
    assert_equal("\\lcm(144, 2988, 37116)", ilcm(144, 2988, 37116))
    assert_equal("\\lcm(144,2988,37116,18,72)", ilcm(144, 2988, 37116, 18, 72))
    assert_equal("\\lcm(144, 2988, 37116, 18, 72, 12, 6)", ilcm(144, 2988, 37116, 18, 72, 12, 6))
    assert_equal("\\lcm(32)", lcm(32, 32))
    assert_equal("\\lcm(-8, 4, -2)", lcm(-8, lcm(4, -2)))
    assert_equal("\\lcm(x, y, z)", lcm(x, lcm(y, z)), symbolically=True)
    assert_equal("\\lcm(6*4, 48, 3)", ilcm(6 * 4, 48, 3))
    assert_equal("\\lcm(2.4, 3.6, 0.6)", lcm(Rational('2.4'), lcm(Rational('3.6'), Rational('0.6'))))
    assert_equal("\\lcm(\\sqrt{3}, \\sqrt{2},\\sqrt{100})", lcm(sqrt(3), lcm(sqrt(2), sqrt(100))))
    assert_equal("\\lcm(1E12, 1E6, 1E3, 10)", ilcm(Rational('1E12'), Rational('1E6'), Rational('1E3'), 10))

    assert_equal("\\operatorname{lcm}(830,450)", lcm(830, 450))
    assert_equal("\\operatorname{lcm}(6,321,429)", ilcm(6, 321, 429))
    assert_equal("\\operatorname{lcm}(14,2324)", lcm(14, 2324))
    assert_equal("\\operatorname{lcm}(3, 6, 2)", ilcm(3, 6, 2))
    assert_equal("\\operatorname{lcm}(8, 9, 21)", ilcm(8, 9, 21))
    assert_equal("\\operatorname{lcm}(144, 2988, 37116)", ilcm(144, 2988, 37116))
    assert_equal("\\operatorname{lcm}(144,2988,37116,18,72)", ilcm(144, 2988, 37116, 18, 72))
    assert_equal("\\operatorname{lcm}(144, 2988, 37116, 18, 72, 12, 6)", ilcm(144, 2988, 37116, 18, 72, 12, 6))
    assert_equal("\\operatorname{lcm}(32)", lcm(32, 32))
    assert_equal("\\operatorname{lcm}(-8, 4, -2)", lcm(-8, lcm(4, -2)))
    assert_equal("\\operatorname{lcm}(x, y, z)", lcm(x, lcm(y, z)), symbolically=True)
    assert_equal("\\operatorname{lcm}(6*4,48, 3)", ilcm(6 * 4, 48, 3))
    assert_equal("\\operatorname{lcm}(2.4, 3.6,0.6)", lcm(Rational('2.4'), lcm(Rational('3.6'), Rational('0.6'))))
    assert_equal("\\operatorname{lcm}(\\sqrt{3}, \\sqrt{2},\\sqrt{100})", lcm(sqrt(3), lcm(sqrt(2), sqrt(100))))
    assert_equal("\\operatorname{lcm}(1E12,1E6, 1E3, 10)", ilcm(Rational('1E12'), Rational('1E6'), Rational('1E3'), 10))
