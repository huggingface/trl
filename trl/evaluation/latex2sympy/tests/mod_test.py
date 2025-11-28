from .context import assert_equal
import pytest
from sympy import Symbol, Rational, Mod, sqrt, nsimplify, pi, GoldenRatio
from sympy.physics.units import hbar

x = Symbol('x', real=True)
y = Symbol('y', real=True)


def test_mod_usual():
    assert_equal("128\\mod 3", Mod(128, 3))
    assert_equal("7\\mod 128", Mod(7, 128))
    assert_equal("5\\mod 10", Mod(5, 10))
    assert_equal("5\\mod 5", Mod(5, 5))
    assert_equal("3\\mod 2", Mod(3, 2))
    assert_equal("0 \\mod 6", Mod(0, 6))
    assert_equal("6109\\mod 28", Mod(6109, 28))
    assert_equal("4000000000\\mod 28791", Mod(4000000000, 28791))
    assert_equal("128*10^300\\mod 876123", Mod(Rational('128E300'), 876123))
    assert_equal("876,123\\mod 128E300)", Mod(876123, Rational('128E300')))


def test_mod_negative():
    assert_equal("-1\\mod 2", Mod(-1, 2))
    assert_equal("-3\\mod 3", Mod(-3, 3))
    assert_equal("-12\\mod -12", Mod(-12, -12))
    assert_equal("-128\\mod 4", Mod(-128, 4))
    assert_equal("9\\mod -213", Mod(9, -213))
    assert_equal("123123\\mod -541", Mod(123123, -541))
    assert_equal("-123123\\mod 541", Mod(-123123, 541))
    assert_equal("-97E34\\mod 7", Mod(Rational('-97E34'), 7))


def test_mod_fraction():
    assert_equal("1/2\\mod 3", Mod(Rational(1, 2), 3))
    assert_equal("6/2\\mod 3", Mod(Rational(6, 2), 3))
    assert_equal("-14/2\\mod 5", Mod(Rational(-14, 2), 5))
    assert_equal("123\\mod (42/6)", Mod(123, Rational(42, 6)))
    assert_equal("431\\mod (2/123)", Mod(431, Rational(2, 123)))
    assert_equal("5/5\\mod (5/5)", Mod(Rational(5, 5), Rational(5, 5)))
    assert_equal("849/-21\\mod (092/2)", Mod(Rational(849, -21), Rational(92, 2)))
    assert_equal("13*10^9\\mod (21/-2)", Mod(13E9, Rational(21, -2)))


def test_mod_float():
    assert_equal("0.41\\mod 2", Mod(Rational('0.41'), 2))
    assert_equal("143E-13\\mod 21", Mod(Rational('143E-13'), 21))
    assert_equal("-9.80665\\mod 9.80665", Mod(-9.80665, 9.80665))
    assert_equal("0.0000923423\\mod -8341.234802909", nsimplify(Mod(0.0000923423, -8341.234802909)))
    assert_equal("\\sqrt{5}\\mod \\sqrt{2}", Mod(sqrt(5), sqrt(2)))
    assert_equal("987\\mod \\pi", Mod(987, pi))
    assert_equal("\\pi\\mod ((1+\\sqrt{5})/2)", Mod(pi, nsimplify(GoldenRatio)), symbolically=True)
    assert_equal("1234\\mod 1E-29", Mod(1234, Rational('1E-29'), evaluate=False))


def test_mod_expr():
    assert_equal("1+1\\mod 2", 1 + Mod(1, 2))
    assert_equal("876123\\mod 128\\times 10^300", Mod(876123, 128) * 1E300)
    assert_equal("141\\mod 9/3", Rational(Mod(141, 9) / 3))
    assert_equal("872 / (12\\mod 9 * 4) * 2", Rational(2 * 872, (Mod(12, 9) * 4)))
    assert_equal("1E-32 * (1E29\\mod 74)", Rational('1E-32') * Mod(Rational('1E29'), 74))
    assert_equal("299,792,458\\mod 9.81", Mod(299792458, Rational('9.81')))


def test_mod_symbol():
    assert_equal("x\\mod y", Mod(x, y))
    assert_equal("2x\\mod y", Mod(2 * x, y))
    assert_equal("y + 3\\mod 2 / 4", y + Rational(Mod(3, 2), 4), symbolically=True)
    assert_equal("0.5x * 2 + \\sqrt{x}\\mod 8y", 0.5 * x * 2 + Mod(sqrt(x), 8 * y), symbolically=True)
    assert_equal("6.673E-11 * ((8.85418782E-12\\mod 9x) + 4) / 2y", Rational('6.673E-11') * (Mod(Rational('8.85418782E-12'), 9 * x) + 4) / (2 * y), symbolically=True)
