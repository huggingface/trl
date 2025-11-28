from .context import assert_equal
import pytest
import hashlib
from sympy import UnevaluatedExpr, Symbol, Mul, Pow, Max, Min, gcd, lcm, floor, ceiling

x = Symbol('x', real=True)
y = Symbol('y', real=True)


def test_variable_letter():
    assert_equal("\\variable{x}", Symbol('x' + hashlib.md5('x'.encode()).hexdigest(), real=True))


def test_variable_digit():
    assert_equal("\\variable{1}", Symbol('1' + hashlib.md5('1'.encode()).hexdigest(), real=True))


def test_variable_letter_subscript():
    assert_equal("\\variable{x_y}", Symbol('x_y' + hashlib.md5('x_y'.encode()).hexdigest(), real=True))


def test_variable_letter_comma_subscript():
    assert_equal("\\variable{x_{i,j}}", Symbol('x_{i,j}' + hashlib.md5('x_{i,j}'.encode()).hexdigest(), real=True))


def test_variable_digit_subscript():
    assert_equal("\\variable{x_1}", Symbol('x_1' + hashlib.md5('x_1'.encode()).hexdigest(), real=True))


def test_variable_after_subscript_required():
    with pytest.raises(Exception):
        assert_equal("\\variable{x_}", Symbol('x_' + hashlib.md5('x_'.encode()).hexdigest(), real=True))


def test_variable_before_subscript_required():
    with pytest.raises(Exception):
        assert_equal("\\variable{_x}", Symbol('_x' + hashlib.md5('_x'.encode()).hexdigest(), real=True))


def test_variable_bad_name():
    with pytest.raises(Exception):
        assert_equal("\\variable{\\sin xy}", None)


def test_variable_in_expr():
    assert_equal("4\\cdot\\variable{x}", 4 * Symbol('x' + hashlib.md5('x'.encode()).hexdigest(), real=True))


def test_variable_greek_letter():
    assert_equal("\\variable{\\alpha }\\alpha", Symbol('\\alpha ' + hashlib.md5('\\alpha '.encode()).hexdigest(), real=True) * Symbol('alpha', real=True))


def test_variable_greek_letter_subscript():
    assert_equal("\\variable{\\alpha _{\\beta }}\\alpha ", Symbol('\\alpha _{\\beta }' + hashlib.md5('\\alpha _{\\beta }'.encode()).hexdigest(), real=True) * Symbol('alpha', real=True))


def test_variable_bad_unbraced_long_subscript():
    with pytest.raises(Exception):
        assert_equal("\\variable{x_yz}", None)


def test_variable_bad_unbraced_long_complex_subscript():
    with pytest.raises(Exception):
        assert_equal("\\variable{x\\beta 10_y\\alpha 20}", None)


def test_variable_braced_subscript():
    assert_equal("\\variable{x\\beta 10_{y\\alpha 20}}", Symbol('x\\beta 10_{y\\alpha 20}' + hashlib.md5('x\\beta 10_{y\\alpha 20}'.encode()).hexdigest(), real=True))


def test_variable_complex_expr():
    assert_equal("4\\cdot\\variable{value1}\\frac{\\variable{value_2}}{\\variable{a}}\\cdot x^2", 4 * Symbol('value1' + hashlib.md5('value1'.encode()).hexdigest(), real=True) * Symbol('value_2' + hashlib.md5('value_2'.encode()).hexdigest(), real=True) / Symbol('a' + hashlib.md5('a'.encode()).hexdigest(), real=True) * x**2)


def test_variable_dollars():
    assert_equal("\\$\\variable{x}", Symbol('x' + hashlib.md5('x'.encode()).hexdigest(), real=True))


def test_variable_percentage():
    assert_equal("\\variable{x}\\%", Mul(Symbol('x' + hashlib.md5('x'.encode()).hexdigest(), real=True), Pow(100, -1, evaluate=False), evaluate=False))


def test_variable_single_arg_func():
    assert_equal("\\floor(\\variable{x})", floor(Symbol('x' + hashlib.md5('x'.encode()).hexdigest(), real=True)))
    assert_equal("\\ceil(\\variable{x})", ceiling(Symbol('x' + hashlib.md5('x'.encode()).hexdigest(), real=True)))


def test_variable_multi_arg_func():
    assert_equal("\\gcd(\\variable{x}, \\variable{y})", UnevaluatedExpr(gcd(Symbol('x' + hashlib.md5('x'.encode()).hexdigest(), real=True), Symbol('y' + hashlib.md5('y'.encode()).hexdigest(), real=True))))
    assert_equal("\\lcm(\\variable{x}, \\variable{y})", UnevaluatedExpr(lcm(Symbol('x' + hashlib.md5('x'.encode()).hexdigest(), real=True), Symbol('y' + hashlib.md5('y'.encode()).hexdigest(), real=True))))
    assert_equal("\\max(\\variable{x}, \\variable{y})", Max(Symbol('x' + hashlib.md5('x'.encode()).hexdigest(), real=True), Symbol('y' + hashlib.md5('y'.encode()).hexdigest(), real=True), evaluate=False))
    assert_equal("\\min(\\variable{x}, \\variable{y})", Min(Symbol('x' + hashlib.md5('x'.encode()).hexdigest(), real=True), Symbol('y' + hashlib.md5('y'.encode()).hexdigest(), real=True), evaluate=False))
