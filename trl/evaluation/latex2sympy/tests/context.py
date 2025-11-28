from sympy import simplify, srepr, Add, Mul, Pow, Rational, pi, sqrt, Symbol
from latex2sympy.latex2sympy2 import latex2sympy as process_sympy
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

x = Symbol('x', real=True)
y = Symbol('y', real=True)

# shorthand definitions


def _Add(a, b):
    return Add(a, b, evaluate=False)


def _Mul(a, b):
    return Mul(a, b, evaluate=False)


def _Pow(a, b):
    return Pow(a, b, evaluate=False)


def get_simple_examples(func):
    '''
    Returns an array of tuples, containing the string `input`, sympy `output` using the provided sympy `func`, and `symbolically` boolean
    for calling `compare`.
    '''
    return [
        ("1.1", func(1.1), False),
        ("6.9", func(6.9), False),
        ("3.5", func(3.5), False),
        ("8", func(8), False),
        ("0", func(0), False),
        ("290348E32", func(Rational('290348E32')), False),
        ("1237.293894239480234", func(Rational('1237.293894239480234')), False),
        ("8623.4592104E-2", func(Rational('8623.4592104E-2')), False),
        ("\\pi ", func(pi), False),
        ("\\sqrt{100}", func(sqrt(100)), False),
        ("12,123.4", func(Rational('12123.4')), False),
        ("-9.4", func(-9.4), False),
        ("-35.9825", func(-35.9825), False),
        ("-\\sqrt{5}", func(-sqrt(5)), False),
        ("-324E-3", func(Rational('-324E-3')), False),
        ("-0.23", func(-0.23), False),
        ("\\frac{1}{2}", func(Rational('1/2')), False),
        ("\\frac{6}{2}", func(Rational('6/2')), False),
        ("\\frac{9}{5}", func(Rational('9/5')), False),
        ("\\frac{-42}{6}", func(Rational('-42/6')), False),
        ("-\\frac{325}{3}", func(Rational('-325/3')), False),
        ("\\frac{\\pi }{2}", func(pi / 2), False),
        ("(1+6)/3", func(Rational(1 + 6, 3)), False),
        ("1+6/3", func(1 + Rational('6/3')), False),
        ("7*4/5", func(7 * 4 / 5), False),
        ("15-2.3", func(15 - Rational('2.3')), False),
        ("x", func(x), True),
        ("x + y", func(x + y), True),
        ("\\frac{9x}{4}", func(9 * x / 4), True),
        ("y\\pi", func(y * pi), True),
        ("2y-y-y", func(2 * y - y - y), True)
    ]


def compare(actual, expected, symbolically=False):
    if symbolically:
        assert simplify(actual - expected) == 0
    else:
        actual_exp_tree = srepr(actual)
        expected_exp_tree = srepr(expected)
        try:
            assert actual_exp_tree == expected_exp_tree
        except Exception:
            if isinstance(actual, int) or isinstance(actual, float) or actual.is_number and isinstance(expected, int) or isinstance(expected, float) or expected.is_number:
                assert actual == expected or actual - expected == 0 or simplify(actual - expected) == 0
            else:
                print('expected_exp_tree = ', expected_exp_tree)
                print('actual exp tree = ', actual_exp_tree)
                raise


def assert_equal(latex, expr, variable_values={}, symbolically=False):
    parsed = process_sympy(latex, variable_values)
    compare(parsed, expr, symbolically)
