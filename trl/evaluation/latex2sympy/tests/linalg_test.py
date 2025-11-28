from .context import assert_equal
import pytest
from sympy import MatMul, Matrix


def test_linalg_placeholder():
    assert_equal("\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}\\cdot\\variable{v}", MatMul(Matrix([[1, 2], [3, 4]]), Matrix([1, 2])), {'v': Matrix([1, 2])})


def test_linalg_placeholder_multiple():
    assert_equal("\\variable{M}\\cdot\\variable{v}", MatMul(Matrix([[1, 2], [3, 4]]), Matrix([1, 2])), {'M': Matrix([[1, 2], [3, 4]]), 'v': Matrix([1, 2])})


def test_linalg_placeholder_multiple_mul():
    assert_equal("\\begin{pmatrix}3&-1\\end{pmatrix}\\cdot\\variable{M}\\cdot\\variable{v}", MatMul(Matrix([[3, -1]]), Matrix([[1, 2], [3, 4]]), Matrix([1, 2])), {'M': Matrix([[1, 2], [3, 4]]), 'v': Matrix([1, 2])})
