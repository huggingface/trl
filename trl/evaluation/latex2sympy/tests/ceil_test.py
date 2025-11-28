from .context import assert_equal, get_simple_examples
import pytest
from sympy import ceiling

examples = get_simple_examples(ceiling)


@pytest.mark.parametrize('input, output, symbolically', examples)
def test_ceil_func(input, output, symbolically):
    assert_equal("\\ceil({input})".format(input=input), output, symbolically=symbolically)


@pytest.mark.parametrize('input, output, symbolically', examples)
def test_ceil_operatorname(input, output, symbolically):
    assert_equal("\\operatorname{{ceil}}({input})".format(input=input), output, symbolically=symbolically)


@pytest.mark.parametrize('input, output, symbolically', examples)
def test_ceil_cmd(input, output, symbolically):
    assert_equal("\\lceil {input}\\rceil".format(input=input), output, symbolically=symbolically)
    assert_equal("\\left\\lceil {input}\\right\\rceil".format(input=input), output, symbolically=symbolically)
    assert_equal("\\mleft\\lceil {input}\\mright\\rceil".format(input=input), output, symbolically=symbolically)


@pytest.mark.parametrize('input, output, symbolically', examples)
def test_ceil_corners(input, output, symbolically):
    assert_equal("\\ulcorner {input}\\urcorner".format(input=input), output, symbolically=symbolically)
    assert_equal("\\left\\ulcorner {input}\\right\\urcorner".format(input=input), output, symbolically=symbolically)
    assert_equal("\\mleft\\ulcorner {input}\\mright\\urcorner".format(input=input), output, symbolically=symbolically)
