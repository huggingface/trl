from .context import assert_equal, get_simple_examples
import pytest
from sympy import floor

examples = get_simple_examples(floor)


@pytest.mark.parametrize('input, output, symbolically', examples)
def test_floor_func(input, output, symbolically):
    assert_equal("\\floor({input})".format(input=input), output, symbolically=symbolically)


@pytest.mark.parametrize('input, output, symbolically', examples)
def test_floor_operatorname(input, output, symbolically):
    assert_equal("\\operatorname{{floor}}({input})".format(input=input), output, symbolically=symbolically)


@pytest.mark.parametrize('input, output, symbolically', examples)
def test_floor_cmd(input, output, symbolically):
    assert_equal("\\lfloor {input}\\rfloor".format(input=input), output, symbolically=symbolically)
    assert_equal("\\left\\lfloor {input}\\right\\rfloor".format(input=input), output, symbolically=symbolically)
    assert_equal("\\mleft\\lfloor {input}\\mright\\rfloor".format(input=input), output, symbolically=symbolically)


@pytest.mark.parametrize('input, output, symbolically', examples)
def test_floor_corners(input, output, symbolically):
    assert_equal("\\llcorner {input}\\lrcorner".format(input=input), output, symbolically=symbolically)
    assert_equal("\\left\\llcorner {input}\\right\\lrcorner".format(input=input), output, symbolically=symbolically)
    assert_equal("\\mleft\\llcorner {input}\\mright\\lrcorner".format(input=input), output, symbolically=symbolically)
