from jax_automin.interpreter import automin_function

def test_jaxpr_to_source_simple():
    import jax
    import jax.numpy as jnp

    def f(x):
        return x + 1

    source = automin_function(f, jnp.array([1, 2, 3]))

    assert source == """def f(a):
    b = (a + 1)
    return b"""


def test_jaxpr_to_source_matmul():
    import jax
    import jax.numpy as jnp

    def f(x, y):
        return jnp.matmul(x, y)

    source = automin_function(f, jnp.array([[1, 2], [3, 4]]), jnp.array([[1, 2], [3, 4]]))

    assert source == """def f(a, b):
    c = jax.numpy.matmul(a, b)
    return c"""

