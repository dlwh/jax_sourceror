import jax





def automin_function(f, *args, **kwargs):
    jaxpr = jax.make_jaxpr(f)(*args, **kwargs)
    source = jaxpr_to_source(jaxpr.jaxpr, fn_name=f.__name__)
    return source

def jaxpr_to_source(jaxpr, fn_name="function"):
    code_lines = []

    # Generate argument declarations
    args = ', '.join([str(var) for var in jaxpr.invars])
    code_lines.append(f"def {fn_name}({args}):")

    # Generate body of the function
    for eqn in jaxpr.eqns:
        prim = str(eqn.primitive)
        if prim in prim_to_python:
            op = prim_to_python[prim](eqn)
            code_lines.append(f"    {op}")
        else:
            invars = ', '.join(str(var) for var in eqn.invars)
            outvars = ', '.join(str(var) for var in eqn.outvars)
            code_lines.append(f"    {outvars} = {prim}({invars})")

    # Generate return statement
    returns = ', '.join([str(var) for var in jaxpr.outvars])
    code_lines.append(f"    return {returns}")

    return '\n'.join(code_lines)




def _automin_dot_general(eqn):
    x, y = eqn.invars
    d = eqn.params['dimension_numbers']
    precision = eqn.params['precision']
    preferred_element_type = eqn.params['preferred_element_type']

    # recognize simple matmul case
    if d == (((1,), (0,)), ((), ())) and precision == None and preferred_element_type == None:
        return f"{eqn.outvars[0]} = jax.numpy.matmul({x}, {y})"

    return f"{eqn.outvars[0]} = jax.lax.dot_general({x}, {y}, {d}, {precision}, {preferred_element_type})"


def binop(fmt: str):
    def binop_fn(eqn):
        fmt.format(*eqn.outvars, *eqn.invars)

    return binop_fn



prim_to_python = {
    'add': binop("{} = {} + {}"),
    'sub': binop("{} = {} - {}"),
    'mul': binop("{} = {} * {}"),
    'div': binop("{} = {} / {}"),
    'neg': binop("{} = -{}"),
    'dot_general': _automin_dot_general,
}
