import jax
from jax._src.core import Literal


def automin_function(f, *args, **kwargs):
    closed_jaxpr = jax.make_jaxpr(f)(*args, **kwargs)
    jaxpr = constant_fold_jaxpr(closed_jaxpr.jaxpr)
    source = jaxpr_to_source(jaxpr, fn_name=f.__name__)
    return source

def jaxpr_to_source(jaxpr, fn_name="function"):
    code_lines = []

    # Generate argument declarations
    args = ', '.join([str(var) for var in jaxpr.invars])
    code_lines.append(f"def {fn_name}({args}):")

    # Generate body of the function
    for eqn in jaxpr.eqns:
        # TODO: constant folding
        # if all(isinstance(v, Literal) for v in eqn.invars):
        #     # go ahead and eval it
        #     out = eqn.primitive.bind(*eqn.invars, **eqn.params)
        #     outvars = ', '.join(str(var) for var in eqn.outvars)
        #     code_lines.append(f"    {outvars} = {out}")
        #     continue
        #

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


def constant_fold_jaxpr(jaxpr: jax.core.Jaxpr):
    """
    Given a jaxpr, return a new jaxpr with all constant folding done.
    """
    env = {}
    new_eqns = []

    def read(var):
        if isinstance(var, Literal):
            return var.val
        else:
            return env.get(var, None)

    def read_or_self(var):
        out = read(var)
        if out is None:
            return var
        else:
            return out

    for eqn in jaxpr.eqns:
        vals = [read(var) for var in eqn.invars]
        if all(val is not None for val in vals):
            # go ahead and eval it
            out = eqn.primitive.bind(*vals, **eqn.params)
            if not isinstance(out, tuple):
                out = (out,)

            for var, val in zip(eqn.outvars, out):
                env[var] = val
        else:
            new_eqns.append(eqn)


    # now that we've evalled everything, inline all the constants
    out_eqns = []
    for eqn in new_eqns:
        eqn = eqn.replace(invars=tuple(read_or_self(var) for var in eqn.invars))
        out_eqns.append(eqn)

    return jaxpr.replace(eqns=out_eqns)



def _automin_dot_general(eqn):
    x, y = eqn.invars
    d = eqn.params['dimension_numbers']
    precision = eqn.params['precision']
    preferred_element_type = eqn.params['preferred_element_type']

    # recognize simple matmul case
    if d == (((1,), (0,)), ((), ())) and precision == None and preferred_element_type == None:
        return f"{eqn.outvars[0]} = jax.numpy.matmul({x}, {y})"

    return f"{eqn.outvars[0]} = jax.lax.dot_general({x}, {y}, {d}, {precision}, {preferred_element_type})"

def _sourcify_dynamic_slice(eqn):
    sliced = eqn.invars[0]
    invars = ', '.join(str(var) for var in eqn.invars[1:])
    outvars = ', '.join(str(var) for var in eqn.outvars)
    params = ', '.join(f"{k}={v}" for k, v in eqn.params.items())
    return f"{outvars} = jax.lax.dynamic_slice({sliced}, ({invars},), {params})"

def binop(fmt: str):
    def binop_fn(eqn):
        return fmt.format(*eqn.outvars, *eqn.invars)

    return binop_fn

def dumb_fn(fn_name):
    def dumb_fn_fn(eqn):
        invars = ', '.join(str(var) for var in eqn.invars)
        outvars = ', '.join(str(var) for var in eqn.outvars)
        if eqn.params:
            params = ', '.join(f"{k}={v}" for k, v in eqn.params.items())
            return f"{outvars} = {fn_name}({invars}, {params})"
        return f"{outvars} = {fn_name}({invars})"

    return dumb_fn_fn

prim_to_python = {
    'add': binop("{} = {} + {}"),
    'sub': binop("{} = {} - {}"),
    'mul': binop("{} = {} * {}"),
    'div': binop("{} = {} / {}"),
    'neg': binop("{} = -{}"),
    'lt': binop("{} = {} < {}"),
    'gt': binop("{} = {} > {}"),
    'le': binop("{} = {} <= {}"),
    'ge': binop("{} = {} >= {}"),
    'eq': binop("{} = {} == {}"),
    'ne': binop("{} = {} != {}"),
    'min': binop("{} = jax.lax.min({}, {})"),
    'max': binop("{} = jax.lax.max({}, {})"),
    'select_n': dumb_fn('jax.lax.select_n'),
    'dynamic_slice': _sourcify_dynamic_slice,
    'squeeze': dumb_fn('jax.lax.squeeze'),
    'dot_general': _automin_dot_general,
}


