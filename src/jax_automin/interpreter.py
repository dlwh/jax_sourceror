import ast
from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
from equinox import is_array
from jax.core import Literal, Var


def automin_function(f, *args, **kwargs):
    closed_jaxpr = jax.make_jaxpr(f)(*args, **kwargs)
    jaxpr = constant_fold_jaxpr(closed_jaxpr.jaxpr)
    source = jaxpr_to_source(jaxpr, fn_name=f.__name__)
    return source

def jaxpr_to_source(jaxpr, fn_name="function"):
    node = jaxpr_to_py_ast(jaxpr, fn_name=fn_name)
    ast.fix_missing_locations(node)
    return ast.unparse(node)


def jaxpr_to_py_ast(jaxpr, fn_name="function"):
    # Generate argument declarations
    # args = ', '.join([str(var) for var in jaxpr.invars])

    ast_args = [ast.arg(arg=str(var), annotation=None) for var in jaxpr.invars]
    ast_args = ast.arguments(args=ast_args, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[])

    stmts = []

    # Generate body of the function
    for eqn in jaxpr.eqns:
        prim = str(eqn.primitive)
        if prim in prim_to_python:
            op = prim_to_python[prim](eqn)
            stmts.append(op)
        else:
            stmts.append(dumb_fn(prim)(eqn))

    # Generate return statement
    # returns = ', '.join([str(var) for var in jaxpr.outvars])
    if len(jaxpr.outvars) == 1:
        returns = ast.Name(id=str(jaxpr.outvars[0]), ctx=ast.Load())
    else:
        returns = ast.Tuple(elts=[ast.Name(id=str(var), ctx=ast.Load()) for var in jaxpr.outvars], ctx=ast.Load())
    stmts.append(ast.Return(value=returns))

    return ast.FunctionDef(name=fn_name, args=ast_args, body=stmts, decorator_list=[])


def constant_fold_jaxpr(jaxpr: jax.core.Jaxpr):
    """
    Given a jaxpr, return a new jaxpr with all constant folding done.
    """
    return eval_jaxpr(jaxpr, {})

def eval_jaxpr(jaxpr, env):
    env = env.copy()
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
            return Literal(out, var.aval)

    for eqn in jaxpr.eqns:
        vals = [read(var) for var in eqn.invars]
        if eqn.primitive.name in constant_fold_blacklist:
            new_eqns.append(eqn)
        elif all(val is not None for val in vals):
            # go ahead and eval it
            out = _eval_eqn(eqn, vals)
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

    # sub in any constants for outvars
    outvars = tuple(read_or_self(var) for var in jaxpr.outvars)

    return jaxpr.replace(eqns=out_eqns, outvars=outvars)


def _eval_eqn(eqn, vals):
    if eqn.primitive.name == "closed_call":
        assert eqn.primitive.call_primitive == True
        assert eqn.primitive.map_primitive == False

        out = eval_jaxpr(eqn.params['call_jaxpr'].jaxpr, { var: val for var, val in zip(eqn.params['call_jaxpr'].jaxpr.invars, vals) })
    else:
        out = eqn.primitive.bind(*vals, **eqn.params)
    return out


def _astify_dot_general(eqn):
    x, y = eqn.invars
    d = eqn.params['dimension_numbers']
    precision = eqn.params['precision']
    preferred_element_type = eqn.params['preferred_element_type']

    # recognize simple matmul case
    if d == (((1,), (0,)), ((), ())) and precision == None and preferred_element_type == None:
        # return f"{eqn.outvars[0]} = jax.numpy.matmul({x}, {y})"
        return ast.Assign(targets=_astify_outvars(eqn.outvars), value=ast.Call(func=ast.Attribute(value=ast.Name(id='jax.numpy', ctx=ast.Load()), attr='matmul', ctx=ast.Load()), args=[_astify_var(x), _astify_var(y)], keywords=[]))

    # return f"{eqn.outvars[0]} = jax.lax.dot_general({x}, {y}, {d}, {precision}, {preferred_element_type})"
    return ast.Assign(targets=_astify_outvars(eqn.outvars), value=ast.Call(func=ast.Attribute(value=ast.Name(id='jax.lax', ctx=ast.Load()), attr='dot_general', ctx=ast.Load()), args=[_astify_var(x), _astify_var(y), _astify_var(d), _astify_var(precision), _astify_var(preferred_element_type)], keywords=[]))

def _sourcify_dynamic_slice(eqn):
    sliced = eqn.invars[0]
    # invars = ', '.join(_astify_var(var) for var in eqn.invars[1:])
    # outvars = ', '.join(repr(var) for var in eqn.outvars)
    # params = ', '.join(f"{k}={v}" for k, v in eqn.params.items())
    # return f"{outvars} = jax.lax.dynamic_slice({sliced}, ({invars},), {params})"
    # now we use ast
    outvars = _astify_outvars(eqn.outvars)
    invars = ast.Tuple(elts=[_astify_var(var) for var in eqn.invars[1:]], ctx=ast.Load())
    params = [ast.keyword(arg=k, value=_astify_var(v)) for k, v in eqn.params.items()]
    return ast.Assign(targets=outvars, value=ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='jax.lax', ctx=ast.Load()),
            attr='dynamic_slice',
            ctx=ast.Load()
        ),
        args=[_astify_var(sliced), invars],
        keywords=params
    ))



def is_array(arr):
    return isinstance(arr, (np.ndarray, np.generic, jnp.ndarray))

def _astify_var(var):
    if isinstance(var, Literal):
        if is_array(var.val):
            if var.val.ndim == 0:
                expr = ast.Constant(value=var.val.item())
                return expr
            else:
                expr = ast.parse(f"jax.numpy.{repr(var.val)}") # TODO: super hacky
                return expr
                # return ast.Constant(value=var.val)
                # need to synthesize a call to jax.numpy.array
                # return ast.Call(
                #     func=ast.Attribute(
                #         value=ast.Name(id='jax.numpy', ctx=ast.Load()),
                #         attr='array',
                #         ctx=ast.Load()
                #     ),
                #     args=
                #     keywords=[]
                # )

        else:
            return ast.parse(repr(var.val))
    elif isinstance(var, Var):
        return ast.Name(id=repr(var))
    else:
        return ast.parse(repr(var)).body[0]


def _astify_outvars(outvars):
    return [ast.Name(repr(v)) for v in outvars]


def assign_expr(call_expr: Callable):
    def binop_fn(eqn):
        return ast.Assign(_astify_outvars(eqn.outvars), call_expr(*[_astify_var(v) for v in eqn.invars],
                                                                  **{k: _astify_var(v) for k, v in eqn.params.items()}
                                                                  ))


    return binop_fn

def binop_fn(op: Union[ast.operator, ast.cmpop]):
    return assign_expr(lambda x, y: ast.BinOp(left=x, op=op, right=y))

def dumb_fn(fn_name):
    return assign_expr(lambda *args, **kwargs: ast.Call(
        func=ast.Name(id=fn_name, ctx=ast.Load()),
        args=list(args),
        keywords=[ast.keyword(arg=k, value=v) for k, v in kwargs.items()]
    ))

def _reduce_fn(fn_name: str):
    def dumb_fn_fn(eqn):
        invars = [_astify_var(v) for v in eqn.invars]
        outvars = _astify_outvars(eqn.outvars)
        if eqn.params:
            params = eqn.params.copy()
            params['axis'] = tuple(params['axes'])
            del params['axes']
            call_op = ast.Call(
                func=ast.Name(id=fn_name, ctx=ast.Load()),
                args=invars,
                keywords=[ast.keyword(arg=k, value=_astify_var(v)) for k, v in params.items()]
            )
        else:
            call_op = ast.Call(
                func=ast.Name(id=fn_name, ctx=ast.Load()),
                args=invars,
                keywords=[]
            )

        return ast.Assign(outvars, call_op)

    return dumb_fn_fn

prim_to_python = {
    'add': binop_fn(ast.Add()),
    'sub': binop_fn(ast.Sub()),
    'mul': binop_fn(ast.Mult()),
    'div': binop_fn(ast.Div()),
    'neg': dumb_fn('jax.lax.neg'),
    'lt': binop_fn(ast.Lt()),
    'gt': binop_fn(ast.Gt()),
    'le': binop_fn(ast.LtE()),
    'ge': binop_fn(ast.GtE()),
    'eq': binop_fn(ast.Eq()),
    'ne': binop_fn(ast.NotEq()),
    'min': dumb_fn('jax.lax.min'),
    'max': dumb_fn('jax.lax.max'),
    'select_n': dumb_fn('jax.lax.select_n'),
    'dynamic_slice': _sourcify_dynamic_slice,
    'squeeze': dumb_fn('jax.lax.squeeze'),
    'dot_general': _astify_dot_general,
    'broadcast_in_dim': dumb_fn('jax.lax.broadcast_in_dim'),
    'broadcast': dumb_fn('jax.lax.broadcast'),
    'reshape': dumb_fn('jax.numpy.reshape'),
    'reduce_sum': _reduce_fn('jax.numpy.sum'),
    'transpose': dumb_fn('jax.lax.transpose'),
}

constant_fold_blacklist = {
    'broadcast_in_dim',
    'broadcast',
}

skolem_id = 0

def _sourcify_scan(eqn):
    global skolem_id
    assert eqn.primitive.name == 'scan'

    jaxpr = eqn.params['jaxpr']
    jaxpr = constant_fold_jaxpr(jaxpr.jaxpr)

    fn_name = f"scan_fn_{skolem_id}"
    fn_source = jaxpr_to_source(jaxpr, fn_name)

    reverse = eqn.params['reverse']
    length = eqn.params['length']
    unroll = eqn.params['unroll']

    return f"""
{fn_source}

    """


