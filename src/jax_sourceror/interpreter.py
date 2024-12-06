import dataclasses
import enum
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union

import ast_comments as ast
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.core import ClosedJaxpr
from jax._src.source_info_util import user_frame, user_frames
from jax.sharding import NamedSharding
from jax._src.custom_derivatives import CustomJVPCallPrimitive
from jax.experimental.pjit import _UNSPECIFIED
from jax.core import Literal, Var, Jaxpr

from jax_sourceror.utils import IdentityMap, IdentitySet

@dataclass
class SourcerorState():
    """State for the auto-minimizer. Basically just in charge of naming variables."""
    _var_names: IdentityMap[Var, str] = dataclasses.field(default_factory=IdentityMap)
    _used_fn_names: set[str] = dataclasses.field(default_factory=set)
    _skolem_count: int = 0
    use_jax_typing: bool = True

    def name(self, var, ctx=ast.Load()) -> ast.Name:
        return ast.Name(id=self.str_name(var), ctx=ctx)

    def str_name(self, var: Var):
        # Names things in a way vaguely compatible with JAX's naming scheme, which is 'a'-'z' followed by 'aa'-'az' etc.
        if var in self._var_names:
            return self._var_names[var]
        else:
            cur_count = len(self._var_names)
            name = ""
            while cur_count >= 26:
                name += chr(ord('a') + cur_count % 26)
                cur_count //= 26

            name += chr(ord('a') + cur_count)

            name = name[::-1]

            if name in ['def', 'if', 'or', 'and', 'not', 'for', 'as', 'in', 'is']:
                name = f"{name}_"


            self._var_names[var] = name

            return name

    def skolem(self, prefix: str):
        self._skolem_count += 1
        return f"{prefix}_{self._skolem_count}"

    def heuristic_fn_skolem(self, jaxpr: Jaxpr, default: Optional[str] = None):
        if default is None:
            default = "fn"

        name = _attempt_to_sniff_fn_name_for_jaxpr(jaxpr) or default
        if name in self._used_fn_names:
            return self.skolem(name)
        else:
            self._used_fn_names.add(name)
            return name



def sourcerize(f, *, use_jax_typing: bool = False):
    def return_fn(*args, **kwargs):
        closed_jaxpr = eqx.filter_make_jaxpr(f)(*args, **kwargs)[0]
        jaxpr = closed_jaxpr.jaxpr
        state = SourcerorState(use_jax_typing=use_jax_typing)
        try:
            name = f.__name__
        except AttributeError:
            name = "unknown"
        node = jaxpr_to_py_ast(state, jaxpr, fn_name=name, unique_fn_name=False)
        node = _maybe_wrap_fn_for_leaves(node, f, len(args) + len(kwargs))
        return _render_ast(node)


    return return_fn


def _render_ast(node):
    ast.fix_missing_locations(node)
    source = ast.unparse(node)
    return source


def register_prim_handler(prim_name, handler):
    """
    Register a handler for a primitive for automin
    :param prim_name:
    :param handler:
    :return:
    """
    if prim_name in prim_to_python:
        warnings.warn(f"Overwriting handler for primitive {prim_name}")
    prim_to_python[prim_name] = handler


def primitive_handler(prim_name):
    """
    Decorator to register a handler for a primitive.
    :param prim_name:
    :return:
    """
    def decorator(fn):
        register_prim_handler(prim_name, fn)
        return fn
    return decorator


def _assign_stmt(call_expr: Callable):
    """
    Create a handler for a primitive that is a simple assignment.
    :param call_expr:
    :return:
    """
    def binop_fn(state, eqn):
        invars = [_astify_atom(state, v) for v in eqn.invars]
        outvars = _astify_outvars(state, eqn.outvars)
        return ast.Assign(outvars, call_expr(*invars,
                                             **{k: _astify_value(v) for k, v in eqn.params.items()}
                                             ))
    return binop_fn

def _binop_fn(op: ast.operator):
    return _assign_stmt(lambda x, y: ast.BinOp(left=x, op=op, right=y))

def _cmpop_fn(op: ast.cmpop):
    return _assign_stmt(lambda x, y: ast.Compare(left=x, ops=[op], comparators=[y]))


def normal_fn(fn_name):
    """
    Create a handler for a normal function call.
    :param fn_name:
    :return:
    """
    return _assign_stmt(lambda *args, **kwargs: ast.Call(
        func=ast.Name(id=fn_name, ctx=ast.Load()),
        args=list(args),
        keywords=[ast.keyword(arg=k, value=v) for k, v in kwargs.items()]
    ))



def _reduce_fn(fn_name: str):
    def reduce_fn_inner(state: SourcerorState, eqn):
        invars = [_astify_atom(state, v) for v in eqn.invars]
        outvars = _astify_outvars(state, eqn.outvars)
        if eqn.params:
            params = eqn.params.copy()
            params['axis'] = tuple(params['axes'])
            del params['axes']
            call_op = ast.Call(
                func=ast.Name(id=fn_name, ctx=ast.Load()),
                args=invars,
                keywords=[ast.keyword(arg=k, value=_astify_value(v)) for k, v in params.items()]
            )
        else:
            call_op = ast.Call(
                func=ast.Name(id=fn_name, ctx=ast.Load()),
                args=invars,
                keywords=[]
            )

        return ast.Assign(outvars, call_op)

    return reduce_fn_inner


prim_to_python = {
}

register_prim_handler('add', _binop_fn(ast.Add()))
register_prim_handler('sub', _binop_fn(ast.Sub()))
register_prim_handler('mul', _binop_fn(ast.Mult()))
register_prim_handler('div', _binop_fn(ast.Div()))
register_prim_handler('lt', _cmpop_fn(ast.Lt()))
register_prim_handler('gt', _cmpop_fn(ast.Gt()))
register_prim_handler('le', _cmpop_fn(ast.LtE()))
register_prim_handler('ge', _cmpop_fn(ast.GtE()))
register_prim_handler('eq', _cmpop_fn(ast.Eq()))
register_prim_handler('ne', _cmpop_fn(ast.NotEq()))
# register_prim_handler('min', normal_fn('jax.lax.min'))
# register_prim_handler('max', normal_fn('jax.lax.max'))
# register_prim_handler('select_n', normal_fn('jax.lax.select_n'))
# register_prim_handler('squeeze', normal_fn('jax.lax.squeeze'))
# register_prim_handler('broadcast', normal_fn('jax.lax.broadcast'))
register_prim_handler('reduce_sum', _reduce_fn('jnp.sum'))
# register_prim_handler('transpose', normal_fn('jax.lax.transpose'))
# register_prim_handler('clamp', normal_fn('jax.lax.clamp'))

normal_fns = {
    'min': 'jax.lax.min',
    'max': 'jax.lax.max',
    'select_n': 'jax.lax.select_n',
    'squeeze': 'jax.lax.squeeze',
    'broadcast': 'jax.lax.broadcast',
    'transpose': 'jax.lax.transpose',
    'clamp': 'jax.lax.clamp',
    # 'reduce_sum': 'jnp.sum',
    'reduce_max': 'jnp.max',
    'reduce_min': 'jnp.min',
    'is_finite': 'jnp.isfinite',
    # misc jax.lax functions
    'integer_pow': 'jax.lax.integer_pow',
    'stop_gradient': 'jax.lax.stop_gradient',
    'neg': 'jnp.negative',
    'abs': 'jnp.abs',
    'sin': 'jnp.sin',
    'cos': 'jnp.cos',
    'tan': 'jnp.tan',
    'asin': 'jnp.arcsin',
    'acos': 'jnp.arccos',
    'atan': 'jnp.arctan',
    'sinh': 'jnp.sinh',
    'cosh': 'jnp.cosh',
    'tanh': 'jnp.tanh',
    'asinh': 'jnp.arcsinh',
    'acosh': 'jnp.arccosh',
    'atanh': 'jnp.arctanh',
    'exp': 'jnp.exp',
    'log': 'jnp.log',
    'log1p': 'jnp.log1p',
    'expm1': 'jnp.expm1',
    'sqrt': 'jnp.sqrt', 
    'square': 'jnp.square',
    'reciprocal': 'jnp.reciprocal',
    'sign': 'jnp.sign',
    'rsqrt': 'jax.lax.rsqrt',
    # 'concatenate': 'jnp.concatenate',
}



for k, v in normal_fns.items():
    register_prim_handler(k, normal_fn(v))


@primitive_handler('cumsum')
def _astify_cumsum(state, eqn):
    invars = [_astify_atom(state, v) for v in eqn.invars]
    outvars = _astify_outvars(state, eqn.outvars)
    axis = eqn.params['axis']
    reverse = eqn.params['reverse']

    if reverse:
        return ast.Assign(outvars, ast.Call(
            func=ast.Name(id='jax.lax.cumsum', ctx=ast.Load()),
            args=[invars[0]],
            keywords=[ast.keyword(arg='axis', value=_astify_value(axis)), ast.keyword(arg='reverse', value=ast.NameConstant(value=True))]
        ))
    else:
        return ast.Assign(outvars, ast.Call(
            func=ast.Name(id='jnp.cumsum', ctx=ast.Load()),
            args=[invars[0]],
            keywords=[ast.keyword(arg='axis', value=_astify_value(axis))]
        ))


@primitive_handler('cumprod')
def _astify_cumprod(state, eqn):
    invars = [_astify_atom(state, v) for v in eqn.invars]
    outvars = _astify_outvars(state, eqn.outvars)
    axis = eqn.params['axis']
    reverse = eqn.params['reverse']

    if reverse:
        return ast.Assign(outvars, ast.Call(
            func=ast.Name(id='jax.lax.cumprod', ctx=ast.Load()),
            args=[invars[0]],
            keywords=[ast.keyword(arg='axis', value=_astify_value(axis)), ast.keyword(arg='reverse', value=ast.NameConstant(value=True))]
        ))
    else:
        return ast.Assign(outvars, ast.Call(
            func=ast.Name(id='jnp.cumprod', ctx=ast.Load()),
            args=[invars[0]],
            keywords=[ast.keyword(arg='axis', value=_astify_value(axis))]
        ))


@primitive_handler('concatenate')
def _astify_concatenate(state, eqn):
    invars = [_astify_atom(state, v) for v in eqn.invars]
    outvars = _astify_outvars(state, eqn.outvars)
    axis = eqn.params['dimension']
    return ast.Assign(outvars, ast.Call(
        func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr='concatenate', ctx=ast.Load()),
        args=[ast.Tuple(elts=invars, ctx=ast.Load())],
        keywords=[ast.keyword(arg='axis', value=_astify_value(axis))]
    ))



def _maybe_wrap_fn_for_leaves(node, f, num_args):
    if len(node.args.args) == num_args:
        return node

    wrapped_node = ast.FunctionDef(name=f.__name__,
                                   args=ast.arguments(
                                       args=[],
                                       vararg=ast.arg(arg="args", annotation=None),
                                       kwarg=ast.arg(arg="kwargs", annotation=None),
                                       kwonlyargs=[], kw_defaults=[], defaults=[],
                                       posonlyargs=[]),
                                   body=[
                                       node,
                                       ast.Return(ast.Call(func=ast.Name(id=node.name, ctx=ast.Load()),
                                                           args=[ast.Starred(ast.Call(func=ast.Attribute(value=ast.Name(id="jax", ctx=ast.Load()),
                                                                                                         attr="tree_leaves",
                                                                                                         ctx=ast.Load()),
                                                                                      args=[ast.Tuple(elts=[ast.Name(id="args", ctx=ast.Load()),
                                                                                                            ast.Name(id="kwargs", ctx=ast.Load())],
                                                                                                      ctx=ast.Load())],
                                                                                      keywords=[]))],
                                                           keywords=[])),
                                   ],
                                   decorator_list=[])

    return wrapped_node


def _astify_jax_typing_annotation(state, aval):
    # jaxtyping annotations are like Float32[Array, "128 32"]
    if not state.use_jax_typing:
        return None

    dtype = aval.dtype
    shape = aval.shape

    if dtype == jnp.float32:
        dtype_str = "Float32"
    elif dtype == jnp.float64:
        dtype_str = "Float64"
    elif dtype == jnp.int32:
        dtype_str = "Int32"
    elif dtype == jnp.int64:
        dtype_str = "Int64"
    elif dtype == jnp.bool_:
        dtype_str = "Bool"
    elif dtype == jnp.bfloat16:
        dtype_str = "BFloat16"
    elif dtype == jnp.float16:
        dtype_str = "Float16"
    else:
        warnings.warn(f"Unknown dtype for jaxtyping {dtype}")
        dtype_str = "Shaped"

    if len(shape) == 0:
        return ast.Subscript(
            value=ast.Name(id="Scalar", ctx=ast.Load()),
            slice=ast.Name(id=dtype_str, ctx=ast.Load()),
        )

    shape_str = " ".join(str(s) for s in shape)

    return ast.Subscript(
        value=ast.Name(id=dtype_str, ctx=ast.Load()),
        slice=ast.Tuple(elts=[ast.Name(id="Array", ctx=ast.Load()), ast.Str(shape_str)], ctx=ast.Load()),
    )



def jaxpr_to_py_ast(state: SourcerorState, jaxpr, fn_name: Optional[str] = None, *, unique_fn_name: bool = True):
    if isinstance(jaxpr, ClosedJaxpr):
        jaxpr = jaxpr.jaxpr
    if fn_name is None or unique_fn_name:
        fn_name = state.heuristic_fn_skolem(jaxpr, default=fn_name)

    # Generate argument declarations
    jaxpr = constant_fold_jaxpr(jaxpr)
    annotations = [_astify_jax_typing_annotation(state, v.aval) for v in jaxpr.invars]
    ast_args = [ast.arg(arg=state.str_name(var), annotation=ann) for var, ann in zip(jaxpr.invars, annotations)]
    ast_args = ast.arguments(args=ast_args, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[])

    stmts = []

    # Generate body of the function
    for eqn in jaxpr.eqns:
        prim = str(eqn.primitive)
        if prim in prim_to_python:
            eqn_stmts = prim_to_python[prim](state, eqn)
        else:
            try:
                eqn_stmts = normal_fn(prim)(state, eqn)
            except Exception:
                raise ValueError(f"Could not handle primitive {prim}")

        if isinstance(eqn_stmts, list):
            stmts.extend(eqn_stmts)
        else:
            stmts.append(eqn_stmts)

    # Generate return statement
    if len(jaxpr.outvars) == 1:
        returns = state.name(jaxpr.outvars[0])
    else:
        returns = ast.Tuple(elts=[_name_or_literal(state, var) for var in jaxpr.outvars], ctx=ast.Load())
    stmts.append(ast.Return(value=returns))

    return ast.FunctionDef(name=fn_name, args=ast_args, body=stmts, decorator_list=[])


def _name_or_literal(state, var):
    if isinstance(var, Literal):
        return _astify_value(var.val)
    else:
        return state.name(var)


def constant_fold_jaxpr(jaxpr: jax.core.Jaxpr):
    """
    Given a jaxpr, return a new jaxpr with all constant folding done.
    """
    return partial_eval_jaxpr(jaxpr, {}, elide_unused_invars=False)

def partial_eval_jaxpr(jaxpr, env, elide_unused_invars):
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
        elif isinstance(out, Var):
            return out
        elif isinstance(out, Literal):
            return Literal(out.val, var.aval)
        else:
            assert not isinstance(out, Jaxpr)
            return Literal(out, var.aval)

    for eqn in jaxpr.eqns:
        vals = [read(var) for var in eqn.invars]
        if eqn.primitive.name in constant_fold_blacklist:
            new_eqns.append(eqn)
        elif all(val is not None for val in vals):
            # go ahead and eval it
            out = _eval_eqn(eqn, vals)

            # two options: either it's a jaxpr result (partial eval) or it's a value or a list of values
            if isinstance(out, Jaxpr):
                # we need to inline this
                new_eqns.extend(out.eqns)
                out = out.outvars
            elif not isinstance(out, tuple) and not isinstance(out, list):
                out = (out,)

            for var, val in zip(eqn.outvars, out):
                assert not isinstance(val, Jaxpr)
                if isinstance(val, Literal):
                    env[var] = val.val
                else:
                    env[var] = val
        else:
            new_eqns.append(eqn)

    # now that we've evaled everything, inline all the constants
    out_eqns = []
    for eqn in new_eqns:
        eqn = eqn.replace(invars=tuple(read_or_self(var) for var in eqn.invars))
        out_eqns.append(eqn)

    invars_still_used = IdentitySet()
    for eqn in out_eqns:
        for var in eqn.invars:
            invars_still_used.add(var)

    if elide_unused_invars:
        invars = tuple(var for var in jaxpr.invars if var in invars_still_used)
    else:
        invars = jaxpr.invars

    # sub in any constants for outvars
    outvars = tuple(read_or_self(var) for var in jaxpr.outvars)

    return jaxpr.replace(eqns=out_eqns, outvars=outvars, invars=invars)


def _eval_eqn(eqn, vals) -> Union[Jaxpr, tuple, list, jnp.ndarray]:
    if eqn.primitive.name == "closed_call":
        assert eqn.primitive.call_primitive == True
        assert eqn.primitive.map_primitive == False

        out = partial_eval_jaxpr(eqn.params['call_jaxpr'].jaxpr, {var: val for var, val in zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)}, elide_unused_invars=True)
    elif eqn.primitive.name == "scan":
        out = eqn.primitive.bind(*vals, **eqn.params)
    elif isinstance(eqn.primitive, CustomJVPCallPrimitive):
        # out = eqn.primitive.bind(*vals, **eqn.params)
        closed_jaxpr = eqn.params['call_jaxpr']
        out = partial_eval_jaxpr(closed_jaxpr.jaxpr, {var: val for var, val in zip(closed_jaxpr.jaxpr.invars, vals)}, elide_unused_invars=True)
    else:
        out = eqn.primitive.bind(*vals, **eqn.params)
    return out


@primitive_handler('dot_general')
def _astify_dot_general(state, eqn):
    x, y = eqn.invars
    d = eqn.params['dimension_numbers']
    precision = eqn.params['precision']
    preferred_element_type = eqn.params['preferred_element_type']

    has_dtype = preferred_element_type is None or x.aval.dtype == y.aval.dtype == preferred_element_type

    # recognize simple matmul case
    if d == (((1,), (0,)), ((), ())) and precision == None:
        invars = [_astify_atom(state, x), _astify_atom(state, y)]
        outvars = _astify_outvars(state, eqn.outvars)
        out = ast.Assign(targets=outvars, value=ast.Call(func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr='matmul', ctx=ast.Load()), args=invars, keywords=[]))
        if not has_dtype:
            out = ast.Assign(targets=outvars, value=ast.Call(func=ast.Attribute(value=out.value, attr='astype', ctx=ast.Load()), args=[_astify_value(preferred_element_type)], keywords=[]))

        return out

    # handle einsum case
    contract_dims, batch_dims = d
    in_specs = [['0']*x.aval.ndim, ['0']*y.aval.ndim]  # the 0's will be replaced with letters
    out_spec = ''
    letter = ord('a')

    # output ordering is batch dims in order, then remaining lhs, then remaining rhs
    for i in range(len(batch_dims[0])):
        in_specs[0][batch_dims[0][i]] = chr(letter)
        in_specs[1][batch_dims[1][i]] = chr(letter)
        out_spec += chr(letter)
        letter += 1

    for i in range(len(contract_dims[0])):
        in_specs[0][contract_dims[0][i]] = chr(letter)
        in_specs[1][contract_dims[1][i]] = chr(letter)
        letter += 1

    # remaining dims are just the rest of the dims
    for i in range(x.aval.ndim):
        if in_specs[0][i] == '0':
            in_specs[0][i] = chr(letter)
            out_spec += chr(letter)
            letter += 1

    for i in range(y.aval.ndim):
        if in_specs[1][i] == '0':
            in_specs[1][i] = chr(letter)
            out_spec += chr(letter)
            letter += 1



    final_spec = f"{''.join(in_specs[0])},{''.join(in_specs[1])}->{out_spec}"
    invars = [_astify_value(final_spec), _astify_atom(state, x), _astify_atom(state, y)]
    outvars = _astify_outvars(state, eqn.outvars)
    keywords = []
    if precision is not None:
        keywords.append(ast.keyword(arg='precision', value=_astify_value(precision)))
    if preferred_element_type is not None:
        keywords.append(ast.keyword(arg='preferred_element_type', value=_astify_value(preferred_element_type)))

    return ast.Assign(targets=outvars, value=ast.Call(func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr='einsum', ctx=ast.Load()), args=invars,
                                                      keywords=keywords))


    # invars = [_astify_atom(state, x), _astify_atom(state, y), _astify_value(d), _astify_value(precision),
    #          _astify_value(preferred_element_type)]
    # outvars = _astify_outvars(state, eqn.outvars)
    # return ast.Assign(targets=outvars, value=ast.Call(func=ast.Attribute(value=ast.Name(id='jax.lax', ctx=ast.Load()), attr='dot_general', ctx=ast.Load()), args=invars, keywords=[]))

@primitive_handler('dynamic_slice')
def _sourcify_dynamic_slice(state, eqn):
    sliced = eqn.invars[0]
    invars = ast.Tuple(elts=[_astify_atom(state, var) for var in eqn.invars[1:]], ctx=ast.Load())
    outvars = _astify_outvars(state, eqn.outvars)
    params = [ast.keyword(arg=k, value=_astify_value(v)) for k, v in eqn.params.items()]
    return ast.Assign(targets=outvars, value=ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='jax.lax', ctx=ast.Load()),
            attr='dynamic_slice',
            ctx=ast.Load()
        ),
        args=[_astify_atom(state, sliced), invars],
        keywords=params
    ))


@primitive_handler('slice')
def _sourcify_slice(state, eqn):
    sliced = eqn.invars[0]
    # invars = ast.Tuple(elts=[_astify_atom(state, var) for var in eqn.invars[1:]], ctx=ast.Load())
    outvars = _astify_outvars(state, eqn.outvars)
    start_indices = eqn.params['start_indices']
    limit_indices = eqn.params['limit_indices']
    strides = eqn.params['strides']
    if strides is None:
        strides = (None,) * len(start_indices)
    indices = [_astify_value(slice(s, e, stride)) for s, e, stride in zip(start_indices, limit_indices, strides)]
    # params = [ast.keyword(arg=k, value=_astify_value(v)) for k, v in eqn.params.items()]
    return ast.Assign(targets=outvars, value=ast.Subscript(
        value=_astify_atom(state, sliced),
        slice=ast.Tuple(elts=indices, ctx=ast.Load()),
        ctx=ast.Load()
    ))


@primitive_handler('dynamic_update_slice')
def _sourcify_dynamic_update_slice(state, eqn):
    sliced = eqn.invars[0]
    # the first two arguments are the sliced array and the update array
    # the remaining are start indices and should be packaged into a tuple
    target = _astify_atom(state, eqn.invars[0])
    update = _astify_atom(state, eqn.invars[1])
    start_indices = maybe_tuple_vars([_astify_atom(state, var) for var in eqn.invars[2:]])
    outvars = _astify_outvars(state, eqn.outvars)

    return ast.Assign(targets=outvars, value=ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='jax.lax', ctx=ast.Load()),
            attr='dynamic_update_slice',
            ctx=ast.Load()
        ),
        args=[target, update, start_indices],
        keywords=[]
    ))


@primitive_handler('convert_element_type')
def _astify_convert_element_type(state, eqn):
    # now we use ast
    outvars = _astify_outvars(state, eqn.outvars)
    assert len(eqn.invars) == 1
    invar = _astify_atom(state, eqn.invars[0])
    dtype = _astify_value(eqn.params['new_dtype'])
    # return ast.Assign(targets=outvars, value=ast.Call(
    #     func=ast.Attribute(
    #         value=ast.Name(id='jax.lax', ctx=ast.Load()),
    #         attr='convert_element_type',
    #         ctx=ast.Load()
    #     ),
    #     args=[invars],
    #     keywords=params
    # ))
    return ast.Assign(targets=outvars, value=ast.Call(
        func=ast.Attribute(
            value=invar,
            attr='astype',
            ctx=ast.Load()
        ),
        args=[dtype],
        keywords=[]
    ))

def is_array(arr):
    return isinstance(arr, (np.ndarray, np.generic, jnp.ndarray))


def _astify_array(value):
    assert is_array(value)
    if isinstance(value, np.int64):
        return ast.Constant(value=int(value))

    if value.ndim == 0 and value.dtype in (jnp.float32, jnp.int32, jnp.bool_, jnp.int64):
        return ast.Constant(value=value.item())

    if value.ndim == 0:
        dtype_value = _astify_value(value.dtype)
        return ast.Call(
            dtype_value,
            args=[ast.Constant(value=value.item())],
            keywords=[],
        )

    values = value.tolist()

    def rec_astify_list(values):
        if isinstance(values, list):
            return ast.List(elts=[rec_astify_list(val) for val in values], ctx=ast.Load())
        else:
            return ast.Constant(value=values)

    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='jnp', ctx=ast.Load()),
            attr='array',
            ctx=ast.Load()
        ),
        args=[rec_astify_list(values)],
        keywords=[ast.keyword(arg='dtype', value=_astify_value(value.dtype))]
    )

def _astify_atom(state: SourcerorState, var: Union[Literal, Var]):
    if isinstance(var, Literal):
        return _astify_value(var.val)
    elif isinstance(var, Var):
        return state.name(var)
    else:
        raise NotImplementedError()

def _astify_value(value):
    assert not isinstance(value, (Literal, Var))

    if is_array(value):
        return _astify_array(value)
    elif isinstance(value, (int, bool, float, str, type(None))):
        return ast.Constant(value=value)
    elif isinstance(value, (tuple, list)):
        return ast.Tuple(elts=[_astify_value(v) for v in value], ctx=ast.Load())
    elif isinstance(value, jnp.dtype):
        # return ast.Call(func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr='dtype', ctx=ast.Load()), args=[ast.Constant(value=str(value))], keywords=[])
        if value.name in ('float32', 'float64', 'int32', 'int64', 'bfloat16', 'float16'):
            # return ast.Constant(value=getattr(jnp, value.name))
            return ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr=value.name, ctx=ast.Load())
        elif value.name == 'bool':
            return ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr='bool_', ctx=ast.Load())
        else:
            return ast.Call(func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr='dtype', ctx=ast.Load()), args=[ast.Constant(value=str(value))], keywords=[])
    elif value is _UNSPECIFIED:
        return ast.Attribute(value=ast.Name(id='jax.experimental.pjit', ctx=ast.Load()), attr='_UNSPECIFIED', ctx=ast.Load())
    elif isinstance(value, jax.lax.GatherScatterMode):
        return ast.Attribute(value=ast.Name("jax.lax.GatherScatterMode", ctx=ast.Load()), attr=value.name, ctx=ast.Load())
    elif isinstance(value, enum.Enum):
        return ast.Attribute(value=ast.Name(id=value.__class__.__qualname__, ctx=ast.Load()), attr=value.name, ctx=ast.Load())
    elif isinstance(value, slice):
        return ast.Call(
            func=ast.Name(id='slice', ctx=ast.Load()),
            args=[_astify_value(value.start), _astify_value(value.stop), _astify_value(value.step)],
            keywords=[]
        )
    elif isinstance(value, NamedSharding):
        # jax.sharding.NamedSharding(mesh=<recurse>, spec=PartitionSpec(*<recurse>)
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='jax.sharding', ctx=ast.Load()), attr='NamedSharding', ctx=ast.Load()),
            args=[_astify_value(value.mesh), _astify_value(value.spec)],
            keywords=[]
        )
    elif isinstance(value, jax.sharding.Mesh):
        return ast.Load(name="TODO_mesh")
    elif isinstance(value, jax.sharding.PartitionSpec):
        return ast.Load(name="TODO_partition_spec")
    elif isinstance(value, bytes):
        return ast.Constant(value=value.decode('utf-8'))
    else:
        warnings.warn(f"Unknown value type {type(value)}")
        raise NotImplementedError(f"Unknown value type {type(value)}")
        return ast.parse(repr(value)).body[0]


def _astify_outvars(state, outvars):
    out = [state.name(v, ctx=ast.Store()) for v in outvars]
    if len(out) == 1:
        return out
    else:
        return [ast.Tuple(elts=out, ctx=ast.Store())]

def maybe_tuple_vars(vars):
    if len(vars) == 1:
        return vars[0]
    else:
        return ast.Tuple(elts=vars, ctx=ast.Load())


def maybe_untuple_vars(var, is_tuple):
    if is_tuple:
        return ast.Starred(value=var, ctx=ast.Load())
    else:
        return var


@primitive_handler('scan')
def _astify_scan(state, eqn):
    assert eqn.primitive.name == 'scan'

    # the args to scan are [constants, carry, xs]
    # constants aren't exposed in the Python API, so we need to handle them specially (we use a lambda)
    num_consts = eqn.params['num_consts']
    num_carry = eqn.params['num_carry']

    # TODO: bring back map
    # if num_carry == 0:
        # this is a map
        # return _astify_map(eqn)

    constant_args = eqn.invars[:num_consts]
    carries = eqn.invars[num_consts:num_consts + num_carry]
    xs = eqn.invars[num_consts + num_carry:]

    jaxpr = eqn.params['jaxpr'].jaxpr

    if num_consts != 0:
        # we want to construct an environment where we partial eval the function using the constants as the env
        env = dict(zip(jaxpr.invars, constant_args))
        jaxpr = partial_eval_jaxpr(jaxpr, env, elide_unused_invars=True)

    fn_ast = jaxpr_to_py_ast(state, jaxpr)
    fn_name = fn_ast.name

    length = _astify_value(eqn.params['length'])
    unroll = _astify_value(eqn.params['unroll'])
    reverse = _astify_value(eqn.params['reverse'])

    stmts = []

    if num_carry != 1 or len(jaxpr.invars) != 2:
        # what we want is something like:
        # fn_name = lambda carry, xs: fn_name(*carry, *xs)
        # jax.lax.scan(fn_name, (carries...), (xs...))

        modified_signature = ast.arguments(
            args=[ast.arg(arg='carry'), ast.arg(arg='x')],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
            posonlyargs=[]
        )

        initial_assign = ast.Assign(
            targets=[ast.Tuple(elts=[ast.Name(a.arg) for a in fn_ast.args.args], ctx=ast.Store())],
            value=ast.Tuple(elts=[maybe_untuple_vars(ast.Name(id='carry', ctx=ast.Load()), num_carry != 1),
                                    maybe_untuple_vars(ast.Name(id='x', ctx=ast.Load()), len(xs) != 1)])
        )

        fn_return = fn_ast.body[-1]
        assert isinstance(fn_return, ast.Return)

        fn_return_value = fn_return.value

        if isinstance(fn_return_value, ast.Tuple):
            fn_return_value = fn_return_value.elts
            ret_carries = maybe_tuple_vars(fn_return_value[:num_carry])
            ret_ys = maybe_tuple_vars(fn_return_value[num_carry:])
        elif num_carry == 0:
            ret_carries = _astify_value(())
            ret_ys = fn_return_value
        else:
            ret_carries = fn_return_value
            ret_ys = _astify_value(())

        scan_return = ast.Return(
            value = ast.Tuple(elts=[ret_carries, ret_ys], ctx=ast.Load())
        )

        new_body = [initial_assign] + list(fn_ast.body[:-1]) + [scan_return]

        fn_ast = ast.FunctionDef(
            name=fn_name,
            args=modified_signature,
            body=new_body,
            decorator_list=[]
        )

        stmts.append(fn_ast)

        scan_call = ast.Assign(
            # targets=_astify_outvars(eqn.outvars),
            targets=[ast.Tuple(elts=[ast.Name(id='final_carry', ctx=ast.Store()), ast.Name(id='ys', ctx=ast.Store())], ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id='jax.lax.scan', ctx=ast.Load()),
                args=[ast.Name(id=fn_name, ctx=ast.Load()),
                      maybe_tuple_vars([_astify_atom(state, v) for v in carries]),
                      maybe_tuple_vars([_astify_atom(state, v) for v in xs])],
                keywords=[ast.keyword(arg='length', value=length), ast.keyword(arg='unroll', value=unroll), ast.keyword(arg='reverse', value=reverse)]
            ))
        stmts.append(scan_call)

        if num_carry > 0:
            assign_carry = ast.Assign(
                targets=_astify_outvars(state, eqn.outvars[:num_carry]),
                value=ast.Name(id='final_carry', ctx=ast.Load())
            )

            stmts.append(assign_carry)

        if num_carry < len(eqn.outvars):
            assign_ys = ast.Assign(
                targets=_astify_outvars(state, eqn.outvars[num_carry:]),
                value=ast.Name(id='ys', ctx=ast.Load())
            )

            stmts.append(assign_ys)
    else:
        stmts.append(fn_ast)

        scan_call = ast.Assign(
            targets=_astify_outvars(state, eqn.outvars),
            value=ast.Call(
                func=ast.Name(id='jax.lax.scan', ctx=ast.Load()),
                args=[ast.Name(id=fn_name, ctx=ast.Load())] + [_astify_atom(state, v) for v in eqn.invars],
                keywords=[ast.keyword(arg='length', value=length), ast.keyword(arg='unroll', value=unroll), ast.keyword(arg='reverse', value=reverse)]
            ))

        stmts.append(scan_call)

    return stmts

def _astify_map(state, eqn):
    assert eqn.primitive.name == 'scan'
    assert eqn.params['num_carry'] == 0

    jaxpr = eqn.params['jaxpr']

    fn_ast = jaxpr_to_py_ast(state, jaxpr)
    fn_name = fn_ast.name

    # map is a bit funny, because the jaxpr takes K args, but the jax.lax.map function takes a single tuple arg
    # so we need to use a lambda to redirect the call
    lam = ast.parse(f"lambda args: {fn_name}(*args)").body[0]

    assign = ast.Assign(
        targets=_astify_outvars(state, eqn.outvars),
        value=ast.Call(
            func=ast.Name(id='jax.lax.map', ctx=ast.Load()),
            args=[lam, ast.Tuple(elts=[_astify_atom(state, v) for v in eqn.invars], ctx=ast.Load())],
            keywords=[]
        ))

    return [fn_ast, assign]


def _attempt_to_sniff_fn_name_for_jaxpr(jaxpr):
    # this is necessarily very hacky.
    eqns = jaxpr.eqns
    if len(eqns) == 0:
        return None
    source_info = eqns[0].source_info
    try:
        name = None
        for frame in user_frames(source_info):
            name = frame.function_name

            if name and "<" not in name:
                return name

        if not name:
            name = frame.file_name
        return name
    except:
        return None





@primitive_handler('closed_call')
def _astify_closed_call(state, eqn):
    # out = partial_eval_jaxpr(eqn.params['call_jaxpr'].jaxpr,
    #                          {var: val for var, val in zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)})
    raw_jaxpr = eqn.params['call_jaxpr'].jaxpr
    literal_args = {k: v.val for k, v in zip(raw_jaxpr.invars, eqn.invars) if isinstance(v, Literal)}
    call_jaxpr = partial_eval_jaxpr(raw_jaxpr, literal_args, elide_unused_invars=False)
    fn_ast = jaxpr_to_py_ast(state, call_jaxpr)
    fn_name = fn_ast.name

    invars = [_astify_atom(state, v) for v in eqn.invars if not isinstance(v, Literal)]
    outvars = _astify_outvars(state, eqn.outvars)

    assign = ast.Assign(
        targets=outvars,
        value=ast.Call(
            func=ast.Name(id=fn_name, ctx=ast.Load()),
            args=invars,
            keywords=[]
        ))

    return [fn_ast, assign]

@primitive_handler('pjit')
def _astify_pjit(state, eqn):
    # this one's a real pain.
    # pjit's params are :
        # jaxpr
        # donated_invars:
        # in_shardings, out_shardings
        # resource env
        # name (yay)
        # keep_unused, inline (which we won't use)

    jaxpr = eqn.params['jaxpr']
    donated_invars = eqn.params['donated_invars']
    in_shardings = eqn.params['in_shardings']
    out_shardings = eqn.params['out_shardings']
    resource_env = eqn.params['resource_env']
    name = eqn.params['name']

    can_ignore_donated = not any(donated_invars)

    keywords = []

    if in_shardings and any(s != jax.experimental.pjit._UNSPECIFIED for s in in_shardings):
        in_shardings = _astify_value(in_shardings)
        keywords.append(ast.keyword(arg='in_shardings', value=in_shardings))

    if out_shardings and any(s != jax.experimental.pjit._UNSPECIFIED for s in out_shardings):
        out_shardings = _astify_value(out_shardings)
        keywords.append(ast.keyword(arg='out_shardings', value=out_shardings))

    if not can_ignore_donated:
        donated_invars = _astify_value(donated_invars)
        keywords.append(ast.keyword(arg='donated_invars', value=donated_invars))


    # preprocess the function
    fn_ast = jaxpr_to_py_ast(state, jaxpr)
    fn_name = fn_ast.name

    jitted_fn = ast.Call(
        func=
        ast.Attribute(
            ast.Name(id='jax', ctx=ast.Load()),
            attr='jit'),
        args=[ast.Name(id=fn_name, ctx=ast.Load())],
        keywords=keywords
    )

    assign = ast.Assign(
        targets=_astify_outvars(state, eqn.outvars),
        value=ast.Call(
            func=jitted_fn,
            args=[_astify_atom(state, v) for v in eqn.invars],
            keywords=[]
        ))

    return [fn_ast, assign]


@primitive_handler('remat2')
def _astify_remat(state: SourcerorState, eqn):
    # out = partial_eval_jaxpr(eqn.params['call_jaxpr'].jaxpr,
    #                          {var: val for var, val in zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)})
    fn_ast = jaxpr_to_py_ast(state, constant_fold_jaxpr(eqn.params['jaxpr']))
    fn_name = fn_ast.name

    invars = [_astify_atom(state, v) for v in eqn.invars]
    outvars = _astify_outvars(state, eqn.outvars)

    prevent_cse = _astify_value(eqn.params.get('prevent_cse', False))

    policy = eqn.params.get('policy')

    if policy is not None:
        warnings.warn(f"Remat2 policy {policy} is not supported.")

    has_args = prevent_cse

    # if we have args, we wrap checkpoint in a partial
    if has_args:
        checkpoint = ast.Call(ast.Name('partial'), [ast.Name(id='jax.checkpoint', ctx=ast.Load())], [ast.keyword(arg='prevent_cse', value=prevent_cse)])
    else:
        checkpoint = ast.Name(id='jax.checkpoint', ctx=ast.Load())

    # apply as a decorator
    fn_ast.decorator_list.append(checkpoint)

    assign = ast.Assign(
        targets=outvars,
        value=ast.Call(
            func=ast.Name(id=fn_name, ctx=ast.Load()),
            args=invars,
            keywords=[]
        ))

    return [fn_ast, assign]




@primitive_handler('custom_vjp_call_jaxpr')
def _astify_custom_vjp_call_jaxpr(state, eqn):
    # out = partial_eval_jaxpr(eqn.params['call_jaxpr'].jaxpr,
    #                          {var: val for var, val in zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)})
    closed_jaxpr = eqn.params['fun_jaxpr']
    fn_ast = jaxpr_to_py_ast(state, closed_jaxpr)
    fn_name = fn_ast.name

    invars = [_astify_atom(state, v) for v in eqn.invars]
    outvars = _astify_outvars(state, eqn.outvars)

    lam = ast.Assign(
        targets=[ast.Name(id=f"vjp_{fn_name}", ctx=ast.Store())],
        value=ast.Lambda(
            args=ast.arguments(
                args=[ast.arg(arg='primals')],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
                posonlyargs=[]
            ),
            body=ast.Call(
                func=ast.Name(id=fn_name, ctx=ast.Load()),
                args=[ast.Name(id='primals', ctx=ast.Load())],
                keywords=[]
            )
        )
    )

    assign = ast.Assign(
        targets=outvars,
        value=ast.Call(
            func=ast.Name(id=f"vjp_{fn_name}", ctx=ast.Load()),
            args=invars,
            keywords=[]
        ))

    return [fn_ast, lam, assign]


@primitive_handler('custom_jvp_call')
def _astify_custom_jvp_call(state, eqn):
    closed_jaxpr = eqn.params['call_jaxpr']
    fn_ast = jaxpr_to_py_ast(state, closed_jaxpr)
    fn_name = fn_ast.name

    invars = [_astify_atom(state, v) for v in eqn.invars]
    outvars = _astify_outvars(state, eqn.outvars)

    # lam = ast.Assign(
    #     targets=[ast.Name(id=f"jvp_{fn_name}", ctx=ast.Store())],
    #     value=ast.Lambda(
    #         args=ast.arguments(
    #             args=[ast.arg(arg='primals', annotation=None), ast.arg(arg='tangents', annotation=None)],
    #             vararg=None,
    #             kwonlyargs=[],
    #             kw_defaults=[],
    #             kwarg=None,
    #             defaults=[],
    #             posonlyargs=[]
    #         ),
    #         body=ast.Call(
    #             func=ast.Name(id=fn_name, ctx=ast.Load()),
    #             args=[ast.Name(id='primals', ctx=ast.Load()), ast.Name(id='tangents', ctx=ast.Load())],
    #             keywords=[]
    #         )
    #     )
    # )
    #
    # assign = ast.Assign(
    #     targets=outvars,
    #     value=ast.Call(
    #         func=ast.Name(id=f"jvp_{fn_name}", ctx=ast.Load()),
    #         args=invars,
    #         keywords=[]
    #     ))

    # return [fn_ast, lam, assign]

    # just call the fn

    assign = ast.Assign(
        targets=outvars,
        value=ast.Call(
            func=ast.Name(id=fn_name, ctx=ast.Load()),
            args=invars,
            keywords=[]
        ))

    return [fn_ast, assign]





@primitive_handler('while')
def _astify_while(state, eqn):
    # out = partial_eval_jaxpr(eqn.params['call_jaxpr'].jaxpr,
    #                          {var: val for var, val in zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)})
    body_jaxpr = eqn.params['body_jaxpr']
    cond_jaxpr = eqn.params['cond_jaxpr']

    body_nconsts = eqn.params['body_nconsts']
    cond_nconsts = eqn.params['cond_nconsts']

    if cond_nconsts != 0:
        env = dict(zip(cond_jaxpr.in_avals, eqn.invars[:cond_nconsts]))
        cond_jaxpr = partial_eval_jaxpr(cond_jaxpr.jaxpr, env, elide_unused_invars=False)

    cond_fn_ast = jaxpr_to_py_ast(state, cond_jaxpr)
    cond_fn_name = cond_fn_ast.name

    if body_nconsts != 0:
        env = dict(zip(body_jaxpr.in_avals, eqn.invars[cond_nconsts:cond_nconsts+body_nconsts]))
        body_jaxpr = partial_eval_jaxpr(body_jaxpr.jaxpr, env, elide_unused_invars=False)

    body_fn_ast = jaxpr_to_py_ast(state, body_jaxpr)
    body_fn_name = body_fn_ast.name

    true_args = eqn.invars[cond_nconsts+body_nconsts:]

    invars = [_astify_atom(state, v) for v in true_args]
    outvars = _astify_outvars(state, eqn.outvars)

    body_lambda = ast.Lambda(
            args=ast.arguments(
                args=[ast.arg(arg='state')],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
                posonlyargs=[]
            ),
            body=ast.Call(
                func=ast.Name(id=body_fn_name, ctx=ast.Load()),
                args=[ast.Starred(value=ast.Name(id='state', ctx=ast.Load()), ctx=ast.Load())],
                keywords=[]
            )
    )

    cond_lam = ast.Lambda(
            args=ast.arguments(
                args=[ast.arg(arg='state')],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
                posonlyargs=[]
            ),
            body=ast.Call(
                func=ast.Name(id=cond_fn_name, ctx=ast.Load()),
                args=[ast.Starred(value=ast.Name(id='state', ctx=ast.Load()), ctx=ast.Load())],
                keywords=[]
            )
    )

    args = ast.Tuple(elts=invars, ctx=ast.Load())

    assign = ast.Assign(
        targets=outvars,
        value=ast.Call(
            func=ast.Name(id='jax.lax.while_loop', ctx=ast.Load()),
            args=[cond_lam, body_lambda, args],
            keywords=[]
        ))

    return [body_fn_ast, cond_fn_ast, assign]


def _astize_fn(state, jaxpr, name):
    return jaxpr_to_py_ast(state, jaxpr, name)


@primitive_handler('cond')
def _astify_cond(state, eqn):
    # out = partial_eval_jaxpr(eqn.params['call_jaxpr'].jaxpr,
    #                          {var: val for var, val in zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)})
    branches = eqn.params['branches']

    ast_branches = [jaxpr_to_py_ast(state, jaxpr) for jaxpr in branches]

    pred_var, *rest_args = eqn.invars
    pred_ast = _astify_atom(state, pred_var)
    invars = [_astify_atom(state, v) for v in rest_args]
    outvars = _astify_outvars(state, eqn.outvars)

    branch_names = [ast.Name(id=ast_branch.name, ctx=ast.Load()) for ast_branch in ast_branches]

    if len(branches) == 2:
        false_fn_ast, true_fn_ast = ast_branches
        false_name, true_name = branch_names

        assign = ast.Assign(
            targets=outvars,
            value=ast.Call(
                func=ast.Name(id='jax.lax.cond', ctx=ast.Load()),
                args=[pred_ast, true_name, false_name, *invars],
                keywords=[]
            ))

        return [true_fn_ast, false_fn_ast, assign]

    else:
        # jax.lax.switch
        assign = ast.Assign(
            targets=outvars,
            value=ast.Call(
                func=ast.Name(id='jax.lax.switch', ctx=ast.Load()),
                args=[pred_ast, ast.List(elts=branch_names), *invars],
                keywords=[]
            ))

        return ast_branches + [assign]




@primitive_handler('iota')
def _astify_iota(state, eqn):
    # iota is a sort of broadcasted arange
    # we can use np.broadcast_to(np.arange(size), shape)
    dimension = eqn.params['dimension']  # axis along which to increment.
    shape = eqn.params['shape']
    dtype = eqn.params['dtype']

    arange = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr='arange',
                    ctx=ast.Load()
                ),
                args=[_astify_value(shape[0])],
                keywords=[ast.keyword(arg='dtype', value=_astify_value(dtype))]
    )

    if len(shape) == 1:
        # this is a simple arange
        return ast.Assign(
            targets=_astify_outvars(state, eqn.outvars),
            value=arange
        )

    broadcast = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="jnp", ctx=ast.Load()),
            attr='broadcast_to',
            ctx=ast.Load()
        ),
        args=[arange, _astify_value(shape)],
        keywords=[]
    )

    return ast.Assign(
        targets=_astify_outvars(state, eqn.outvars),
        value=broadcast
    )



@primitive_handler('reshape')
def _astify_reshape(state, eqn):
    # the lax reshape is a bit different, because it can combine a transpose and reshape into one.
    # np.reshape(np.transpose(operand, dimensions), new_sizes)
    dimensions = eqn.params['dimensions']
    new_sizes = eqn.params['new_sizes']

    source = _astify_atom(state, eqn.invars[0])

    if dimensions is not None:
        source = ast.Call(
            func=ast.Name(id='jnp.transpose', ctx=ast.Load()),
            args=[source, _astify_value(dimensions)],
            keywords=[]
        )

    assign = ast.Assign(
        targets=_astify_outvars(state, eqn.outvars),
        value=ast.Call(
            func=ast.Name(id='jnp.reshape', ctx=ast.Load()),
            args=[source, _astify_value(new_sizes)],
            keywords=[]
        ))

    return [assign]


@primitive_handler('add_any')
def _astify_add_any(state, eqn):
    # add_any is a weird undocumented jax primitive. best guess is it adds?
    return _binop_fn(ast.Add())(state, eqn)


@primitive_handler('broadcast_in_dim')
def _astify_broadcast_in_dim(state, eqn):
    # broadcast_in_dim is how zeros, ones, full, etc are implemented,
    # so we prefer to use those where possible
    assert len(eqn.invars) == 1
    value = eqn.invars[0]
    shape = eqn.params['shape']
    broadcast_dimensions = eqn.params['broadcast_dimensions']

    if not isinstance(value, Literal) or broadcast_dimensions != ():
        return normal_fn('jax.lax.broadcast_in_dim')(state, eqn)

    if not isinstance(value.val, np.ndarray) or value.val.ndim != 0:
        return normal_fn('jax.lax.broadcast_in_dim')(state, eqn)
    else:
        constant_value = value.val.item()
        if constant_value == 0:
            call = ast.Call(
                ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr='zeros',
                    ctx=ast.Load()
                ),
                args=[_astify_value(shape), _astify_value(value.val.dtype)],
                keywords=[]
            )
        elif constant_value == 1:
            call = ast.Call(
                ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr='ones',
                    ctx=ast.Load()
                ),
                args=[_astify_value(shape), _astify_value(value.val.dtype)],
                keywords=[]
            )
        else:
            call = ast.Call(
                ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr='full',
                    ctx=ast.Load()
                ),
                args=[_astify_value(shape), _astify_value(constant_value), _astify_value(value.val.dtype)],
                keywords=[]
            )

        return [ast.Assign(
            targets=_astify_outvars(state, eqn.outvars),
            value=call
        )]

@primitive_handler('random_wrap')
def _astify_random_wrap(state, eqn):
    # we treat this as a noop
    return ast.Assign(
        targets=_astify_outvars(state, eqn.outvars),
        value=_astify_atom(state, eqn.invars[0])
    )




constant_fold_blacklist = {
    'broadcast_in_dim',
    'broadcast',
    'iota',
}