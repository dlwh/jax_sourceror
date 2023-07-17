import ast
import dataclasses
import warnings
from dataclasses import dataclass
from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Literal, Var, Jaxpr

from jax_automin.utils import IdentityMap, IdentitySet

@dataclass
class _SourcererState():
    _var_names: IdentityMap[Var, str] = dataclasses.field(default_factory=IdentityMap)
    _skolem_count: int = 0

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

            self._var_names[var] = name

            return name

    def skolem(self, prefix: str):
        self._skolem_count += 1
        return f"{prefix}_{self._skolem_count}"


def automin_function(f, *args, **kwargs):
    closed_jaxpr = jax.make_jaxpr(f)(*args, **kwargs)
    jaxpr = constant_fold_jaxpr(closed_jaxpr.jaxpr)
    state = _SourcererState()
    node = jaxpr_to_py_ast(state, jaxpr, fn_name=f.__name__)
    node = _maybe_wrap_fn_for_leaves(node, f, len(args) + len(kwargs))
    ast.fix_missing_locations(node)
    source = ast.unparse(node)
    return source


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

def jaxpr_to_py_ast(state: _SourcererState, jaxpr, fn_name="function"):
    # Generate argument declarations
    ast_args = [ast.arg(arg=state.str_name(var), annotation=None) for var in jaxpr.invars]
    ast_args = ast.arguments(args=ast_args, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[])

    stmts = []

    # Generate body of the function
    for eqn in jaxpr.eqns:
        prim = str(eqn.primitive)
        if prim in prim_to_python:
            eqn_stmts = prim_to_python[prim](state, eqn)
        else:
            eqn_stmts = normal_fn(prim)(state, eqn)

        if isinstance(eqn_stmts, list):
            stmts.extend(eqn_stmts)
        else:
            stmts.append(eqn_stmts)

    # Generate return statement
    if len(jaxpr.outvars) == 1:
        returns = state.name(jaxpr.outvars[0])
    else:
        returns = ast.Tuple(elts=[state.name(var) for var in jaxpr.outvars], ctx=ast.Load())
    stmts.append(ast.Return(value=returns))

    return ast.FunctionDef(name=fn_name, args=ast_args, body=stmts, decorator_list=[])


def constant_fold_jaxpr(jaxpr: jax.core.Jaxpr):
    """
    Given a jaxpr, return a new jaxpr with all constant folding done.
    """
    return partial_eval_jaxpr(jaxpr, {})

def partial_eval_jaxpr(jaxpr, env):
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
        else:
            assert not isinstance(out, Jaxpr)
            assert not isinstance(out, Literal)
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

    invars = tuple(var for var in jaxpr.invars if var in invars_still_used)

    # sub in any constants for outvars
    outvars = tuple(read_or_self(var) for var in jaxpr.outvars)

    return jaxpr.replace(eqns=out_eqns, outvars=outvars, invars=invars)


def _eval_eqn(eqn, vals) -> Union[Jaxpr, tuple, list, jnp.ndarray]:
    if eqn.primitive.name == "closed_call":
        assert eqn.primitive.call_primitive == True
        assert eqn.primitive.map_primitive == False

        out = partial_eval_jaxpr(eqn.params['call_jaxpr'].jaxpr, {var: val for var, val in zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)})
    elif eqn.primitive.name == "scan":
        out = eqn.primitive.bind(*vals, **eqn.params)
    else:
        out = eqn.primitive.bind(*vals, **eqn.params)
    return out


def _astify_dot_general(state, eqn):
    x, y = eqn.invars
    d = eqn.params['dimension_numbers']
    precision = eqn.params['precision']
    preferred_element_type = eqn.params['preferred_element_type']

    # recognize simple matmul case
    if d == (((1,), (0,)), ((), ())) and precision == None and preferred_element_type == None:
        invars = [_astify_atom(state, x), _astify_atom(state, y)]
        outvars = _astify_outvars(state, eqn.outvars)
        return ast.Assign(targets=outvars, value=ast.Call(func=ast.Attribute(value=ast.Name(id='jax.numpy', ctx=ast.Load()), attr='matmul', ctx=ast.Load()), args=invars, keywords=[]))

    invars = [_astify_atom(state, x), _astify_atom(state, y), _astify_value(d), _astify_value(precision),
             _astify_value(preferred_element_type)]
    outvars = _astify_outvars(state, eqn.outvars)
    return ast.Assign(targets=outvars, value=ast.Call(func=ast.Attribute(value=ast.Name(id='jax.lax', ctx=ast.Load()), attr='dot_general', ctx=ast.Load()), args=invars, keywords=[]))

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


def _astify_convert_element_type(state, eqn):
    # now we use ast
    outvars = _astify_outvars(state, eqn.outvars)
    invars = [_astify_atom(state, var) for var in eqn.invars]
    params = [ast.keyword(arg=k, value=_astify_value(v)) for k, v in eqn.params.items() if k != 'weak_type']
    return ast.Assign(targets=outvars, value=ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='jax.lax', ctx=ast.Load()),
            attr='convert_element_type',
            ctx=ast.Load()
        ),
        args=[invars],
        keywords=params
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
            value=ast.Name(id='jax.numpy', ctx=ast.Load()),
            attr='array',
            ctx=ast.Load()
        ),
        args=[rec_astify_list(values)],
        keywords=[ast.keyword(arg='dtype', value=_astify_value(value.dtype))]
    )

def _astify_atom(state: _SourcererState, var: Union[Literal, Var]):
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
        # return ast.Call(func=ast.Attribute(value=ast.Name(id='jax.numpy', ctx=ast.Load()), attr='dtype', ctx=ast.Load()), args=[ast.Constant(value=str(value))], keywords=[])
        if value.name in ('float32', 'float64', 'int32', 'int64', 'bfloat16', 'float16'):
            # return ast.Constant(value=getattr(jnp, value.name))
            return ast.Attribute(value=ast.Name(id='jax.numpy', ctx=ast.Load()), attr=value.name, ctx=ast.Load())
        elif value.name == 'bool':
            return ast.Attribute(value=ast.Name(id='jax.numpy', ctx=ast.Load()), attr='bool_', ctx=ast.Load())
        else:
            return ast.Call(func=ast.Attribute(value=ast.Name(id='jax.numpy', ctx=ast.Load()), attr='dtype', ctx=ast.Load()), args=[ast.Constant(value=str(value))], keywords=[])
    else:
        warnings.warn(f"Unknown value type {type(value)}")
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



def assign_stmt(call_expr: Callable):
    def binop_fn(state, eqn):
        invars = [_astify_atom(state, v) for v in eqn.invars]
        outvars = _astify_outvars(state, eqn.outvars)
        return ast.Assign(outvars, call_expr(*invars,
                                             **{k: _astify_value(v) for k, v in eqn.params.items()}
                                             ))
    return binop_fn

def binop_fn(op: ast.operator):
    return assign_stmt(lambda x, y: ast.BinOp(left=x, op=op, right=y))

def cmpop_fn(op: ast.cmpop):
    return assign_stmt(lambda x, y: ast.Compare(left=x, ops=[op], comparators=[y]))

def normal_fn(fn_name):
    return assign_stmt(lambda *args, **kwargs: ast.Call(
        func=ast.Name(id=fn_name, ctx=ast.Load()),
        args=list(args),
        keywords=[ast.keyword(arg=k, value=v) for k, v in kwargs.items()]
    ))

def _reduce_fn(fn_name: str):
    def reduce_fn_inner(state: _SourcererState, eqn):
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

    jaxpr = eqn.params['jaxpr']
    jaxpr = constant_fold_jaxpr(jaxpr.jaxpr)

    stmts = []

    fn_name = state.skolem('fn')
    fn_ast = jaxpr_to_py_ast(state, jaxpr, fn_name)

    length = _astify_value(eqn.params['length'])
    unroll = _astify_value(eqn.params['unroll'])
    reverse = _astify_value(eqn.params['reverse'])

    if num_consts > 0 or num_carry != 1 or len(jaxpr.invars) != 2:
        # what we want is something like:
        # fn_name = lambda carry, xs: fn_name(constants..., *carry, *xs)
        # jax.lax.scan(fn_name, (carries...), (xs...))
        constant_args = eqn.invars[:num_consts]
        carries = eqn.invars[num_consts:num_consts + num_carry]
        xs = eqn.invars[num_consts + num_carry:]

        constant_args = [_astify_atom(state, v) for v in constant_args]
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
            value=ast.Tuple(elts=[*constant_args,
                                  maybe_untuple_vars(ast.Name(id='carry', ctx=ast.Load()), num_carry != 1),
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
    jaxpr = constant_fold_jaxpr(jaxpr.jaxpr)

    fn_name = state.skolem('fn')
    fn_ast = jaxpr_to_py_ast(state, jaxpr, fn_name)

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


def _astify_closed_call(state, eqn):
    # out = partial_eval_jaxpr(eqn.params['call_jaxpr'].jaxpr,
    #                          {var: val for var, val in zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)})
    call_japr = constant_fold_jaxpr(eqn.params['call_jaxpr'].jaxpr)
    fn_name = state.skolem('fn')

    fn_ast = jaxpr_to_py_ast(state, call_japr, fn_name)

    invars = [_astify_atom(state, v) for v in eqn.invars]
    outvars = _astify_outvars(state, eqn.outvars)

    assign = ast.Assign(
        targets=outvars,
        value=ast.Call(
            func=ast.Name(id=fn_name, ctx=ast.Load()),
            args=invars,
            keywords=[]
        ))

    return [fn_ast, assign]


def _astify_remat(state: _SourcererState, eqn):
    # out = partial_eval_jaxpr(eqn.params['call_jaxpr'].jaxpr,
    #                          {var: val for var, val in zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)})
    call_japr = constant_fold_jaxpr(eqn.params['jaxpr'])
    fn_name = state.skolem('fn')

    fn_ast = jaxpr_to_py_ast(state, call_japr, fn_name)

    invars = [_astify_atom(state, v) for v in eqn.invars]
    outvars = _astify_outvars(state, eqn.outvars)

    lam = ast.Assign(
        targets=[ast.Name(id=f"ckpt_{fn_name}", ctx=ast.Store())],
        # value=ast.parse(f"jax.checkpoint({fn_name})").body[0]
        value=ast.Call(
            func=ast.Name(id='jax.checkpoint', ctx=ast.Load()),
            args=[ast.Name(id=fn_name, ctx=ast.Load())],
            keywords=[])
    )

    assign = ast.Assign(
        targets=outvars,
        value=ast.Call(
            func=ast.Name(id=f"ckpt_{fn_name}"),
            args=invars,
            keywords=[]
        ))

    return [fn_ast, lam, assign]

def _astify_reshape(state, eqn):
    # the lax reshape is a bit different, because it can combine a transpose and reshape into one.
    # np.reshape(np.transpose(operand, dimensions), new_sizes)
    dimensions = eqn.params['dimensions']
    new_sizes = eqn.params['new_sizes']

    source = _astify_atom(state, eqn.invars[0])

    if dimensions is not None:
        source = ast.Call(
            func=ast.Name(id='jax.numpy.transpose', ctx=ast.Load()),
            args=[source, _astify_value(dimensions)],
            keywords=[]
        )

    assign = ast.Assign(
        targets=_astify_outvars(state, eqn.outvars),
        value=ast.Call(
            func=ast.Name(id='jax.numpy.reshape', ctx=ast.Load()),
            args=[source, _astify_value(new_sizes)],
            keywords=[]
        ))

    return [assign]


def _astify_add_any(state, eqn):
    # add_Any is a weird undocumented jax primitive. best guess is it adds?
    return binop_fn(ast.Add())(state, eqn)


prim_to_python = {
    'add': binop_fn(ast.Add()),
    'sub': binop_fn(ast.Sub()),
    'mul': binop_fn(ast.Mult()),
    'div': binop_fn(ast.Div()),
    'neg': normal_fn('jax.lax.neg'),
    'lt': cmpop_fn(ast.Lt()),
    'gt': cmpop_fn(ast.Gt()),
    'le': cmpop_fn(ast.LtE()),
    'ge': cmpop_fn(ast.GtE()),
    'eq': cmpop_fn(ast.Eq()),
    'ne': cmpop_fn(ast.NotEq()),
    'min': normal_fn('jax.lax.min'),
    'max': normal_fn('jax.lax.max'),
    'convert_element_type': _astify_convert_element_type,
    'select_n': normal_fn('jax.lax.select_n'),
    'dynamic_slice': _sourcify_dynamic_slice,
    'dynamic_update_slice': _sourcify_dynamic_update_slice,
    'squeeze': normal_fn('jax.lax.squeeze'),
    'dot_general': _astify_dot_general,
    'broadcast_in_dim': normal_fn('jax.lax.broadcast_in_dim'),
    'broadcast': normal_fn('jax.lax.broadcast'),
    'reduce_sum': _reduce_fn('jax.numpy.sum'),
    'transpose': normal_fn('jax.lax.transpose'),
    'scan': _astify_scan,
    'closed_call': _astify_closed_call,
    'remat2': _astify_remat,
    'reshape': _astify_reshape,
    'add_any': _astify_add_any,
}

constant_fold_blacklist = {
    'broadcast_in_dim',
    'broadcast',
}