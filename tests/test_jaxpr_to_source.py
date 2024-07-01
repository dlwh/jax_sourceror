import functools
import textwrap
from functools import partial

import pytest

from jax_sourceror.interpreter import sourcerize
import jax
import jaxtyping
import jax.numpy as jnp


def test_jaxpr_to_source_simple():
    import jax.numpy as jnp

    def f(x):
        return x + 1

    source = sourcerize(f, use_jax_typing=False)(jnp.array([1, 2, 3]))

    assert source == """def f(a):
    b = a + 1
    return b"""


def test_jaxpr_to_source_matmul():
    import jax.numpy as jnp

    def f(x, y):
        return jnp.matmul(x, y)

    source = sourcerize(f, use_jax_typing=False)(jnp.array([[1, 2], [3, 4]]), jnp.array([[1, 2], [3, 4]]))

    assert source == """def f(a, b):
    c = jnp.matmul(a, b)
    return c"""

    check_roundtrip(f)(jnp.array([[1, 2], [3, 4]]), jnp.array([[1, 2], [3, 4]]))


def check_roundtrip(f, **config_kwargs):
    def return_function(*args, **kwargs):
        source = sourcerize(f, **config_kwargs)(*args, **kwargs)
        f2 = _parse_sandboxed(source, f.__name__)

        f_results = f(*args, **kwargs)
        f2_results = f2(*args, **kwargs)

        if isinstance(f_results, tuple):
            assert isinstance(f2_results, tuple)
            assert len(f_results) == len(f2_results)
            for a, b in zip(f_results, f2_results):
                assert jnp.all(a == b)
        else:
            assert jnp.all(f_results == f2_results)
        return f2

    return return_function


def _parse_sandboxed(source, fn_name):
    g = {'jax': jax, 'jaxtyping': jaxtyping, 'jnp': jnp, 'functools': functools, 'partial': partial}
    l = {}
    source = f"""
from jaxtyping import *

{source}"""
    exec(source, g, l)
    return l[fn_name]


def test_slice_squeeze():
    def f(x):
        return x[0:2, 0:1, 3]

    f2 = check_roundtrip(f)(jnp.arange(4 * 5 * 6).reshape(4, 5, 6))
    check_roundtrip(f2)(jnp.arange(4 * 5 * 6).reshape(4, 5, 6))


def test_pseudo_sliding_window_attn_block():
    block_len = 64
    seq_len = 128
    batch = 4
    num_heads = 2
    embed_size = 32
    num_layers = 2
    head_size = 16

    def block(x):
        query_block = x
        weights = jnp.sum(query_block, axis=3)  # [batch, len, num_heads]
        weights = jax.lax.broadcast_in_dim(weights, (batch, block_len, num_heads, block_len),
                                           (0, 1, 2))  # [batch, len, num_heads, len]
        # weights = jax.lax.with_sharding_constraint(weights, PartitionSpec('data', None, None, None))
        # without "bias", no boom
        bias = jnp.ones(block_len)
        bias = jnp.broadcast_to(bias, (batch, block_len, num_heads, block_len))
        weights = weights + bias
        return jnp.einsum('bqhk,bkhd->bqhd', weights, query_block).astype(query_block.dtype)

    x = jnp.arange(batch * block_len * num_heads * head_size).reshape(batch, block_len, num_heads, head_size).astype(jnp.float32)

    mesh = jax.sharding.Mesh(jax.devices('cpu'), ('data',))
    with mesh:
        f2 = check_roundtrip(block)(x)

def test_scan():
    def scanfn(x, y):
        return x + y, x * y

    x = jnp.arange(10)
    y = jnp.arange(10)

    def f(x, y):
        return jax.lax.scan(scanfn, x, y)

    f2 = check_roundtrip(f)(x, y)

    assert jnp.all(f(x, y)[0] == f2(x, y)[0])
    assert jnp.all(f(x, y)[1] == f2(x, y)[1])


def test_map():
    def f(x):
        return x + 1

    x = jnp.arange(10)

    def g(x):
        return jax.lax.map(f, x)

    g2 = check_roundtrip(g)(x)

    assert jnp.all(g(x) == g2(x))


def test_map_pytree():
    def f(x):
        return x[0] + 1, x[1] + 1

    x = jnp.arange(10)

    def g(x, y):
        return jax.lax.map(f, (x, y))

    g2 = check_roundtrip(g)(x, x)

    assert jnp.all(g(x, x)[0] == g2(x, x)[0])
    assert jnp.all(g(x, x)[1] == g2(x, x)[1])


def test_pseudo_sliding_window_attention():
    block_len = 64
    seq_len = 128
    batch = 4
    num_heads = 2
    embed_size = 32
    num_layers = 2
    head_size = 16
    def pseudo_sliding_window_attention(x):
        # (this is not attention, but is minimized from attn)
        # dims are [batch, len, num_heads, head_dim]
        # having num_heads is important. num_heads = 1, no boom
        def block(block_idx):
            query_block = jax.lax.dynamic_slice_in_dim(x, block_idx, block_len, axis=1)
            weights = jnp.sum(query_block, axis=3)  # [batch, len, num_heads]
            weights = jax.lax.broadcast_in_dim(weights, (batch, block_len, num_heads, block_len),
                                               (0, 1, 2))  # [batch, len, num_heads, len]
            # weights = with_sharding_constraint(weights, P('data', None, None, None))
            # without "bias", no boom
            bias = jnp.ones(block_len)
            bias = jnp.broadcast_to(bias, (batch, block_len, num_heads, block_len))
            weights = weights + bias
            return jnp.einsum('bqhk,bkhd->bqhd', weights, query_block).astype(query_block.dtype)

        num_blocks = seq_len // block_len
        blocked_attn = jax.lax.map(block, jnp.arange(0, num_blocks))  # [num_blocks, batch, len, num_heads, head_dim]
        blocked_attn = jnp.concatenate(blocked_attn, axis=1)

        return blocked_attn

    def fwd(params, x):
        @partial(jax.checkpoint, prevent_cse=False)
        def layer(x, params):
            qkv, o = params
            y = jnp.einsum('bte,hde->bthd', x, qkv)
            y = pseudo_sliding_window_attention(y)
            z = jnp.einsum('bthd,hde->bte', y, o)
            return z, None

        x, _ = jax.lax.scan(layer, x, params)

        return x

    def loss_fn(params, x):
        x = fwd(params, x)
        l = jnp.mean(x)
        return l

    def grad_fn(params, x):
        loss, grad = jax.value_and_grad(loss_fn)(params, x)
        # we can't reasonably sourcerize pytrees so just get the leaves
        return loss, *jax.tree_util.tree_leaves(grad)

    qkv = jnp.ones((num_layers, num_heads, head_size, embed_size), dtype=jnp.bfloat16)
    o = jnp.ones((num_layers, num_heads, head_size, embed_size), dtype=jnp.bfloat16)
    x = jnp.ones((batch, seq_len, embed_size), dtype=jnp.bfloat16)

    params = (qkv, o)

    f2 = check_roundtrip(grad_fn)(params, x)

def test_einsum():
    def f(x, y):
        return jnp.einsum('cij,cjk->ik', x, y)

    x = jnp.arange(8).reshape(2, 2, 2)
    y = jnp.arange(8).reshape(2, 2, 2)

    check_roundtrip(f)(x, y)

    source = sourcerize(f, use_jax_typing=True)(x, y)

    assert source.strip() == \
    textwrap.dedent("""
    def f(a: Int32[Array, '2 2 2'], b: Int32[Array, '2 2 2']):
    c = jnp.einsum('acb,abd->cd', a, b, preferred_element_type=jax.numpy.int32)
    return c""".strip())


def test_while_loop():
    def f(y):
        def loop(args):
            x, y = args
            return x + y, y

        def cond(args):
            return args[0] < 10

        return jax.lax.while_loop(cond, loop, (0, y))

    y = jnp.array(1)
    check_roundtrip(f)(y)


def test_cond():
    def f(x):
        def true_fn(x):
            return x + 1

        def false_fn(x):
            return x + 2

        return jax.lax.cond(x > 0, true_fn, false_fn, x)

    x = jnp.array(1)
    check_roundtrip(f)(x)


def test_switch():
    def f(x):
        def fn_a(x):
            return x + 1

        def fn_b(x):
            return x + 2

        def fn_c(x):
            return x + 3

        return jax.lax.switch(x, [fn_a, fn_b, fn_c], x)


    x = jnp.array(1)

    check_roundtrip(f)(x)


def test_gather():
    def f(x, y):
        return x[y]

    x = jnp.arange(8).reshape(2, 2, 2)
    y = jnp.array([0, 1])

    check_roundtrip(f)(x, y)

@pytest.mark.parametrize('fn', [jnp.cumsum, jnp.cumprod])
def test_cumulative(fn):
    def f(x):
        return fn(x, axis=0)

    x = jnp.arange(8).reshape(2, 2, 2) + 3

    check_roundtrip(f)(x)

def test_concatenate():
    def f(x, y):
        return jnp.concatenate([x, y], axis=0)

    x = jnp.arange(8).reshape(2, 2, 2)
    y = jnp.arange(8).reshape(2, 2, 2)

    check_roundtrip(f)(x, y)



# jax.lax.fori_loop
# jax.lax.dynamic_update_slice
# jax.lax.dynamic_update_index_in_dim
# jax.lax.dynamic_slice
# jax.lax.dynamic_index_in_dim
# jax.lax.dynamic_update_index_in_dim
# jax.lax.dynamic_slice_in_dim
# jax.lax.dynamic_update_slice_in_dim
# jax.lax.gather
# jax.lax.scatter
# jax.lax.scatter_add
# jax.lax.scatter_mul






# want to handle this (complex) case:
# { lambda a:u32[2] b:f32[128,72] c:f32[16,72] d:i32[4] e:f32[4,72] f:f32[4,72] g:f32[4,72,3,8,9]
#     h:f32[4,3,8,9] i:f32[4,8,9,72] j:f32[4,72] k:f32[4,72] l:f32[4,72] m:f32[4,72,288]
#     n:f32[4,288] o:f32[4,288,72] p:f32[4,72] q:bool[16,16] r:f32[72] s:f32[72]; t:i32[16]. let
#     u:key<fry>[] = random_wrap[impl=fry] a
#     v:key<fry>[2] = random_split[count=2] u
#     w:u32[2,2] = random_unwrap v
#     x:u32[1,2] = slice[limit_indices=(1, 2) start_indices=(0, 0) strides=(1, 1)] w
#     _:u32[2] = squeeze[dimensions=(0,)] x
#     y:u32[1,2] = slice[limit_indices=(2, 2) start_indices=(1, 0) strides=(1, 1)] w
#     z:u32[2] = squeeze[dimensions=(0,)] y
#     ba:f32[16,72] = pjit[
#       jaxpr={ lambda ; bb:f32[128,72] bc:i32[16]. let
#           bd:bool[16] = lt bc 0
#           be:i32[16] = add bc 128
#           bf:i32[16] = pjit[
#             jaxpr={ lambda ; bg:bool[16] bh:i32[16] bi:i32[16]. let
#                 bj:i32[16] = select_n bg bi bh
#               in (bj,) }
#             name=_where
#           ] bd be bc
#           bk:i32[16,1] = broadcast_in_dim[
#             broadcast_dimensions=(0,)
#             shape=(16, 1)
#           ] bf
#           bl:f32[16,72] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 72)
#             unique_indices=False
#           ] bb bk
#         in (bl,) }
#       name=_take
#     ] b t
#     bm:f32[16,72] = add ba c
#     bn:key<fry>[] = random_wrap[impl=fry] z
#     bo:key<fry>[4] = random_split[count=4] bn
#     bp:u32[4,2] = random_unwrap bo
#     _:f32[72] = pjit[
#       jaxpr={ lambda ; bq:f32[4,72] br:i32[]. let
#           bs:bool[] = lt br 0
#           bt:i32[] = add br 4
#           bu:i32[] = pjit[
#             jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                 by:i32[] = select_n bv bx bw
#               in (by,) }
#             name=_where
#           ] bs bt br
#           bz:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] bu
#           ca:f32[72] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0,), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 72)
#             unique_indices=False
#           ] bq bz
#         in (ca,) }
#       name=_take
#     ] e 0
#     _:f32[72] = pjit[
#       jaxpr={ lambda ; bq:f32[4,72] br:i32[]. let
#           bs:bool[] = lt br 0
#           bt:i32[] = add br 4
#           bu:i32[] = pjit[
#             jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                 by:i32[] = select_n bv bx bw
#               in (by,) }
#             name=_where
#           ] bs bt br
#           bz:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] bu
#           ca:f32[72] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0,), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 72)
#             unique_indices=False
#           ] bq bz
#         in (ca,) }
#       name=_take
#     ] f 0
#     _:f32[72,3,8,9] = pjit[
#       jaxpr={ lambda ; cb:f32[4,72,3,8,9] cc:i32[]. let
#           cd:bool[] = lt cc 0
#           ce:i32[] = add cc 4
#           cf:i32[] = pjit[
#             jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                 by:i32[] = select_n bv bx bw
#               in (by,) }
#             name=_where
#           ] cd ce cc
#           cg:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] cf
#           ch:f32[72,3,8,9] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0, 1, 2, 3), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 72, 3, 8, 9)
#             unique_indices=False
#           ] cb cg
#         in (ch,) }
#       name=_take
#     ] g 0
#     _:f32[3,8,9] = pjit[
#       jaxpr={ lambda ; ci:f32[4,3,8,9] cj:i32[]. let
#           ck:bool[] = lt cj 0
#           cl:i32[] = add cj 4
#           cm:i32[] = pjit[
#             jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                 by:i32[] = select_n bv bx bw
#               in (by,) }
#             name=_where
#           ] ck cl cj
#           cn:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] cm
#           co:f32[3,8,9] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0, 1, 2), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 3, 8, 9)
#             unique_indices=False
#           ] ci cn
#         in (co,) }
#       name=_take
#     ] h 0
#     _:f32[8,9,72] = pjit[
#       jaxpr={ lambda ; cp:f32[4,8,9,72] cq:i32[]. let
#           cr:bool[] = lt cq 0
#           cs:i32[] = add cq 4
#           ct:i32[] = pjit[
#             jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                 by:i32[] = select_n bv bx bw
#               in (by,) }
#             name=_where
#           ] cr cs cq
#           cu:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] ct
#           cv:f32[8,9,72] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0, 1, 2), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 8, 9, 72)
#             unique_indices=False
#           ] cp cu
#         in (cv,) }
#       name=_take
#     ] i 0
#     _:f32[72] = pjit[
#       jaxpr={ lambda ; bq:f32[4,72] br:i32[]. let
#           bs:bool[] = lt br 0
#           bt:i32[] = add br 4
#           bu:i32[] = pjit[
#             jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                 by:i32[] = select_n bv bx bw
#               in (by,) }
#             name=_where
#           ] bs bt br
#           bz:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] bu
#           ca:f32[72] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0,), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 72)
#             unique_indices=False
#           ] bq bz
#         in (ca,) }
#       name=_take
#     ] j 0
#     _:f32[72] = pjit[
#       jaxpr={ lambda ; bq:f32[4,72] br:i32[]. let
#           bs:bool[] = lt br 0
#           bt:i32[] = add br 4
#           bu:i32[] = pjit[
#             jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                 by:i32[] = select_n bv bx bw
#               in (by,) }
#             name=_where
#           ] bs bt br
#           bz:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] bu
#           ca:f32[72] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0,), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 72)
#             unique_indices=False
#           ] bq bz
#         in (ca,) }
#       name=_take
#     ] k 0
#     _:f32[72] = pjit[
#       jaxpr={ lambda ; bq:f32[4,72] br:i32[]. let
#           bs:bool[] = lt br 0
#           bt:i32[] = add br 4
#           bu:i32[] = pjit[
#             jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                 by:i32[] = select_n bv bx bw
#               in (by,) }
#             name=_where
#           ] bs bt br
#           bz:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] bu
#           ca:f32[72] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0,), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 72)
#             unique_indices=False
#           ] bq bz
#         in (ca,) }
#       name=_take
#     ] l 0
#     _:f32[72,288] = pjit[
#       jaxpr={ lambda ; cw:f32[4,72,288] cx:i32[]. let
#           cy:bool[] = lt cx 0
#           cz:i32[] = add cx 4
#           da:i32[] = pjit[
#             jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                 by:i32[] = select_n bv bx bw
#               in (by,) }
#             name=_where
#           ] cy cz cx
#           db:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] da
#           dc:f32[72,288] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0, 1), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 72, 288)
#             unique_indices=False
#           ] cw db
#         in (dc,) }
#       name=_take
#     ] m 0
#     _:f32[288] = pjit[
#       jaxpr={ lambda ; dd:f32[4,288] de:i32[]. let
#           df:bool[] = lt de 0
#           dg:i32[] = add de 4
#           dh:i32[] = pjit[
#             jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                 by:i32[] = select_n bv bx bw
#               in (by,) }
#             name=_where
#           ] df dg de
#           di:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] dh
#           dj:f32[288] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0,), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 288)
#             unique_indices=False
#           ] dd di
#         in (dj,) }
#       name=_take
#     ] n 0
#     _:f32[288,72] = pjit[
#       jaxpr={ lambda ; dk:f32[4,288,72] dl:i32[]. let
#           dm:bool[] = lt dl 0
#           dn:i32[] = add dl 4
#           do:i32[] = pjit[
#             jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                 by:i32[] = select_n bv bx bw
#               in (by,) }
#             name=_where
#           ] dm dn dl
#           dp:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] do
#           dq:f32[288,72] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0, 1), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 288, 72)
#             unique_indices=False
#           ] dk dp
#         in (dq,) }
#       name=_take
#     ] o 0
#     _:f32[72] = pjit[
#       jaxpr={ lambda ; bq:f32[4,72] br:i32[]. let
#           bs:bool[] = lt br 0
#           bt:i32[] = add br 4
#           bu:i32[] = pjit[
#             jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                 by:i32[] = select_n bv bx bw
#               in (by,) }
#             name=_where
#           ] bs bt br
#           bz:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] bu
#           ca:f32[72] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0,), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 72)
#             unique_indices=False
#           ] bq bz
#         in (ca,) }
#       name=_take
#     ] p 0
#     _:i32[] = pjit[
#       jaxpr={ lambda ; dr:i32[4] ds:i32[]. let
#           dt:bool[] = lt ds 0
#           du:i32[] = add ds 4
#           dv:i32[] = pjit[
#             jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                 by:i32[] = select_n bv bx bw
#               in (by,) }
#             name=_where
#           ] dt du ds
#           dw:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] dv
#           dx:i32[] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=-2147483648
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1,)
#             unique_indices=False
#           ] dr dw
#         in (dx,) }
#       name=_take
#     ] d 0
#     dy:f32[16,72] = scan[
#       jaxpr={ lambda ; dz:bool[16,16] ea:f32[16,72] eb:f32[72] ec:f32[72] ed:f32[72,3,8,9]
#           ee:f32[3,8,9] ef:f32[8,9,72] eg:f32[72] eh:f32[72] ei:f32[72] ej:f32[72,288]
#           ek:f32[288] el:f32[288,72] em:f32[72] en:i32[] eo:u32[2]. let
#           ep:key<fry>[] = random_wrap[impl=fry] eo
#           eq:key<fry>[3] = random_split[count=3] ep
#           er:u32[3,2] = random_unwrap eq
#           es:u32[1,2] = slice[
#             limit_indices=(1, 2)
#             start_indices=(0, 0)
#             strides=(1, 1)
#           ] er
#           _:u32[2] = squeeze[dimensions=(0,)] es
#           et:u32[1,2] = slice[
#             limit_indices=(2, 2)
#             start_indices=(1, 0)
#             strides=(1, 1)
#           ] er
#           _:u32[2] = squeeze[dimensions=(0,)] et
#           eu:u32[1,2] = slice[
#             limit_indices=(3, 2)
#             start_indices=(2, 0)
#             strides=(1, 1)
#           ] er
#           _:u32[2] = squeeze[dimensions=(0,)] eu
#           ev:f32[16] = reduce_sum[axes=(1,)] ea
#           ew:f32[16] = div ev 72.0
#           ex:f32[16] = pjit[
#             jaxpr={ lambda ; ey:f32[16,72] ez:i32[]. let
#                 fa:f32[16] = reduce_sum[axes=(1,)] ey
#                 fb:f32[16,1] = broadcast_in_dim[
#                   broadcast_dimensions=(0,)
#                   shape=(16, 1)
#                 ] fa
#                 fc:f32[16,1] = div fb 72.0
#                 fd:f32[16,72] = sub ey fc
#                 fe:f32[16,72] = integer_pow[y=2] fd
#                 ff:f32[] = convert_element_type[
#                   new_dtype=float32
#                   weak_type=False
#                 ] ez
#                 fg:f32[] = sub 72.0 ff
#                 fh:f32[16] = reduce_sum[axes=(1,)] fe
#                 fi:f32[16] = div fh fg
#               in (fi,) }
#             name=_var
#           ] ea 0
#           fj:f32[16] = add ex 9.999999747378752e-06
#           fk:f32[16] = rsqrt fj
#           fl:f32[72,16] = broadcast_in_dim[
#             broadcast_dimensions=(1,)
#             shape=(72, 16)
#           ] ew
#           fm:f32[16,72] = transpose[permutation=(1, 0)] fl
#           fn:f32[16,72] = sub ea fm
#           fo:f32[72,16] = broadcast_in_dim[
#             broadcast_dimensions=(1,)
#             shape=(72, 16)
#           ] fk
#           fp:f32[16,72] = transpose[permutation=(1, 0)] fo
#           fq:f32[16,72] = mul fn fp
#           fr:f32[16,72] = broadcast_in_dim[
#             broadcast_dimensions=(1,)
#             shape=(16, 72)
#           ] eb
#           fs:f32[16,72] = mul fr fq
#           ft:f32[16,72] = broadcast_in_dim[
#             broadcast_dimensions=(1,)
#             shape=(16, 72)
#           ] ec
#           fu:f32[16,72] = add fs ft
#           fv:f32[16,3,8,9] = dot_general[
#             dimension_numbers=(([1], [0]), ([], []))
#           ] fu ed
#           fw:f32[16,3,8,9] = broadcast_in_dim[
#             broadcast_dimensions=(1, 2, 3)
#             shape=(16, 3, 8, 9)
#           ] ee
#           fx:f32[16,3,8,9] = add fv fw
#           fy:f32[3,8,16,9] = transpose[permutation=(1, 2, 0, 3)] fx
#           fz:f32[8,16,9] = pjit[
#             jaxpr={ lambda ; ga:f32[3,8,16,9] gb:i32[]. let
#                 gc:bool[] = lt gb 0
#                 gd:i32[] = add gb 3
#                 ge:i32[] = pjit[
#                   jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                       by:i32[] = select_n bv bx bw
#                     in (by,) }
#                   name=_where
#                 ] gc gd gb
#                 gf:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] ge
#                 gg:f32[8,16,9] = gather[
#                   dimension_numbers=GatherDimensionNumbers(offset_dims=(0, 1, 2), collapsed_slice_dims=(0,), start_index_map=(0,))
#                   fill_value=nan
#                   indices_are_sorted=False
#                   mode=GatherScatterMode.FILL_OR_DROP
#                   slice_sizes=(1, 8, 16, 9)
#                   unique_indices=False
#                 ] ga gf
#               in (gg,) }
#             name=_take
#           ] fy 0
#           gh:f32[8,16,9] = pjit[
#             jaxpr={ lambda ; ga:f32[3,8,16,9] gb:i32[]. let
#                 gc:bool[] = lt gb 0
#                 gd:i32[] = add gb 3
#                 ge:i32[] = pjit[
#                   jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                       by:i32[] = select_n bv bx bw
#                     in (by,) }
#                   name=_where
#                 ] gc gd gb
#                 gf:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] ge
#                 gg:f32[8,16,9] = gather[
#                   dimension_numbers=GatherDimensionNumbers(offset_dims=(0, 1, 2), collapsed_slice_dims=(0,), start_index_map=(0,))
#                   fill_value=nan
#                   indices_are_sorted=False
#                   mode=GatherScatterMode.FILL_OR_DROP
#                   slice_sizes=(1, 8, 16, 9)
#                   unique_indices=False
#                 ] ga gf
#               in (gg,) }
#             name=_take
#           ] fy 1
#           gi:f32[8,16,9] = pjit[
#             jaxpr={ lambda ; ga:f32[3,8,16,9] gb:i32[]. let
#                 gc:bool[] = lt gb 0
#                 gd:i32[] = add gb 3
#                 ge:i32[] = pjit[
#                   jaxpr={ lambda ; bv:bool[] bw:i32[] bx:i32[]. let
#                       by:i32[] = select_n bv bx bw
#                     in (by,) }
#                   name=_where
#                 ] gc gd gb
#                 gf:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] ge
#                 gg:f32[8,16,9] = gather[
#                   dimension_numbers=GatherDimensionNumbers(offset_dims=(0, 1, 2), collapsed_slice_dims=(0,), start_index_map=(0,))
#                   fill_value=nan
#                   indices_are_sorted=False
#                   mode=GatherScatterMode.FILL_OR_DROP
#                   slice_sizes=(1, 8, 16, 9)
#                   unique_indices=False
#                 ] ga gf
#               in (gg,) }
#             name=_take
#           ] fy 2
#           gj:f32[] = rsqrt 9.0
#           gk:f32[] = convert_element_type[new_dtype=float32 weak_type=False] gj
#           gl:f32[8,16,9] = mul fz gk
#           gm:f32[8,16,16] = dot_general[
#             dimension_numbers=(([2], [2]), ([0], [0]))
#           ] gl gh
#           gn:f32[16,16] = convert_element_type[new_dtype=float32 weak_type=True] dz
#           go:f32[16,16] = sub 1.0 gn
#           gp:f32[16,16] = mul go -1000000000.0
#           gq:f32[8,16,16] = broadcast_in_dim[
#             broadcast_dimensions=(1, 2)
#             shape=(8, 16, 16)
#           ] gp
#           gr:f32[8,16,16] = convert_element_type[
#             new_dtype=float32
#             weak_type=False
#           ] gq
#           gs:f32[8,16,16] = add gm gr
#           gt:f32[8,16] = reduce_max[axes=(2,)] gs
#           gu:f32[8,16,1] = broadcast_in_dim[
#             broadcast_dimensions=(0, 1)
#             shape=(8, 16, 1)
#           ] gt
#           gv:f32[8,16,1] = stop_gradient gu
#           gw:f32[8,16,16] = sub gs gv
#           gx:f32[8,16,16] = exp gw
#           gy:f32[8,16] = reduce_sum[axes=(2,)] gx
#           gz:f32[8,16,1] = broadcast_in_dim[
#             broadcast_dimensions=(0, 1)
#             shape=(8, 16, 1)
#           ] gy
#           ha:f32[8,16,16] = div gx gz
#           hb:f32[8,16,9] = dot_general[
#             dimension_numbers=(([2], [1]), ([0], [0]))
#           ] ha gi
#           hc:f32[16,72] = dot_general[
#             dimension_numbers=(([0, 2], [0, 1]), ([], []))
#           ] hb ef
#           hd:f32[16,72] = broadcast_in_dim[
#             broadcast_dimensions=(1,)
#             shape=(16, 72)
#           ] eg
#           he:f32[16,72] = add hc hd
#           hf:f32[16,72] = add ea he
#           hg:f32[16] = reduce_sum[axes=(1,)] hf
#           hh:f32[16] = div hg 72.0
#           hi:f32[16] = pjit[
#             jaxpr={ lambda ; ey:f32[16,72] ez:i32[]. let
#                 fa:f32[16] = reduce_sum[axes=(1,)] ey
#                 fb:f32[16,1] = broadcast_in_dim[
#                   broadcast_dimensions=(0,)
#                   shape=(16, 1)
#                 ] fa
#                 fc:f32[16,1] = div fb 72.0
#                 fd:f32[16,72] = sub ey fc
#                 fe:f32[16,72] = integer_pow[y=2] fd
#                 ff:f32[] = convert_element_type[
#                   new_dtype=float32
#                   weak_type=False
#                 ] ez
#                 fg:f32[] = sub 72.0 ff
#                 fh:f32[16] = reduce_sum[axes=(1,)] fe
#                 fi:f32[16] = div fh fg
#               in (fi,) }
#             name=_var
#           ] hf 0
#           hj:f32[16] = add hi 9.999999747378752e-06
#           hk:f32[16] = rsqrt hj
#           hl:f32[72,16] = broadcast_in_dim[
#             broadcast_dimensions=(1,)
#             shape=(72, 16)
#           ] hh
#           hm:f32[16,72] = transpose[permutation=(1, 0)] hl
#           hn:f32[16,72] = sub hf hm
#           ho:f32[72,16] = broadcast_in_dim[
#             broadcast_dimensions=(1,)
#             shape=(72, 16)
#           ] hk
#           hp:f32[16,72] = transpose[permutation=(1, 0)] ho
#           hq:f32[16,72] = mul hn hp
#           hr:f32[16,72] = broadcast_in_dim[
#             broadcast_dimensions=(1,)
#             shape=(16, 72)
#           ] eh
#           hs:f32[16,72] = mul hr hq
#           ht:f32[16,72] = broadcast_in_dim[
#             broadcast_dimensions=(1,)
#             shape=(16, 72)
#           ] ei
#           hu:f32[16,72] = add hs ht
#           hv:f32[16,288] = dot_general[dimension_numbers=(([1], [0]), ([], []))] hu
#             ej
#           hw:f32[16,288] = broadcast_in_dim[
#             broadcast_dimensions=(1,)
#             shape=(16, 288)
#           ] ek
#           hx:f32[16,288] = add hv hw
#           hy:f32[16,288] = integer_pow[y=3] hx
#           hz:f32[16,288] = mul 0.044714998453855515 hy
#           ia:f32[16,288] = add hx hz
#           ib:f32[16,288] = mul 0.7978845834732056 ia
#           ic:f32[16,288] = tanh ib
#           id:f32[16,288] = add 1.0 ic
#           ie:f32[16,288] = mul 0.5 id
#           if:f32[16,288] = mul hx ie
#           ig:f32[16,72] = dot_general[dimension_numbers=(([1], [0]), ([], []))] if
#             el
#           ih:f32[16,72] = broadcast_in_dim[
#             broadcast_dimensions=(1,)
#             shape=(16, 72)
#           ] em
#           ii:f32[16,72] = add ig ih
#           ij:f32[16,72] = add hf ii
#         in (ij,) }
#       length=4
#       linear=(False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False)
#       num_carry=1
#       num_consts=1
#       reverse=False
#       unroll=1
#     ] q bm e f g h i j k l m n o p d bp
#     ik:f32[16] = reduce_sum[axes=(1,)] dy
#     il:f32[16] = div ik 72.0
#     im:f32[16] = pjit[
#       jaxpr={ lambda ; ey:f32[16,72] ez:i32[]. let
#           fa:f32[16] = reduce_sum[axes=(1,)] ey
#           fb:f32[16,1] = broadcast_in_dim[
#             broadcast_dimensions=(0,)
#             shape=(16, 1)
#           ] fa
#           fc:f32[16,1] = div fb 72.0
#           fd:f32[16,72] = sub ey fc
#           fe:f32[16,72] = integer_pow[y=2] fd
#           ff:f32[] = convert_element_type[new_dtype=float32 weak_type=False] ez
#           fg:f32[] = sub 72.0 ff
#           fh:f32[16] = reduce_sum[axes=(1,)] fe
#           fi:f32[16] = div fh fg
#         in (fi,) }
#       name=_var
#     ] dy 0
#     in:f32[16] = add im 9.999999747378752e-06
#     io:f32[16] = rsqrt in
#     ip:f32[72,16] = broadcast_in_dim[broadcast_dimensions=(1,) shape=(72, 16)] il
#     iq:f32[16,72] = transpose[permutation=(1, 0)] ip
#     ir:f32[16,72] = sub dy iq
#     is:f32[72,16] = broadcast_in_dim[broadcast_dimensions=(1,) shape=(72, 16)] io
#     it:f32[16,72] = transpose[permutation=(1, 0)] is
#     iu:f32[16,72] = mul ir it
#     iv:f32[16,72] = broadcast_in_dim[broadcast_dimensions=(1,) shape=(16, 72)] r
#     iw:f32[16,72] = mul iv iu
#     ix:f32[16,72] = broadcast_in_dim[broadcast_dimensions=(1,) shape=(16, 72)] s
#     iy:f32[16,72] = add iw ix
#     iz:f32[16,128] = dot_general[dimension_numbers=(([1], [1]), ([], []))] iy b
#   in (iz,) }

### Jaxpr for gpt2 train_step
# let _take = { lambda ; a:f32[2,32] b:i32[]. let
#     c:bool[] = lt b 0
#     d:i32[] = add b 2
#     e:i32[] = pjit[
#       name=_where
#       jaxpr={ lambda ; f:bool[] g:i32[] h:i32[]. let
#           i:i32[] = select_n f h g
#         in (i,) }
#     ] c d b
#     j:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] e
#     k:f32[32] = gather[
#       dimension_numbers=GatherDimensionNumbers(offset_dims=(0,), collapsed_slice_dims=(0,), start_index_map=(0,))
#       fill_value=nan
#       indices_are_sorted=False
#       mode=GatherScatterMode.FILL_OR_DROP
#       slice_sizes=(1, 32)
#       unique_indices=False
#     ] a j
#   in (k,) } in
# let _var = { lambda ; l:f32[32,512,32] m:i32[]. let
#     n:f32[32,512] = reduce_sum[axes=(2,)] l
#     o:f32[32,512,1] = broadcast_in_dim[
#       broadcast_dimensions=(0, 1)
#       shape=(32, 512, 1)
#     ] n
#     p:f32[32,512,1] = div o 32.0
#     q:f32[32,512,32] = sub l p
#     r:f32[32,512,32] = integer_pow[y=2] q
#     s:f32[] = convert_element_type[new_dtype=float32 weak_type=False] m
#     t:f32[] = sub 32.0 s
#     u:f32[32,512] = reduce_sum[axes=(2,)] r
#     v:f32[32,512] = div u t
#   in (v,) } in
# let _where = { lambda ; f:bool[] g:i32[] h:i32[]. let
#     i:i32[] = select_n f h g
#   in (i,) } in
# let _take1 = { lambda ; w:f32[32,3,4,512,8] x:i32[]. let
#     y:bool[] = lt x 0
#     z:i32[] = add x 3
#     ba:i32[] = pjit[name=_where jaxpr=_where] y z x
#     bb:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] ba
#     bc:f32[32,4,512,8] = gather[
#       dimension_numbers=GatherDimensionNumbers(offset_dims=(0, 1, 2, 3), collapsed_slice_dims=(1,), start_index_map=(1,))
#       fill_value=nan
#       indices_are_sorted=False
#       mode=GatherScatterMode.FILL_OR_DROP
#       slice_sizes=(32, 1, 4, 512, 8)
#       unique_indices=False
#     ] w bb
#   in (bc,) } in
# { lambda ; bd:f32[2,32] be:f32[2,32] bf:f32[2,32,3,4,8] bg:f32[2,3,4,8] bh:f32[2,4,8,32]
#     bi:f32[2,32] bj:f32[2,32] bk:f32[2,32] bl:f32[2,32,128] bm:f32[2,128] bn:f32[2,128,32]
#     bo:f32[2,32] bp:f32[32] bq:f32[32] br:f32[50257,32] bs:f32[512,32] bt:i32[32,512]
#     bu:f32[32,512]. let
#     bv:f32[32,512,32] = pjit[
#       name=_take
#       jaxpr={ lambda ; bw:f32[50257,32] bx:i32[32,512]. let
#           by:bool[32,512] = lt bx 0
#           bz:i32[32,512] = add bx 50257
#           ca:i32[32,512] = pjit[
#             name=_where
#             jaxpr={ lambda ; cb:bool[32,512] cc:i32[32,512] cd:i32[32,512]. let
#                 ce:i32[32,512] = select_n cb cd cc
#               in (ce,) }
#           ] by bz bx
#           cf:i32[32,512,1] = broadcast_in_dim[
#             broadcast_dimensions=(0, 1)
#             shape=(32, 512, 1)
#           ] ca
#           cg:f32[32,512,32] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(2,), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 32)
#             unique_indices=False
#           ] bw cf
#         in (cg,) }
#     ] br bt
#     ch:i32[512] = iota[dimension=0 dtype=int32 shape=(512,)]
#     ci:i32[512] = mul ch 1
#     cj:i32[512] = add ci 0
#     ck:f32[512,32] = pjit[
#       name=_take
#       jaxpr={ lambda ; cl:f32[512,32] cm:i32[512]. let
#           cn:bool[512] = lt cm 0
#           co:i32[512] = add cm 512
#           cp:i32[512] = pjit[
#             name=_where
#             jaxpr={ lambda ; cq:bool[512] cr:i32[512] cs:i32[512]. let
#                 ct:i32[512] = select_n cq cs cr
#               in (ct,) }
#           ] cn co cm
#           cu:i32[512,1] = broadcast_in_dim[
#             broadcast_dimensions=(0,)
#             shape=(512, 1)
#           ] cp
#           cv:f32[512,32] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 32)
#             unique_indices=False
#           ] cl cu
#         in (cv,) }
#     ] bs cj
#     cw:f32[32,512,32] = broadcast_in_dim[
#       broadcast_dimensions=(1, 2)
#       shape=(32, 512, 32)
#     ] ck
#     cx:f32[32,512,32] = add bv cw
#     cy:i32[2] = iota[dimension=0 dtype=int32 shape=(2,)]
#     cz:i32[2] = mul cy 1
#     da:i32[2] = add cz 0
#     _:f32[32] = pjit[name=_take jaxpr=_take] bd 0
#     _:f32[32] = pjit[name=_take jaxpr=_take] be 0
#     _:f32[32,3,4,8] = pjit[
#       name=_take
#       jaxpr={ lambda ; db:f32[2,32,3,4,8] dc:i32[]. let
#           dd:bool[] = lt dc 0
#           de:i32[] = add dc 2
#           df:i32[] = pjit[name=_where jaxpr=_where] dd de dc
#           dg:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] df
#           dh:f32[32,3,4,8] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0, 1, 2, 3), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 32, 3, 4, 8)
#             unique_indices=False
#           ] db dg
#         in (dh,) }
#     ] bf 0
#     _:f32[3,4,8] = pjit[
#       name=_take
#       jaxpr={ lambda ; di:f32[2,3,4,8] dj:i32[]. let
#           dk:bool[] = lt dj 0
#           dl:i32[] = add dj 2
#           dm:i32[] = pjit[name=_where jaxpr=_where] dk dl dj
#           dn:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] dm
#           do:f32[3,4,8] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0, 1, 2), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 3, 4, 8)
#             unique_indices=False
#           ] di dn
#         in (do,) }
#     ] bg 0
#     _:f32[4,8,32] = pjit[
#       name=_take
#       jaxpr={ lambda ; dp:f32[2,4,8,32] dq:i32[]. let
#           dr:bool[] = lt dq 0
#           ds:i32[] = add dq 2
#           dt:i32[] = pjit[name=_where jaxpr=_where] dr ds dq
#           du:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] dt
#           dv:f32[4,8,32] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0, 1, 2), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 4, 8, 32)
#             unique_indices=False
#           ] dp du
#         in (dv,) }
#     ] bh 0
#     _:f32[32] = pjit[name=_take jaxpr=_take] bi 0
#     _:f32[32] = pjit[name=_take jaxpr=_take] bj 0
#     _:f32[32] = pjit[name=_take jaxpr=_take] bk 0
#     _:f32[32,128] = pjit[
#       name=_take
#       jaxpr={ lambda ; dw:f32[2,32,128] dx:i32[]. let
#           dy:bool[] = lt dx 0
#           dz:i32[] = add dx 2
#           ea:i32[] = pjit[name=_where jaxpr=_where] dy dz dx
#           eb:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] ea
#           ec:f32[32,128] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0, 1), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 32, 128)
#             unique_indices=False
#           ] dw eb
#         in (ec,) }
#     ] bl 0
#     _:f32[128] = pjit[
#       name=_take
#       jaxpr={ lambda ; ed:f32[2,128] ee:i32[]. let
#           ef:bool[] = lt ee 0
#           eg:i32[] = add ee 2
#           eh:i32[] = pjit[name=_where jaxpr=_where] ef eg ee
#           ei:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] eh
#           ej:f32[128] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0,), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 128)
#             unique_indices=False
#           ] ed ei
#         in (ej,) }
#     ] bm 0
#     _:f32[128,32] = pjit[
#       name=_take
#       jaxpr={ lambda ; ek:f32[2,128,32] el:i32[]. let
#           em:bool[] = lt el 0
#           en:i32[] = add el 2
#           eo:i32[] = pjit[name=_where jaxpr=_where] em en el
#           ep:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] eo
#           eq:f32[128,32] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(0, 1), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=nan
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1, 128, 32)
#             unique_indices=False
#           ] ek ep
#         in (eq,) }
#     ] bn 0
#     _:f32[32] = pjit[name=_take jaxpr=_take] bo 0
#     _:i32[] = pjit[
#       name=_take
#       jaxpr={ lambda ; er:i32[2] es:i32[]. let
#           et:bool[] = lt es 0
#           eu:i32[] = add es 2
#           ev:i32[] = pjit[name=_where jaxpr=_where] et eu es
#           ew:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] ev
#           ex:i32[] = gather[
#             dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,))
#             fill_value=-2147483648
#             indices_are_sorted=False
#             mode=GatherScatterMode.FILL_OR_DROP
#             slice_sizes=(1,)
#             unique_indices=False
#           ] er ew
#         in (ex,) }
#     ] da 0
#     ey:f32[32,512,32] = scan[
#       _split_transpose=False
#       jaxpr={ lambda ; ez:f32[32,512,32] fa:f32[32] fb:f32[32] fc:f32[32,3,4,8] fd:f32[3,4,8]
#           fe:f32[4,8,32] ff:f32[32] fg:f32[32] fh:f32[32] fi:f32[32,128] fj:f32[128]
#           fk:f32[128,32] fl:f32[32] fm:i32[]. let
#           fn:f32[32,512,32] = remat2[
#             differentiated=False
#             jaxpr={ lambda ; fo:f32[32,512,32] fp:f32[32] fq:f32[32] fr:f32[32,3,4,8]
#                 fs:f32[3,4,8] ft:f32[4,8,32] fu:f32[32] fv:f32[32] fw:f32[32] fx:f32[32,128]
#                 fy:f32[128] fz:f32[128,32] ga:f32[32] gb:i32[]. let
#                 gc:f32[32,512] = reduce_sum[axes=(2,)] fo
#                 gd:f32[32,512] = div gc 32.0
#                 ge:f32[32,512] = pjit[name=_var jaxpr=_var] fo 0
#                 gf:f32[32,512] = add ge 9.999999747378752e-06
#                 gg:f32[32,512] = rsqrt gf
#                 gh:f32[32,32,512] = broadcast_in_dim[
#                   broadcast_dimensions=(1, 2)
#                   shape=(32, 32, 512)
#                 ] gd
#                 gi:f32[32,512,32] = transpose[permutation=(1, 2, 0)] gh
#                 gj:f32[32,512,32] = sub fo gi
#                 gk:f32[32,32,512] = broadcast_in_dim[
#                   broadcast_dimensions=(1, 2)
#                   shape=(32, 32, 512)
#                 ] gg
#                 gl:f32[32,512,32] = transpose[permutation=(1, 2, 0)] gk
#                 gm:f32[32,512,32] = mul gj gl
#                 gn:f32[32,512,32] = broadcast_in_dim[
#                   broadcast_dimensions=(2,)
#                   shape=(32, 512, 32)
#                 ] fp
#                 go:f32[32,512,32] = mul gn gm
#                 gp:f32[32,512,32] = broadcast_in_dim[
#                   broadcast_dimensions=(2,)
#                   shape=(32, 512, 32)
#                 ] fq
#                 gq:f32[32,512,32] = add go gp
#                 gr:f32[32,512,3,4,8] = dot_general[
#                   dimension_numbers=(([2], [0]), ([], []))
#                   preferred_element_type=float32
#                 ] gq fr
#                 gs:f32[32,512,3,4,8] = broadcast_in_dim[
#                   broadcast_dimensions=(2, 3, 4)
#                   shape=(32, 512, 3, 4, 8)
#                 ] fs
#                 gt:f32[32,512,3,4,8] = add gr gs
#                 gu:f32[32,3,4,512,8] = transpose[permutation=(0, 2, 3, 1, 4)] gt
#                 gv:f32[32,4,512,8] = pjit[name=_take jaxpr=_take1] gu 0
#                 gw:f32[32,4,512,8] = pjit[name=_take jaxpr=_take1] gu 1
#                 gx:f32[32,4,512,8] = pjit[name=_take jaxpr=_take1] gu 2
#                 gy:i32[512] = iota[dimension=0 dtype=int32 shape=(512,)]
#                 gz:i32[512] = mul gy 1
#                 ha:i32[512] = add gz 0
#                 hb:i32[512] = iota[dimension=0 dtype=int32 shape=(512,)]
#                 hc:i32[512] = mul hb 1
#                 hd:i32[512] = add hc 0
#                 he:i32[512,512] = broadcast_in_dim[
#                   broadcast_dimensions=(1,)
#                   shape=(512, 512)
#                 ] hd
#                 hf:i32[512,512] = broadcast_in_dim[
#                   broadcast_dimensions=(1,)
#                   shape=(512, 512)
#                 ] ha
#                 hg:i32[512,512] = transpose[permutation=(1, 0)] hf
#                 hh:bool[512,512] = ge hg he
#                 hi:f32[] = sqrt 8.0
#                 hj:f32[] = convert_element_type[
#                   new_dtype=float32
#                   weak_type=False
#                 ] hi
#                 hk:f32[32,4,512,8] = div gv hj
#                 hl:f32[32,4,512,512] = dot_general[
#                   dimension_numbers=(([3], [3]), ([0, 1], [0, 1]))
#                   preferred_element_type=float32
#                 ] hk gw
#                 hm:bool[32,4,512,512] = broadcast_in_dim[
#                   broadcast_dimensions=(2, 3)
#                   shape=(32, 4, 512, 512)
#                 ] hh
#                 hn:f32[32,4,512,512] = pjit[
#                   name=_where
#                   jaxpr={ lambda ; ho:bool[32,4,512,512] hp:f32[32,4,512,512] hq:f32[]. let
#                       hr:f32[] = convert_element_type[
#                         new_dtype=float32
#                         weak_type=False
#                       ] hq
#                       hs:f32[32,4,512,512] = broadcast_in_dim[
#                         broadcast_dimensions=()
#                         shape=(32, 4, 512, 512)
#                       ] hr
#                       ht:f32[32,4,512,512] = select_n ho hs hp
#                     in (ht,) }
#                 ] hm hl -1000000000.0
#                 hu:f32[32,4,512,512] = custom_jvp_call[
#                   call_jaxpr={ lambda ; hv:f32[32,4,512,512]. let
#                       hw:f32[32,4,512] = reduce_max[axes=(3,)] hv
#                       hx:f32[32,4,512,1] = broadcast_in_dim[
#                         broadcast_dimensions=(0, 1, 2)
#                         shape=(32, 4, 512, 1)
#                       ] hw
#                       hy:f32[32,4,512,512] = sub hv hx
#                       hz:f32[32,4,512,512] = exp hy
#                       ia:f32[32,4,512] = reduce_sum[axes=(3,)] hz
#                       ib:f32[32,4,512,1] = broadcast_in_dim[
#                         broadcast_dimensions=(0, 1, 2)
#                         shape=(32, 4, 512, 1)
#                       ] ia
#                       ic:f32[32,4,512,512] = div hz ib
#                     in (ic,) }
#                   jvp_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x348214b80>
#                   num_consts=0
#                   symbolic_zeros=False
#                 ] hn
#                 id:f32[32,4,512,8] = dot_general[
#                   dimension_numbers=(([3], [2]), ([0, 1], [0, 1]))
#                   preferred_element_type=float32
#                 ] hu gx
#                 ie:f32[32,512,32] = dot_general[
#                   dimension_numbers=(([1, 3], [0, 1]), ([], []))
#                   preferred_element_type=float32
#                 ] id ft
#                 if:f32[32,512,32] = broadcast_in_dim[
#                   broadcast_dimensions=(2,)
#                   shape=(32, 512, 32)
#                 ] fu
#                 ig:f32[32,512,32] = add ie if
#                 ih:f32[32,512,32] = add fo ig
#                 ii:f32[32,512] = reduce_sum[axes=(2,)] ih
#                 ij:f32[32,512] = div ii 32.0
#                 ik:f32[32,512] = pjit[name=_var jaxpr=_var] ih 0
#                 il:f32[32,512] = add ik 9.999999747378752e-06
#                 im:f32[32,512] = rsqrt il
#                 in:f32[32,32,512] = broadcast_in_dim[
#                   broadcast_dimensions=(1, 2)
#                   shape=(32, 32, 512)
#                 ] ij
#                 io:f32[32,512,32] = transpose[permutation=(1, 2, 0)] in
#                 ip:f32[32,512,32] = sub ih io
#                 iq:f32[32,32,512] = broadcast_in_dim[
#                   broadcast_dimensions=(1, 2)
#                   shape=(32, 32, 512)
#                 ] im
#                 ir:f32[32,512,32] = transpose[permutation=(1, 2, 0)] iq
#                 is:f32[32,512,32] = mul ip ir
#                 it:f32[32,512,32] = broadcast_in_dim[
#                   broadcast_dimensions=(2,)
#                   shape=(32, 512, 32)
#                 ] fv
#                 iu:f32[32,512,32] = mul it is
#                 iv:f32[32,512,32] = broadcast_in_dim[
#                   broadcast_dimensions=(2,)
#                   shape=(32, 512, 32)
#                 ] fw
#                 iw:f32[32,512,32] = add iu iv
#                 ix:f32[32,512,128] = dot_general[
#                   dimension_numbers=(([2], [0]), ([], []))
#                   preferred_element_type=float32
#                 ] iw fx
#                 iy:f32[32,512,128] = broadcast_in_dim[
#                   broadcast_dimensions=(2,)
#                   shape=(32, 512, 128)
#                 ] fy
#                 iz:f32[32,512,128] = add ix iy
#                 ja:f32[32,512,128] = integer_pow[y=3] iz
#                 jb:f32[32,512,128] = mul 0.044714998453855515 ja
#                 jc:f32[32,512,128] = add iz jb
#                 jd:f32[32,512,128] = mul 0.7978845834732056 jc
#                 je:f32[32,512,128] = tanh jd
#                 jf:f32[32,512,128] = add 1.0 je
#                 jg:f32[32,512,128] = mul 0.5 jf
#                 jh:f32[32,512,128] = mul iz jg
#                 ji:f32[32,512,32] = dot_general[
#                   dimension_numbers=(([2], [0]), ([], []))
#                   preferred_element_type=float32
#                 ] jh fz
#                 jj:f32[32,512,32] = broadcast_in_dim[
#                   broadcast_dimensions=(2,)
#                   shape=(32, 512, 32)
#                 ] ga
#                 jk:f32[32,512,32] = add ji jj
#                 jl:f32[32,512,32] = add ih jk
#               in (jl,) }
#             policy=None
#             prevent_cse=False
#           ] ez fa fb fc fd fe ff fg fh fi fj fk fl fm
#         in (fn,) }
#       length=2
#       linear=(False, False, False, False, False, False, False, False, False, False, False, False, False, False)
#       num_carry=1
#       num_consts=0
#       reverse=False
#       unroll=1
#     ] cx bd be bf bg bh bi bj bk bl bm bn bo da
#     jm:f32[32,512] = reduce_sum[axes=(2,)] ey
#     jn:f32[32,512] = div jm 32.0
#     jo:f32[32,512] = pjit[name=_var jaxpr=_var] ey 0
#     jp:f32[32,512] = add jo 9.999999747378752e-06
#     jq:f32[32,512] = rsqrt jp
#     jr:f32[32,32,512] = broadcast_in_dim[
#       broadcast_dimensions=(1, 2)
#       shape=(32, 32, 512)
#     ] jn
#     js:f32[32,512,32] = transpose[permutation=(1, 2, 0)] jr
#     jt:f32[32,512,32] = sub ey js
#     ju:f32[32,32,512] = broadcast_in_dim[
#       broadcast_dimensions=(1, 2)
#       shape=(32, 32, 512)
#     ] jq
#     jv:f32[32,512,32] = transpose[permutation=(1, 2, 0)] ju
#     jw:f32[32,512,32] = mul jt jv
#     jx:f32[32,512,32] = broadcast_in_dim[
#       broadcast_dimensions=(2,)
#       shape=(32, 512, 32)
#     ] bp
#     jy:f32[32,512,32] = mul jx jw
#     jz:f32[32,512,32] = broadcast_in_dim[
#       broadcast_dimensions=(2,)
#       shape=(32, 512, 32)
#     ] bq
#     ka:f32[32,512,32] = add jy jz
#     kb:f32[32,512,50257] = dot_general[
#       dimension_numbers=(([2], [1]), ([], []))
#       preferred_element_type=float32
#     ] ka br
#     kc:i32[32,512] = pjit[
#       name=_roll_static
#       jaxpr={ lambda ; kd:i32[32,512]. let
#           ke:i32[32,511] = slice[
#             limit_indices=(32, 512)
#             start_indices=(0, 1)
#             strides=(1, 1)
#           ] kd
#           kf:i32[32,1] = slice[
#             limit_indices=(32, 1)
#             start_indices=(0, 0)
#             strides=(1, 1)
#           ] kd
#           kg:i32[32,512] = concatenate[dimension=1] ke kf
#         in (kg,) }
#     ] bt
#     kh:f32[32,512,50257] = pjit[
#       name=_one_hot
#       jaxpr={ lambda ; ki:i32[32,512]. let
#           kj:i32[32,512,1] = broadcast_in_dim[
#             broadcast_dimensions=(0, 1)
#             shape=(32, 512, 1)
#           ] ki
#           kk:i32[1,1,50257] = iota[dimension=2 dtype=int32 shape=(1, 1, 50257)]
#           kl:bool[32,512,50257] = eq kj kk
#           km:f32[32,512,50257] = convert_element_type[
#             new_dtype=float32
#             weak_type=False
#           ] kl
#         in (km,) }
#     ] kc
#     kn:f32[32,512] = reduce_max[axes=(2,)] kb
#     ko:bool[32,512] = is_finite kn
#     kp:f32[32,512] = broadcast_in_dim[broadcast_dimensions=() shape=(32, 512)] 0.0
#     kq:f32[32,512] = select_n ko kp kn
#     kr:f32[32,512] = stop_gradient kq
#     ks:f32[32,512,1] = broadcast_in_dim[
#       broadcast_dimensions=(0, 1)
#       shape=(32, 512, 1)
#     ] kr
#     kt:f32[32,512,50257] = sub kb ks
#     ku:f32[32,512,50257] = exp kt
#     kv:f32[32,512] = reduce_sum[axes=(2,)] ku
#     _:f32[32,512] = sign kv
#     kw:f32[32,512] = abs kv
#     kx:f32[32,512] = log kw
#     ky:f32[32,512] = add kx kr
#     kz:f32[50257,32,512] = broadcast_in_dim[
#       broadcast_dimensions=(1, 2)
#       shape=(50257, 32, 512)
#     ] ky
#     la:f32[32,512,50257] = transpose[permutation=(1, 2, 0)] kz
#     lb:f32[32,512,50257] = sub la kb
#     lc:f32[32,512] = dot_general[
#       dimension_numbers=(([2], [2]), ([0, 1], [0, 1]))
#       preferred_element_type=float32
#     ] kh lb
#     ld:f32[] = reduce_sum[axes=(0, 1)] bu
#     le:f32[32,512] = pjit[
#       name=_where
#       jaxpr={ lambda ; lf:f32[32,512] lg:f32[32,512] lh:f32[]. let
#           li:bool[32,512] = ne lf 0.0
#           lj:f32[32,512] = broadcast_in_dim[
#             broadcast_dimensions=()
#             shape=(32, 512)
#           ] lh
#           lk:f32[32,512] = select_n li lj lg
#         in (lk,) }
#     ] bu lc 0.0
#     ll:f32[] = reduce_sum[axes=(0, 1)] le
#     lm:f32[] = div ll ld
#   in (lm,) }