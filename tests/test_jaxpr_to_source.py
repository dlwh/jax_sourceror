from functools import partial

from jax_automin.interpreter import automin_function
import jax
import jax.numpy as jnp


def test_jaxpr_to_source_simple():
    import jax.numpy as jnp

    def f(x):
        return x + 1

    source = automin_function(f, jnp.array([1, 2, 3]))

    assert source == """def f(a):
    b = a + 1
    return b"""


def test_jaxpr_to_source_matmul():
    import jax.numpy as jnp

    def f(x, y):
        return jnp.matmul(x, y)

    source = automin_function(f, jnp.array([[1, 2], [3, 4]]), jnp.array([[1, 2], [3, 4]]))

    assert source == """def f(a, b):
    c = jax.numpy.matmul(a, b)
    return c"""

    check_roundtrip(f, jnp.array([[1, 2], [3, 4]]), jnp.array([[1, 2], [3, 4]]))


def check_roundtrip(f, *args, **kwargs):
    source = automin_function(f, *args, **kwargs)
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


def _parse_sandboxed(source, fn_name):
    g = {'jax': jax}
    l = {}
    exec(source, g, l)
    return l[fn_name]


def test_slice_squeeze():
    def f(x):
        return x[0:2, 0:1, 3]

    f2 = check_roundtrip(f, jnp.arange(4 * 5 * 6).reshape(4, 5, 6))
    check_roundtrip(f2, jnp.arange(4 * 5 * 6).reshape(4, 5, 6))


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
        # weights = with_sharding_constraint(weights, P('data', None, None, None))
        # without "bias", no boom
        bias = jnp.ones(block_len).broadcast_in_dim((batch, block_len, num_heads, block_len), (1,))
        weights = weights + bias
        return jnp.einsum('bqhk,bkhd->bqhd', weights, query_block).astype(query_block.dtype)

    x = jnp.arange(batch * block_len * num_heads * head_size).reshape(batch, block_len, num_heads, head_size).astype(jnp.float32)

    f2 = check_roundtrip(block, x)

def test_scan():
    def scanfn(x, y):
        return x + y, x * y

    x = jnp.arange(10)
    y = jnp.arange(10)

    def f(x, y):
        return jax.lax.scan(scanfn, x, y)

    f2 = check_roundtrip(f, x, y)

    assert jnp.all(f(x, y)[0] == f2(x, y)[0])
    assert jnp.all(f(x, y)[1] == f2(x, y)[1])


def test_map():
    def f(x):
        return x + 1

    x = jnp.arange(10)

    def g(x):
        return jax.lax.map(f, x)

    g2 = check_roundtrip(g, x)

    assert jnp.all(g(x) == g2(x))


def test_map_pytree():
    def f(x):
        return x[0] + 1, x[1] + 1

    x = jnp.arange(10)

    def g(x, y):
        return jax.lax.map(f, (x, y))

    g2 = check_roundtrip(g, x, x)

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
            bias = jnp.ones(block_len).broadcast_in_dim((batch, block_len, num_heads, block_len), (1,))
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

    f2 = check_roundtrip(grad_fn, params, x)





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