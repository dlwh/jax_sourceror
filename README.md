# Jax Sourceror

Jax Sourceror is a Python library that allows you to recreate JAX source code from a jitted jax function (specifically its `jaxpr`)
and a set of inputs. This is useful for minimizing bugs, debugging, teaching, and understanding how JAX works under the hood.

The code this generates is definitely not going to be clean, idiomatic, or sometimes even correct, but it should be a good starting point for understanding what's going on.

I created it mostly as a learning exercise and to minimize bugs in framework-heavy code (i.e. removing layers of equinox or flax abstraction to get to the JAX code).

This is more of a "submit a PR" or "fork it" repo than a "this doesn't work for me" repo, but I'm happy to help out if you're stuck.

## Example

Jax Sourceror can turn this:

```python
import jax
import jax.numpy as jnp

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

grad_fn(params, x)
```

into this:

```python
def grad_fn(*args, **kwargs):

    def grad_fn(a, b, c):
        d = jax.numpy.zeros((4, 128, 32), jax.numpy.bfloat16)
        e = jax.numpy.ones((64,), jax.numpy.float32)
        f = jax.lax.broadcast_in_dim(e, shape=(4, 64, 2, 64), broadcast_dimensions=(3,))

        def fn_1(carry, x):
            # (I would like to make this part nicer)
            (g, h, i) = (carry, *x)
            j = jax.lax.dot_general(g, h, (((2,), (2,)), ((), ())), None, jax.numpy.bfloat16)

            def fn_2(k, l):

                def fn_3(carry, x):
                    (m,) = (*carry, x)
                    n = jax.lax.dynamic_slice(l, (0, m, 0, 0), slice_sizes=(4, 64, 2, 16))
                    o = n.astype(jax.numpy.float32)
                    p = jax.numpy.sum(o, axis=(3,))
                    q = p.astype(jax.numpy.bfloat16)
                    r = jax.lax.broadcast_in_dim(q, shape=(4, 64, 2, 64), broadcast_dimensions=(0, 1, 2))
                    s = r.astype(jax.numpy.float32)
                    t = s + k
                    u = jax.lax.dot_general(n, t, (((1,), (3,)), ((0, 2), (0, 2))), None, jax.numpy.float32)
                    v = jax.lax.transpose(u, permutation=(0, 3, 1, 2))
                    w = v.astype(jax.numpy.bfloat16)
                    return ((), w)
                (final_carry, ys) = jax.lax.scan(fn_3, (), jax.numpy.array([0, 1], dtype=jax.numpy.int32), length=2, unroll=1, reverse=False)
                x = ys
                return x
            y = fn_2(f, j)
            z = jax.numpy.reshape(jax.numpy.transpose(y, (1, 0, 2, 3, 4)), (4, 128, 2, 16))
            ba = jax.lax.dot_general(z, i, (((3, 2), (1, 0)), ((), ())), None, jax.numpy.bfloat16)
            return (ba, g)
        (final_carry, ys) = jax.lax.scan(fn_1, c, (a, b), length=2, unroll=1, reverse=False)
        bb = final_carry
        bc = ys
        bd = bb.astype(jax.numpy.float32)
        be = jax.numpy.sum(bd, axis=(0, 1, 2))
        bf = be / 16384.0
        bg = bf.astype(jax.numpy.bfloat16)
        bh = jax.lax.broadcast_in_dim(6.103515625e-05, shape=(4, 128, 32), broadcast_dimensions=())
        bi = bh.astype(jax.numpy.bfloat16)

        def fn_4(carry, x):
            (bj, bk, bl, bm) = (carry, *x)

            def fn_5(bn, bo, bp, bq):
                br = jax.lax.dot_general(bn, bo, (((2,), (2,)), ((), ())), None, jax.numpy.bfloat16)
                bs = jax.numpy.ones((64,), jax.numpy.float32)
                bt = jax.lax.broadcast_in_dim(bs, shape=(4, 64, 2, 64), broadcast_dimensions=(3,))

                def fn_6(carry, x):
                    (bu,) = (*carry, x)
                    bv = bu < 0
                    bw = bu + 128
                    bx = jax.lax.select_n(bv, bu, bw)
                    by = jax.lax.dynamic_slice(br, (0, bx, 0, 0), slice_sizes=(4, 64, 2, 16))
                    bz = by.astype(jax.numpy.float32)
                    ca = jax.numpy.sum(bz, axis=(3,))
                    cb = ca.astype(jax.numpy.bfloat16)
                    cc = jax.lax.broadcast_in_dim(cb, shape=(4, 64, 2, 64), broadcast_dimensions=(0, 1, 2))
                    cd = cc.astype(jax.numpy.float32)
                    ce = cd + bt
                    cf = jax.lax.dot_general(by, ce, (((1,), (3,)), ((0, 2), (0, 2))), None, jax.numpy.float32)
                    cg = jax.lax.transpose(cf, permutation=(0, 3, 1, 2))
                    ch = cg.astype(jax.numpy.bfloat16)
                    return ((), (ch, bx, ce, by))
                (final_carry, ys) = jax.lax.scan(fn_6, (), jax.numpy.array([0, 1], dtype=jax.numpy.int32), length=2, unroll=1, reverse=False)
                (ci, cj, ck, cl) = ys
                cm = jax.numpy.reshape(jax.numpy.transpose(ci, (1, 0, 2, 3, 4)), (4, 128, 2, 16))
                cn = jax.lax.dot_general(bq, cm, (((0, 1), (0, 1)), ((), ())), None, jax.numpy.bfloat16)
                co = jax.lax.transpose(cn, permutation=(1, 2, 0))
                cp = jax.lax.dot_general(bq, bp, (((2,), (2,)), ((), ())), None, jax.numpy.bfloat16)
                cq = jax.numpy.reshape(cp, (4, 2, 64, 2, 16))
                cr = jax.lax.transpose(cq, permutation=(1, 0, 2, 3, 4))
                cs = jax.numpy.zeros((4, 128, 2, 16), jax.numpy.bfloat16)

                def fn_7(carry, x):
                    (ct, cu, cv, cw, cx) = (carry, *x)
                    cy = cu.astype(jax.numpy.float32)
                    cz = jax.lax.transpose(cy, permutation=(0, 2, 3, 1))
                    da = jax.lax.dot_general(cz, cx, (((2,), (3,)), ((0, 1), (0, 2))), None, jax.numpy.float32)
                    db = jax.lax.transpose(da, permutation=(0, 2, 1, 3))
                    dc = db.astype(jax.numpy.bfloat16)
                    dd = jax.numpy.sum(dc, axis=(3,))
                    de = dd.astype(jax.numpy.float32)
                    df = jax.lax.broadcast_in_dim(de, shape=(4, 64, 2, 16), broadcast_dimensions=(0, 1, 2))
                    dg = df.astype(jax.numpy.bfloat16)
                    dh = jax.lax.dot_general(cz, cw, (((3,), (1,)), ((0, 1), (0, 2))), None, jax.numpy.float32)
                    di = jax.lax.transpose(dh, permutation=(0, 3, 1, 2))
                    dj = di.astype(jax.numpy.bfloat16)
                    dk = dg + dj
                    dl = jax.numpy.zeros((4, 128, 2, 16), jax.numpy.bfloat16)
                    dm = jax.lax.dynamic_update_slice(dl, dk, (0, cv, 0, 0))
                    dn = ct + dm
                    return (dn, ())
                (final_carry, ys) = jax.lax.scan(fn_7, cs, (cr, cj, ck, cl), length=2, unroll=1, reverse=True)
                do = final_carry
                dp = jax.lax.dot_general(do, bn, (((0, 1), (0, 1)), ((), ())), None, jax.numpy.bfloat16)
                dq = jax.lax.dot_general(do, bo, (((2, 3), (0, 1)), ((), ())), None, jax.numpy.bfloat16)
                return (dq, dp, co)
            ckpt_fn_5 = jax.checkpoint(fn_5)
            (dr, ds, dt) = ckpt_fn_5(bk, bl, bm, bj)
            return (dr, (ds, dt))
        (final_carry, ys) = jax.lax.scan(fn_4, bi, (bc, a, b), length=2, unroll=1, reverse=True)
        du = final_carry
        (dv, dw) = ys
        return (bg, dv, dw)
    return grad_fn(*jax.tree_leaves((args, kwargs)))
```

Is this pretty code? No. Is it even readable? If you try hard enough.
Is it correct? I think so. (It definitely passes my unit test!)


## Usage

```python
from jax_sourceror import sourcerize

source_code = sourcerize(grad_fn)(*args, **kwargs)

print(source_code)
```