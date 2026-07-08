# July 1, 2026
# Speed test for solving 2x2s
# used in solving the fast version of newton iteration

import jax
from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr
import time

key1 = jr.PRNGKey(1)

count = int(1e7)

testmats = jr.uniform(key1, shape=(2,2,count))
testbs = jr.uniform(key1, shape=(2,count))

def solve1(A, by):
    [a,b],[c,d] = A
    e,f = by
    x2 = (a*f - e*c)/(a*d-b*c)
    x1 = (e-b*x2)/a
    return jnp.array([x1, x2])
def solve2(A, by):
    [a,b],[c,d] = A
    e,f = by
    x2 = (a*f - e*c)/(a*d-b*c)
    x1 = (e*d-b*f)/(a*d-b*c)
    return jnp.array([x1, x2])
def solve3(A,b):
    return jnp.linalg.solve(A, b)

jsolve1 = jit(solve1)
jsolve2 = jit(solve2)
jsolve3 = jit(solve3)
jsolve1(testmats[:,:,0], testbs[:,0])
jsolve2(testmats[:,:,0], testbs[:,0])
jsolve3(testmats[:,:,0], testbs[:,0])



start1 = time.time()

out1= vmap(jsolve1,in_axes=(-1, -1))(testmats, testbs)
jax.block_until_ready(out1)
time1 = time.time()-start1

start2 = time.time()

out2=vmap(jsolve2,in_axes=(-1, -1))(testmats, testbs)
jax.block_until_ready(out2)
time2 = time.time()-start2

start3 = time.time()

out3 = vmap(jsolve3,in_axes=(-1, -1))(testmats, testbs)
jax.block_until_ready(out3)
time3 = time.time()-start3

print(time1/count, time2/count, time3/count)
