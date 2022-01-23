import tvm
import tvm.testing
from tvm import te
import numpy as np

tgt = tvm.target.Target(target="llvm -mcpu=skylake-avx512", host="llvm -mcpu=skylake-avx512")

n = te.var("n")
A = te.placeholder((n,), dtype='int8', name="A")
B = te.placeholder((n,), dtype='int8', name="B")
C = te.compute(A.shape, lambda i: A[i] == B[i], name="C")
s = te.create_schedule(C.op)

dev = tvm.device(tgt.kind.name, 0)
f_eq_cmp = tvm.build(s, [A, B, C], tgt, name="eq_cmp")

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
f_eq_cmp(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() == b.numpy())

def evaluate_addition(n, func, target, optimization, log):
    dev = tvm.device(target.kind.name, 0)
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=100)
    mean_time = evaluator(a, b, c).mean
    print(f'{optimization}: {a.shape} : {mean_time:.9f}')
    #print(tvm.lower(s, [A, B, C], simple_mode=True))

    log.append((optimization, n, mean_time))

log = []
ns = [1000000, 8000000, 30000000, 50000000, 75000000, 100000000]

for n in ns:
    evaluate_addition(n, f_eq_cmp, tgt, "naive", log=log)

timings = []

print(f'{"Operator": >20}{"Number": >20}\t{"Timing": >20}')
for result in log:
    print(
        f"{result[0]: >20}\t{result[1]:>20}\t{result[2]:>20.9f}"
    )
    timings.append(result[2])

print(f'{result[0]}\t{timings}')

log = []
timings = []

s[C].parallel(C.op.axis[0])
f_eq_cmp_parallel = tvm.build(s, [A, B, C], tgt, name="eq_cmp__parallel")
f_eq_cmp_parallel(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() == b.numpy())
for n in ns:
    evaluate_addition(n, f_eq_cmp_parallel, tgt, "parallel", log=log)

print(f'{"Operator": >20}{"Number": >20}\t{"Timing": >20}')
for result in log:
    print(
        f"{result[0]: >20}\t{result[1]:>20}\t{result[2]:>20.9f}"
    )

print(f'{"Operator": >20}{"Number": >20}\t{"Timing": >20}')
for result in log:
    print(
        f"{result[0]: >20}\t{result[1]:>20}\t{result[2]:>20.9f}"
    )
    timings.append(result[2])

print(f'{result[0]}\t{timings}')



