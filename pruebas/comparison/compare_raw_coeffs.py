import numpy as np
import sys

file_a = sys.argv[1]
file_b = sys.argv[2]

a = np.load(file_a)
b = np.load(file_b)

keys_a = set(a.files)
keys_b = set(b.files)

print("Same keys:", keys_a == keys_b)
print("N keys A:", len(keys_a))
print("N keys B:", len(keys_b))

if keys_a != keys_b:
    print("Only in A:", keys_a - keys_b)
    print("Only in B:", keys_b - keys_a)

all_close = True
max_abs = 0.0
max_rel = 0.0

for k in sorted(keys_a & keys_b):
    arr_a = a[k]
    arr_b = b[k]

    if arr_a.shape != arr_b.shape:
        print(f"Shape mismatch: {k}")
        print(arr_a.shape, arr_b.shape)
        all_close = False
        continue

    diff = arr_a - arr_b
    abs_diff = np.max(np.abs(diff)) if diff.size else 0.0

    denom = np.maximum(np.abs(arr_a), np.abs(arr_b))
    mask = denom > 0
    rel_diff = np.max(np.abs(diff[mask]) / denom[mask]) if np.any(mask) else 0.0

    if not np.allclose(arr_a, arr_b, rtol=1e-10, atol=1e-12):
        print(f"Mismatch: {k}")
        print("  max abs diff:", abs_diff)
        print("  max rel diff:", rel_diff)
        all_close = False

    max_abs = max(max_abs, abs_diff)
    max_rel = max(max_rel, rel_diff)

print("All close:", all_close)
print("Global max abs diff:", max_abs)
print("Global max rel diff:", max_rel)