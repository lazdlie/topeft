import gzip
import pickle
import numpy as np
import sys

pkl_path = sys.argv[1]
out_path = sys.argv[2]

with gzip.open(pkl_path, "rb") as f:
    data = pickle.load(f)

h = data["njets"]

raw = h.view(flow=True)

arrays = {}
for key, arr in raw.items():
    arrays[str(key)] = np.asarray(arr)

np.savez_compressed(out_path, **arrays)

print("Input PKL:", pkl_path)
print("Saved:", out_path)
print("Keys in PKL:", data.keys())
print("Number of category keys:", len(arrays))

first_key = next(iter(arrays))
print("First key:", first_key)
print("First array shape:", arrays[first_key].shape)