import numpy as np

# Inspect one real file
z_real = np.load("preprocessed/posture1_no_exo.npz", allow_pickle=True)
print("REAL keys:", z_real.files)
for k in z_real.files:
    print(k, z_real[k].shape)

# Inspect synthetic file
z_fake = np.load("timegan_runs/posture1_no_exo/synthetic.npz", allow_pickle=True)
print("FAKE keys:", z_fake.files)
for k in z_fake.files:
    arr = z_fake[k]
    print(k, getattr(arr, "shape", None), getattr(arr, "ndim", None))
