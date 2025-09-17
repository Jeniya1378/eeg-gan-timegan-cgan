import numpy as np
from glob import glob

condition = "no_exo"  # or "with_exo"
data_files = sorted(glob(f"./preprocessed/posture*_{condition}.npz"))  # adjust path as needed
X_list = []
y_list = []  # posture labels
for fpath in data_files:
    npz = np.load(fpath)
    X = npz['X']          # shape (N, 768, 14) for this posture
    posture_label = int(npz['posture'])  # e.g., 1, 2, ..., 9
    # Transpose X to shape (N, 14, 768) for convenience (channels first)
    X = X.transpose(0, 2, 1).astype(np.float32)
    # Create label array
    y = np.full((X.shape[0],), posture_label, dtype=np.int64)
    X_list.append(X)
    y_list.append(y)
# Combine all postures for this condition
X_all = np.concatenate(X_list, axis=0)   # shape (total_N, 14, 768)
y_all = np.concatenate(y_list, axis=0)   # shape (total_N,)
print("Combined data shape:", X_all.shape, "Labels shape:", y_all.shape)
# (Optional) Shuffle the combined dataset
perm = np.random.permutation(X_all.shape[0])
X_all = X_all[perm]
y_all = y_all[perm]
