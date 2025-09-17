from scipy.signal import coherence

coherence_diff = {}
ch1, ch2 = 0, 13  # indices for AF3 and AF4 for example (based on EPOC_CHS ordering in pre-processing script)
for posture in range(1, num_classes+1):
    real_idx = np.where(y_all == posture)[0]
    if len(real_idx) == 0:
        continue
    X_real = X_all[real_idx]
    X_fake = generate_synthetic_data(posture, num_samples=X_real.shape[0]).reshape(-1, 14, 768)
    # Compute mean coherence for the channel pair
    coh_real_list = []
    coh_fake_list = []
    for x in X_real:
        f, Cxy = coherence(x[ch1], x[ch2], fs=fs, nperseg=256)
        coh_real_list.append(Cxy)
    for x in X_fake:
        f, Cxy_f = coherence(x[ch1], x[ch2], fs=fs, nperseg=256)
        coh_fake_list.append(Cxy_f)
    mean_coh_real = np.mean(coh_real_list, axis=0)
    mean_coh_fake = np.mean(coh_fake_list, axis=0)
    # Difference metric, e.g., mean absolute difference
    coh_diff_val = np.mean(np.abs(mean_coh_real - mean_coh_fake))
    coherence_diff[posture] = coh_diff_val
    print(f"Posture {posture}: Mean |coherence_diff| between AF3-AF4 = {coh_diff_val:.4f}")
