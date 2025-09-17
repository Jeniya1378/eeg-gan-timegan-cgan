from scipy.signal import welch

psd_diff = {}  # store PSD difference metrics per posture
acf_diff = {}  # store ACF difference metrics per posture

fs = 128.0  # sampling rate (or use actual fs from data if provided)
for posture in range(1, num_classes+1):
    real_idx = np.where(y_all == posture)[0]
    if len(real_idx) == 0:
        continue
    X_real = X_all[real_idx]  # (N_real, 14, 768)
    X_fake = generate_synthetic_data(posture, num_samples=X_real.shape[0])
    X_fake = X_fake.reshape(X_fake.shape[0], 14, 768)
    # Compute average PSD for each channel
    psd_diff_post = []
    acf_diff_post = []
    for ch in range(14):
        # Real PSD (average across epochs)
        psds = []
        for x in X_real:
            f, Pxx = welch(x[ch], fs=fs, nperseg=256)
            psds.append(Pxx)
        mean_psd_real = np.mean(psds, axis=0)
        # Fake PSD
        psds_f = []
        for x in X_fake:
            f, Pxx_f = welch(x[ch], fs=fs, nperseg=256)
            psds_f.append(Pxx_f)
        mean_psd_fake = np.mean(psds_f, axis=0)
        # Compute difference (e.g., L2 norm of PSD difference)
        psd_diff_val = np.linalg.norm(mean_psd_real - mean_psd_fake) / len(mean_psd_real)
        psd_diff_post.append(psd_diff_val)
        # Autocorrelation (normalized) for one epoch mean
        # Compute mean autocorrelation across epochs
        def autocorr(x):
            n = len(x)
            x = x - np.mean(x)
            result = np.correlate(x, x, mode='full')
            result = result[result.size//2:]  # second half
            result = result / result[0]       # normalize
            return result
        acfs = [autocorr(x[ch]) for x in X_real]
        acf_real = np.mean(acfs, axis=0)
        acfs_f = [autocorr(x[ch]) for x in X_fake]
        acf_fake = np.mean(acfs_f, axis=0)
        # e.g., take first 50 lags for comparison
        L = 50
        acf_diff_val = np.linalg.norm(acf_real[:L] - acf_fake[:L]) / L
        acf_diff_post.append(acf_diff_val)
    # store average difference across channels for this posture
    psd_diff[posture] = np.mean(psd_diff_post)
    acf_diff[posture] = np.mean(acf_diff_post)
    print(f"Posture {posture}: Avg PSD curve L2 diff = {psd_diff[posture]:.4f}, Avg ACF diff = {acf_diff[posture]:.4f}")
