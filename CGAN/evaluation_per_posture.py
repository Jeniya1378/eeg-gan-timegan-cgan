from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

target_ch = "AF4"
# Find index of target channel (assuming ch_names list exists from the data)
ch_index = [i for i, ch in enumerate(npz_data['ch_names']) if ch == target_ch]
if ch_index:
    target_idx = ch_index[0]
else:
    target_idx = -1  # if not found (should not happen for AF4 given the EPOC channels)
print("Target channel index:", target_idx)

predictive_results = {}
for posture in range(1, num_classes+1):
    # Real data for this posture
    real_idx = np.where(y_all == posture)[0]
    if len(real_idx) == 0:
        continue
    X_real = X_all[real_idx]  # shape (N_real, 14, 768)
    # Synthetic data for this posture (generate same number of samples as real for fairness)
    X_fake = generate_synthetic_data(posture, num_samples=X_real.shape[0])
    X_fake = X_fake.reshape(X_fake.shape[0], 14, 768)
    # Prepare regression dataset: flatten time and sample dimensions
    # Each "example" will be one time-point from one epoch, with 13 features (channels except target) predicting target channel.
    N_real, _, T = X_real.shape
    N_fake = X_fake.shape[0]
    # Create feature matrix and target vector for real and fake
    # For real:
    real_features = X_real[:, :, :].transpose(0, 2, 1).reshape(-1, 14)  # shape (N_real*T, 14)
    real_target = real_features[:, target_idx]
    real_features = np.delete(real_features, target_idx, axis=1)  # remove target channel from features (shape: (N_real*T, 13))
    # For synthetic:
    fake_features = X_fake.transpose(0, 2, 1).reshape(-1, 14)
    fake_target = fake_features[:, target_idx]
    fake_features = np.delete(fake_features, target_idx, axis=1)
    # Train on synthetic (fake) and test on real
    if fake_features.shape[0] == 0:
        continue
    reg = Ridge(alpha=1.0)  # using Ridge regression for stability
    # Sample a subset if data is huge (but here it might be okay because N_real and T are moderate)
    reg.fit(fake_features, fake_target)
    # Evaluate on real
    pred_real = reg.predict(real_features)
    rmse = np.sqrt(mean_squared_error(real_target, pred_real))
    r2 = r2_score(real_target, pred_real)
    # Train on real and test on synthetic (for reference)
    reg2 = Ridge(alpha=1.0)
    reg2.fit(real_features, real_target)
    pred_fake = reg2.predict(fake_features)
    rmse_trts = np.sqrt(mean_squared_error(fake_target, pred_fake))
    r2_trts = r2_score(fake_target, pred_fake)
    predictive_results[posture] = {"TSTR_RMSE": rmse, "TSTR_R2": r2, 
                                   "TRTS_RMSE": rmse_trts, "TRTS_R2": r2_trts}
    print(f"Posture {posture}: TSTR RMSE={rmse:.4f}, R²={r2:.3f} | TRTS RMSE={rmse_trts:.4f}, R²={r2_trts:.3f}")
