from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Prepare a combined dataset of real + synthetic for visualization
# Take, say, 100 real and 100 synthetic from each posture (if available) to have a manageable size
vis_samples = []
vis_labels = []
vis_domain = []  # real vs fake indicator
samples_per_class = 100
for posture in range(1, num_classes+1):
    real_idx = np.where(y_all == posture)[0]
    n_real = len(real_idx)
    if n_real == 0:
        continue
    # take up to samples_per_class real
    real_take = real_idx[:min(n_real, samples_per_class)]
    for i in real_take:
        vis_samples.append(X_all[i].reshape(-1))  # flatten
        vis_labels.append(posture)
        vis_domain.append(0)  # 0 for real
    # generate synthetic samples_per_class
    X_fake = generate_synthetic_data(posture, num_samples=samples_per_class)
    for j in range(min(samples_per_class, X_fake.shape[0])):
        vis_samples.append(X_fake[j])
        vis_labels.append(posture)
        vis_domain.append(1)  # 1 for fake

vis_samples = np.array(vis_samples)
vis_labels = np.array(vis_labels)
vis_domain = np.array(vis_domain)

# PCA to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(vis_samples)
print("Explained variance by first 2 PCs:", pca.explained_variance_ratio_)

# t-SNE to 2 components (this can be slow, use a subset if needed)
tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
X_tsne = tsne.fit_transform(vis_samples)

# Now X_pca and X_tsne are 2D embeddings. We can visualize them (for example, using matplotlib):
import matplotlib.pyplot as plt

# PCA scatter
plt.figure(figsize=(8,6))
for posture in range(1, num_classes+1):
    mask_real = (vis_labels == posture) & (vis_domain == 0)
    mask_fake = (vis_labels == posture) & (vis_domain == 1)
    plt.scatter(X_pca[mask_real, 0], X_pca[mask_real, 1], label=f"Real Posture{posture}", marker='o', s=20, alpha=0.7)
    plt.scatter(X_pca[mask_fake, 0], X_pca[mask_fake, 1], label=f"Fake Posture{posture}", marker='x', s=20, alpha=0.7)
plt.title("PCA of Real vs Fake EEG Data")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# t-SNE scatter
plt.figure(figsize=(8,6))
for posture in range(1, num_classes+1):
    mask_real = (vis_labels == posture) & (vis_domain == 0)
    mask_fake = (vis_labels == posture) & (vis_domain == 1)
    plt.scatter(X_tsne[mask_real, 0], X_tsne[mask_real, 1], label=f"Real Posture{posture}", marker='o', s=20, alpha=0.7)
    plt.scatter(X_tsne[mask_fake, 0], X_tsne[mask_fake, 1], label=f"Fake Posture{posture}", marker='x', s=20, alpha=0.7)
plt.title("t-SNE of Real vs Fake EEG Data")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
