import numpy as np
import os
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

folder_path = './samples/cifar_cond_vanilla'

# 从文件夹中找到最新的文件
all_samples = []
all_labels = []

for filename in os.listdir(folder_path):
    if filename.endswith(".npz"):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)

        samples = data["samples"]
        all_samples.append(samples)

        if "label" in data:
            labels = data["label"]
            all_labels.append(labels)

samples = np.concatenate(all_samples, axis=0)
labels = np.concatenate(all_labels, axis=0)
labels = torch.tensor(labels)
labels_indices = torch.argmax(labels, dim=1).numpy()

num_samples_per_class = 100
selected_samples = []
selected_labels = []

for i in range(10):  # Assuming there are 10 classes
    class_indices = np.where(labels_indices == i)[0]
    selected_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)
    selected_samples.append(samples[selected_indices])
    selected_labels.append(labels_indices[selected_indices])

samples = np.concatenate(selected_samples, axis=0)
labels_indices = np.concatenate(selected_labels, axis=0)

# Flatten samples
samples_flat = torch.tensor(samples).reshape(samples.shape[0], -1).numpy()

# Normalize samples using L2 norm
samples_norm = samples_flat / np.linalg.norm(samples_flat, axis=1, keepdims=True)

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(samples_norm)

# Perform t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
tsne_result = tsne.fit_transform(samples_norm)

# Plot PCA results
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels_indices, cmap='viridis', s=25)
plt.title("PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("pca_result.png")
plt.show()

# Plot t-SNE results
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels_indices, cmap='viridis', s=25)
plt.title("t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.savefig("tsne_result.png")
plt.show()
