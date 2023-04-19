import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                          shuffle=False, num_workers=2)

# Extract samples and labels
images, labels = next(iter(trainloader))

# Flatten and normalize images
images_flat = images.view(images.size(0), -1).numpy()

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(images_flat)

# Perform t-SNE
tsne = TSNE(n_components=2, perplexity=50, n_iter=2000)
tsne_result = tsne.fit_transform(images_flat)

# Plot PCA results
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', s=25)
plt.title("PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.savefig("pca_result_true.png")
plt.show()

# Plot t-SNE results
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', s=25)
plt.title("t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar()
plt.savefig("tsne_result_true.png")
plt.show()
