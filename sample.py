import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# Choose a random subset of the data

# Load CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                          shuffle=False, num_workers=2)

# Extract samples and labels
images, labels = next(iter(trainloader))

# Flatten and normalize images
images_flat = images.view(images.size(0), -1).numpy()

subset_size = 200
indices = np.random.choice(images_flat.shape[0], subset_size, replace=False)
images_flat_subset = images_flat[indices]
labels_subset = labels[indices]

# Perform t-SNE with adjusted parameters
tsne = TSNE(n_components=2, perplexity=50, n_iter=2000)
tsne_result = tsne.fit_transform(images_flat_subset)

# Plot t-SNE results
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels_subset, cmap='viridis', s=25)
plt.title("t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar()
plt.savefig("tsne_result_subset.png")
plt.show()
