import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

# Perform t-SNE with 3 components
tsne = TSNE(n_components=3, perplexity=30, n_iter=1000)
tsne_result = tsne.fit_transform(images_flat)

# Plot t-SNE results in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2], c=labels, cmap='viridis', s=25)
plt.title("t-SNE 3D")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
fig.colorbar(scatter)
plt.savefig("tsne_result_3d.png")
plt.show()
