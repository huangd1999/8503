# import torch
# import torch.backends.cudnn as cudnn
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.models as models
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import argparse
# import os

# import numpy as np
# import pandas as pd
# # from tsne import bh_sne
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import seaborn as sns
# import matplotlib.pyplot as plt

# parser = argparse.ArgumentParser(description='PyTorch t-SNE for STL10')
# parser.add_argument('--save-dir', type=str, default='./results', help='path to save the t-sne image')
# parser.add_argument('--batch-size', type=int, default=128, help='batch size (default: 128)')
# parser.add_argument('--seed', type=int, default=1, help='random seed value (default: 1)')

# args = parser.parse_args()

# if not os.path.exists(args.save_dir):
#     os.makedirs(args.save_dir)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # set seed
# torch.manual_seed(args.seed)
# if device == 'cuda':
#     torch.cuda.manual_seed(args.seed)

# # set dataset
# transform = transforms.Compose([    
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])

# # dataset = torchvision.datasets.STL10(root='~/Downloads', split='train', download=True, transform=transform)
# # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# # dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
# #                                         download=True, transform=transform)
# # dataloader = torch.utils.data.DataLoader(dataset, batch_size=512,
# #                                           shuffle=False, num_workers=2)


# folder_path = './samples/cifar_cond'
# # 从文件夹中找到最新的文件
# all_samples = []
# all_labels = []

# for filename in os.listdir(folder_path):
#     if filename.endswith(".npz"):
#         file_path = os.path.join(folder_path, filename)
#         data = np.load(file_path)

#         samples = data["samples"]
#         all_samples.append(samples)

#         if "label" in data:
#             labels = data["label"]
#             all_labels.append(labels)


# samples = np.concatenate(all_samples, axis=0)
# labels = np.concatenate(all_labels, axis=0)


# # Convert NumPy arrays to PyTorch tensors
# samples_tensor = torch.from_numpy(samples).float()
# labels_tensor = torch.from_numpy(labels).long()

# # Create a TensorDataset
# dataset = TensorDataset(samples_tensor, labels_tensor)

# # Create a DataLoader
# batch_size = 500
# # transform = transforms.Compose(
# #     [transforms.ToTensor(),
# #     #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# #      ])

# # dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
# #                                         download=False, transform=transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# # set model
# net = models.resnet18(pretrained=True)
# net.fc = torch.nn.Linear(512, 10)
# net.load_state_dict(torch.load('./cifar10_resnet18.pth'))
# net = net.to(device)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

# def gen_features():
#     net.eval()
#     targets_list = []
#     outputs_list = []

#     with torch.no_grad():
#         for idx, (inputs, targets) in enumerate(dataloader):
#             inputs = inputs.to(device)
#             inputs = inputs.permute(0, 3, 1, 2)
#             targets = targets.to(device)
#             targets_np = targets.data.cpu().numpy()

#             outputs = net(inputs)
#             outputs_np = outputs.data.cpu().numpy()
            
#             targets_list.append(targets_np[:, np.newaxis])
#             outputs_list.append(outputs_np)
            
#             if ((idx+1) % 10 == 0) or (idx+1 == len(dataloader)):
#                 print(idx+1, '/', len(dataloader))
#             # if idx==90:
#             #     break

#     targets = np.concatenate(targets_list, axis=0)
#     outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

#     return targets, outputs

# def tsne_plot(save_dir, targets, outputs):
#     print('generating t-SNE plot...')
#     targets = torch.squeeze(torch.from_numpy(targets))
#     targets = torch.argmax(targets, dim=1).numpy()
#     # tsne_output = bh_sne(outputs)
#     tsne = TSNE(random_state=0)
#     tsne_output = tsne.fit_transform(outputs)

#     df = pd.DataFrame(tsne_output, columns=['x', 'y'])
#     df['targets'] = targets

#     plt.rcParams['figure.figsize'] = 10, 10
#     sns.scatterplot(
#         x='x', y='y',
#         hue='targets',
#         palette=sns.color_palette("hls", 10),
#         data=df,
#         marker='o',
#         legend="full",
#         alpha=0.5
#     )

#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel('')
#     plt.ylabel('')

#     plt.savefig(os.path.join(save_dir,'tsne_ddpm.pdf'), bbox_inches='tight')
#     print('done!')

# def pca_plot(save_dir, targets, outputs):
#     print('generating PCA plot...')
#     targets = torch.squeeze(torch.from_numpy(targets))
#     targets = torch.argmax(targets, dim=1).numpy()
#     pca = PCA(n_components=2)
#     pca_output = pca.fit_transform(outputs)

#     df = pd.DataFrame(pca_output, columns=['x', 'y'])
#     df['targets'] = targets

#     plt.rcParams['figure.figsize'] = 10, 10
#     sns.scatterplot(
#         x='x', y='y',
#         hue='targets',
#         palette=sns.color_palette("hls", 10),
#         data=df,
#         marker='o',
#         legend="full",
#         alpha=0.5
#     )

#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel('')
#     plt.ylabel('')

#     plt.savefig(os.path.join(save_dir,'pca_ddpm.pdf'), bbox_inches='tight')
#     print('done!')

# targets, outputs = gen_features()
# tsne_plot(args.save_dir, targets, outputs)
# pca_plot(args.save_dir, targets, outputs)




import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.decomposition import PCA
import argparse
import os

import numpy as np
import pandas as pd
# from tsne import bh_sne
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser(description='PyTorch t-SNE for STL10')
parser.add_argument('--save-dir', type=str, default='./results', help='path to save the t-sne image')
parser.add_argument('--batch-size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('--seed', type=int, default=1, help='random seed value (default: 1)')

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set seed
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)

# Create a DataLoader
batch_size = 512
transform = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# set model
net = models.resnet18(pretrained=True)
net.fc = torch.nn.Linear(512, 10)
net.load_state_dict(torch.load('./cifar10_resnet18.pth'))
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

def gen_features():
    net.eval()
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_np = targets.data.cpu().numpy()

            outputs = net(inputs)
            outputs_np = outputs.data.cpu().numpy()
            
            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)
            
            if ((idx+1) % 10 == 0) or (idx+1 == len(dataloader)):
                print(idx+1, '/', len(dataloader))
            # if idx==20:
            #     break

    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

    return targets, outputs

def tsne_plot(save_dir, targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join(save_dir,'tsne_clean.pdf'), bbox_inches='tight')
    print('done!')

def pca_plot(save_dir, targets, outputs):
    print('generating PCA plot...')
    targets = torch.squeeze(torch.from_numpy(targets))
    # targets = torch.argmax(targets, dim=1).numpy()
    pca = PCA(n_components=2)
    pca_output = pca.fit_transform(outputs)

    df = pd.DataFrame(pca_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join(save_dir,'pca_clean.pdf'), bbox_inches='tight')
    print('done!')


# pca 3d

def pca_plot_3d(save_dir, targets, outputs):
    print('generating PCA plot...')
    targets = torch.squeeze(torch.from_numpy(targets))
    # targets = torch.argmax(targets, dim=1).numpy()
    pca = PCA(n_components=3)
    pca_output = pca.fit_transform(outputs)

    df = pd.DataFrame(pca_output, columns=['x', 'y', 'z'])
    df['targets'] = targets

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['x'], df['y'], df['z'], c=df['targets'], cmap=plt.cm.Spectral)
    plt.savefig(os.path.join(save_dir,'pca_clean_3d.pdf'), bbox_inches='tight')
    print('done!')

targets, outputs = gen_features()
# tsne_plot(args.save_dir, targets, outputs)
# pca_plot(args.save_dir, targets, outputs)
pca_plot_3d(args.save_dir, targets, outputs)