import torch
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np

from random import Random

import torch.utils.data

import sys
sys.path.append("./")
from metrics.pytorch_gan_metrics.utils import get_inception_score_from_directory, get_fid_from_directory


def partition(dataset, num_label, download=True, transform=transforms.ToTensor()):
    if dataset == 'mnist':
        train_partition, test_partition = partition_mnist(path='./data',
                                                     num_label=num_label,
                                                     download=download,
                                                     transform=transform)
        return train_partition, test_partition
    else:
        raise Exception("dataset is not supported.")


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index  # index actually the index within samples.

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class GeneratorDataset(object):
    def __init__(self, Generator, z_dim, device):
        self.Generator = Generator
        self.z_dim = z_dim
        self.num_sample = num_sample
        self.device = device

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        return self.Generator(torch.randn(1, self.z_dim).to(self.device))[0]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        overall_indices = [x for x in range(0, data_len)]
        targets = torch.tensor(self.data.targets).clone().detach()
        labels = torch.unique(targets)
        label_indices = []
        for label in labels:
            indices = torch.squeeze((targets == label).nonzero(as_tuple=False))
            label_indices.append(indices)
        self.partitions = label_indices

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_mnist(path, transform, num_label, download=True):
    """ Partitioning mnist """
    # return: list of dataParitioner


    train_transform = transform

    train_dataset = datasets.MNIST(
        path,
        train=True,
        download=download,
        transform=train_transform,
    )

    test_transform = transform
    test_dataset = datasets.MNIST(
        path,
        train=False,
        download=download,
        transform=test_transform,
    )

    train_partition = DataPartitioner(train_dataset)
    train_partition = [train_partition.use(i) for i in range(num_label)]
    test_partition = DataPartitioner(test_dataset)
    test_partition = [test_partition.use(i) for i in range(num_label)]
    return train_partition, test_partition


def parse_log(log):
    lines = open(log, "r")
    d_loss = []
    g_loss = []
    d_value = []
    z_value = []
    result = []
    acc = []
    for line in lines:
        split = line.replace("]"," ").replace("["," ").replace(","," ").split()
        #print(split)
        if split[0].startswith('epoch'):
            d_loss.append(float(split[4]))
            z_value.append(float(split[9]))
            d_value.append(float(split[14]))
            g_loss.append(float(split[17]))
        if split[0].startswith('classification'):
            acc.append(float(split[1]))

    return d_loss, g_loss, d_value, z_value, acc


def smooth(line, p=1):
    s_line = line[0:p]
    for i in range(p, len(line)-p):
        s_line += [np.mean(np.array(line[i-p:i+p]))]
    s_line += line[len(line)-p:len(line)]
    return s_line


def inception_score(path='./images', device='cpu', batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (channelsxHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """


    #assert 0 <= imgs.min() and imgs.max() <= 1

    # Set up dtype
    dtype = torch.cuda.FloatTensor

    # Set up dataloader
    #dataset = GeneratorDataset(Generator, latent_dim, device)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)

    # Inception Score
    IS, IS_std = get_inception_score_from_directory(path, device=device)
    #FID = get_fid_from_directory(path, './test.npz', device=device)

    return IS, IS_std#, FID


""" 
if __name__ == '__main__':
    print(inception_score(path='./images', device='cpu', batch_size=32, resize=False, splits=1))
"""  
if __name__ == "__main__":
    import sys
    sys.path.append("./")
    import matplotlib.pyplot as plt
    d_loss, g_loss, d_value, z_value, acc = parse_log("gan.log")


    plt.figure(figsize=(15, 5))
    plt.title("training loss on model 9")
    plt.subplot(1, 3, 1)
    plt.title("loss")
    plt.plot(smooth(d_loss))
    plt.plot(smooth(g_loss))
    plt.legend([r'discriminator$', r'generator'])
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 3, 2)
    plt.title(r"discriminator  value, $\sum_iD_i(\hat x)$")
    plt.plot(smooth(d_value))
    plt.plot(smooth(z_value))
    plt.legend([r'$\hat x$', r'$\hat x=g(z)$'])
    plt.xlabel("epoch")
    plt.ylabel("probability")

    plt.subplot(1, 3, 3)
    plt.title("accuracy of classification")
    plt.plot(smooth(acc,1))
    plt.xlabel("epoch")
    plt.ylabel("Top-1 test accuracy")


    plt.savefig("result.png")
    plt.show()

