import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data.dataset import Dataset


class missing(Dataset):

    def __init__(self, data, n, p):
        self.ins_list = []
        self.label_list = []
        self.miss_list = []
        num_sample = len(data)
        indice = torch.randperm(num_sample)
        for i in range(num_sample):
            image, label = data[indice[i]]
            if i <= n*num_sample:
                miss_image, identity = self.miss(image, p)
                self.miss_list.append(identity)
                self.ins_list.append(miss_image)
            else:
                self.miss_list.append(torch.ones(image.shape))
                self.ins_list.append(image)

            self.label_list.append(label)
            #self.ins_list_np.append(image)
        #self.ins_np = torch.stack(self.ins_list_np, dim=0)
        #self.label_np = torch.tensor(self.label_list).flatten()  # reshape(1,len(self.label_list))

    def miss(self, image, p):
        dim = image.shape
        length = 1
        identity = torch.ones(image.shape)
        for j in dim:
            length *= j
        for feature in range(length):
            toss = torch.rand(1)
            if toss <= p:
                image[0, int(feature / dim[1]), int(feature % dim[2])] = 0 #instead of float('nan') is convenient for later combinations with noise.
                identity[0, int(feature / dim[1]), int(feature % dim[2])] = 0
        return image, identity

    def __getitem__(self, index):
        ins = self.ins_list[index]
        label = self.label_list[index]
        missing_indentity = self.miss_list[index]
        return ins, label, missing_indentity

    def __len__(self):
        return len(self.label_list)


def data_processor(dataset):
    if dataset == 'mnist':
        train_data = torchvision.datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
        )
        test_data = torchvision.datasets.MNIST(
            "../../data/mnist",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
        )

    else:
        raise ValueError("Dataset isn't supported yet.")
    return train_data, test_data

if __name__=="__main__":
    train_data, test_data = data_processor('mnist')
    dataloader = torch.utils.data.DataLoader(
        missing(train_data, 0.01, 0.1),
        batch_size = 128,
        shuffle = True,
    )
    print(test[0][0].shape)