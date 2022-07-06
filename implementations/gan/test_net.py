import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.linear1 = nn.Sequential(
                                    nn.Linear(input_size, hidden_dim),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(hidden_dim, hidden_dim),
        )

        self.conv = nn.Conv1d(1, 1, 16)

        self.linear2 = nn.Sequential(
                                    nn.Linear(241, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, output_size),)
                                    

    def forward(self, x):

        #x = x.view(-1, self.input_size)
        f1 = self.linear1(x)
        f2 = self.conv(f1.unsqueeze(1)).squeeze(1)
        return f1#self.linear2(f2)


class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()

        self.linear1 = nn.Sequential(
                                    nn.Linear(input_size, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, output_size),
                                    nn.Tanh())
        self.conv = nn.Conv1d(1, 1, 16)

        self.linear2 = nn.Sequential(
                                    nn.Linear(769, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, output_size),
                                    nn.Tanh())
        

    def forward(self, x):
        f1 = self.linear1(x)
        #print(f1.shape)
        #f2 = self.conv(f1.unsqueeze(1))
        f2 = self.conv(f1.unsqueeze(1)).squeeze(1)

        #print(f2.shape)
        return self.linear2(f2)



# Calculate losses
def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
    # smooth, real labels = 0.9
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)  # real labels = 1

    # numerically stable loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)  # fake labels = 0
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def criterion(outputs, targets, n_class):
    # outputs: (num_sample, (n_class+1)*num_dim*)
    # targets: (num_sample, num_dim)
    # n_class: int

    predict = nn.Softmax(dim=1)
    num_sample, num_dim = targets.shape
    loss = torch.zeros(1, requires_grad=False)
    for i in range(num_sample):
        prob = predict(outputs[i].reshape((num_dim, n_class+1)))
        tmp_loss = torch.zeros(1, requires_grad=False)
        for j in range(num_dim):
            #index = j*(n_class+1) + targets[i,j]
            tmp_loss += torch.log(prob[j, int(targets[i,j])])
        loss += tmp_loss

    return loss/num_sample









