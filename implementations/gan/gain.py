import argparse
import os
import numpy as np
import math

from torchvision.utils import save_image

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torch.autograd import Variable



import time

import sys
sys.path.append("./")
from mnist_gan import  real_loss, fake_loss
#from gcan import Generator, Discriminator
from test_net import Generator, Discriminator, criterion
from utils import *
import copy

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--num_label", type=int, default=10, help="number of cpu threads to use during batch generation")
parser.add_argument("--prob", type=float, default=0, help="probability of missing value")

#parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)


# 1
input_size = 28*28
# 2
d_output_size =  (opt.num_label+1)*input_size
print(opt.num_label)
# 3
d_hidden_size = 256#32
# Generator hyperparams
# 4
z_size = 64
# 5
g_output_size = 28*28
# 6
g_hidden_size = d_hidden_size


cuda = True if torch.cuda.is_available() else False


# Loss function
criterion = criterion

# Initialize generator and discriminator
generator = Generator(z_size, g_hidden_size, g_output_size)
discriminator = Discriminator(input_size, d_hidden_size, d_output_size)
print(generator)
print(discriminator)

if cuda:
    generator.cuda()
    discriminator.cuda()
    criterion.cuda()
    device = 'cuda:0'
else:
    device = 'cpu'



# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)

#train_transform = transform
_, test_partition = partition('mnist', num_label=opt.num_label, download=True, transform=transforms.ToTensor())
#train_dataloader_list = [DataLoader(train_partition[i], batch_size=opt.batch_size, shuffle=True,) for i in range(opt.num_label)]
test_dataloader_list = [DataLoader(test_partition[i], batch_size=opt.batch_size, shuffle=True) for i in range(opt.num_label)]
train_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.ToTensor()),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
batch = 0
step = 1
stamp = time.time()

predict = torch.nn.Softmax(dim=1)
for epoch in range(opt.n_epochs):
    for idx, (imgs, labels) in enumerate(train_dataloader):
        batch += 1
        imgs = imgs.view(-1, input_size)
        # create missing values mask
        m = torch.bernoulli(torch.ones(imgs.shape)*(1-opt.prob))
        #print(m.sum().item()/m.shape[0]/m.shape[1])
        # Configure input
        # Sample noise as generator input
        
        z = Variable(Tensor(np.random.normal(0, 1, (int(imgs.shape[0]), z_size))))
        gen_imgs = generator(z)
        # configure
        x_hat = imgs*m + gen_imgs*(1-m)
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        optimizer_G.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        # configure new label to be d-dimensional
        num_sample, num_dim = imgs.shape
        dim_labels = []
        for i in range(num_sample):
            y = labels[i] * torch.ones(num_dim)
            fake = opt.num_label * torch.ones(num_dim)
            dim_labels.append(y*m[i,:] + fake*(1-m[i,:]))
        dim_labels = torch.stack(dim_labels, dim=0)

        loss = -criterion(discriminator(x_hat), dim_labels, opt.num_label)

        loss.backward()
        optimizer_D.step()
        
        for param in generator.parameters():
            param.grad.data = -param.grad.data.clone()
        optimizer_G.step()


        # -----------------
        #  Test
        # -----------------
        if batch % 10 == 0:

            correct = 0
            total = 0
            for label in range(opt.num_label):
                for real_imgs, targets in test_dataloader_list[label]:
                    #print(real_imgs.shape)
                    real_imgs = real_imgs.view(-1, input_size)
                    num_sample, num_dim = real_imgs.shape
                    outputs = discriminator(real_imgs)
                    for i in range(num_sample):
                        prob = predict(outputs[i].reshape((num_dim, opt.num_label+1)))
                        for j in range(num_dim):
                            prob[j, :] /= 1- prob[j, opt.num_label]
                        map_hat = torch.mean(prob, dim=0)
                        #_, predicted = torch.max(prob[:, 0:opt.num_label], 1)
                        #label = torch.argmax(torch.bincount(predicted, minlength=opt.num_label))
                        _, label = torch.max(map_hat, 1)
                        correct += (targets[i] == label).item()
                    total += num_sample
                    break
                    # break

            print("classification: {}".format(correct / total))
        if batch % 100 == 0:
            print("[batch: %d] [loss: %f]" % (batch, loss.item()))
            z = Variable(Tensor(np.random.normal(0, 1, (1, z_size))))
            gen_imgs = generator(z).reshape((28, 28))
            save_image(gen_imgs.data, "images/%d.png" % (batch), normalize=True)


