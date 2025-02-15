import argparse
import os
import numpy as np
import math

from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


import time

import sys
sys.path.append("./")
from mnist_gan import  real_loss, fake_loss
#from gcan import Generator, Discriminator
from test_net import Generator, Discriminator
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
#parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)


# 1
input_size = 28*28
# 2
d_output_size = opt.num_label+1
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
criterion = torch.nn.CrossEntropyLoss(reduction='sum')#torch.nn.BCELoss()
#criterion = torch.nn.BCELoss()

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
for image_8, _ in test_dataloader_list[7]:
    break
softmax = torch.nn.Softmax()
for epoch in range(opt.n_epochs):
    for idx, (imgs, labels) in enumerate(train_dataloader):
        batch += 1


            # Configure input
        fake_labels = Variable(torch.ones(imgs.shape[0]).long() * opt.num_label, requires_grad=False)
            # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], z_size))))
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        gen_imgs = generator(z)

        # Measure discriminator's ability to classify real from generated samples
        real = criterion(discriminator(imgs), labels)
        fake = criterion(discriminator(gen_imgs), fake_labels)
        d_loss = (real + fake)/(imgs.shape[0]+gen_imgs.shape[0])
   

        d_loss.backward()
        optimizer_D.step()

        if batch % 1 == 0:
            # -----------------
            #  Train Generator
            # -----------------
            
            for param in generator.parameters():
                param.grad.data = - param.grad.data
            # Generate a batch of images
            #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], z_size))))
            #gen_imgs = generator(z)
            #g_loss = torch.log(softmax(discriminator(gen_imgs))[:, opt.num_label]).mean()
            #g_loss.backward()
            optimizer_G.step()

        # -----------------
        #  Test
        # -----------------
        if batch % len(train_dataloader) == 0:

            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], z_size))))
            gen_imgs = generator(z)
            print(
                "[epoch: %d] [D loss: %f] [D value on g(z): %f] [D value on x: %f] [G loss: %f]"
                % (epoch, d_loss.item(), torch.mean(softmax(discriminator(gen_imgs))[:, 0:opt.num_label].sum(axis=1)).item(),
                   torch.mean(softmax(discriminator(image_8))[:, 0:opt.num_label].sum(axis=1)).item(),
                   fake.item()/gem_imgs.shape[0])
            )
            z = Variable(Tensor(np.random.normal(0, 1, (1, z_size))))
            gen_imgs = generator(z).reshape((28, 28))
            save_image(gen_imgs.data, "images/%d.png" % (batch), normalize=True)


            correct = 0
            total = 0
            for label in range(opt.num_label):
                for real_imgs, labels in test_dataloader_list[label]:
                    prob = softmax(discriminator(real_imgs))[:, 0:opt.num_label]
                    for i in range(prob.shape[0]):
                        prob[i, :] /= prob[i, :].sum()
                    _, predicted = torch.max(prob, 1)
                    correct += (predicted == labels).sum().item()
                    total += len(real_imgs)
                    # break

            print("classification: {}".format(correct / total))

