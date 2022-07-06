import sys
sys.path.append("./")
import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
import torchvision
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from data.utils import *

""" 
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
opt = parser.parse_args()
print(opt)
"""
img_shape = (1, 28, 28)#(opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(784, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    #logistic regression
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = torch.nn.Linear(784, 10)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, img, label):
        y_pred = self.predict(img.view(-1))
        return self.criterion(y_pred, label)

    def predict(self, x):
        y_pred = self.linear(x.view(-1))
        return y_pred


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader

train_dataset, test_dataset = data_processor("mnist")

missing_dataloader = torch.utils.data.DataLoader(
    missing(train_dataset, 1, 0.1),
    batch_size=128,
    shuffle=True,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=0.1)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.1)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

batches_done = 0
print("start training")
for epoch in range(20):

    for i, (imgs, labels), (m_imgs, m_labels, miss) in enumerate(zip(train_dataloader, missing_dataloader)):

        # Configure input

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (m_img.shape))))
        # combined with imputation to missing value
        imputed_img = Variable(m_imgs.type(Tensor)) * miss + z * (1-miss)

        # Generate a batch of images
        fake_imgs = generator(imputed_img).detach()
        #fake_imgs = generator(z).detach()
        # imputation
        fake_imgs = (miss == 0)*fake_imgs + (miss == 1)

        # Adversarial loss: get the worst case for current generator.
        loss_D = -torch.mean(discriminator(imgs, labels)) + torch.mean(discriminator(fake_imgs, m_labels))

        loss_D.backward()
        optimizer_D.step()  # update discriminator

        # Clip weights of discriminator
        #for p in discriminator.parameters():
        #    p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % 1 == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step() # update generator

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, 20, batches_done % len(train_dataloader), len(train_dataloader), loss_D.item(), loss_G.item())
            )

        #if batches_done % opt.sample_interval == 0:
            #save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        #batches_done += 1
