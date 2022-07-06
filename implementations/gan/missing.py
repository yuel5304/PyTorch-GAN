import argparse
import os
import numpy as np
import math

from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


import time

#import sys
#sys.path.append("./")
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

img_size = 28*28
# 1
input_size = img_size*2 #(hidden, hinting)
# 2
d_output_size = opt.num_label+1
# 3
d_hidden_size = 256
# Generator hyperparams
# 4
z_size = img_size
g_input_size = z_size * 2 #(x, m)
# 5
g_output_size = 28*28
# 6
g_hidden_size = d_hidden_size


cuda = True if torch.cuda.is_available() else False


# Loss function
criterion = torch.nn.CrossEntropyLoss(reduction='sum')#torch.nn.BCELoss()
#criterion = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(g_input_size, g_hidden_size, g_output_size)
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

print(device)

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

def sample_hint(batch_size, dimension, p):
    A = torch.rand(size = [batch_size, dimension])
    B = A > p
    C = 1.*B
    return C
# ----------
#  Training
# ----------
batch = 0
step = 1
stamp = time.time()

p = 0.5
p_hint = 0.9
softmax = torch.nn.Softmax()
for image_8, _ in test_dataloader_list[7]:
    image_8 = image_8.view(-1, img_size).to(device)
    m = torch.ones(image_8.shape).to(device)
    h = sample_hint(image_8.shape[0], image_8.shape[1], 1-p_hint).to(device)*m
    image_8 = torch.cat((image_8, h), dim=1)
    break
for epoch in range(opt.n_epochs):
    for idx, (imgs, labels) in enumerate(train_dataloader):
        batch += 1
        imgs, labels = imgs.to(device), labels.to(device)
        length = imgs.shape[0]
        imgs = imgs.view(-1, img_size)

        # create disjoint data paritition 
        

        # create missing values mask
        missing = imgs[np.arange(length*p, length),: ]
        missing_labels = Variable(torch.ones(missing.shape[0]).long() * opt.num_label, requires_grad=False).to(device)
        m = torch.bernoulli(torch.ones(missing.shape)*(1-0.5)).to(device)
        missing_h = sample_hint(missing.shape[0], missing.shape[1], 1-p_hint).to(device)*m

        # configure imputed data
        z = Variable(Tensor(np.random.normal(0, 1, (int(missing.shape[0]), z_size)))).to(device)
        noise = missing*m + z*(1-m)
        x_hat = generator(torch.cat((noise, m), dim=1))
        #print(x_hat.shape)
        fake = criterion(discriminator(torch.cat((x_hat, missing_h), dim=1)), missing_labels)#torch.log(softmax(discriminator(x_hat))[:, opt.num_label]).mean()


        normal = imgs[np.arange(length*p),:]
        normal_label = labels[np.arange(length*p)]
        m = torch.ones(normal.shape).to(device)
        normal_h = sample_hint(normal.shape[0], normal.shape[1], 1-p_hint).to(device)*m
        #print(discriminator(torch.cat((normal, normal_h), dim=1)).shape)
        #print(normal_label.shape)
        real = criterion(discriminator(torch.cat((normal, normal_h), dim=1)), normal_label)
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        #optimizer_G.zero_grad()
        #gen_imgs = generator(z)

        # Measure discriminator's ability to classify real from generated samples

        d_loss = (real + fake)/length


        d_loss.backward()
        optimizer_D.step()

        if batch % 1 == 0:
            # -----------------
            #  Train Generator via gradient w.r.t old discriminator.
            # -----------------
            #for param in generator.parameters():
            #    param.grad.data = -param.grad.data.clone()
            
            optimizer_G.zero_grad()
            m = torch.bernoulli(torch.ones(missing.shape)*(1-0.5)).to(device)
            missing_h = sample_hint(missing.shape[0], missing.shape[1], 1-p_hint).to(device)*m
            z = Variable(Tensor(np.random.normal(0, 1, (int(missing.shape[0]), z_size)))).to(device)
            noise = missing*m + z*(1-m)
            x_hat = generator(torch.cat((noise, m), dim=1))
            fake = -criterion(discriminator(torch.cat((x_hat, missing_h), dim=1)), missing_labels)/len(x_hat)
            fake.backward()

            optimizer_G.step()

        # -----------------
        #  Test
        # -----------------
        if batch % len(train_dataloader) == 0:

            z = Variable(Tensor(np.random.normal(0, 1, (1, z_size)))).to(device)
            m = torch.zeros(z.shape).to(device)
            h = sample_hint(z.shape[0], z.shape[1], 1-p_hint).to(device)*m
            gen_imgs = generator(torch.cat((z, m), dim=1))

            print(
                "[epoch: %d] [D loss: %f] [D value on g(z): %f] [D value on x: %f] [G loss: %f]"
                % (epoch, d_loss.item(), torch.mean(softmax(discriminator(torch.cat((gen_imgs, h), dim=1)))[:, 0:opt.num_label].sum(axis=1)).item(),
                   torch.mean(softmax(discriminator(image_8.to(device)))[:, 0:opt.num_label].sum(axis=1)).item(),
                   fake.item()/missing.shape[0])
            )
            
            save_image(gen_imgs.reshape((28, 28)).data, "images/%d.png" % (batch), normalize=True)


            correct = 0
            total = 0
            for h in [0.5]:#,0.3,0.4,0.5]:
                for label in range(opt.num_label):
                    for real_imgs, labels in test_dataloader_list[label]:
                        real_imgs, labels = real_imgs.to(device), labels.to(device)
                        real_imgs = real_imgs.view(-1, img_size)

                        m = torch.bernoulli(torch.ones(real_imgs.shape)*(1-h)).to(device)
                        z = Variable(Tensor(np.random.normal(0, 1, (int(real_imgs.shape[0]), z_size)))).to(device)
                        noise = real_imgs*m + z*(1-m)
                        x_hat = generator(torch.cat((noise, m), dim=1))
                        hint = sample_hint(x_hat.shape[0], x_hat.shape[1], 1-p_hint).to(device)*m
                        # test on incomplete samples
                        prob = softmax(discriminator(torch.cat((x_hat, hint), dim=1)))[:, 0:opt.num_label]
                        #print(prob.shape)
                        for i in range(prob.shape[0]):
                            prob[i, :] /= prob[i, :].sum()
                        _, predicted = torch.max(prob, 1)
                        #print(predicted.shape)
                        #print(labels.shape)
                        correct += (predicted == labels).sum().item()
                        total += len(real_imgs)
                        # break

                print("p: {}, classification: {}".format(h, correct / total))

