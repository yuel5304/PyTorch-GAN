import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, input_size, latent_dim, channels):
        super(Generator, self).__init__()

        #self.init_size = img_size // 4
        #self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        #self.conv_blocks = nn.Sequential(
        #    nn.BatchNorm2d(128),
        #    nn.Upsample(scale_factor=2),
        #    nn.Conv2d(128, 128, 3, stride=1, padding=1),
        #    nn.BatchNorm2d(128, 0.8),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    nn.Upsample(scale_factor=2),
        #    nn.Conv2d(128, 64, 3, stride=1, padding=1),
        #    nn.BatchNorm2d(64, 0.8),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    nn.Conv2d(64, channels, 3, stride=1, padding=1),
        #    nn.Tanh(),
        #)
        n_nodes = 128 * 7 * 7
        self.dense = nn.Sequential(nn.Linear(input_size, n_nodes), nn.LeakyReLU(0.2, inplace=True))

        def generator_block(in_filters, out_filters, bn=False):
            block = [nn.ConvTranspose2d(in_filters, out_filters, (4, 4), (2, 2)), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        self.conv_blocks = nn.Sequential(
            *generator_block(7, 128),
            *generator_block(128, 128)
        )
        self.output = nn.Sequential(nn.Conv2d(128, channels, (7, 7)), nn.Tanh())

    def forward(self, z):
        out = self.dense(z)
        #print(z.shape[0])
        #print(out.shape)
        out = out.reshape((z.shape[0], 7, 7, 128))
        out = self.conv_blocks(out)
        img = self.output(out),
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, latent_dim, channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=False):
            block = [nn.Conv2d(in_filters, out_filters, (4, 4), (2, 2)), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(28*28, 64, bn=False),
            *discriminator_block(64, 64),
            #*discriminator_block(32, 64),
            #*discriminator_block(64, 128),
        )
        # The height and width of downsampled image
        #ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img.view(-1, 28*28))
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity