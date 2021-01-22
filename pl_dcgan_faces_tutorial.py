# -*- coding: utf-8 -*-

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

hparams = {
    'dataroot': "celeba",
    # Number of workers for dataloader
    'workers': 0,
    # Batch size during training
    'batch_size': 128,
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    'image_size': 64,
    # Number of channels in the training images. For color images this is 3
    'nc': 3,
    # Size of z latent vector (i.e. size of generator input)
    'nz': 100,
    # Size of feature maps in generator
    'ngf': 64,
    # Size of feature maps in discriminator
    'ndf': 64,
    # Number of training epochs
    # instead of num_epochs set max_epochs to desired num
    # pl will run num epochs if criteria of earlyStopping are not met
    'max_epochs': 5,
    # Learning rate for optimizers
    'lr': 0.0002,
    # Beta1 hyperparam for Adam optimizers
    'beta1': 0.5,
    # Number of GPUs available. Use 0 for CPU mode.
    'ngpu': 1
}


class DCGanDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.batch_size = self.hparams['batch_size']

        self.dataroot = self.hparams['dataroot']
        self.train_transforms = transforms.Compose([
                                                  transforms.Resize(self.hparams['image_size']),
                                                  transforms.CenterCrop(self.hparams['image_size']),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ])

    def setup(self, stage=None):
        # stage assigns train/val datasets for use in dataloaders
        # this example don't have train/val splits of data, so its not used
        self.dataset = dset.ImageFolder(root=self.dataroot,
                                        transform=self.train_transforms)

    def train_dataloader(self):
        # Create the dataloader
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                           shuffle=True, num_workers=self.hparams['workers'])

    def example_show(self):
        # Plot some training images
        dataloader = self.train_dataloader()
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True), (1, 2, 0)))
        plt.show()


class Generator(nn.Module):
    def __init__(self,hparams):
        super(Generator, self).__init__()
        self.hparams = hparams
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.hparams.nz, self.hparams.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hparams.ngf * 8),
            nn.ReLU(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.hparams.ngf * 8, self.hparams.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.ngf * 4),
            nn.ReLU(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.hparams.ngf * 4, self.hparams.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.ngf * 2),
            nn.ReLU(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.hparams.ngf * 2, self.hparams.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.ngf),
            nn.ReLU(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.hparams.ngf, self.hparams.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, hparams):
        super(Discriminator, self).__init__()
        self.hparams = hparams
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.hparams.nc, self.hparams.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.hparams.ndf, self.hparams.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.ndf * 2),
            nn.LeakyReLU(0.2),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.hparams.ndf * 2, self.hparams.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.ndf * 4),
            nn.LeakyReLU(0.2),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.hparams.ndf * 4, self.hparams.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.ndf * 8),
            nn.LeakyReLU(0.2),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.hparams.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class DCGan_model(pl.LightningModule):
    def __init__(self, hparams):
        super(DCGan_model, self).__init__()
        # call this to save hyperparams to the checkpoint
        # you can save only params that you need:
        # self.save_hyperparameters('num_epochs')
        self.save_hyperparameters(hparams)
        # after this can access things like: self.hparams.batch_size
        # so the step below is not necessary
        # self.hparams = hparams
        self.generator = Generator(self.hparams).apply(self.weights_init)
        self.discriminator = Discriminator(self.hparams).apply(self.weights_init)
        self.fixed_noise = torch.randn(64, self.hparams.nz, 1, 1)
        self.real_label = 1.
        self.fake_label = 0.

    def forward(self, z):
        return self.generator(z)

    def bin_cross_entropy_loss(self, logit, label):
        return nn.BCELoss()(logit, label)

    def discriminator_step(self, x):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Forward pass real batch through D
        b_size = x.size(0)
        label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
        # for multiple gpus
        label = label.type_as(x)
        output = self.discriminator(x).view(-1)
        # Calculate loss on all-real batch
        loss_d_true = self.bin_cross_entropy_loss(output, label)

        ## Train with all-fake batch

        # label.fill_(self.fake_label) <--- need to create new object because of optimizer step
        label = torch.full((b_size,), self.fake_label, dtype=torch.float, device=self.device)
        label = label.type_as(x)

        # Generate fake image batch with G
        noise = torch.randn(b_size, self.hparams.nz, 1, 1)
        noise = noise.type_as(x)
        fake = self.generator(noise)
        # Classify all fake batch with D
        output = self.discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        loss_d_fake = self.bin_cross_entropy_loss(output, label)
        loss = loss_d_fake + loss_d_true
        return loss

    def generator_step(self, x):
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # Generate fake image batch with G
        b_size = x.size(0)
        noise = torch.randn(b_size, self.hparams.nz, 1, 1, device=self.device)
        noise = noise.type_as(x)
        fake = self.generator(noise)

        # fake labels are real for generator cost
        label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
        label = label.type_as(x)

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        loss = self.bin_cross_entropy_loss(output, label)
        return loss

    def training_step(self, training_batch, batch_idx, optimizer_idx):
        # Format batch

        # Generate batch of latent vectors
        if optimizer_idx == 0:
            loss_d = self.discriminator_step(training_batch[0])
            self.log('loss_d', loss_d, on_epoch=True, prog_bar=True)
            return loss_d

        if optimizer_idx == 1:
            loss_g = self.generator_step(training_batch[0])
            self.log('loss_g', loss_g, on_epoch=True, prog_bar=True)
            return loss_g

    def on_epoch_end(self):
        z = self.fixed_noise.type_as(self.generator.main[0].weight)
        # show sampled images
        sample_imgs = self(z).detach()
        plt.imshow(np.transpose(vutils.make_grid(sample_imgs, padding=2, normalize=True).cpu().numpy(), (1, 2, 0)))  # !!!
        plt.show()
        # log sampled images if you specified logger
        # grid = vutils.make_grid(sample_imgs, padding=2, normalize=True)
        # self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

    def configure_optimizers(self):
        lr = self.hparams.lr
        optimizerD = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(self.hparams.beta1, 0.999))
        optimizerG = optim.Adam(self.generator.parameters(), lr=lr, betas=(self.hparams.beta1, 0.999))
        return optimizerD, optimizerG

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

# TO DO: add dataparallel
if __name__ == '__main__':
    model = DCGan_model(hparams)
    dm = DCGanDataModule(hparams)
    dm.setup()
    dm.example_show()
    trainer = pl.Trainer(gpus=hparams['ngpu'],
                         max_epochs=hparams['max_epochs'],
                         )
    trainer.fit(model, dm)

