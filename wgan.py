import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import modèle as mods
import time

from pathlib import Path

# Define WGAN Generator
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, nz=100):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            # Fully connected layer to transform input noise vector to a high-dimensional feature map
            nn.Linear(nz, 512 * 4 * 4),  # Transform to 512 feature maps of 4x4
            nn.BatchNorm1d(512 * 4 * 4),  # Use BatchNorm1d for 1D output from Linear
            nn.ReLU(True),

            # Reshape to 4D tensor (batch_size, 512, 4, 4)
            nn.Unflatten(1, (512, 4, 4)),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 (output size)
            nn.Tanh()
        )

    def forward(self, x):
      x = x.view(x.size(0), -1)
      return self.model(x)



# Define WGAN Discriminator (Critic)
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.model(x).view(-1)

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters and DataLoader
dataroot = Path("/content/drive/MyDrive/dataset_gan")
batch_size = 48
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 150
lr = 0.001
ngpu = 1
critic_iterations = 5  # Number of updates for Discriminator per Generator update
weight_clip = 0.01     # Clipping parameter for weight constraints in WGAN

# DataLoader
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
netD = Discriminator(ngpu).to(device)
netG = Generator(ngpu).to(device)

# Initialize weights for models
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netD.apply(weights_init)
netG.apply(weights_init)

# Optimizers
optimizerD = optim.Adam(netD.parameters(), lr=0.000005, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.9))


# Training Loop
print("Starting Training Loop...")
t0 = time.time()
G_losses, D_losses, img_list = [], [], []
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# parameter lambda of GP
lambda_gp = 5

def compute_gradient_penalty(critic, real_samples, fake_samples):

    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device).expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)


    d_interpolates = critic(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)

    # calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # calculate gradients penalties
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network (Critic): maximize D(x) - D(G(z))
        ###########################
        for _ in range(critic_iterations):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            output_real = netD(real_cpu)
            errD_real = -torch.mean(output_real)

            # Generate fake image batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            output_fake = netD(fake.detach())
            errD_fake = torch.mean(output_fake)

            #print(f"Real Samples Shape: {real_cpu.shape}, Fake Samples Shape: {fake.shape}")
            gradient_penalty = compute_gradient_penalty(netD, real_cpu, fake)
            d_loss = errD_fake + errD_real + gradient_penalty
            d_loss.backward()

            # Update Discriminator
            optimizerD.step()

            # Clip weights for WGAN
            # for p in netD.parameters():
                # p.data.clamp_(-weight_clip, weight_clip)

        D_losses.append(errD_real.item() + errD_fake.item())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
        netG.zero_grad()
        fake = netG(noise)
        output = netD(fake)
        errG = -torch.mean(output)
        errG.backward()
        optimizerG.step()

        G_losses.append(errG.item())

        # Output training stats
        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {D_losses[-1]:.4f}\tLoss_G: {errG.item():.4f}')

        # Save generator output for visualization
        if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

torch.save(netG.state_dict(), "generator.pth")

# Display fake images and loss
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.subplot(1,2,2)
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()

plt.subplots_adjust(wspace=0.5)

plt.show()
print("Training Time: ", time.time() - t0)


