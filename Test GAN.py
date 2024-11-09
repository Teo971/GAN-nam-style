#%matplotlib inline
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
import modèle as mods
from IPython.display import HTML
from pathlib import Path

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Root directory for dataset
dataroot = Path("C:/Users/pzx27/OneDrive/document/dataset_gan/train")
# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

if __name__ == "__main__":
    # Your existing code goes here
    # For example:
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
    
    noise_dim = 100          

    generator = mods.Generator(noise_dim=noise_dim, out_channels=3)
    discriminator = mods.Discriminator(in_channels=3)

    # Loss fct
    criterion = nn.BCELoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


    for epoch in range(num_epochs):
        for real_images, _ in dataloader:  
        
            real_labels = torch.ones(batch_size, 1, device=device)  
            fake_labels = torch.zeros(batch_size, 1, device=device)  
            
            real_images = real_images.to(device)
            
        
            outputs = discriminator(real_images).view(-1, 1)
            d_loss_real = criterion(outputs, real_labels)
            
        
            noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)  
            fake_images = generator(noise)
            
        
            outputs = discriminator(fake_images.detach()).view(-1, 1)  
            d_loss_fake = criterion(outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            
            # BP
            optimizer_d.zero_grad()
            d_loss.backward()  # calculer le gradient
            optimizer_d.step()  # update disc parametres

            
            fake_labels = torch.ones(batch_size, 1, device=device)
            
            noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            fake_images = generator(noise)
            
            outputs = discriminator(fake_images).view(-1, 1)
            g_loss = criterion(outputs, fake_labels)
            
            optimizer_g.zero_grad()
            g_loss.backward()  
            optimizer_g.step()  

        print(f"Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")
