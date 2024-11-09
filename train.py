import torch
import torch.nn as nn
import torch.optim as optim
import modèle as mods


num_epochs = 100         
learning_rate = 0.0002   
batch_size = 64          
noise_dim = 100          

generator = mods.Generator(noise_dim=noise_dim, out_channels=3)
discriminator = mods.Discriminator(in_channels=3)

# Loss fct
criterion = nn.BCELoss()

optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))


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
