# enregistrer les modeles
torch.save(netG.state_dict(), "generator.pth")
torch.save(netD.state_dict(), "discriminator.pth")
# torch.save(netG, "generator_full.pth")
# torch.save(netD, "discriminator_full.pth")

# chargement de modele
netG = Generator(ngpu).to(device)
netD = Discriminator(ngpu).to(device)

netG.load_state_dict(torch.load("generator.pth"))
netD.load_state_dict(torch.load("discriminator.pth"))

# netG = torch.load("generator_full.pth")
# netD = torch.load("discriminator_full.pth")

# mode d'evaluation
netG.eval()
netD.eval()


# test
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
with torch.no_grad():  
    fake_images = netG(fixed_noise).detach().cpu()

import torchvision.utils as vutils
import matplotlib.pyplot as plt

grid = vutils.make_grid(fake_images, normalize=True, nrow=8)
plt.imshow(grid.permute(1, 2, 0))  
plt.axis("off")
plt.show()
