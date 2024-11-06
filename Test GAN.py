import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the input and output sizes
input_size = 784
hidden_size = 256
output_size = 1
# Root directory for dataset
dataroot = ""
dataset = torchvision.datasets.MNIST(root=dataroot, download=True, transform=transforms.Compose([])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# Create the discriminator and generator
discriminator = Discriminator(input_size, hidden_size, output_size).to(device)
generator = Generator(input_size, hidden_size, output_size).to(device)

# Set the loss function and optimizers
loss_fn = nn.BCEWithLogitsLoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

# Set the number of epochs and the noise size
num_epochs = 200
noise_size = 100

# Training loop
for epoch in range(num_epochs):
  for i, (real_images, _) in enumerate(dataloader):
    # Get the batch size
    batch_size = real_images.size(0)

# Generate fake images
noise = torch.randn(batch_size, noise_size).to(device)
fake_images = generator(noise)

# Train the discriminator on real and fake images
d_real = discriminator(real_images)
d_fake = discriminator(fake_images)

# Calculate the loss
real_loss = loss_fn(d_real, torch.ones_like(d_real))
fake_loss = loss_fn(d_fake, torch.zeros_like(d_fake))
d_loss = real_loss + fake_loss

# Backpropagate and optimize
d_optimizer.zero_grad()
d_loss.backward()
d_optimizer.step()

# Train the generator
d_fake = discriminator(fake_images)
g_loss = loss_fn(d_fake, torch.ones_like(d_fake))

# Backpropagate and optimize
g_optimizer.zero_grad()
g_loss.backward()
g_optimizer.step()

# Print the loss every 50 batches
if (i+1) % 50 == 0:
    print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}' 
            .format(epoch+1, num_epochs, i+1, len(dataloader), d_loss.item(), g_loss.item()))