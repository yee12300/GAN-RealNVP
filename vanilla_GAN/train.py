import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from nn_model import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0002
z_dim = 100
img_dim = 784
batch_size = 64
num_epochs = 50

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

generator = Generator(z_dim, img_dim).to(device)
discriminator = Discriminator(img_dim).to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for _, (real_images, _) in enumerate(dataloader):
        real_images = real_images.view(-1, img_dim).to(device)
        batch_size = real_images.shape[0]

        noise = torch.randn(batch_size, z_dim).to(device)
        fake_images = generator(noise)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        optimizer_D.zero_grad()
        real_outputs = discriminator(real_images)
        fake_outputs = discriminator(fake_images.detach())
        loss_real = criterion(real_outputs, real_labels)
        loss_fake = criterion(fake_outputs, fake_labels)
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        fake_outputs = discriminator(fake_images)
        loss_G = criterion(fake_outputs, real_labels)
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss D: {loss_D.item():.4f} Loss G: {loss_G.item():.4f}")

torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')