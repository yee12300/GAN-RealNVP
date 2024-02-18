import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from nn_model import RealNVP, RealNVPLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0002
z_dim = 100
img_dim = 784
batch_size = 64
num_epochs = 10

# Load MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

model = RealNVP(num_block=8, in_dim=28 * 28, hidden_dim=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    for _, (real_images, _) in enumerate(dataloader):
        real_images = real_images.view(-1, img_dim).to(device)
        batch_size = real_images.shape[0]

        optimizer.zero_grad()
        z, log_det_sum = model(real_images)
        loss = RealNVPLoss(z, log_det_sum)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}')

torch.save(model.state_dict(), 'RealNVP.pth')

