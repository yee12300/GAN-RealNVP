import torch
from matplotlib import pyplot as plt
from nn_model import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0002
z_dim = 100
img_dim = 784
batch_size = 32
num_epochs = 100
img_num = 16

def generate_samples(model, num_samples=16, device='cpu'):
    with torch.no_grad():
        noise = torch.randn(num_samples, z_dim).to(device)
        samples = model(noise)
        samples = samples.cpu().numpy()
    return samples

generator = Generator(z_dim, img_dim).to(device)
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()

# Generate samples
num_samples = 16
generated_samples = generate_samples(generator, num_samples)

# Display generated samples using Matplotlib
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i in range(4):
    for j in range(4):
        index = i * 4 + j
        sample = generated_samples[index].reshape(28, 28)
        axes[i, j].imshow(sample, cmap='gray')
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()