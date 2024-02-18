import torch
from matplotlib import pyplot as plt

from nn_model import RealNVP

def generate_samples(model, num_samples=16, device='cpu'):
    with torch.no_grad():
        z = torch.randn(num_samples, 28*28).to(device)
        samples, _ = model(z, reverse=True)
        samples = samples.cpu().numpy()
    return samples

model = RealNVP(num_block=8, in_dim=28 * 28, hidden_dim=256)
model.load_state_dict(torch.load('RealNVP.pth'))
model.eval()

# Generate samples
num_samples = 16
generated_samples = generate_samples(model, num_samples)

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
