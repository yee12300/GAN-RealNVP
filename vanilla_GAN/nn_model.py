import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.model = nn.Sequential(
            Maxout(img_dim, 512, 3),
            nn.Dropout(0.3),
            Maxout(512, 256, 3),
            nn.Dropout(0.3),
            Maxout(256, 1, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Maxout(nn.Module):
    def __init__(self, input_dim, output_dim, pool_size):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pool_size = pool_size
        self.affines = nn.Linear(input_dim, output_dim * pool_size)

    def forward(self, x):
        x = self.affines(x)
        x, _ = x.view(x.size(0), self.output_dim, self.pool_size).max(-1)
        return x