import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ndf=16, nc=1):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, (4, 4), (2, 2), (1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(ndf * 4, 1, (2, 4), (1, 1), (0, 0), bias=False),
        )

       
        with torch.no_grad():
            dummy_input = torch.randn(1, nc, 16, 305) 
            dummy_output = self.main(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]

        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.main(x)
        features = features.view(features.size(0), -1)
        output = self.final(features)
        return output, features
