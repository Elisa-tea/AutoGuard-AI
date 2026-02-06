import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super(Generator, self).__init__()
        self.nz = nz  # latent vector size

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=(4, 4), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
           

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
           

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=(4, 6), stride=(2, 2), padding=(1, 2), bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
          

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=(1, 16), stride=(1, 16), padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # pixel-wise probabilities
            nn.ConvTranspose2d(ngf, nc, kernel_size=(1, 50), stride=(1, 1), padding=0, bias=False),
            nn.Sigmoid()
           
        )

    """
    def forward(self, input):
        prob_output = self.main(input)             
        binary_output = torch.bernoulli(prob_output)  
        return binary_output
    """
    def forward(self, input, hard=True, stochastic=True):
        probs = self.main(input)

        if not hard:
            return probs  # soft mode

        if stochastic:
            # stochastic binarization
            with torch.no_grad():
                hard_sample = (probs > torch.rand_like(probs)).float()
        else:
            # deterministic threshold
            with torch.no_grad():
                hard_sample = (probs > 0.5).float()

        # straight-through: forward = hard, backward = identity on probs
        return hard_sample + (probs - probs.detach())

