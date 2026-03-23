import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)

class DisentanglingAutoEncoder(nn.Module):
    def __init__(self, input_dim=192, latent_dim=512):
        super(DisentanglingAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.Linear(latent_dim, input_dim)
        )
        self.latent_split = latent_dim // 2

    def forward(self, x, swap_codes=False, x_swap=None):
        z = self.encoder(x)
        e_spk = F.normalize(z[:, :self.latent_split], p=1, dim=1)
        e_env = F.normalize(z[:, self.latent_split:], p=1, dim=1)
        
        if swap_codes and x_swap is not None:
            z_swap = self.encoder(x_swap)
            e_spk_swap = F.normalize(z_swap[:, :self.latent_split], p=1, dim=1)
            z_recombined = torch.cat((e_spk_swap, e_env), dim=1)
        else:
            z_recombined = torch.cat((e_spk, e_env), dim=1)
            
        return e_spk, e_env, self.decoder(z_recombined)

class Discriminator(nn.Module):
    def __init__(self, input_dim=256, output_dim=128, is_speaker=False, num_speakers=100):
        super(Discriminator, self).__init__()
        if is_speaker:
            self.net = nn.Linear(input_dim, num_speakers)
        else:
            self.net = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim),
                nn.ELU(),
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, output_dim)
            )
    def forward(self, x):
        return self.net(x)

class MAPCLoss(nn.Module):
    def forward(self, e_spk, e_env):
        e_spk_c = e_spk - e_spk.mean(dim=0, keepdim=True)
        e_env_c = e_env - e_env.mean(dim=0, keepdim=True)
        cov = torch.matmul(e_spk_c.t(), e_env_c) / (e_spk.size(0) - 1)
        std_spk = e_spk_c.std(dim=0, keepdim=True)
        std_env = e_env_c.std(dim=0, keepdim=True)
        correlation = cov / (torch.matmul(std_spk.t(), std_env) + 1e-8)
        return torch.abs(correlation).mean()

class DynamicLossWeighter(nn.Module):
    """Part D Improvement: Homoscedastic uncertainty weighting for 5 losses."""
    def __init__(self, num_losses=5):
        super(DynamicLossWeighter, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        
    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss